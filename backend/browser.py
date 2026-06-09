"""Playwright browser controller.

Runs a headless, persistent Chromium streamed to the dashboard over a WebSocket.
The frame is captured at a fixed viewport (config.VIEWPORT_W/H, device_scale_factor
1) so a click in the preview maps 1:1 to page CSS pixels. Manual takeover (click /
type / scroll) happens right in the dashboard preview.

Two responsibilities:
  observe()  -> read the page into a numbered list of interactive elements
                (text-based; Qwen3 has no vision) + draws an on-page overlay so
                the human sees exactly what the AI sees.
  act(d)     -> execute one decision dict (click / type / scroll / navigate / ...).

All page access goes through a single asyncio.Lock so the agent loop and the
control panel's screenshot endpoint never touch the page concurrently.
"""
from __future__ import annotations

import asyncio
import logging
import re

from playwright.async_api import async_playwright

import config

log = logging.getLogger(__name__)

# Injected into the page each observe(): tags visible interactive elements with a
# stable data-ai-index (so we can act on them by index) and paints a labelled
# overlay. Page text is captured BEFORE the overlay is added so it stays clean.
#
# The boxes are position:fixed (viewport coords) and a requestAnimationFrame loop
# re-pins each one to its element's LIVE getBoundingClientRect every frame. That
# way they track the element in real time no matter what moves it — window scroll,
# an inner overflow:auto container (LinkedIn feed / Tokopedia / most SPAs, where
# window.scrollY never changes), reflow, lazy-load, or CSS animation — instead of
# being frozen at the positions captured the moment observe() ran (which lagged:
# "bbox telat gak sesuai sama tombolnya").
_OBSERVE_JS = r"""
() => {
  const text = ((document.body && document.body.innerText) || '')
      .replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').slice(0, 4000);

  // Tear down the previous overlay AND its live-tracking loop/listeners first.
  if (window.__aiOverlayCleanup) { try { window.__aiOverlayCleanup(); } catch (e) {} }
  document.querySelectorAll('[data-ai-index]').forEach(e => e.removeAttribute('data-ai-index'));
  const prev = document.getElementById('__ai_overlay__'); if (prev) prev.remove();

  const sel = 'a,button,input,select,textarea,summary,[role=button],[role=link],' +
              '[role=tab],[role=checkbox],[role=radio],[role=menuitem],[role=option],' +
              '[onclick],[contenteditable=""],[contenteditable=true]';
  const nodes = Array.from(document.querySelectorAll(sel));
  const vw = window.innerWidth, vh = window.innerHeight;

  const overlay = document.createElement('div');
  overlay.id = '__ai_overlay__';
  // position:fixed → boxes are placed directly in viewport coords; the rAF loop
  // below keeps them pinned to each element's live rect (see header comment).
  overlay.style.cssText = 'position:fixed;top:0;left:0;width:0;height:0;' +
      'pointer-events:none;z-index:2147483647';

  const out = [];
  const tracked = [];  // {el, box} pairs the live loop re-pins each frame
  let idx = 0;
  for (const el of nodes) {
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') continue;  // ARIA-disabled too; ignore 'false'/null
    const r = el.getBoundingClientRect();
    if (r.width < 5 || r.height < 5) continue;
    if (r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || cs.opacity === '0') continue;

    el.setAttribute('data-ai-index', idx);
    let label = (el.getAttribute('aria-label') || el.innerText || el.value ||
                 el.getAttribute('placeholder') || el.getAttribute('title') ||
                 el.getAttribute('alt') || el.getAttribute('name') || '')
                 .replace(/\s+/g, ' ').trim().slice(0, 120);
    out.push({
      index: idx,
      tag: el.tagName.toLowerCase(),
      type: el.getAttribute('type') || '',
      label: label,
      w: Math.round(r.width), h: Math.round(r.height),  // size → lets the model tell a product card from a tiny icon (shot_of)
      href: el.href || el.getAttribute('href') || '',  // el.href is absolute for links
      role: el.getAttribute('role') || '',  // real control type behind a generic <div>/<span> (tab/menuitem/…)
      // Behavior token — what a CLICK will DO, so the text brain understands the control
      // BEFORE acting (it can't see it). Pure attribute reads (no layout), first match wins,
      // '' when nothing notable. Rendered compactly + omit-when-empty in llm._elements_block.
      act: (() => {
        if (el.target === '_blank' || el.getAttribute('target') === '_blank') return 'tab';
        if (el.tagName === 'SUMMARY') { const d = el.closest('details'); if (d) return d.open ? 'menu-open' : 'menu'; }
        const ex = el.getAttribute('aria-expanded');
        if (ex === 'true') return 'menu-open';   // already expanded → just read, don't re-click
        if (ex === 'false') return 'menu';        // collapsed → a click reveals it
        const hp = el.getAttribute('aria-haspopup'), ro = el.getAttribute('role') || '';
        if ((hp && hp !== 'false') || ro === 'menu' || ro === 'listbox' || ro === 'combobox') return 'menu';
        const pr = el.getAttribute('aria-pressed'), ck = el.getAttribute('aria-checked');
        if (pr === 'true' || ck === 'true') return 'tgl:on';
        if (pr === 'false' || ck === 'false') return 'tgl:off';   // 'mixed' → no token
        if (el.type === 'submit' || (el.tagName === 'BUTTON' && el.closest('form'))) return 'submit';
        const lab = (label || '').toLowerCase();
        if (/(…|see more|show more|read more|selengkapnya)\s*$/.test(lab)) return 'expand';
        return '';
      })()
    });

    const box = document.createElement('div');
    box.style.cssText = 'position:fixed;left:' + r.left + 'px;top:' + r.top + 'px;width:' +
        r.width + 'px;height:' + r.height + 'px;border:2px solid #E8FF3A;box-sizing:border-box;';
    const tagEl = document.createElement('div');
    tagEl.textContent = idx;
    tagEl.style.cssText = 'position:absolute;left:0;top:0;transform:translateY(-100%);' +
        'background:#E8FF3A;color:#000;font:bold 11px monospace;padding:0 3px;line-height:1.3;';
    box.appendChild(tagEl);
    overlay.appendChild(box);
    tracked.push({ el: el, box: box });

    idx++;
    if (idx >= 120) break;
  }
  if (document.body) document.body.appendChild(overlay);

  // ---- live tracking: pin each box to its element's current viewport rect -----
  let rafId = 0;
  const reposition = () => {
    const W = window.innerWidth, H = window.innerHeight;
    // Read ALL rects first, then write ALL styles — interleaving read/write would
    // force a synchronous re-layout per element (thrash) on every frame.
    const rects = new Array(tracked.length);
    for (let i = 0; i < tracked.length; i++) rects[i] = tracked[i].el.getBoundingClientRect();
    for (let i = 0; i < tracked.length; i++) {
      const rr = rects[i], s = tracked[i].box.style;
      // Hide boxes whose element scrolled out of view or got detached (0-size).
      if (rr.width < 1 || rr.height < 1 ||
          rr.bottom < 0 || rr.top > H || rr.right < 0 || rr.left > W) {
        if (s.display !== 'none') s.display = 'none';
        continue;
      }
      if (s.display === 'none') s.display = '';
      s.left = rr.left + 'px'; s.top = rr.top + 'px';
      s.width = rr.width + 'px'; s.height = rr.height + 'px';
    }
  };
  const tick = () => {
    if (!overlay.isConnected) { cleanup(); return; }
    reposition();
    rafId = requestAnimationFrame(tick);
  };
  // Capture-phase catches scrolls from inner containers too (scroll doesn't bubble,
  // but a capture-phase listener on window sees every scroll on the page). passive:
  // we only read rects, never preventDefault.
  const onMove = () => reposition();
  function cleanup() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    window.removeEventListener('scroll', onMove, true);
    window.removeEventListener('resize', onMove, true);
    if (window.__aiOverlayCleanup === cleanup) window.__aiOverlayCleanup = null;
  }
  window.addEventListener('scroll', onMove, { capture: true, passive: true });
  window.addEventListener('resize', onMove, { capture: true, passive: true });
  window.__aiOverlayCleanup = cleanup;
  rafId = requestAnimationFrame(tick);

  return {
    url: location.href, title: document.title, elements: out, text: text,
    scrollY: Math.round(window.scrollY), scrollH: Math.round(document.body ? document.body.scrollHeight : 0)
  };
}
"""


# Opaque schemes (no `://`) we must reject explicitly — they slip past the
# `scheme://` check below. Note we DON'T blanket-reject any `word:` prefix, so
# host:port like `localhost:8080` still works.
_BAD_SCHEME_RE = re.compile(
    r"^\s*(javascript|data|file|blob|vbscript|about|chrome|chrome-extension|ftp):",
    re.IGNORECASE,
)


# Scroll the RIGHT thing by `dy`: walk up from the page-centre element to the
# nearest real scroll container and scroll it; fall back to the window. Returns
# whether anything actually moved (so the agent knows when it hit the end). Setting
# scrollTop / window.scrollBy fires 'scroll' events, so infinite-scroll still loads.
_SCROLL_JS = r"""
(dy) => {
  const cx = (window.innerWidth / 2) | 0, cy = (window.innerHeight / 2) | 0;
  let n = document.elementFromPoint(cx, cy);
  while (n && n !== document.body && n !== document.documentElement) {
    const s = getComputedStyle(n);
    if (/(auto|scroll|overlay)/.test(s.overflowY) && n.scrollHeight > n.clientHeight + 4) {
      const t0 = n.scrollTop;
      n.scrollTop += dy;
      if (n.scrollTop !== t0) return true;
    }
    n = n.parentElement;
  }
  const y0 = window.scrollY;
  window.scrollBy(0, dy);
  return window.scrollY !== y0;
}
"""


def _normalize(url: str) -> str:
    u = (url or "").strip()
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", u)
    if m:
        # Block file:// / ftp:// etc. — only real web navigation.
        if m.group(1).lower() not in ("http", "https"):
            raise ValueError(f"URL scheme not allowed: {m.group(1)}://")
        return u
    if _BAD_SCHEME_RE.match(u):
        raise ValueError(f"URL scheme not allowed: {u.split(':', 1)[0]}")
    return "https://" + u


class Browser:
    # AI scroll speed → distance per scroll, animation steps/step-wait, and the
    # settle pause afterwards (slower = smaller jumps + longer waits so lazy content
    # loads and nothing is skipped).
    SCROLL_PROFILES = {
        "slow":   {"amount": 400,  "steps": 8, "wait": 90, "settle": 1100},
        "medium": {"amount": 700,  "steps": 6, "wait": 70, "settle": 500},
        "fast":   {"amount": 1100, "steps": 5, "wait": 45, "settle": 200},
    }

    def __init__(self) -> None:
        self._pw = None
        self._ctx = None
        self.page = None
        self._lock = asyncio.Lock()
        self._started = False
        # CDP screencast: the browser pushes JPEG frames as it renders (smooth,
        # low-CPU) instead of us polling page.screenshot. latest_frame_b64 is the
        # last frame (base64 jpeg); frame_seq bumps on each new frame.
        self.latest_frame_b64: str | None = None
        self.frame_seq = 0
        self._cdp = None
        self._cdp_page = None
        self._channel = None  # resolved browser channel (chrome / bundled)
        self.scroll_speed = config.SCROLL_SPEED  # slow | medium | fast
        self.scroll_delay = config.SCROLL_DELAY  # seconds; 0 = use the preset settle
        self.scroll_distance = config.SCROLL_DISTANCE  # px/scroll; 0 = use the preset

    async def start(self) -> None:
        if self._started:
            return
        self._pw = await async_playwright().start()
        launch_kwargs = dict(
            user_data_dir=config.USER_DATA_DIR,
            headless=config.HEADLESS,
            viewport={"width": config.VIEWPORT_W, "height": config.VIEWPORT_H},
            device_scale_factor=1,  # frame px == CSS px → 1:1 click mapping
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
            ignore_default_args=["--enable-automation"],
            user_agent=config.USER_AGENT or None,
        )
        self._ctx = await self._open(launch_kwargs)
        self.page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()

        # Chrome headless still puts "HeadlessChrome" in the UA, which anti-bot
        # systems (DataDome on Tokopedia etc.) block at the network layer. Strip it
        # — keeping the real version so client hints stay consistent — then relaunch
        # with the clean UA (context-level, so it applies to every request/tab).
        if not config.USER_AGENT:
            try:
                ua = await self.page.evaluate("() => navigator.userAgent")
                if "Headless" in ua:
                    launch_kwargs["user_agent"] = ua.replace("HeadlessChrome", "Chrome").replace("Headless", "")
                    await self._ctx.close()
                    self._ctx = await self._open(launch_kwargs)
                    self.page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
            except Exception as e:  # noqa: BLE001
                log.warning("UA cleanup skipped: %s", e)

        self._started = True
        log.info("Browser started (channel=%s, headless=%s, profile=%s)",
                 self._channel or "chromium", config.HEADLESS, config.USER_DATA_DIR)

    async def _open(self, kwargs):
        """Launch the persistent context, preferring the configured channel (real
        Chrome) and falling back to bundled Chromium. The resolved channel is cached
        in self._channel so a relaunch (UA cleanup) reuses the same one."""
        if self._channel is None:
            chan = config.BROWSER_CHANNEL
            if chan:
                try:
                    ctx = await self._pw.chromium.launch_persistent_context(channel=chan, **kwargs)
                    self._channel = chan
                    return ctx
                except Exception as e:  # noqa: BLE001
                    log.warning("Browser channel '%s' unavailable (%s) — using bundled Chromium", chan, e)
            self._channel = ""
        if self._channel:
            return await self._pw.chromium.launch_persistent_context(channel=self._channel, **kwargs)
        return await self._pw.chromium.launch_persistent_context(**kwargs)

    @property
    def started(self) -> bool:
        return self._started

    def current_url(self) -> str:
        """Live URL of the active page (sync property — safe without the lock).
        Reflects manual navigation immediately, unlike the loop's last_url."""
        try:
            return self.page.url if self.page else ""
        except Exception:  # noqa: BLE001
            return ""

    def _sync_page(self) -> None:
        """Follow popups / new tabs: keep `self.page` pointing at the newest page."""
        if self._ctx and self._ctx.pages:
            newest = self._ctx.pages[-1]
            if self.page is not newest:
                self.page = newest

    # ---- live-preview input forwarding (manual takeover in the dashboard) -----
    @staticmethod
    def _num(v, cap: float) -> float:
        """Coerce a client-supplied number, dropping NaN/inf, clamped to ±cap."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        return max(-cap, min(v, cap))

    def _xy(self, x, y) -> tuple[float, float]:
        return (max(0.0, self._num(x, config.VIEWPORT_W)),
                max(0.0, self._num(y, config.VIEWPORT_H)))

    async def apply_input(self, msg: dict) -> None:
        """Apply ONE manual input event. Caller MUST already hold self._lock and
        have verified the AI isn't running. All client-supplied values are clamped."""
        self._sync_page()
        t = msg.get("t")
        if t == "click":
            x, y = self._xy(msg.get("x"), msg.get("y"))
            btn = msg.get("button", "left")
            await self.page.mouse.click(
                x, y, button=btn if btn in ("left", "right", "middle") else "left",
                click_count=max(1, min(int(msg.get("clicks", 1) or 1), 3)))
        elif t == "move":
            x, y = self._xy(msg.get("x"), msg.get("y"))
            await self.page.mouse.move(x, y)
        elif t == "scroll":
            # Move to the cursor first so the wheel scrolls the content under it,
            # not whatever sits at (0,0) (often a sticky header that won't scroll).
            x, y = msg.get("x"), msg.get("y")
            if x is not None and y is not None:
                cx, cy = self._xy(x, y)
                await self.page.mouse.move(cx, cy)
            await self.page.mouse.wheel(self._num(msg.get("dx", 0), 5000),
                                        self._num(msg.get("dy", 0), 5000))
        elif t == "key":
            await self.page.keyboard.press(str(msg.get("key", ""))[:32])
        elif t == "text":
            await self.page.keyboard.type(str(msg.get("text", ""))[:2000])

    # ---- CDP screencast (the live stream) ------------------------------------
    def _on_screencast_frame(self, params) -> None:
        self.latest_frame_b64 = params.get("data")
        self.frame_seq += 1
        sid = params.get("sessionId")
        if sid is not None:
            asyncio.create_task(self._ack(sid))  # must ack or frames stop coming

    async def _ack(self, sid) -> None:
        try:
            if self._cdp:
                await self._cdp.send("Page.screencastFrameAck", {"sessionId": sid})
        except Exception:  # noqa: BLE001
            pass

    async def _stop_screencast(self) -> None:
        if self._cdp:
            try:
                await self._cdp.send("Page.stopScreencast")
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._cdp.detach()
            except Exception:  # noqa: BLE001
                pass
        self._cdp = None
        self._cdp_page = None

    async def stop_stream(self) -> None:
        """Stop the screencast (called when no dashboard is watching, to save CPU)."""
        await self._stop_screencast()

    async def ensure_screencast(self) -> None:
        """(Re)attach the CDP screencast to the current page. Cheap to call on a
        loop — it only re-attaches when the active page changes (new tab/popup)."""
        if not self._started:
            return
        self._sync_page()
        if self._cdp is not None and self._cdp_page is self.page:
            return
        await self._stop_screencast()
        try:
            cdp = await self._ctx.new_cdp_session(self.page)
            cdp.on("Page.screencastFrame", self._on_screencast_frame)
            mw = config.STREAM_MAX_WIDTH
            mh = max(1, round(mw / config.VIEWPORT_W * config.VIEWPORT_H))
            await cdp.send("Page.startScreencast", {
                "format": "jpeg", "quality": config.STREAM_QUALITY,
                "maxWidth": mw, "maxHeight": mh,
                "everyNthFrame": config.STREAM_EVERY_NTH,
            })
            self._cdp = cdp
            self._cdp_page = self.page
        except Exception as e:  # noqa: BLE001
            log.warning("screencast attach failed: %s", e)

    async def observe(self) -> dict:
        # Wait for load OUTSIDE the lock so a slow page can't hold it for 8s.
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        async with self._lock:
            self._sync_page()
            try:
                return await self.page.evaluate(_OBSERVE_JS)
            except Exception as e:  # noqa: BLE001
                log.warning("observe failed: %s", e)
                return {"url": self.page.url, "title": "", "elements": [], "text": "",
                        "scrollY": 0, "scrollH": 0}

    async def clear_overlay(self) -> None:
        async with self._lock:
            try:
                await self.page.evaluate(
                    "() => { if (window.__aiOverlayCleanup) { try { window.__aiOverlayCleanup(); } catch (e) {} }"
                    " const o = document.getElementById('__ai_overlay__'); if (o) o.remove(); }"
                )
            except Exception:
                pass

    async def _goto(self, url: str) -> None:
        await self.page.goto(_normalize(url), wait_until="domcontentloaded", timeout=30000)

    async def goto(self, url: str) -> None:
        async with self._lock:
            await self._goto(url)

    async def screenshot(self, quality: int = 55) -> bytes:
        async with self._lock:
            self._sync_page()
            return await self.page.screenshot(type="jpeg", quality=quality, full_page=False)

    async def marked_shot(self, quality: int | None = None) -> bytes:
        """JPEG of the current viewport WITH the numbered overlay boxes left visible
        — the Set-of-Marks image the vision model reads (numbers = element indices).
        The overlay is drawn by the most recent observe() and the rAF loop keeps it
        pinned, so the marks line up with the elements at capture time."""
        q = config.VISION_JPEG_QUALITY if quality is None else quality
        async with self._lock:
            self._sync_page()
            return await self.page.screenshot(type="jpeg", quality=q, full_page=False)

    # Given a too-small target element (e.g. the model picked a 17x17 wishlist icon
    # instead of the product card for a shot_of), climb to the NEAREST ancestor that
    # is at least min_w x min_h but not the whole grid/page (<= maxFrac of viewport),
    # and return its viewport-clamped clip rect. Null if no good ancestor → caller
    # falls back to the element itself. This is the auto safety-net for shot_of so a
    # wrong-but-small index still yields the product card, not a blank icon.
    _CARD_CLIP_JS = r"""
    (el, a) => {
      const vw = innerWidth, vh = innerHeight;
      let n = el, best = null;
      while (n && n !== document.body && n !== document.documentElement) {
        const r = n.getBoundingClientRect();
        if (r.width >= a.minW && r.height >= a.minH &&
            r.width <= vw * a.maxFW && r.height <= vh * a.maxFH) { best = r; break; }
        n = n.parentElement;
      }
      if (!best) return null;
      const x = Math.max(0, best.left), y = Math.max(0, best.top);
      const w = Math.min(best.right, vw) - x, h = Math.min(best.bottom, vh) - y;
      return (w < 2 || h < 2) ? null : { x: x, y: y, width: w, height: h };
    }
    """

    async def capture(self, index=None, region=None, full=False, min_w=0, min_h=0) -> bytes:
        """PNG screenshot of a specific element (by data-ai-index), a region clip
        {x,y,width,height}, the full page, or (default) the current viewport. The
        AI overlay boxes are hidden during the shot so they don't pollute it.

        min_w/min_h (used by shot_of): if the indexed element is smaller than this,
        climb to the nearest sizable ancestor (the product card) and clip that
        instead — so a mis-picked tiny icon still captures the real product."""
        async with self._lock:
            self._sync_page()
            await self.page.evaluate(
                "() => { const o = document.getElementById('__ai_overlay__'); if (o) o.style.display = 'none'; }"
            )
            try:
                if index is not None:
                    loc = self.page.locator(f'[data-ai-index="{int(index)}"]').first
                    if min_w or min_h:
                        try:
                            await loc.scroll_into_view_if_needed(timeout=2000)
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            clip = await loc.evaluate(
                                self._CARD_CLIP_JS,
                                {"minW": min_w, "minH": min_h, "maxFW": 0.85, "maxFH": 0.85},
                            )
                        except Exception:  # noqa: BLE001
                            clip = None
                        if clip and all(k in clip for k in ("x", "y", "width", "height")):
                            return await self.page.screenshot(type="png", clip=clip)
                    return await loc.screenshot(type="png", timeout=8000)
                if region and all(k in region for k in ("x", "y", "width", "height")):
                    clip = {k: float(region[k]) for k in ("x", "y", "width", "height")}
                    return await self.page.screenshot(type="png", clip=clip)
                return await self.page.screenshot(type="png", full_page=bool(full))
            finally:
                await self.page.evaluate(
                    "() => { const o = document.getElementById('__ai_overlay__'); if (o) o.style.display = ''; }"
                )

    async def _robust_click(self, loc) -> str:
        """Click that survives elements which RESOLVE but never become 'actionable'.
        On infinite-scroll feeds (LinkedIn) the constant reflow defeats Playwright's
        stability check, and toggles like 'see more' can fail its hit-test — so a
        plain click() just times out. We try a normal click first (realistic event
        sequence, handles navigation/focus), and on failure fall back to firing the
        element's OWN click() in the page — which bypasses stability AND hit-testing,
        so it reaches the exact element whether it was reflowing or covered (a force
        click can't: it would land on whatever sits on top). Returns the path used."""
        try:
            await loc.scroll_into_view_if_needed(timeout=2500)
        except Exception:  # noqa: BLE001
            pass
        try:
            await loc.click(timeout=4000)
            return "click"
        except Exception:  # noqa: BLE001
            pass
        # Fallback 1: fire the element's OWN click() in-page — bypasses stability AND
        # hit-testing, so it reaches the exact element whether it was reflowing or
        # covered. Single fire (a toggle flips exactly once). Wrapped in a timeout:
        # on a busy/navigating page evaluate() can otherwise hang the full 30s default
        # (seen on LinkedIn feeds) and stall the whole step.
        try:
            await asyncio.wait_for(loc.evaluate("el => el.click()"), timeout=4)
            return "dom-click"
        except Exception:  # noqa: BLE001
            pass
        # Fallback 2: a real mouse click at the element's centre — a different path
        # again (full pointer-event sequence some widgets require). Last resort; if the
        # element has no box it's gone, so raise a clear error the model can act on.
        box = await loc.bounding_box()
        if not box:
            raise RuntimeError("element is not clickable (no box — it may have detached or be hidden)")
        await self.page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return "coord-click"

    # Cheap page fingerprint to tell whether a click actually did anything: URL +
    # title + interactive-element count + scroll + text length. Compared before/after
    # a click; identical = the click had no visible effect (wrong/covered element).
    _FINGERPRINT_JS = (
        "() => { const b = document.body; return location.href + '|' + document.title"
        " + '|' + (b ? b.querySelectorAll('a,button,input,select,textarea,[role]').length : 0)"
        " + '|' + Math.round(window.scrollY) + '|' + (b ? b.innerText.length : 0); }"
    )

    async def _fingerprint(self) -> str | None:
        try:
            return await self.page.evaluate(self._FINGERPRINT_JS)
        except Exception:  # noqa: BLE001
            return None

    async def _settle(self, url0: str, floor_ms: int = 300, nav_ms: int = 600) -> None:
        """Navigation-aware settle after a click / Enter, replacing a blind fixed
        sleep. The common case — a click that focuses a field, opens a dropdown,
        toggles 'see more', ticks a filter — does NOT navigate, so a flat 700/800ms
        is pure waste; a real navigation can need longer. So:
          1. a small floor (let JS handlers + the overlay rAF paint);
          2. if a full-page navigation is committing, wait for domcontentloaded
             (returns instantly when nothing is loading);
          3. if the URL changed but no load fired (SPA route change — e.g. LinkedIn,
             the case _robust_click exists for), a brief extra settle for the new view.
        observe()'s own domcontentloaded wait then absorbs whatever remains. The
        url0 snapshot is what makes this safe: without it we'd risk observe() reading
        a stale/half-rendered DOM, and one stale read costs a whole ~5-10s step."""
        await self.page.wait_for_timeout(floor_ms)
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=nav_ms)
        except Exception:  # noqa: BLE001 — already loaded / nothing pending
            pass
        try:
            if self.page.url != url0:
                await self.page.wait_for_timeout(250)
        except Exception:  # noqa: BLE001 — detached page mid-navigation
            pass

    async def act(self, d: dict) -> str:
        async with self._lock:
            self._sync_page()
            a = d.get("action")

            if a == "navigate":
                await self.page.goto(_normalize(d.get("url", "")), wait_until="domcontentloaded", timeout=30000)
                return f"navigated to {d.get('url')}"

            if a == "click":
                i = int(d["index"])
                url0 = self.page.url
                fp0 = await self._fingerprint()
                loc = self.page.locator(f'[data-ai-index="{i}"]').first
                how = await self._robust_click(loc)
                await self._settle(url0)
                self._sync_page()
                msg = f"clicked [{i}]" + ("" if how == "click" else f" ({how})")
                # Tell the model when the click changed nothing — the #1 reason it gets
                # "stuck": it clicks a wrong/covered element, sees no error, and repeats.
                fp1 = await self._fingerprint()
                if fp0 is not None and fp0 == fp1:
                    msg += (" — but the page did NOT visibly change (same URL, elements & text). "
                            "The click likely hit the wrong or a covered element. Do NOT just "
                            "repeat it: use `look` to SEE the page, pick a DIFFERENT element index, "
                            "or scroll/close an overlay first.")
                return msg

            if a == "type":
                i = int(d["index"])
                text = str(d.get("text", ""))
                loc = self.page.locator(f'[data-ai-index="{i}"]').first
                await self._robust_click(loc)
                # fill() only works on input/textarea/contenteditable; on a button
                # or link it throws. Surface a clear error so the model retries with
                # a 'click' (the error is fed back via history).
                fillable = await loc.evaluate(
                    "el => { const t = el.tagName.toLowerCase();"
                    " return t === 'input' || t === 'textarea' || el.isContentEditable; }"
                )
                if not fillable:
                    raise ValueError(f"element [{i}] is not a text field — use 'click', not 'type'")
                await loc.fill(text)
                if d.get("submit"):
                    url0 = self.page.url
                    await loc.press("Enter")
                    await self._settle(url0)
                    self._sync_page()
                return f"typed into [{i}]: {text!r}" + (" + Enter" if d.get("submit") else "")

            if a == "scroll":
                prof = self.SCROLL_PROFILES.get(self.scroll_speed, self.SCROLL_PROFILES["medium"])
                # Amount precedence: the user's FIXED setting wins when set (so "set the
                # scroll px" is actually respected), else the model's adaptive amount,
                # else the speed preset. (0 / unset falls through to the next.)
                try:
                    amt = int(self.scroll_distance or d.get("amount") or prof["amount"])
                except (TypeError, ValueError):
                    amt = self.scroll_distance or prof["amount"]
                amt = max(50, min(abs(amt), 4000))
                if str(d.get("direction", "down")).lower() == "up":
                    amt = -amt
                # Stepped (visibly animated) scroll of the correct scroll container.
                moved = False
                for _ in range(prof["steps"]):
                    if await self.page.evaluate(_SCROLL_JS, amt / prof["steps"]):
                        moved = True
                    await self.page.wait_for_timeout(prof["wait"])
                # Settle pause precedence: the user's FIXED delay wins when set (so
                # "set the delay myself" is respected), else the model's own "wait" (s),
                # else the speed preset.
                if self.scroll_delay:
                    settle = int(min(max(self.scroll_delay, 0), 30) * 1000)
                elif d.get("wait") is not None:
                    try:
                        settle = int(min(max(float(d["wait"]), 0), 30) * 1000)
                    except (TypeError, ValueError):
                        settle = prof["settle"]
                else:
                    settle = prof["settle"]
                await self.page.wait_for_timeout(settle)
                return (f"scrolled {amt}px ({self.scroll_speed})"
                        + ("" if moved else " — page did NOT move (likely the end / nothing more to load)"))

            if a == "go_back":
                await self.page.go_back(timeout=15000)
                await self.page.wait_for_timeout(500)
                return "went back"

            if a == "wait":
                s = max(0.0, min(float(d.get("seconds", 1)), 10.0))
                await self.page.wait_for_timeout(int(s * 1000))
                return f"waited {s}s"

            if a == "key":
                key = str(d.get("key", "Enter"))
                await self.page.keyboard.press(key)
                await self.page.wait_for_timeout(400)
                return f"pressed {key}"

            return f"unknown action: {a!r}"

    async def close(self) -> None:
        try:
            await self._stop_screencast()
            if self._ctx:
                await self._ctx.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._started = False
