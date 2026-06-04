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
_OBSERVE_JS = r"""
() => {
  const text = ((document.body && document.body.innerText) || '')
      .replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').slice(0, 4000);

  document.querySelectorAll('[data-ai-index]').forEach(e => e.removeAttribute('data-ai-index'));
  const prev = document.getElementById('__ai_overlay__'); if (prev) prev.remove();

  const sel = 'a,button,input,select,textarea,summary,[role=button],[role=link],' +
              '[role=tab],[role=checkbox],[role=radio],[role=menuitem],[role=option],' +
              '[onclick],[contenteditable=""],[contenteditable=true]';
  const nodes = Array.from(document.querySelectorAll(sel));
  const vw = window.innerWidth, vh = window.innerHeight;

  const overlay = document.createElement('div');
  overlay.id = '__ai_overlay__';
  overlay.style.cssText = 'position:fixed;left:0;top:0;width:100%;height:100%;' +
      'pointer-events:none;z-index:2147483647';

  const out = [];
  let idx = 0;
  for (const el of nodes) {
    if (el.disabled) continue;
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
      href: el.getAttribute('href') || ''
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

    idx++;
    if (idx >= 120) break;
  }
  if (document.body) document.body.appendChild(overlay);

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
                    "() => { const o = document.getElementById('__ai_overlay__'); if (o) o.remove(); }"
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

    async def capture(self, index=None, region=None, full=False) -> bytes:
        """PNG screenshot of a specific element (by data-ai-index), a region clip
        {x,y,width,height}, the full page, or (default) the current viewport. The
        AI overlay boxes are hidden during the shot so they don't pollute it."""
        async with self._lock:
            self._sync_page()
            await self.page.evaluate(
                "() => { const o = document.getElementById('__ai_overlay__'); if (o) o.style.display = 'none'; }"
            )
            try:
                if index is not None:
                    loc = self.page.locator(f'[data-ai-index="{int(index)}"]').first
                    return await loc.screenshot(type="png", timeout=8000)
                if region and all(k in region for k in ("x", "y", "width", "height")):
                    clip = {k: float(region[k]) for k in ("x", "y", "width", "height")}
                    return await self.page.screenshot(type="png", clip=clip)
                return await self.page.screenshot(type="png", full_page=bool(full))
            finally:
                await self.page.evaluate(
                    "() => { const o = document.getElementById('__ai_overlay__'); if (o) o.style.display = ''; }"
                )

    async def act(self, d: dict) -> str:
        async with self._lock:
            self._sync_page()
            a = d.get("action")

            if a == "navigate":
                await self.page.goto(_normalize(d.get("url", "")), wait_until="domcontentloaded", timeout=30000)
                return f"navigated to {d.get('url')}"

            if a == "click":
                i = int(d["index"])
                await self.page.locator(f'[data-ai-index="{i}"]').first.click(timeout=8000)
                await self.page.wait_for_timeout(700)
                self._sync_page()
                return f"clicked [{i}]"

            if a == "type":
                i = int(d["index"])
                text = str(d.get("text", ""))
                loc = self.page.locator(f'[data-ai-index="{i}"]').first
                await loc.click(timeout=8000)
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
                    await loc.press("Enter")
                    await self.page.wait_for_timeout(800)
                    self._sync_page()
                return f"typed into [{i}]: {text!r}" + (" + Enter" if d.get("submit") else "")

            if a == "scroll":
                amt = int(d.get("amount", 600))
                if str(d.get("direction", "down")).lower() == "up":
                    amt = -abs(amt)
                # Smooth (animated) scroll so you can actually watch it move in the
                # live preview instead of it teleporting to the new position.
                await self.page.evaluate(
                    "(y) => window.scrollBy({ top: y, left: 0, behavior: 'smooth' })", amt)
                await self.page.wait_for_timeout(650)  # let the animation play + stream
                return f"scrolled {amt}px"

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
