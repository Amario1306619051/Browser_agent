"""Playwright browser controller.

Runs a HEADED, persistent Chromium so the human can watch the agent and take
over directly in the window (e.g. to solve a Cloudflare / CAPTCHA challenge).

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

    async def start(self) -> None:
        if self._started:
            return
        self._pw = await async_playwright().start()
        self._ctx = await self._pw.chromium.launch_persistent_context(
            user_data_dir=config.USER_DATA_DIR,
            headless=config.HEADLESS,
            no_viewport=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
                "--no-sandbox",
            ],
            ignore_default_args=["--enable-automation"],
            user_agent=config.USER_AGENT or None,
        )
        self.page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
        self._started = True
        log.info("Browser started (headless=%s, profile=%s)", config.HEADLESS, config.USER_DATA_DIR)

    def _sync_page(self) -> None:
        """Follow popups / new tabs: keep `self.page` pointing at the newest page."""
        if self._ctx and self._ctx.pages:
            newest = self._ctx.pages[-1]
            if self.page is not newest:
                self.page = newest

    async def observe(self) -> dict:
        async with self._lock:
            self._sync_page()
            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=8000)
            except Exception:
                pass
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

    async def goto(self, url: str) -> None:
        async with self._lock:
            await self.page.goto(_normalize(url), wait_until="domcontentloaded", timeout=30000)

    async def screenshot(self) -> bytes:
        async with self._lock:
            self._sync_page()
            return await self.page.screenshot(type="jpeg", quality=55, full_page=False)

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
                await self.page.evaluate("(y) => window.scrollBy(0, y)", amt)
                await self.page.wait_for_timeout(300)
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
            if self._ctx:
                await self._ctx.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._started = False
