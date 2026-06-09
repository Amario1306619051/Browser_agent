"""The agent session: a single autonomous observe -> think -> act loop, with a
human takeover toggle (Cloudflare / CAPTCHA / manual steps).

State machine:  idle -> running <-> paused -> done | error

The takeover toggle is an asyncio.Event `ai_enabled`:
  set()    = AI active  (loop runs)
  clear()  = manual mode (loop blocks at the top of the next step; the human
             drives the headed Chromium window directly, then resumes)

Only ONE task runs at a time — this is a personal, single-browser tool.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import time

import config
import exporter
import llm
import vision
from browser import Browser
from memory import store

log = logging.getLogger(__name__)


def _fmt_action(d: dict, obs: dict | None = None) -> str:
    a = d.get("action")

    def _lbl(idx) -> str:
        # Resolve an element index to its label from THIS step's observation, so a
        # logged action reads "click [5] \"Search\"" not a bare "click [5]". The
        # bracketed index goes stale next step (the list is rebuilt every observe),
        # so the label is what keeps the history line meaningful for self-checking.
        if not obs:
            return ""
        for e in obs.get("elements", []):
            if e.get("index") == idx:
                return (e.get("label") or "").strip()[:60]
        return ""

    if a == "navigate":
        return f"navigate → {d.get('url')}"
    if a == "click":
        lbl = _lbl(d.get("index"))
        return f"click [{d.get('index')}]" + (f' "{lbl}"' if lbl else "")
    if a == "type":
        lbl = _lbl(d.get("index"))
        tgt = f"[{d.get('index')}]" + (f' "{lbl}"' if lbl else "")
        return f'type {tgt} "{d.get("text", "")}"' + (" + Enter" if d.get("submit") else "")
    if a == "scroll":
        return f"scroll {d.get('direction', 'down')}"
    if a == "wait":
        return f"wait {d.get('seconds')}s"
    if a == "go_back":
        return "go back"
    if a == "request_manual":
        return f"request manual: {d.get('reason', '')}"
    if a == "record_rows":
        return f"record {len(d.get('rows') or [])} row(s)"
    if a == "export":
        return f"export {d.get('format', 'xlsx')}: {d.get('filename', 'export')}"
    if a == "screenshot":
        if d.get("index") is not None:
            return f"screenshot element [{d.get('index')}]"
        return "screenshot (full page)" if d.get("full") else "screenshot (view)"
    if a == "look":
        return f"look: {d.get('question', '')}"
    if a == "done":
        return f"done: {d.get('answer', '')}"
    return str(a)


class AgentSession:
    def __init__(self) -> None:
        self.browser = Browser()
        self.state = "idle"  # idle | running | paused | done | error
        self.task = ""
        self.step = 0
        self.logs: list[dict] = []
        self.result = ""
        self.last_url = ""
        self.last_title = ""
        self.ai_enabled: asyncio.Event | None = None
        self._stop = False
        self._runner: asyncio.Task | None = None
        self._safety_ack = False  # set when a sensitive action was already flagged
        # Data the agent collects during a run, for CSV/XLSX export.
        self.data_rows: list[dict] = []
        self.data_columns: list[str] = []
        self._row_seen: set = set()  # row signatures, for dedupe across scrolls
        self.last_export: dict | None = None
        self.shots: list[dict] = []  # saved screenshots ({filename, url})
        # Short-term memory: tasks done earlier in this thread_id (cross-task context).
        self.thread_id = ""
        self.thread_memory: list[dict] = []
        self.thread_count = 0
        self.unlimited = False  # ignore the MAX_STEPS cap for this run
        self.smart = True       # LLM thinking ON (smarter, slower)
        # Vision ("eyes"): let the text model call `look` to SEE the page via the VL
        # model. last_vision holds the most recent observation, tagged with its step
        # so decide() only feeds it forward while it's still fresh (and it's cleared
        # the moment an action changes the page, since its numbered refs go stale).
        # When ON, auto-look is coupled to the reasoning mode (see _needs_auto_look):
        # thinking OFF → look EVERY step (the eyes stand in for the missing brain);
        # thinking ON → look ON DEMAND only (sparse elements, a prior error, truncated
        # content). The model can still call `look` itself either way. last_vision
        # holds that observation; it's cleared after each act.
        self.vision = True
        self.last_vision = ""
        self._vision_step = -10
        # Circuit breaker: if the VL endpoint fails repeatedly, stop calling it for the
        # rest of the run so a dead endpoint doesn't add a timeout to every step.
        self._vision_fails = 0
        # No-progress guard: the direction of the last scroll that hit the end (page
        # didn't move). A repeat scroll the SAME way is short-circuited (see _loop).
        self._scroll_stuck_dir: str | None = None
        # No-progress guard for clicks: the index of the last click that changed nothing.
        # A repeat click on the SAME index is intercepted — force the eyes + a nudge —
        # instead of re-firing a click that already did nothing (the classic stuck loop).
        self._click_noeffect_idx: int | None = None

    # Sensitive actions we never auto-confirm — pause for a human first. The
    # phrase list covers English and Indonesian site labels to keep it useful on
    # local e-commerce; on resume the action is let through once (see _loop).
    _DANGER_RE = re.compile(
        r"(buy now|place order|pay now|complete (purchase|order)|confirm (payment|order|purchase)"
        r"|checkout|proceed to pay|subscribe|delete account|close account|deactivate|confirm delete"
        r"|bayar sekarang|beli sekarang|pesan sekarang|konfirmasi (pembayaran|pesanan)|hapus akun"
        r"|berlangganan)",
        re.IGNORECASE,
    )

    # The AI may only screenshot when the task explicitly asks for it — otherwise
    # the screenshot action is rejected (see _loop). EN + ID terms.
    _SCREENSHOT_RE = re.compile(
        r"screenshot|screen ?shot|screen ?capture|screen ?grab|snapshot|capture|"
        r"tangkap(?:an)? layar|potret layar|cuplikan layar|ss layar",
        re.IGNORECASE,
    )

    def _wants_screenshot(self) -> bool:
        return bool(self._SCREENSHOT_RE.search(self.task or ""))

    def _is_dangerous(self, decision: dict, obs: dict) -> bool:
        """True if a click/type targets an element whose label looks like a
        payment / purchase / account-deletion control."""
        if decision.get("action") not in ("click", "type"):
            return False
        idx = decision.get("index")
        for e in obs.get("elements", []):
            if e.get("index") == idx:
                return bool(self._DANGER_RE.search(e.get("label", "") or ""))
        return False

    def _export(self, filename: str, fmt: str, columns=None) -> dict:
        ref = exporter.write_table(
            self.data_rows, filename, fmt, columns or self.data_columns or None
        )
        self.last_export = ref
        return ref

    @staticmethod
    def _resolve_refs(row: dict, obs: dict) -> dict:
        """Resolve {"href_of": <element index>} values to that element's EXACT full
        href from the observation — so link columns get the real URL, no copy/
        truncation errors and no grabbing the wrong (visible-text) link."""
        by_idx = {e.get("index"): e for e in obs.get("elements", [])}
        out = {}
        for k, v in row.items():
            if isinstance(v, dict) and "href_of" in v:
                e = by_idx.get(v.get("href_of"))
                out[k] = (e.get("href") or "") if e else ""
            else:
                out[k] = v
        return out

    async def _resolve_shots(self, row: dict, obs: dict) -> dict:
        """Resolve {"shot_of": N} cells into a SAVED SCREENSHOT of element N — the
        "image column is a picture, not a URL" case (e.g. a product photo the user
        wants captured). Screenshots element N as PNG, writes it to output/, and
        replaces the cell with the file's /output/ URL: CSV shows the path, XLSX
        embeds the actual image (see exporter.write_table). Best-effort — a capture
        that fails leaves the cell empty rather than dropping the whole row.

        Async + page-touching, so it runs in the loop (not the sync _resolve_refs),
        and only on rows already confirmed NEW so we never screenshot a duplicate."""
        out = {}
        for k, v in row.items():
            if isinstance(v, dict) and "shot_of" in v:
                idx = v.get("shot_of")
                try:
                    # min 80x80 → if the model picked a tiny icon, capture() climbs to
                    # the nearest product-card-sized ancestor instead of a blank thumb.
                    png = await self.browser.capture(index=idx, min_w=80, min_h=80)
                    ref = exporter.save_image(png, f"img_{idx}")
                    out[k] = ref["url"]
                except Exception as e:  # noqa: BLE001 — bad index / detached node
                    self._log("error", f"shot_of [{idx}] failed: {e}")
                    out[k] = ""
            else:
                out[k] = v
        return out

    def _remember(self) -> None:
        """Persist this finished task + its result into the thread's memory."""
        if self.thread_id and self.result:
            store.append(self.thread_id, self.task, self.result)
            self.thread_memory.append({"task": self.task, "result": self.result, "ts": time.time()})
            self.thread_count = len(self.thread_memory)

    def _save_pending(self, fmt: str = "csv") -> None:
        """Write collected rows on any terminal path (done / stop / step-limit) so
        a run never loses data it already gathered."""
        if self.data_rows and not self.last_export:
            try:
                ref = self._export("export", fmt)
                self._log("file", f"auto-exported {ref['rows']} row(s) → {ref['filename']}")
            except Exception as e:  # noqa: BLE001
                self._log("error", f"auto-export failed: {e}")

    def export_now(self, fmt: str = "csv") -> dict | None:
        """Manual export, triggered from the panel. Works regardless of run state
        (idle / paused / stopped) as long as rows were collected."""
        if not self.data_rows:
            return None
        ref = self._export("export", fmt)
        self._log("file", f"exported {ref['rows']} row(s) → {ref['filename']}")
        return ref

    # ---- logging -------------------------------------------------------------
    def _log(self, kind: str, text: str, **extra) -> None:
        entry = {"ts": time.time(), "step": self.step, "kind": kind, "text": text}
        entry.update(extra)
        self.logs.append(entry)
        self.logs = self.logs[-300:]
        log.info("[%s] %s", kind, text)

    # ---- lifecycle -----------------------------------------------------------
    async def start(self, task: str, start_url: str | None = None, thread_id: str | None = None,
                    unlimited: bool = False, scroll_speed: str | None = None,
                    scroll_delay: float | None = None, smart: bool = True,
                    vision_on: bool = True, scroll_distance: int | None = None) -> None:
        if self.state in ("running", "paused"):
            raise RuntimeError("A task is already running. Stop it before starting a new one.")
        # A previous run may still be finishing its terminal step (state already
        # done/error/idle, task not yet awaited). Stop + await it so two _loop()
        # tasks never run concurrently (which would interleave stale element indices).
        if self._runner is not None and not self._runner.done():
            self._stop = True
            if self.ai_enabled:
                self.ai_enabled.set()
            try:
                await self._runner
            except Exception:  # noqa: BLE001
                pass
        self.task = task.strip()
        if not self.task:
            raise RuntimeError("Task is empty.")
        self.step = 0
        self.logs = []
        self.result = ""
        self._stop = False
        self._safety_ack = False
        self.unlimited = bool(unlimited)
        self.smart = bool(smart)
        self.vision = bool(vision_on) and vision.enabled()
        self.last_vision = ""
        self._vision_step = -10
        self._vision_fails = 0
        if scroll_speed in Browser.SCROLL_PROFILES:
            self.browser.scroll_speed = scroll_speed
        if scroll_delay is not None:
            try:
                self.browser.scroll_delay = max(0.0, float(scroll_delay))
            except (TypeError, ValueError):
                pass
        if scroll_distance is not None:
            try:
                self.browser.scroll_distance = max(0, min(int(scroll_distance), 4000))
            except (TypeError, ValueError):
                pass
        self.data_rows = []
        self.data_columns = []
        self._row_seen = set()
        self.last_export = None
        self.shots = []
        self._scroll_stuck_dir = None
        self._click_noeffect_idx = None
        # Load short-term memory for this thread (earlier tasks → context).
        self.thread_id = store.norm(thread_id)
        self.thread_memory = store.load(self.thread_id) if self.thread_id else []
        self.thread_count = len(self.thread_memory)
        if self.ai_enabled is None:
            self.ai_enabled = asyncio.Event()
        self.ai_enabled.set()
        self.state = "running"
        self._log("info", f"Task started: {self.task}")
        if self.thread_id:
            self._log("info", f"🧠 Thread '{self.thread_id}': recalling {self.thread_count} earlier task(s).")

        await self.browser.start()
        if start_url:
            self._log("action", f"navigate → {start_url}")
            try:
                await self.browser.goto(start_url)
            except Exception as e:  # noqa: BLE001
                self._log("error", f"Failed to open {start_url}: {e}")

        self._runner = asyncio.create_task(self._loop())

    # Expand/truncation controls whose presence means the on-screen text the task
    # needs is probably cut off — worth a look to find which NUMBER reveals it.
    _EXPAND_LABELS = (
        "see more", "show more", "read more", "…more", "...more", "more",
        "see full text", "lihat selengkapnya", "selengkapnya", "baca selengkapnya",
        "tampilkan lebih", "lihat lainnya",
    )

    def _needs_auto_look(self, obs: dict) -> bool:
        """Whether to spend an auto vision call THIS step. Vision is the agent's
        eyes, but each call is a whole VL inference (~2-5s on a real page).

        Coupled to the reasoning mode:
          - thinking OFF (smart=False): the cheap no-reason decision leans hard on
            the eyes, so look EVERY step to compensate for the missing brain;
          - thinking ON (smart=True): on-demand only — the reasoning brain doesn't
            need a fresh picture every step, so we look only when the DOM text is
            likely NOT enough on its own:
              1. the element list is near-empty (blank page / canvas / iframe-app /
                 SPA still mounting) — the picture is the only way forward;
              2. the previous act errored — let the eyes help diagnose / recover;
              3. content the task needs is truncated behind an expand control.
        Either way the model keeps the explicit `look` action, so it can still ask
        for eyes whenever the text genuinely isn't enough."""
        if not self.smart:
            return True
        els = obs.get("elements") or []
        if len(els) < 3:
            return True
        last = next((l for l in reversed(self.logs) if l.get("kind") in ("result", "error")), None)
        if last is not None and last.get("kind") == "error":
            return True
        text = (obs.get("text") or "").rstrip()
        if text.endswith("…") or text.endswith("..."):
            return True
        for e in els:
            if (e.get("label") or "").strip().lower() in self._EXPAND_LABELS:
                return True
        return False

    async def _do_look(self, question: str, obs: dict) -> bool:
        """Capture a Set-of-Marks screenshot (numbered overlay) and ask the VL model
        `question`. The answer is stored as last_vision and fed into the next
        decide(). Best-effort — never raises into the loop."""
        if not self.vision:
            return False
        try:
            png = await self.browser.marked_shot()
        except Exception as e:  # noqa: BLE001
            self._log("error", f"vision screenshot failed: {e}")
            return False
        b64 = base64.b64encode(png).decode()
        # vision.look is a blocking HTTP call — run it off the event loop so the live
        # stream and status endpoint stay responsive while the VL model thinks.
        ans = await asyncio.to_thread(
            vision.look, self.task, question, b64, obs.get("elements"), obs.get("url", "")
        )
        # Cap the note before it re-enters the text prompt (every other prompt input
        # is bounded too) so a runaway VL response can't blow up the next call.
        ans = (ans or "").strip()[:config.VISION_MAX_CHARS]
        if ans:
            self._vision_fails = 0
            self.last_vision = ans
            self._vision_step = self.step
            self._log("vision", ans)
            return True
        # No answer (timeout / endpoint down / empty). After a couple in a row, stop
        # calling the VL model for the rest of this run so it can't stall every step.
        self._vision_fails += 1
        if self._vision_fails >= 2:
            self.vision = False
            self._log("error", "👁 vision disabled for this run — the VL endpoint isn't responding. Continuing DOM-only.")
        else:
            self._log("error", "vision returned nothing — proceeding with the DOM text only")
        return False

    async def _loop(self) -> None:
        try:
            # Unlimited mode: huge cap so the loop runs until done / stop / error.
            cap = 10 ** 9 if self.unlimited else config.MAX_STEPS
            for _ in range(cap):
                # Manual-mode / pause gate. Loop only blocks here, between steps,
                # so the page is never touched while the human drives it.
                if not self.ai_enabled.is_set():
                    self.state = "paused"
                    await self.browser.clear_overlay()
                    await self.ai_enabled.wait()
                    if self._stop:
                        break
                    self.state = "running"
                    self._log("info", "▶ AI mode resumed.")
                if self._stop:
                    break

                self.step += 1
                obs = await self.browser.observe()
                self.last_url, self.last_title = obs.get("url", ""), obs.get("title", "")

                # On-demand eyes: only spend a vision call when the DOM text is likely
                # insufficient (sparse/odd element list, the last act errored, or content
                # is truncated behind an expand control) — not blindly every step, which
                # doubled per-step latency for no gain on DOM-clear pages. The model can
                # still request `look` itself for anything else. The circuit breaker in
                # _do_look turns vision off if the VL endpoint stops answering.
                if self.vision and self._needs_auto_look(obs):
                    await self._do_look(
                        f"For this task: {self.task}\nBriefly (1-3 sentences): the page's "
                        "state; is any text the task needs truncated, and which NUMBER "
                        "expands it ('…more'/'see more'); is anything (modal, popup, cookie "
                        "banner, login) covering the content, and which NUMBER dismisses it; "
                        "which numbered elements matter for the next step.",
                        obs,
                    )

                # Feed this step's fresh visual note into the decision (cleared after the
                # act, so a note whose numbered refs predate a page change is never reused).
                vnote = self.last_vision if (self.last_vision and self.step - self._vision_step <= 1) else ""
                # Reasoning stays full (smart=ON) — it's the brain; cutting it made the
                # agent noticeably worse at picking the right action. The speedup comes
                # from on-demand vision above, not from dumbing the decision down.
                try:
                    decision = llm.decide(self.task, obs, self.logs, self.thread_memory,
                                          self.smart, vnote, self.vision)
                except Exception as e:  # noqa: BLE001
                    self._log("error", f"LLM failed to decide an action: {e}")
                    await asyncio.sleep(1.0)
                    continue

                if decision.get("thought"):
                    self._log("think", str(decision["thought"]))
                self._log("action", _fmt_action(decision, obs))

                action = decision.get("action")

                if action == "done":
                    self._save_pending()  # don't lose data if the model forgot to export
                    self.result = str(decision.get("answer", "(done)"))
                    self._remember()
                    self.state = "done"
                    self._log("done", self.result)
                    return

                if action == "request_manual":
                    self.ai_enabled.clear()
                    self._log("manual", f"⏸ AI requested manual takeover: {decision.get('reason', '')}")
                    continue  # top of loop handles overlay-clear + wait

                if action == "record_rows":
                    rows = decision.get("rows")
                    if isinstance(rows, list):
                        new = dup = 0
                        for r in rows:
                            if not isinstance(r, dict):
                                continue
                            r = self._resolve_refs(r, obs)  # {"href_of": N} → real URL
                            # Dedup on the STABLE text columns only — exclude {"shot_of": N}
                            # image cells, whose element index drifts across scrolls and
                            # whose saved filename is unique per capture (would defeat dedup).
                            sig = tuple(sorted(
                                (str(k), str(v).strip()) for k, v in r.items()
                                if not (isinstance(v, dict) and "shot_of" in v)
                            ))
                            if sig in self._row_seen:  # dedupe re-records across scrolls
                                dup += 1
                                continue
                            self._row_seen.add(sig)
                            # Only NOW (row confirmed new) capture screenshots, so we never
                            # screenshot a duplicate. {"shot_of": N} → saved /output/ image.
                            r = await self._resolve_shots(r, obs)
                            self.data_rows.append(r)
                            for k in r.keys():
                                if str(k) not in self.data_columns:
                                    self.data_columns.append(str(k))
                            new += 1
                        msg = f"recorded {new} new row(s) (total {len(self.data_rows)}"
                        msg += f", skipped {dup} dupe(s))" if dup else ")"
                        self._log("result", msg)
                    else:
                        self._log("error", "record_rows needs a 'rows' list")
                    continue

                if action == "export":
                    try:
                        ref = self._export(
                            decision.get("filename") or "export",
                            decision.get("format") or "xlsx",
                            decision.get("columns"),
                        )
                        self._log("file", f"exported {ref['rows']} row(s) → {ref['filename']}")
                    except Exception as e:  # noqa: BLE001
                        self._log("error", f"export failed: {e}")
                    continue

                if action == "screenshot":
                    if not self._wants_screenshot():
                        # Hard guard: never screenshot unless the task asked for it.
                        # Logged as an error so the model sees it and stops trying.
                        self._log("error", "screenshot rejected: the task did not ask for a screenshot — do not use the screenshot action")
                        continue
                    try:
                        png = await self.browser.capture(
                            decision.get("index"), decision.get("region"), decision.get("full")
                        )
                        ref = exporter.save_image(png, decision.get("filename") or "shot")
                        self.shots.append(ref)
                        self._log("file", f"screenshot saved → {ref['filename']}")
                    except Exception as e:  # noqa: BLE001
                        self._log("error", f"screenshot failed: {e}")
                    continue

                if action == "look":
                    if not self.vision:
                        self._log("error", "look rejected: vision is off — rely on the DOM text + element list")
                    else:
                        await self._do_look(decision.get("question", ""), obs)
                    continue

                # Sensitive-action guard: pause once for the human. On resume the
                # loop re-decides; if it picks the same flagged action again the ack
                # lets it through (reset below), so we never loop forever on it.
                if self._is_dangerous(decision, obs) and not self._safety_ack:
                    self.ai_enabled.clear()
                    self._safety_ack = True
                    self._log("manual", f"⏸ Sensitive action detected ({_fmt_action(decision, obs)}) — AI paused for your confirmation. Click Resume AI if you really want to proceed.")
                    continue
                self._safety_ack = False

                # No-progress guard: if the previous scroll THIS direction already hit
                # the end (browser.act reported the page didn't move), don't burn another
                # scroll + settle wait re-confirming it — nudge the model to do something
                # else. Hard evidence, so near-zero false positives (a scroll up resets).
                if action == "scroll":
                    dirn = str(decision.get("direction", "down")).lower()
                    if self._scroll_stuck_dir == dirn:
                        self._log("result", f"skipped scroll {dirn}: already at the end last time (page did not move) — "
                                            "do something else now (expand/click an item, paginate, export, or done).")
                        continue

                # No-progress guard for clicks: the model is about to re-click the exact
                # element whose last click changed nothing → that path is a dead loop.
                # Intercept it: don't re-fire; force the eyes (if on) so next step the
                # model can SEE what's wrong, and nudge it to pick a different element.
                if action == "click" and decision.get("index") == self._click_noeffect_idx:
                    self._log("result", f"skipped re-click [{decision.get('index')}]: the last click on it "
                                        "changed nothing — pick a DIFFERENT element, close any overlay, or scroll.")
                    self._click_noeffect_idx = None  # one-shot, so we don't loop on the guard itself
                    if self.vision:
                        await self._do_look(
                            f"For this task: {self.task}\nClicking element "
                            f"[{decision.get('index')}] did nothing. What is actually there — is it "
                            "covered by a popup/modal/overlay (which NUMBER closes it)? Which NUMBER "
                            "is the correct element to click instead?",
                            obs,
                        )
                    continue

                try:
                    res = await self.browser.act(decision)
                    self._log("result", res)
                    # The act changed the page → the element indices the vision note
                    # referenced are now stale. Drop it so it can't mislead next step.
                    self.last_vision = ""
                    # Track whether a scroll hit the end, so the next same-direction
                    # scroll is short-circuited above (any other action resets it).
                    if action == "scroll":
                        self._scroll_stuck_dir = (str(decision.get("direction", "down")).lower()
                                                  if "— page did NOT move" in res else None)
                    else:
                        self._scroll_stuck_dir = None
                    # Remember a click that changed nothing, so an immediate re-click of
                    # the SAME index is intercepted above (any other action clears it).
                    if action == "click":
                        self._click_noeffect_idx = (decision.get("index")
                                                    if "did NOT visibly change" in res else None)
                    else:
                        self._click_noeffect_idx = None
                except Exception as e:  # noqa: BLE001
                    self._log("error", f"Action failed: {e}")
            else:
                self._save_pending()
                self.result = self.result or f"Auto-stopped: reached the {config.MAX_STEPS}-step limit before the task finished."
                self._remember()
                self._log("info", self.result)

            if self._stop:
                self._save_pending()  # stopping mid-run still saves what was gathered
                if not self.result:
                    self.result = (f"⏹ Stopped — collected {len(self.data_rows)} row(s)."
                                   if self.data_rows else "⏹ Stopped before finishing.")
                self._remember()  # keep the (partial) task in the chat history
                self.state = "idle"
                self._log("info", "Stopped by user.")
            elif self.state not in ("done", "error"):
                self.state = "done"
        except Exception as e:  # noqa: BLE001
            self.state = "error"
            self._log("error", f"Loop error: {e}")

    # ---- controls (called from API; flags only, no page access) --------------
    def pause(self) -> None:
        if self.ai_enabled:
            self.ai_enabled.clear()
        if self.state == "running":
            self.state = "paused"
        self._log("manual", "✋ Manual mode active — AI paused. Do what you need in the browser window, then click Resume AI.")

    def resume(self) -> None:
        if self.ai_enabled:
            self.ai_enabled.set()

    async def stop(self) -> None:
        self._stop = True
        if self.ai_enabled:
            self.ai_enabled.set()  # unblock a paused loop so it can see _stop
        self._log("info", "Stop requested…")

    # ---- manual takeover in the dashboard preview ----------------------------
    async def manual_input(self, msg: dict) -> None:
        """Forward one dashboard input event to the page — only when the AI isn't
        running. The state is re-checked INSIDE the lock to close the TOCTOU with
        the agent loop (which acquires the same lock to act)."""
        if not self.browser.started or self.state == "running":
            return
        async with self.browser._lock:
            if self.state == "running":
                return
            try:
                await self.browser.apply_input(msg)
            except Exception:  # noqa: BLE001 — detached page / bad event; drop it
                pass

    async def manual_goto(self, url: str) -> None:
        if not self.browser.started:
            raise RuntimeError("no browser — start a task first")
        async with self.browser._lock:
            if self.state == "running":
                raise RuntimeError("AI is running — take over first")
            await self.browser._goto(url)

    def status(self) -> dict:
        return {
            "state": self.state,
            "task": self.task,
            "step": self.step,
            "max_steps": config.MAX_STEPS,
            "unlimited": self.unlimited,
            "vision": self.vision,
            "ai_enabled": bool(self.ai_enabled and self.ai_enabled.is_set()),
            # Live page URL (reflects manual navigation); falls back to the loop's last.
            "url": self.browser.current_url() or self.last_url,
            "title": self.last_title,
            "result": self.result,
            "data_rows": len(self.data_rows),
            "export": self.last_export,
            "shots": self.shots,
            "thread_id": self.thread_id,
            "thread_count": self.thread_count,
            "logs": self.logs[-100:],
        }
