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
import logging
import re
import time

import config
import exporter
import llm
from browser import Browser

log = logging.getLogger(__name__)


def _fmt_action(d: dict) -> str:
    a = d.get("action")
    if a == "navigate":
        return f"navigate → {d.get('url')}"
    if a == "click":
        return f"click [{d.get('index')}]"
    if a == "type":
        return f'type [{d.get("index")}] "{d.get("text", "")}"' + (" + Enter" if d.get("submit") else "")
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
    async def start(self, task: str, start_url: str | None = None) -> None:
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
        self.data_rows = []
        self.data_columns = []
        self._row_seen = set()
        self.last_export = None
        self.shots = []
        if self.ai_enabled is None:
            self.ai_enabled = asyncio.Event()
        self.ai_enabled.set()
        self.state = "running"
        self._log("info", f"Task started: {self.task}")

        await self.browser.start()
        if start_url:
            self._log("action", f"navigate → {start_url}")
            try:
                await self.browser.goto(start_url)
            except Exception as e:  # noqa: BLE001
                self._log("error", f"Failed to open {start_url}: {e}")

        self._runner = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        try:
            for _ in range(config.MAX_STEPS):
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

                try:
                    decision = llm.decide(self.task, obs, self.logs)
                except Exception as e:  # noqa: BLE001
                    self._log("error", f"LLM failed to decide an action: {e}")
                    await asyncio.sleep(1.0)
                    continue

                if decision.get("thought"):
                    self._log("think", str(decision["thought"]))
                self._log("action", _fmt_action(decision))

                action = decision.get("action")

                if action == "done":
                    self._save_pending()  # don't lose data if the model forgot to export
                    self.result = str(decision.get("answer", "(done)"))
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
                            sig = tuple(sorted((str(k), str(v).strip()) for k, v in r.items()))
                            if sig in self._row_seen:  # dedupe re-records across scrolls
                                dup += 1
                                continue
                            self._row_seen.add(sig)
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

                # Sensitive-action guard: pause once for the human. On resume the
                # loop re-decides; if it picks the same flagged action again the ack
                # lets it through (reset below), so we never loop forever on it.
                if self._is_dangerous(decision, obs) and not self._safety_ack:
                    self.ai_enabled.clear()
                    self._safety_ack = True
                    self._log("manual", f"⏸ Sensitive action detected ({_fmt_action(decision)}) — AI paused for your confirmation. Click Resume AI if you really want to proceed.")
                    continue
                self._safety_ack = False

                try:
                    res = await self.browser.act(decision)
                    self._log("result", res)
                except Exception as e:  # noqa: BLE001
                    self._log("error", f"Action failed: {e}")
            else:
                self._save_pending()
                self.result = self.result or f"Auto-stopped: reached the {config.MAX_STEPS}-step limit before the task finished."
                self._log("info", self.result)

            if self._stop:
                self._save_pending()  # stopping mid-run still saves what was gathered
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
            "ai_enabled": bool(self.ai_enabled and self.ai_enabled.is_set()),
            # Live page URL (reflects manual navigation); falls back to the loop's last.
            "url": self.browser.current_url() or self.last_url,
            "title": self.last_title,
            "result": self.result,
            "data_rows": len(self.data_rows),
            "export": self.last_export,
            "shots": self.shots,
            "logs": self.logs[-100:],
        }
