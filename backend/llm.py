"""vLLM client (OpenAI-compatible endpoint). This is the *brain* of the browser
agent: given the current page state it decides the single next action as a JSON
object.
"""
from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

import config

log = logging.getLogger(__name__)

_client_singleton: OpenAI | None = None


def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI(
            api_key=config.VLLM_API_KEY,
            base_url=config.VLLM_BASE_URL,
            timeout=90,
        )
    return _client_singleton


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Qwen3 is a reasoning model: with thinking ON it spends minutes emitting a
# <think> block and can 504 the gateway. We disable it and instead ask the model
# to put a one-line "thought" inside the JSON.
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
_MAX_ATTEMPTS = 3

_SYSTEM = (
    "You are an autonomous web-browsing agent driving a REAL Chromium browser.\n"
    "Each turn you get the current page (URL, title, visible text) and a NUMBERED "
    "list of interactive elements. Reply with EXACTLY ONE JSON object for the next "
    "single action. No prose, no markdown fences — JSON only.\n\n"
    "Shape: {\"thought\": \"<one short sentence, in the user's language>\", "
    "\"action\": \"<name>\", ...args}\n\n"
    "Actions:\n"
    "- {\"action\":\"navigate\",\"url\":\"https://...\"}            open a URL\n"
    "- {\"action\":\"click\",\"index\":N}                           click element N\n"
    "- {\"action\":\"type\",\"index\":N,\"text\":\"...\",\"submit\":true}  type into field N (submit = press Enter)\n"
    "- {\"action\":\"scroll\",\"direction\":\"down|up\",\"amount\":600}    scroll (amount px is OPTIONAL — adapt it: small to nudge, larger to move faster)\n"
    "- {\"action\":\"go_back\"}                                      browser back\n"
    "- {\"action\":\"wait\",\"seconds\":2}                            wait for the page to settle\n"
    "- {\"action\":\"request_manual\",\"reason\":\"...\"}             hand control to the human\n"
    "- {\"action\":\"record_rows\",\"rows\":[{\"col\":\"val\",...}]}    collect data rows for a table/CSV/Excel\n"
    "- {\"action\":\"export\",\"format\":\"xlsx|csv\",\"filename\":\"...\",\"columns\":[...]}  write the collected rows to a file\n"
    "- {\"action\":\"screenshot\",\"index\":N,\"filename\":\"...\"}    save a PNG of element N (omit index = current view; add \"full\":true for the whole page)\n"
    "- {\"action\":\"done\",\"success\":true,\"answer\":\"<result for the user>\"}  task finished\n\n"
    "Rules:\n"
    "- If the task asks to compile/collect data into a table, list, CSV, or Excel: as you "
    "read each item (scroll / paginate as needed), call record_rows with one object per "
    "item, using EXACTLY the column names the user asked for as keys. Call record_rows "
    "across as many steps as you need, then call export once at the end (format 'xlsx' for "
    "Excel, 'csv' otherwise). Don't dump everything in one giant action — record in batches.\n"
    "- Be THOROUGH on 'collect / find all' tasks: keep scrolling and recording NEW items "
    "until you've gone through a good amount of the results (scroll several screens) or you "
    "stop finding new relevant ones. Do NOT export and call done after just a handful. If the "
    "user gave a target number, collect at least that many before finishing.\n"
    "- Fill EVERY requested column with the item's real visible value — never leave one blank.\n"
    "- For a link/URL column, set the value to {\"href_of\": N} where N is the element index "
    "whose link you want — the system fills in its EXACT full URL (no copy errors). Pick the "
    "link to the ITEM ITSELF: a post's own permalink (usually its TIMESTAMP link like '18h' / "
    "'2w' / 'edited'), a product's page, a job's page — NOT the author's name, their profile "
    "(/in/...), or a company page (/company/...). If truly no such element exists, copy the "
    "full href shown after '->'.\n"
    "- Only record items you haven't recorded yet — after scrolling, record the NEW items.\n"
    "- Record AT MOST 5 items per record_rows call (fewer is fine). Recording too many "
    "at once makes the JSON response too long and it gets cut off — use several smaller "
    "record_rows calls across steps instead.\n"
    "- Use ONLY element indices that appear in the current list. The list is rebuilt "
    "every turn — never reuse an old index.\n"
    "- One clear step at a time. After typing a search query, set submit:true.\n"
    "- If you see a CAPTCHA, Cloudflare 'verify you are human', a bot-check, or a "
    "login wall you cannot pass, DO NOT try to solve it — return request_manual with "
    "a short reason. A human will solve it and resume you.\n"
    "- If navigating to a site fails (error page, blank, or it won't load) DON'T just "
    "retry the same URL. Try another route: search for it on https://www.google.com "
    "(or use the site's own search), click the result, or go to a different but valid "
    "URL for the same site. If it still won't work after a couple of tries, request_manual.\n"
    "- NEVER perform payments, purchases, deletions, or other irreversible / sensitive "
    "submits on your own — return request_manual so the human can confirm.\n"
    "- ONLY use the screenshot action when the task EXPLICITLY asks to screenshot / "
    "capture / 'tangkap layar' something. For any other task, never take a screenshot.\n"
    "- Before acting, check RECENT ACTIONS: if the task is already accomplished "
    "(e.g. the screenshot was saved, the file was exported, the answer was found), "
    "call done immediately — NEVER repeat an action that already succeeded.\n"
    "- If a scroll result says the page did NOT move, you've hit the bottom (nothing more "
    "to load) — stop scrolling: export/finish or try something else, don't keep scrolling.\n"
    "- When the goal is achieved, return done with a clear answer for the user."
)


def _strip(s: str) -> str:
    s = _THINK_RE.sub("", s).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json(s: str) -> dict:
    """Pull the first balanced {...} object out of the model output. Brace
    counting is string-aware so braces inside JSON string values (e.g. a thought
    like 'click the {save} button') don't end the object early."""
    s = _strip(s)
    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in: {s[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    raise ValueError(f"unbalanced JSON in: {s[:200]!r}")


def _short_href(href: str) -> str:
    # Drop query string / fragment (mostly tracking params) so the model sees the
    # full, clean URL path instead of a path truncated mid-slug.
    href = href.split("#", 1)[0].split("?", 1)[0]
    return href[:200]


def _elements_block(elements: list[dict]) -> str:
    lines = []
    for e in elements:
        tag = e.get("tag", "")
        if e.get("type"):
            tag += " " + e["type"]
        label = (e.get("label") or "").strip()
        line = f'[{e.get("index")}] <{tag}> "{label}"'
        href = e.get("href") or ""
        if href:
            line += f" -> {_short_href(href)}"
        lines.append(line)
    return "\n".join(lines) or "(no interactive elements found)"


def _history_block(logs: list[dict]) -> str:
    # Include errors so the model can self-correct (e.g. after an out-of-range
    # index or a type into a non-fillable element).
    recent = [l for l in logs if l.get("kind") in ("action", "result", "manual", "error")][-12:]
    if not recent:
        return "(none yet)"
    return "\n".join(f"- {l['text']}" for l in recent)


def _validate(decision: dict, obs: dict) -> None:
    """Reject malformed decisions so the call retries instead of failing at the
    Playwright layer (bad index -> 8s timeout; missing url -> goto('')) ."""
    action = decision.get("action")
    if action in ("click", "type"):
        idx = decision.get("index")
        valid = {e.get("index") for e in obs.get("elements", [])}
        if isinstance(idx, bool) or not isinstance(idx, int) or idx not in valid:
            raise ValueError(f"action '{action}' has invalid/out-of-range index {idx!r}")
    if action == "navigate" and not str(decision.get("url", "")).strip():
        raise ValueError("action 'navigate' is missing 'url'")
    if action == "record_rows" and not isinstance(decision.get("rows"), list):
        raise ValueError("action 'record_rows' requires a 'rows' array")


def _memory_block(memory: list[dict] | None) -> str:
    if not memory:
        return ""
    lines = []
    for i, t in enumerate(memory, 1):
        result = (t.get("result") or "").strip().replace("\n", " ")[:600]
        lines.append(f"{i}. Task: {(t.get('task') or '').strip()[:300]}\n   Result: {result}")
    return (
        "CONVERSATION MEMORY — earlier tasks you completed in this thread (oldest first). "
        "Use it for context and to resolve references like 'it' / 'that one':\n"
        + "\n".join(lines)
        + "\n\n"
    )


def decide(task: str, obs: dict, logs: list[dict], memory: list[dict] | None = None) -> dict:
    """Ask the model for the next action. Returns the parsed JSON dict.
    Raises on repeated failure so the agent loop can log + skip the step."""
    user = (
        _memory_block(memory)
        + f"TASK: {task}\n\n"
        f"CURRENT PAGE\nURL: {obs.get('url','')}\nTitle: {obs.get('title','')}\n"
        f"scroll {obs.get('scrollY',0)} / {obs.get('scrollH',0)} px\n\n"
        f"VISIBLE PAGE TEXT (truncated):\n{(obs.get('text') or '')[:2500]}\n\n"
        f"INTERACTIVE ELEMENTS (index: tag \"label\"):\n{_elements_block(obs.get('elements', []))}\n\n"
        f"RECENT ACTIONS:\n{_history_block(logs)}\n\n"
        "Reply with ONE JSON action object now."
    )

    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = _client().chat.completions.create(
                model=config.VLLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                # Generous so a record_rows batch (rows of long names + URLs) is
                # never cut off mid-JSON, which would fail to parse.
                max_tokens=3000,
                extra_body=_NO_THINK,
            )
            raw = resp.choices[0].message.content or ""
            decision = _extract_json(raw)
            if not decision.get("action"):
                raise ValueError(f"missing 'action' in {decision!r}")
            _validate(decision, obs)
            return decision
        except Exception as e:  # noqa: BLE001 — cold start 504 / bad JSON: retry
            last = e
            log.warning("decide() failed (attempt %d/%d): %s", attempt + 1, _MAX_ATTEMPTS, e)
    raise last  # type: ignore[misc]
