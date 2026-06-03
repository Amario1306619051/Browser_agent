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
    "- {\"action\":\"scroll\",\"direction\":\"down|up\",\"amount\":600}    scroll the page\n"
    "- {\"action\":\"go_back\"}                                      browser back\n"
    "- {\"action\":\"wait\",\"seconds\":2}                            wait for the page to settle\n"
    "- {\"action\":\"request_manual\",\"reason\":\"...\"}             hand control to the human\n"
    "- {\"action\":\"done\",\"success\":true,\"answer\":\"<result for the user>\"}  task finished\n\n"
    "Rules:\n"
    "- Use ONLY element indices that appear in the current list. The list is rebuilt "
    "every turn — never reuse an old index.\n"
    "- One clear step at a time. After typing a search query, set submit:true.\n"
    "- If you see a CAPTCHA, Cloudflare 'verify you are human', a bot-check, or a "
    "login wall you cannot pass, DO NOT try to solve it — return request_manual with "
    "a short reason. A human will solve it and resume you.\n"
    "- NEVER perform payments, purchases, deletions, or other irreversible / sensitive "
    "submits on your own — return request_manual so the human can confirm.\n"
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
            line += f" -> {href[:60]}"
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


def decide(task: str, obs: dict, logs: list[dict]) -> dict:
    """Ask the model for the next action. Returns the parsed JSON dict.
    Raises on repeated failure so the agent loop can log + skip the step."""
    user = (
        f"TASK: {task}\n\n"
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
                max_tokens=700,
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
