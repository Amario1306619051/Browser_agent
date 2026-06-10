"""Vision client — the agent's *eyes*.

The text model (llm.py) stays the decision-maker; this module is only called when
that model needs to actually SEE the page (visual layout matters, it can't tell
which element to use, or something is visually blocking the content). It sends a
Set-of-Marks screenshot — the live numbered overlay boxes from observe() — to a
vision-language model (OpenAI-compatible vLLM endpoint) and returns a short textual
observation that gets fed back into the text model's next decision.

Separate base_url / model from the text LLM (config.VISION_*), so the fast VL model
and the big reasoning model can live on different endpoints.
"""
from __future__ import annotations

import logging

from openai import OpenAI

import config

log = logging.getLogger(__name__)

_client_singleton: OpenAI | None = None


def enabled() -> bool:
    return bool(config.VISION_BASE_URL and config.VISION_MODEL)


def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI(
            api_key=config.VISION_API_KEY,
            base_url=config.VISION_BASE_URL,
            # Used on every step in always-on mode, so keep it tight — a screenshot
            # describe is a few seconds; a long hang would stall each step.
            timeout=30,
        )
    return _client_singleton


_SYSTEM = (
    "You are the EYES of a web-browsing agent. You receive a screenshot of the "
    "current browser viewport. Interactive elements are outlined with YELLOW boxes, "
    "each tagged with a NUMBER at its top-left corner — those numbers are the element "
    "indices the agent acts on (e.g. it can click number 5).\n\n"
    "Answer the agent's question about what is actually on screen, concretely:\n"
    "- Refer to elements by their NUMBER whenever you can.\n"
    "- If asked which element to use, name the exact number.\n"
    "- If a popup, modal, cookie banner, login wall, or overlay is covering the "
    "content, say so and give the number of the button that dismisses or passes it.\n"
    "- If the page is empty/blank/still loading, say that.\n"
    "Be specific and brief (under ~120 words). Plain text, no markdown."
)


def _elements_text(elements: list[dict], url: str = "") -> str:
    """Render the element list for the eyes using the SAME format the text brain reads
    (llm._elements_block: index, tag/role, label, (#k/N), WxH, href, [act]). Sharing one
    vocabulary means when the VL model says 'use number N' it maps to the exact element
    the brain will act on — no eyes/brain mismatch. No import cycle: llm imports config
    only, never vision. The 120-cap lives here (the brain caps elsewhere)."""
    import llm
    return llm._elements_block((elements or [])[:120], url)


def look(task: str, question: str, image_b64: str, elements: list[dict] | None = None,
         url: str = "") -> str:
    """Ask the VL model the question about the marked screenshot. Returns a short
    text observation (or an empty string on failure — the caller treats vision as
    best-effort and proceeds without it)."""
    if not enabled():
        return ""
    q = (question or "").strip() or "Describe what is on screen and how to proceed."
    user_text = (
        f"The agent's overall TASK: {task}\n"
        f"Current URL: {url}\n\n"
        f"Elements the DOM detected (number: tag \"label\" (#k/N) WxH -> href [behavior]):\n{_elements_text(elements, url)}\n\n"
        f"AGENT'S QUESTION: {q}"
    )
    try:
        resp = _client().chat.completions.create(
            model=config.VISION_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
            temperature=0.1,
            max_tokens=400,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — vision is best-effort; never break the loop
        log.warning("vision.look failed: %s", e)
        return ""
