"""Config loader. Reads from environment, with an optional `.env` in the
project root (browser_agent/.env). Point VLLM_BASE_URL / VLLM_MODEL at any
OpenAI-compatible endpoint (vLLM, etc.).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where exported CSV/XLSX files are written (and served from).
OUTPUT_DIR = BASE_DIR / "output"


def _load_dotenv() -> None:
    """Minimal .env loader (no python-dotenv dependency needed). Lines like
    KEY=VALUE; ignores blanks and # comments. Does not override real env vars."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

# ===== LLM (any OpenAI-compatible endpoint, e.g. vLLM) =====
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "your-model-name")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")

# ===== Vision model (optional; the agent's "eyes") =====
# An OpenAI-compatible VL endpoint. When set, the text model can call the `look`
# action to actually SEE the page (Set-of-Marks screenshot) — useful when the DOM
# text isn't enough (visual layout, ambiguous elements, a modal blocking content).
# Leave VISION_BASE_URL / VISION_MODEL empty to disable vision entirely.
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "").strip()
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", os.getenv("VLLM_API_KEY", "dummy"))
# JPEG quality of the screenshot sent to the VL model (legible marks vs payload).
VISION_JPEG_QUALITY = max(30, min(int(os.getenv("VISION_JPEG_QUALITY", "70")), 95))
# Cap the VL model's answer before it re-enters the text prompt (keeps it in line
# with the other bounded prompt inputs: page text 2500, memory 600, etc.).
VISION_MAX_CHARS = max(200, int(os.getenv("VISION_MAX_CHARS", "900")))

# ===== Agent / browser tuning =====
# Hard cap on autonomous steps so a confused model can't loop forever.
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "30"))
# Default AI scroll speed: slow | medium | fast (per-task override from the UI).
SCROLL_SPEED = os.getenv("AGENT_SCROLL_SPEED", "medium").strip().lower()
# Pause (seconds) after each AI scroll. 0 = use the speed preset's settle pause.
SCROLL_DELAY = float(os.getenv("AGENT_SCROLL_DELAY", "0") or 0)
# Fixed AI scroll distance in px per scroll (overrides the speed preset's amount).
# 0 = auto (use the speed preset / let the model decide). Per-task override in the UI.
SCROLL_DISTANCE = int(os.getenv("AGENT_SCROLL_DISTANCE", "0") or 0)
# Headless by default — the browser is streamed into the dashboard (no separate
# window), and you take over right there in the live preview.
HEADLESS = os.getenv("AGENT_HEADLESS", "true").strip().lower() in ("1", "true", "yes")
# Fixed viewport so the streamed frame size is predictable (needed to map a click
# in the dashboard preview back to page coordinates).
VIEWPORT_W = int(os.getenv("AGENT_VIEWPORT_W", "1280"))
VIEWPORT_H = int(os.getenv("AGENT_VIEWPORT_H", "800"))
# Live-stream tuning. Lower quality / higher everyNth = lighter on the CPU (helps on
# laptops / heavy pages); the screencast only captures while the dashboard is open.
STREAM_QUALITY = int(os.getenv("AGENT_STREAM_QUALITY", "45"))
STREAM_EVERY_NTH = max(1, int(os.getenv("AGENT_STREAM_EVERY_NTH", "2")))
# Downscale the streamed frame (the browser viewport stays VIEWPORT_W, so clicks
# still map correctly). Smaller = much smaller/faster frames, blurrier preview.
STREAM_MAX_WIDTH = max(200, min(int(os.getenv("AGENT_STREAM_MAX_WIDTH", str(VIEWPORT_W))), VIEWPORT_W))
# Prefer real Google Chrome — its (new) headless mode passes anti-bot checks that
# block Playwright's bundled "headless shell" (e.g. Tokopedia resets the HTTP/2
# connection). Empty = always use bundled Chromium. Falls back automatically if the
# channel isn't installed.
BROWSER_CHANNEL = os.getenv("AGENT_BROWSER_CHANNEL", "chrome").strip()
# Persistent Chrome profile so logins / Cloudflare clearance survive restarts.
USER_DATA_DIR = os.getenv("AGENT_USER_DATA_DIR", str(BASE_DIR / ".profile"))
# Optional custom UA (left blank = Playwright default Chromium UA).
USER_AGENT = os.getenv("AGENT_USER_AGENT", "").strip()
# ===== Short-term memory (per thread_id) =====
# Empty = SQLite (zero-setup, file below). Set to postgresql://user:pass@host/db
# to use Postgres instead (needs the 'psycopg' package).
DATABASE_URL = os.getenv("AGENT_DATABASE_URL", "").strip()
MEMORY_DB_PATH = BASE_DIR / "memory.db"
# How many recent tasks-in-a-thread to feed back to the model as context.
MEMORY_TURNS = int(os.getenv("AGENT_MEMORY_TURNS", "10"))

# Where the control panel listens.
def _port() -> int:
    try:
        p = int(os.getenv("AGENT_PORT", "8001"))
    except ValueError:
        return 8001
    return p if 0 <= p <= 65535 else 8001


PORT = _port()
