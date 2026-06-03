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

# ===== Agent / browser tuning =====
# Hard cap on autonomous steps so a confused model can't loop forever.
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "30"))
# Headless by default — the browser is streamed into the dashboard (no separate
# window), and you take over right there in the live preview.
HEADLESS = os.getenv("AGENT_HEADLESS", "true").strip().lower() in ("1", "true", "yes")
# Fixed viewport so the streamed frame size is predictable (needed to map a click
# in the dashboard preview back to page coordinates).
VIEWPORT_W = int(os.getenv("AGENT_VIEWPORT_W", "1280"))
VIEWPORT_H = int(os.getenv("AGENT_VIEWPORT_H", "800"))
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
