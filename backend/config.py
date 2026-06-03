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
# Headed by default — the human needs to SEE the browser to take over (Cloudflare etc.).
HEADLESS = os.getenv("AGENT_HEADLESS", "false").strip().lower() in ("1", "true", "yes")
# Persistent Chrome profile so logins / Cloudflare clearance survive restarts.
USER_DATA_DIR = os.getenv("AGENT_USER_DATA_DIR", str(BASE_DIR / ".profile"))
# Optional custom UA (left blank = Playwright default Chromium UA).
USER_AGENT = os.getenv("AGENT_USER_AGENT", "").strip()
# Where the control panel listens.
def _port() -> int:
    try:
        p = int(os.getenv("AGENT_PORT", "8001"))
    except ValueError:
        return 8001
    return p if 0 <= p <= 65535 else 8001


PORT = _port()
