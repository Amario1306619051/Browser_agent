# AI Browser Agent

A browser driven by an LLM through a **prompt**. You give it a task (e.g. "find
the cheapest iPhone 15 on Amazon"), and the AI opens pages, types, clicks, and
scrolls until it's done. It uses any OpenAI-compatible LLM endpoint (vLLM,
Ollama, llama.cpp, …) as its brain.

Key feature: a **Take Over (Manual)** button. When you hit a Cloudflare / CAPTCHA
/ login wall, click it — the AI pauses, you solve it yourself in the real Chromium
window, then click **Resume AI** to continue the task from where it left off.

```
[Control panel @ localhost:8001] ──HTTP──> [FastAPI] ──> [Agent loop] ──> [Playwright / headed Chromium]
        prompt · log · preview           start/pause/resume/stop      observe → LLM → act
```

## How it works

1. **observe** — JS is injected into the page to list every visible interactive
   element, number them, and draw a yellow labelled overlay (so you see exactly
   what the AI "sees").
2. **think** — the element list + page text is sent to the LLM, which returns a
   single JSON action.
3. **act** — Playwright executes it: click / type / scroll / navigate / etc.
4. Repeat until `done`, the `AGENT_MAX_STEPS` cap, or you stop it.

If the agent runs into a bot-check, it can choose the `request_manual` action
itself — which auto-pauses and asks you to take over.

### CSV / Excel export

Ask it to compile data and it will (e.g. *"go through this product list and make
an Excel with columns name, price, link"*). As the agent reads each item it
collects rows (`record_rows`), then writes a `.xlsx` or `.csv` file (`export`) to
`output/`. A **Download** button appears in the panel; files are also served at
`/output/<filename>`.

You don't have to wait for the agent to finish: once any rows are collected,
**Export CSV / Export Excel** buttons appear in the panel so you can dump the data
at any time — even after you pause or stop the run. Stopping a run also
auto-saves whatever was gathered, so data is never lost.

## Setup

```bash
cd browser_agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium    # download the browser (once)
cp .env.example .env                      # then edit .env with your endpoint
cd backend && python main.py              # → http://127.0.0.1:8001
```

Open <http://127.0.0.1:8001>, type a task, click **Start**. A separate Chromium
window opens — that's where you can take over manually when needed.

## Configuration (.env)

| Var | Default | Purpose |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLLM_MODEL` | `your-model-name` | model name |
| `VLLM_API_KEY` | `dummy` | API key (use `dummy` for local endpoints) |
| `AGENT_MAX_STEPS` | `30` | max autonomous steps |
| `AGENT_HEADLESS` | `false` | **keep false** so manual takeover works |
| `AGENT_PORT` | `8001` | control-panel port |

## Tech stack

- **Backend**: Python 3.11, FastAPI + uvicorn, Playwright (headed Chromium),
  `openai` client for the LLM.
- **Frontend**: vanilla HTML/CSS/JS, no framework, no build step.
- No database, no auth — single user, one browser, one task at a time.

## Notes & limitations

- **Headed is required** for manual takeover, so this needs a display (run it on a
  desktop, not a headless server). The Chrome profile is stored in `.profile/` so
  logins and Cloudflare clearance persist across runs.
- The agent reads the page through the DOM (text-based), so content inside
  cross-origin iframes, shadow DOM, or pure-canvas apps may be invisible to it.
- Page content is fed to the LLM, so a malicious page could attempt prompt
  injection. Run it only on sites you trust, and keep an eye on it.
- It won't auto-confirm payments / purchases / account deletions — it pauses and
  asks you first.
- The first step can take ~30–60s if the model is cold-starting; calls retry
  automatically until the model is warm.
