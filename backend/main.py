"""FastAPI control panel for the AI browser agent.

Serves the vanilla frontend and a tiny API:
  POST /api/start      {task, start_url?}   begin an autonomous run
  POST /api/pause                            manual takeover (AI waits)
  POST /api/resume                           hand control back to the AI
  POST /api/stop                             abort the run
  GET  /api/status                           full state + log feed (polled)
  GET  /api/screenshot                       live JPEG preview of the page
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config
import exporter
from agent import AgentSession
from models import StartReq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

session = AgentSession()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Close Chromium on server shutdown so we don't leak browser processes.
    await session.browser.close()


app = FastAPI(title="AI Browser Agent", lifespan=lifespan)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/start")
async def start(req: StartReq):
    try:
        await session.start(req.task, req.start_url)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/pause")
async def pause():
    session.pause()
    return {"ok": True}


@app.post("/api/resume")
async def resume():
    session.resume()
    return {"ok": True}


@app.post("/api/stop")
async def stop():
    await session.stop()
    return {"ok": True}


@app.get("/api/status")
async def status():
    return session.status()


@app.post("/api/export")
async def export(fmt: str = "csv"):
    ref = session.export_now("xlsx" if fmt.lower() == "xlsx" else "csv")
    if ref is None:
        return JSONResponse({"ok": False, "error": "no data collected yet"}, status_code=400)
    return {"ok": True, "export": ref}


@app.post("/api/capture")
async def capture():
    """Manual screenshot of the current view, saved to output/."""
    try:
        png = await session.browser.capture()
        ref = exporter.save_image(png, "manual")
        session.shots.append(ref)
        return {"ok": True, "shot": ref}
    except Exception as e:  # noqa: BLE001 — no browser yet / mid-navigation
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/output/{name}")
async def download(name: str):
    # Path(name).name strips any path components — only serve flat files in OUTPUT_DIR.
    path = config.OUTPUT_DIR / Path(name).name
    if not path.is_file():
        return Response(status_code=404)
    return FileResponse(path, filename=path.name)


@app.get("/api/screenshot")
async def screenshot():
    try:
        png = await session.browser.screenshot()
        return Response(content=png, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except Exception:  # noqa: BLE001 — no page yet / mid-navigation
        return Response(status_code=204)


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=config.PORT)
