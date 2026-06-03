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

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config
import exporter
from agent import AgentSession
from memory import store
from models import GotoReq, StartReq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

session = AgentSession()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Stop the frame producer and close Chromium on shutdown.
    if _producer is not None:
        _producer.cancel()
    await session.browser.close()


app = FastAPI(title="AI Browser Agent", lifespan=lifespan)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/start")
async def start(req: StartReq):
    try:
        await session.start(req.task, req.start_url, req.thread_id)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/thread")
async def thread(id: str = ""):
    """Indicator for the UI: does this thread_id already have memory?"""
    c = store.count(id)
    return {"id": store.norm(id), "exists": c > 0, "count": c}


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


@app.post("/api/goto")
async def goto(req: GotoReq):
    """Manual navigation from the dashboard address bar (only when AI isn't running)."""
    try:
        await session.manual_goto(req.url)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


def _input_allowed() -> bool:
    # Manual interaction is allowed whenever the AI is NOT actively running a step.
    return session.browser.started and session.state != "running"


# One shared frame producer feeds ALL dashboard tabs: a single lock-free screenshot
# loop, so N WebSocket clients don't each contend on the browser lock, and the
# stream keeps flowing even while a long navigation holds the lock for the agent.
_frame: dict = {"data": None}
_producer: asyncio.Task | None = None


async def _frame_producer():
    while True:
        try:
            if session.browser.started:
                _frame["data"] = await session.browser.frame_bytes(quality=45)
        except Exception:  # noqa: BLE001 — mid-navigation / closing page
            pass
        await asyncio.sleep(0.1 if _input_allowed() else 0.25)


def _ensure_producer() -> None:
    global _producer
    if _producer is None or _producer.done():
        _producer = asyncio.create_task(_frame_producer())


@app.websocket("/ws/screen")
async def ws_screen(ws: WebSocket):
    """Live browser stream into the dashboard + manual input forwarding.

    The server pushes JPEG frames (from the shared producer); the client sends
    click/scroll/key/text events, applied only when the AI isn't running (manual
    takeover, re-checked under the lock in session.manual_input).
    """
    await ws.accept()
    _ensure_producer()

    async def pump():
        while True:
            data = _frame["data"]
            if data is not None:
                try:
                    await ws.send_json({
                        "t": "frame",
                        "img": base64.b64encode(data).decode("ascii"),
                        "w": config.VIEWPORT_W,
                        "h": config.VIEWPORT_H,
                        "interactive": _input_allowed(),
                    })
                except Exception:  # noqa: BLE001 — client gone
                    return
            await asyncio.sleep(0.1 if _input_allowed() else 0.25)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            await session.manual_input(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass


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
