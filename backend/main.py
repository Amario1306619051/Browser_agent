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
    # Stop the screencast manager and close Chromium on shutdown.
    if _manager is not None:
        _manager.cancel()
    await session.browser.close()


app = FastAPI(title="AI Browser Agent", lifespan=lifespan)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/start")
async def start(req: StartReq):
    try:
        await session.start(req.task, req.start_url, req.thread_id, req.unlimited)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/thread")
async def thread(id: str = ""):
    """Indicator: does this thread_id already have memory?"""
    c = store.count(id)
    return {"id": store.norm(id), "exists": c > 0, "count": c}


@app.get("/api/threads")
async def threads():
    """Sidebar list of chats (one per thread_id)."""
    return {"threads": store.list_threads()}


@app.get("/api/thread/history")
async def thread_history(id: str = ""):
    """Full transcript (task + result turns) for one thread."""
    return {"id": store.norm(id), "turns": store.history(id)}


@app.delete("/api/thread")
async def thread_delete(id: str = ""):
    store.delete(id)
    return {"ok": True}


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


# Single CDP-screencast manager: re-attaches the browser's screencast to the active
# page (handles new tabs/popups). Frames arrive into browser.latest_frame_b64 as the
# page renders. Per-tab pumps just forward the latest frame — no screenshotting — so
# the stream is smooth and N tabs don't multiply the cost.
_manager: asyncio.Task | None = None
_ws_clients = 0  # number of dashboards currently watching


async def _screencast_manager():
    while True:
        try:
            if _ws_clients > 0:
                await session.browser.ensure_screencast()
            elif session.browser.started:
                await session.browser.stop_stream()  # nobody watching → save CPU
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.5)


def _ensure_manager() -> None:
    global _manager
    if _manager is None or _manager.done():
        _manager = asyncio.create_task(_screencast_manager())


@app.websocket("/ws/screen")
async def ws_screen(ws: WebSocket):
    """Live browser stream into the dashboard + manual input forwarding.

    Frames come from a CDP screencast (the browser pushes them as it renders); the
    client sends click/scroll/key/text events, applied only when the AI isn't
    running (manual takeover, re-checked under the lock in session.manual_input).
    """
    global _ws_clients
    await ws.accept()
    _ws_clients += 1
    _ensure_manager()

    async def pump():
        last_seq, last_int = -1, None
        while True:
            b = session.browser
            inter = _input_allowed()
            if b.latest_frame_b64 is not None and (b.frame_seq != last_seq or inter != last_int):
                last_seq, last_int = b.frame_seq, inter
                try:
                    await ws.send_json({
                        "t": "frame",
                        "img": b.latest_frame_b64,
                        "w": config.VIEWPORT_W,
                        "h": config.VIEWPORT_H,
                        "interactive": inter,
                    })
                except Exception:  # noqa: BLE001 — client gone
                    return
            await asyncio.sleep(0.05)  # cap forwarding at ~20fps (lighter on the client)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            await session.manual_input(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        _ws_clients -= 1
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
