"""
web.app -- FastAPI application serving the BeaconLock dashboard.

Routes
------
GET  /                     single-page dashboard (web/static/index.html)
GET  /static/*             local JS/CSS (no CDNs -> works offline)
GET  /video.mjpg           live camera feed with overlay (MJPEG stream)
GET  /api/frame.jpg        latest camera frame (single JPEG)
GET  /api/minimap.jpg      latest world minimap (single JPEG)
GET  /api/state            live stats + error history (JSON)
GET  /api/config           current settings, defaults and limits
POST /api/config           apply settings (resets the run)
POST /api/disturbances     change disturbance sliders live (no reset)
POST /api/start | /api/stop | /api/reset
GET  /api/report           ZIP: frames.csv + summary.txt + summary.json + config.json

Create with :func:`create_app`; ``run.py`` serves it with uvicorn.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sim.engine import TRAJECTORY_PRESETS

from .runner import DEFAULT_SETTINGS, LIMITS, SimRunner


def static_dir() -> Path:
    """Location of the dashboard assets (works from source and PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "web" / "static"


def create_app(runner: Optional[SimRunner] = None, start_thread: bool = True) -> FastAPI:
    runner = runner or SimRunner()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_thread:
            runner.start_thread()
        try:
            yield
        finally:
            runner.stop_thread()

    app = FastAPI(title="BeaconLock dashboard", version="1.0", lifespan=lifespan)
    app.state.runner = runner
    sdir = static_dir()
    app.mount("/static", StaticFiles(directory=str(sdir)), name="static")

    def bad(e: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(e))

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(sdir / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state() -> Dict[str, Any]:
        return runner.state()

    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        return {"settings": runner.settings, "defaults": DEFAULT_SETTINGS,
                "limits": {k: list(v) for k, v in LIMITS.items()}, "trajectories": list(TRAJECTORY_PRESETS)}

    @app.post("/api/config")
    def set_config(settings: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return {"settings": runner.reset(settings)}
        except (ValueError, TypeError) as e:
            raise bad(e)

    @app.post("/api/disturbances")
    def set_disturbances(values: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            return {"settings": runner.set_live(values)}
        except (ValueError, TypeError) as e:
            raise bad(e)

    @app.post("/api/start")
    def start() -> Dict[str, Any]:
        runner.start()
        return {"running": True}

    @app.post("/api/stop")
    def stop() -> Dict[str, Any]:
        runner.stop()
        return {"running": False}

    @app.post("/api/reset")
    def reset() -> Dict[str, Any]:
        return {"settings": runner.reset()}

    @app.get("/api/frame.jpg")
    def frame() -> Response:
        return Response(runner.camera_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/minimap.jpg")
    def minimap() -> Response:
        return Response(runner.minimap_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/video.mjpg")
    async def video(request: Request) -> StreamingResponse:
        async def gen():
            last, last_sent = -1, 0.0
            loop = asyncio.get_running_loop()
            while True:
                if await request.is_disconnected():
                    break
                seq = runner.frame_seq
                # Re-send the current frame every 0.5 s when idle: browsers only paint a
                # multipart part once the next one arrives, so a paused/idle stream would
                # otherwise show a black feed on first load.
                if seq != last or loop.time() - last_sent > 0.5:
                    last, last_sent = seq, loop.time()
                    jpg = runner.camera_jpeg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(1 / 60)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/report")
    def report() -> Response:
        data = runner.report_zip()
        name = f"beaconlock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.exception_handler(HTTPException)
    async def http_exc(_req: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app
