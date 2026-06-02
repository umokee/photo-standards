import os
from contextlib import asynccontextmanager

import structlog
import uvicorn
from app.asyncio_compat import configure_asyncio_policy
from app.config import settings
from app.db import dispose_engines
from app.exception_handlers import register_exception_handlers
from app.import_models import import_models
from app.live import ConnectionManager, event_bus
from app.live.router import router as live_router
from app.logging import (
    build_logging_config,
    configure_logging,
)
from app.observability import RequestContextMiddleware, log_event
from app.router import api_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from infra.queue.runner import TaskQueueRunner
from modules.cameras.live_status import CameraLiveStatusService
from modules.cameras.service import close_all_camera_preview_peers
from modules.cameras.streaming.manager import CameraStreamManager
from modules.yolo.inspection.realtime.streamer import InspectionStreamer
from modules.yolo.inspection.realtime.webrtc import close_all_realtime_webrtc_peers

configure_asyncio_policy()
logger = structlog.get_logger(__name__)


def ensure_storage_dirs() -> None:
    settings.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    settings.standards_storage_path.mkdir(parents=True, exist_ok=True)
    settings.inspections_storage_path.mkdir(parents=True, exist_ok=True)
    settings.models_storage_path.mkdir(parents=True, exist_ok=True)
    settings.logs_storage_path.mkdir(parents=True, exist_ok=True)
    (settings.STORAGE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
    (settings.STORAGE_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_event(logger, "info", "app.startup.started")
    ensure_storage_dirs()
    import_models()

    await event_bus.start()
    connection_manager = ConnectionManager(event_bus)
    await connection_manager.start()
    app.state.connection_manager = connection_manager

    task_queue_runner = TaskQueueRunner()
    await task_queue_runner.start()
    app.state.task_queue_runner = task_queue_runner

    camera_stream_manager = CameraStreamManager()
    await camera_stream_manager.start()
    app.state.camera_stream_manager = camera_stream_manager

    camera_live_status = CameraLiveStatusService()
    app.state.camera_live_status = camera_live_status

    inspection_streamer = InspectionStreamer()
    app.state.inspection_streamer = inspection_streamer

    try:
        log_event(logger, "info", "app.startup.finished")
        yield
    finally:
        log_event(logger, "info", "app.shutdown.started")
        await close_all_realtime_webrtc_peers()
        await close_all_camera_preview_peers()
        await task_queue_runner.stop()
        inspection_streamer.stop_all()
        await camera_stream_manager.stop()
        await connection_manager.stop()
        await event_bus.stop()
        await dispose_engines()
        log_event(logger, "info", "app.shutdown.finished")


def create_app() -> FastAPI:
    configure_logging(debug=settings.DEBUG)

    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api")
    app.include_router(live_router)
    app.mount("/storage", StaticFiles(directory=settings.STORAGE_ROOT), name="storage")
    app.state.settings = settings
    return app


app = create_app()


def _get_server_host() -> str:
    return os.getenv("SERVER_HOST", "0.0.0.0")


def _get_server_port() -> int:
    raw_port = os.getenv("SERVER_PORT", "3001")
    return int(raw_port)


def _get_server_reload() -> bool:
    raw_value = os.getenv("SERVER_RELOAD")
    if raw_value is None:
        return settings.DEBUG

    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        app_dir="src",
        host=_get_server_host(),
        port=_get_server_port(),
        reload=_get_server_reload(),
        log_config=build_logging_config(debug=settings.DEBUG),
        access_log=True,
    )
