from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from modules.cameras.live_status import (
    CameraLiveStatusService,
    normalize_camera_status_scope,
)
from modules.system.service import get_system_stats
from app.observability import bind_context, clear_context, log_event
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["live"])


@router.websocket("/tasks/{task_id}")
async def task_ws(ws: WebSocket, task_id: UUID) -> None:
    manager = ws.app.state.connection_manager
    await ws.accept()
    bind_context(task_id=task_id)

    async with manager.subscribe_task(task_id, ws):
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log_event(
                logger,
                "error",
                "live.ws.task.error",
                task_id=task_id,
                error_type=type(exc).__name__,
                exception=exc,
            )
        finally:
            clear_context()


@router.websocket("/system/stats")
async def system_stats_ws(ws: WebSocket) -> None:
    await ws.accept()

    try:
        while True:
            payload = await get_system_stats()
            await ws.send_json(payload.model_dump(mode="json"))
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log_event(
            logger,
            "error",
            "live.ws.system_stats.error",
            error_type=type(exc).__name__,
            exception=exc,
        )


@router.websocket("/cameras/status")
async def cameras_status_ws(ws: WebSocket) -> None:
    await ws.accept()
    scope = normalize_camera_status_scope(ws.query_params.get("scope"))
    live_status: CameraLiveStatusService = ws.app.state.camera_live_status

    try:
        async for event in live_status.watch(scope=scope):
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log_event(
            logger,
            "error",
            "live.ws.cameras_status.error",
            scope=scope,
            error_type=type(exc).__name__,
            exception=exc,
        )
