from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from app.config import settings
from app.exception import ConflictError, ValidationError
from fastapi import UploadFile
from modules.cameras.service import get_camera
from modules.cameras.streaming.manager import CameraStreamManager
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.service import create_task
from modules.yolo.inspection.adapters import queue, storage
from modules.yolo.inspection.adapters.context import (
    InspectionContext,
    load_inspection_context,
)
from modules.yolo.inspection.adapters.payloads import build_inspection_payload
from modules.yolo.inspection.constants import inspections as inspections_constants
from modules.yolo.inspection.realtime.browser_source import BrowserFrameSource
from modules.yolo.inspection.realtime.streamer import InspectionStreamer
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(slots=True, frozen=True)
class InspectionStartResult:
    kind: Literal["task", "session"]
    status: str
    message: str
    task_id: UUID | None = None
    session_id: UUID | None = None


async def start_inspection(
    db: AsyncSession,
    *,
    standard_id: UUID,
    mode: str,
    selected_segment_class_ids: list[UUID],
    camera_id: UUID | None,
    notes: str | None,
    image: UploadFile | None,
    stream_manager: CameraStreamManager,
    inspection_streamer: InspectionStreamer,
) -> InspectionStartResult:
    if mode not in inspections_constants.modes:
        raise ValidationError("Некорректный режим проверки")

    context = await load_inspection_context(
        db,
        standard_id=standard_id,
        selected_segment_class_ids=selected_segment_class_ids,
    )

    if mode == inspections_constants.modes.realtime:
        return await _start_realtime_session(
            db,
            context=context,
            selected_segment_class_ids=selected_segment_class_ids,
            camera_id=camera_id,
            stream_manager=stream_manager,
            inspection_streamer=inspection_streamer,
        )

    task_id = uuid4()
    source = await storage.acquire_inspection_image_source(
        db=db,
        task_id=task_id,
        mode=mode,
        image=image,
        camera_id=camera_id,
        stream_manager=stream_manager,
    )

    return await _start_batch_inspection_task(
        db,
        task_id=task_id,
        context=context,
        mode=mode,
        selected_segment_class_ids=selected_segment_class_ids,
        notes=notes,
        source=source,
    )


async def _start_realtime_session(
    db: AsyncSession,
    *,
    context: InspectionContext,
    selected_segment_class_ids: list[UUID],
    camera_id: UUID | None,
    stream_manager: CameraStreamManager,
    inspection_streamer: InspectionStreamer,
) -> InspectionStartResult:
    if inspection_streamer.active_count >= settings.MAX_REALTIME_INSPECTIONS:
        raise ConflictError("Достигнут лимит активных realtime-проверок")

    if camera_id is None:
        frame_source = BrowserFrameSource()
        session = inspection_streamer.create_session(
            context=context,
            camera=None,
            selected_class_ids=selected_segment_class_ids,
            frame_source=frame_source,
        )

        return InspectionStartResult(
            kind="session",
            session_id=session.session_id,
            status="warming_up",
            message="Realtime-сессия запущена. Ожидаем кадры камеры устройства",
        )

    camera = await get_camera(db, camera_id)
    if not camera.is_active:
        raise ValidationError("Камера отключена")

    frame_source = await stream_manager.acquire_source(camera_id)

    try:
        await frame_source.wait_for_frame(
            timeout=max(float(camera.timeout_sec), 5.0),
        )
        session = inspection_streamer.create_session(
            context=context,
            camera=camera,
            selected_class_ids=selected_segment_class_ids,
            frame_source=frame_source,
        )
    except Exception:
        await frame_source.close()
        raise

    return InspectionStartResult(
        kind="session",
        session_id=session.session_id,
        status="warming_up",
        message="Realtime-сессия запущена",
    )


async def _start_batch_inspection_task(
    db: AsyncSession,
    *,
    task_id: UUID,
    context: InspectionContext,
    mode: str,
    selected_segment_class_ids: list[UUID],
    notes: str | None,
    source: storage.InspectionImageSource,
) -> InspectionStartResult:
    await queue.request_training_pause_for_inspection(db)

    try:
        task = await create_task(
            db,
            task_id=task_id,
            type=tasks_constants.types.inspection,
            status=tasks_constants.statuses.pending,
            queue=tasks_constants.queues.gpu,
            priority=tasks_constants.priorities.inspection,
            entity_type="standard",
            entity_id=context.standard.id,
            group_id=context.standard.group_id,
            payload=build_inspection_payload(
                context=context,
                camera_id=source.camera_id,
                mode=mode,
                notes=notes,
                filename=source.filename,
                content_type=source.content_type,
                image_path=source.image_path,
                selected_segment_class_ids=selected_segment_class_ids,
            ),
        )

        task = await queue.enqueue_inspection(db, task)

        return InspectionStartResult(
            kind="task",
            task_id=task.id,
            status=task.status,
            message=task.message or "Проверка поставлена в очередь",
        )
    except Exception:
        await db.rollback()
        storage.unlink_storage_file(
            source.image_path,
            log_message="Failed to cleanup orphan inspection image %s",
        )
        raise
