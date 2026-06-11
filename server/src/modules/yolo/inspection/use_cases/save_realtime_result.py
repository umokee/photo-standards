from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
from app.exception import ConflictError
from modules.yolo.inspection.adapters import repository, storage
from modules.yolo.inspection.adapters.context import InspectionContext
from modules.yolo.inspection.adapters.entities import (
    build_inspection_entities,
    build_matches_from_details,
)
from modules.yolo.inspection.adapters.payloads import build_realtime_save_payload
from modules.yolo.inspection.domain.overlay import render_overlay
from modules.yolo.inspection.models import InspectionResult
from modules.yolo.inspection.realtime.streamer import InspectionStreamer
from sqlalchemy.ext.asyncio import AsyncSession

from .realtime_session import get_realtime_session_or_raise

if TYPE_CHECKING:
    from modules.yolo.inspection.realtime.frame_result import (
        FrameResult as RealtimeFrameResult,
    )


async def save_realtime_session_snapshot(
    db: AsyncSession,
    *,
    streamer: InspectionStreamer,
    session_id: UUID,
    notes: str | None,
) -> InspectionResult:
    session = get_realtime_session_or_raise(
        streamer=streamer,
        session_id=session_id,
    )

    frame, result = session.get_frame_for_save()
    if frame is None or result is None:
        raise ConflictError("Дождитесь первой проверки кадра")

    return await _save_realtime_snapshot_inspection(
        db,
        context=session.context,
        camera_id=session.camera.id if session.camera is not None else None,
        frame=frame,
        result=result,
        notes=notes,
    )


async def _save_realtime_snapshot_inspection(
    db: AsyncSession,
    *,
    context: InspectionContext,
    camera_id: UUID | None,
    frame: np.ndarray,
    result: RealtimeFrameResult,
    notes: str | None,
) -> InspectionResult:
    relative_path: str | None = None
    result_image_path: str | None = None
    inspection_id = uuid4()

    try:
        relative_path = storage.persist_frame_as_inspection_image(
            frame,
            image_id=inspection_id,
        )

        rendered_frame = render_overlay(
            frame,
            build_matches_from_details(result.details),
        )
        result_image_path = storage.persist_frame_as_inspection_result(
            rendered_frame,
            image_id=inspection_id,
        )

        data = build_realtime_save_payload(
            context=context,
            camera_id=camera_id,
            image_path=relative_path,
            result=result,
            result_image_path=result_image_path,
        )

        inspection, segment_results = build_inspection_entities(
            data=data,
            notes=notes,
            inspection_id=inspection_id,
        )

        return await repository.create_inspection(
            db,
            inspection=inspection,
            segment_results=segment_results,
        )

    except Exception:
        storage.unlink_storage_file(
            relative_path,
            log_message="Failed to cleanup orphan realtime inspection image %s",
        )
        storage.unlink_storage_file(
            result_image_path,
            log_message="Failed to cleanup orphan realtime inspection result image %s",
        )
        raise
