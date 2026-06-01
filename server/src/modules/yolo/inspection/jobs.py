import asyncio
import time
from pathlib import Path
from uuid import UUID

import structlog
import torch
from app.config import settings
from app.db import AsyncSessionLocal
from app.observability import bind_context, elapsed_ms, log_event
from infra.queue.scheduler import maybe_resume_paused_training
from infra.storage.file_storage import resolve_storage_path
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from modules.tasks.service import update_task_status
from modules.yolo.inspection.adapters.storage import persist_frame_as_inspection_result

from .adapters.context import load_inspection_context
from .adapters.payloads import build_result_payload_from_frame_result
from .use_cases.inspect_frame import inspect_image_path

logger = structlog.get_logger(__name__)


async def execute_inspection(*, task_id: str) -> None:
    log_event(
        logger,
        "info",
        "inspection.started",
        task_id=task_id,
        queue=tasks_constants.queues.gpu,
    )

    try:
        await process_inspection_task(task_id)
    except Exception:
        raise


async def process_inspection_task(task_id: str) -> None:
    task_uuid = UUID(task_id)
    started_at = time.perf_counter()

    async with AsyncSessionLocal() as db:
        task = await db.get(Task, task_uuid)
        if task is None:
            return

        try:
            payload = task.payload or {}
            selected_ids = [
                UUID(class_id)
                for class_id in payload.get("selected_segment_class_ids", [])
            ]

            context = await load_inspection_context(
                db,
                standard_id=UUID(payload["standard_id"]),
                selected_segment_class_ids=selected_ids,
            )
            bind_context(
                standard_id=context.standard.id,
                group_id=context.standard.group_id,
                model_id=context.model.id,
            )
            log_event(
                logger,
                "info",
                "inspection.context.loaded",
                task_id=task_uuid,
                standard_id=context.standard.id,
                group_id=context.standard.group_id,
                model_id=context.model.id,
                reference_image_id=context.reference_image.id,
                selected_class_count=len(context.selected_classes),
                reference_feature_count=context.reference_features.count,
            )

            image_path = resolve_storage_path(payload["image_path"])

            await update_task_status(
                db,
                task_id=task_uuid,
                status=tasks_constants.statuses.running,
                stage="Проверка",
                message="Выполняем YOLO, выравнивание и сравнение с эталоном",
            )

            frame_result = await asyncio.to_thread(
                inspect_image_path,
                context=context,
                image_path=Path(image_path),
                render=True,
                profile_enabled=True,
            )
            alignment = frame_result.alignment
            log_event(
                logger,
                "info" if alignment.is_success else "warning",
                (
                    "inspection.alignment.success"
                    if alignment.is_success
                    else "inspection.alignment.failed"
                ),
                task_id=task_uuid,
                method=alignment.method,
                status=alignment.status.value,
                stage=alignment.stage,
                reason=alignment.reason,
                raw_match_count=alignment.raw_match_count,
                inlier_count=alignment.inlier_count,
                median_error=alignment.median_error,
                reference_feature_count=alignment.reference_feature_count,
                frame_feature_count=alignment.frame_feature_count,
                reference_size=alignment.reference_size,
                frame_size=alignment.frame_size,
                duration_ms=frame_result.profile.get("inspect_alignment_ms"),
            )
            log_event(
                logger,
                "info",
                "inspection.yolo.inference.finished",
                task_id=task_uuid,
                group_id=context.standard.group_id,
                model_id=context.model.id,
                device=(
                    "cuda:0"
                    if settings.YOLO_DEVICE != "cpu" and torch.cuda.is_available()
                    else "cpu"
                ),
                half=bool(settings.YOLO_HALF and torch.cuda.is_available()),
                imgsz=context.model.imgsz,
                weights_path=context.model.weights_path,
                detections_count=len(frame_result.detections),
                raw_class_counts=frame_result.raw_class_counts,
                duration_ms=frame_result.profile.get("inspect_detection_ms"),
            )
            log_event(
                logger,
                "info",
                "inspection.matching.finished",
                task_id=task_uuid,
                matches_count=len(frame_result.matches),
                expected_count=len(frame_result.expected_segments),
                matched_count=frame_result.matched,
                missing_count=len(frame_result.missing),
                duration_ms=frame_result.profile.get("inspect_matching_ms"),
            )

            result_image_path = None

            if frame_result.rendered_frame is not None:
                result_image_path = persist_frame_as_inspection_result(
                    frame_result.rendered_frame,
                    image_id=task_uuid,
                )

            await update_task_status(
                db,
                task_id=task_uuid,
                status=tasks_constants.statuses.succeeded,
                stage="Готово",
                message="Проверка завершена",
                result=build_result_payload_from_frame_result(
                    task_id=task_uuid,
                    context=context,
                    payload=payload,
                    frame_result=frame_result,
                    result_image_path=result_image_path,
                ),
            )
            log_event(
                logger,
                "info",
                "inspection.result.saved",
                task_id=task_uuid,
                status=frame_result.inspection_status,
                result_image_path=result_image_path,
                duration_ms=elapsed_ms(started_at),
            )
            log_event(
                logger,
                "info",
                "inspection.finished",
                task_id=task_uuid,
                status=frame_result.inspection_status,
                duration_ms=elapsed_ms(started_at),
            )

        except Exception as exc:
            log_event(
                logger,
                "error",
                "inspection.failed",
                task_id=task_uuid,
                duration_ms=elapsed_ms(started_at),
                error_type=type(exc).__name__,
                exception=exc,
            )
            await update_task_status(
                db,
                task_id=task_uuid,
                status=tasks_constants.statuses.failed,
                stage="Ошибка",
                message="Проверка прервана из-за ошибки",
                error=str(exc),
            )
        finally:
            await maybe_resume_paused_training(
                db,
                exclude_inspection_task_id=task_uuid,
            )
