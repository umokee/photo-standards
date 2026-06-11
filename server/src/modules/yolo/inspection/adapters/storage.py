from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any
from uuid import UUID

import structlog
from app.exception import ValidationError
from app.observability import log_event
from fastapi import UploadFile
from infra.storage.file_storage import (
    delete_storage_file,
    ensure_parent_dir,
    resolve_storage_path,
)
from modules.cameras.service import take_snapshot
from modules.cameras.streaming.manager import CameraStreamManager
from modules.yolo.inspection.constants import inspections as inspections_constants
from sqlalchemy.ext.asyncio import AsyncSession

from .repository import (
    has_inspection_with_image_path,
    has_inspection_with_result_image_path,
)

logger = structlog.get_logger(__name__)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True, frozen=True)
class InspectionImageSource:
    image_path: str
    filename: str | None
    content_type: str | None
    camera_id: UUID | None


async def acquire_inspection_image_source(
    *,
    db: AsyncSession,
    task_id: UUID,
    mode: str,
    image: UploadFile | None,
    camera_id: UUID | None,
    stream_manager: CameraStreamManager,
) -> InspectionImageSource:
    if mode == inspections_constants.modes.photo:
        if image is None:
            raise ValidationError("Для режима photo нужно передать изображение")

        image_path = await persist_uploaded_inspection_image(image, task_id=task_id)
        return InspectionImageSource(
            image_path=image_path,
            filename=image.filename,
            content_type=image.content_type,
            camera_id=None,
        )

    if mode == inspections_constants.modes.snapshot:
        if image is not None:
            image_path = await persist_uploaded_inspection_image(image, task_id=task_id)
            return InspectionImageSource(
                image_path=image_path,
                filename=image.filename,
                content_type=image.content_type,
                camera_id=None,
            )

        if camera_id is None:
            raise ValidationError("Для режима snapshot нужно выбрать камеру")

        snapshot = await take_snapshot(
            db,
            camera_id,
            task_id=task_id,
            stream_manager=stream_manager,
        )

        return InspectionImageSource(
            image_path=snapshot.image_path,
            filename=Path(snapshot.image_path).name,
            content_type="image/jpeg",
            camera_id=camera_id,
        )

    if mode == inspections_constants.modes.realtime:
        raise ValidationError("Режим realtime запускается отдельной realtime-сессией")

    raise ValidationError("Некорректный режим проверки")


async def persist_uploaded_inspection_image(
    image: UploadFile,
    *,
    task_id: UUID,
) -> str:
    suffix = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        suffix = ".jpg"

    relative_path = _inspection_source_rel_path(task_id, suffix=suffix)
    absolute_path = resolve_storage_path(relative_path)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(await image.read())
    return relative_path


def persist_frame_as_inspection_image(
    frame,
    *,
    image_id: UUID,
) -> str:
    import cv2

    relative_path = _inspection_source_rel_path(image_id, suffix=".jpg")
    absolute_path = resolve_storage_path(relative_path)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(absolute_path), frame)
    if not ok:
        raise RuntimeError("Не удалось сохранить кадр")

    return relative_path


def persist_frame_as_inspection_result(
    frame,
    *,
    image_id: UUID,
) -> str:
    import cv2

    relative_path = _inspection_result_rel_path(image_id, suffix=".jpg")
    absolute_path = resolve_storage_path(relative_path)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(absolute_path), frame)
    if not ok:
        raise RuntimeError("Не удалось сохранить изображение результата проверки")

    return relative_path


async def cleanup_unreferenced_inspection_task_files(
    db: AsyncSession,
    *,
    payload: Any,
    result: Any,
) -> None:
    image_path = _get_dict_value(payload, "image_path")
    result_image_path = _get_dict_value(result, "result_image_path")

    if image_path:
        is_referenced = await has_inspection_with_image_path(
            db,
            image_path=image_path,
        )

        if not is_referenced:
            unlink_storage_file(
                image_path,
                log_message="Failed to delete orphan inspection image %s",
            )

    if result_image_path:
        is_result_referenced = await has_inspection_with_result_image_path(
            db,
            result_image_path=result_image_path,
        )

        if not is_result_referenced:
            unlink_storage_file(
                result_image_path,
                log_message="Failed to delete orphan inspection result image %s",
            )


def unlink_storage_file(
    relative_path: str | None,
    *,
    log_message: str,
) -> None:
    if not relative_path:
        return

    try:
        delete_storage_file(relative_path)
    except Exception as exc:
        log_event(
            logger,
            "warning",
            "inspection.storage.cleanup_failed",
            path=relative_path,
            error_type=type(exc).__name__,
            exception=exc,
        )


def _get_dict_value(data: Any, key: str) -> str | None:
    if not isinstance(data, dict):
        return None

    value = data.get(key)

    if not isinstance(value, str) or not value:
        return None

    return value


def materialize_saved_inspection_files(
    *,
    inspection_id: UUID,
    image_path: str,
    result_image_path: str | None,
) -> tuple[str, str | None]:
    image_suffix = Path(image_path).suffix.lower() or ".jpg"
    target_image_path = _inspection_source_rel_path(inspection_id, suffix=image_suffix)
    target_result_path: str | None = None

    try:
        _copy_storage_file(image_path, target_image_path)

        if result_image_path:
            result_suffix = Path(result_image_path).suffix.lower() or ".jpg"
            target_result_path = _inspection_result_rel_path(
                inspection_id,
                suffix=result_suffix,
            )
            _copy_storage_file(result_image_path, target_result_path)
    except Exception:
        delete_storage_file(target_image_path)
        delete_storage_file(target_result_path)
        raise

    return target_image_path, target_result_path


def _copy_storage_file(source_relative_path: str, target_relative_path: str) -> None:
    source = resolve_storage_path(source_relative_path)
    target = resolve_storage_path(target_relative_path)

    if source == target:
        return

    ensure_parent_dir(target)
    shutil.copy2(source, target)


def _inspection_source_rel_path(artifact_id: UUID, *, suffix: str) -> str:
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return (Path("inspections") / str(artifact_id) / f"source{normalized_suffix}").as_posix()


def _inspection_result_rel_path(artifact_id: UUID, *, suffix: str) -> str:
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return (Path("inspections") / str(artifact_id) / f"result{normalized_suffix}").as_posix()
