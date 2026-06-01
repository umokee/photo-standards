from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from uuid import UUID

import structlog
from app.exception import NotFoundError, ValidationError
from app.observability import bind_context, elapsed_ms, log_event
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from modules.tasks.service import update_task_status
from modules.yolo.interop.adapters import repository, storage, yolo
from modules.yolo.interop.domain import mapping as mapping_domain
from modules.yolo.interop.domain.types import ResolvedImportMapping, SegmentClassRef
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


async def run_persisted_import(
    db: AsyncSession,
    *,
    task_id: UUID,
) -> None:
    started_at = time.perf_counter()

    task = await db.get(Task, task_id)
    if task is None:
        return

    if task.status == tasks_constants.statuses.cancelled or task.abort_requested:
        await update_task_status(
            db,
            task_id=task_id,
            status=tasks_constants.statuses.cancelled,
            stage="Отменено",
            message="Импорт модели остановлен пользователем",
        )
        return

    model = await _load_model_or_fail(db, task)
    if model is None:
        return

    bind_context(group_id=model.group_id, model_id=model.id)

    try:
        payload = storage.parse_import_task_payload(task.payload or {})
    except Exception as exc:  # noqa: BLE001
        await _fail_task(db, task_id, error="Некорректный payload задачи импорта")
        raise ValidationError("Некорректный payload задачи импорта") from exc

    task_paths = storage.ImportTaskPaths.from_payload(payload)
    artifact_paths = storage.build_model_artifact_paths(
        group_id=payload.group_id,
        version=payload.version,
    )

    try:
        await update_task_status(
            db,
            task_id=task_id,
            status=tasks_constants.statuses.running,
            stage="Чтение модели",
            message="Проверяем .pt и читаем классы",
        )
        native_classes = await asyncio.to_thread(
            yolo.read_native_classes, task_paths.source_pt
        )

        await repository.ensure_group_exists(db, group_id=payload.group_id)
        categories, ungrouped_classes = await repository.load_group_segment_tree(
            db,
            group_id=payload.group_id,
        )

        await update_task_status(
            db,
            task_id=task_id,
            status=tasks_constants.statuses.running,
            stage="Сопоставление",
            message="Подготавливаем классы модели",
        )
        resolved = await _resolve_import_mappings(
            db,
            group_id=payload.group_id,
            native_classes=native_classes,
            mappings=payload.mappings,
            categories=categories,
            ungrouped_classes=ungrouped_classes,
        )
        log_event(
            logger,
            "info",
            "model.import.classes.resolved",
            task_id=task_id,
            group_id=payload.group_id,
            model_id=model.id,
            native_class_count=len(native_classes),
            mapped_class_count=len(resolved.class_keys),
            ignored_native_class_count=len(resolved.ignored_native_classes),
        )

        await update_task_status(
            db,
            task_id=task_id,
            status=tasks_constants.statuses.running,
            stage="Сохранение модели",
            message="Сохраняем .pt модель",
        )

        storage.move_file(source=task_paths.source_pt, target=artifact_paths.pt)

        repository.apply_import_to_model(
            model,
            version=payload.version,
            architecture=payload.architecture,
            imgsz=payload.imgsz,
            batch_size=payload.batch_size,
            weights_path=artifact_paths.pt_rel,
            class_keys=resolved.class_keys,
            class_meta=resolved.class_meta,
        )
        model.trained_at = _now()

        await asyncio.to_thread(
            storage.write_class_manifest,
            model=model,
            artifact_paths=artifact_paths,
        )
        await db.commit()

        if payload.activate:
            model = await repository.activate_model(db, model=model)

        await update_task_status(
            db,
            task_id=task_id,
            status=tasks_constants.statuses.succeeded,
            stage="Готово",
            message="Модель импортирована",
            result={
                "model_id": str(model.id),
                "weights_path": artifact_paths.pt_rel,
                "source_pt_path": artifact_paths.pt_rel,
                "manifest_path": artifact_paths.manifest_rel,
                "ignored_native_classes": resolved.ignored_native_classes,
            },
        )
        log_event(
            logger,
            "info",
            "model.import.saved",
            task_id=task_id,
            group_id=payload.group_id,
            model_id=model.id,
            weights_path=artifact_paths.pt_rel,
            duration_ms=elapsed_ms(started_at),
        )
    except Exception as exc:  # noqa: BLE001
        await db.rollback()
        await repository.reset_failed_model(db, model_id=model.id)

        for path in (artifact_paths.pt, artifact_paths.manifest):
            storage.unlink_path(path)

        await _fail_task(
            db,
            task_id,
            error=str(exc),
            message="Импорт модели завершился с ошибкой",
        )
        log_event(
            logger,
            "error",
            "model.import.failed",
            task_id=task_id,
            group_id=payload.group_id,
            model_id=model.id,
            duration_ms=elapsed_ms(started_at),
            error_type=type(exc).__name__,
            exception=exc,
        )
    finally:
        storage.remove_tree(task_paths.task_root)


async def _load_model_or_fail(
    db: AsyncSession,
    task: Task,
) -> MlModel | None:
    if task.entity_id is None:
        await _fail_task(db, task.id, error="В задаче отсутствует ссылка на модель")
        return None

    try:
        return await repository.get_model(db, model_id=task.entity_id)
    except NotFoundError:
        await _fail_task(db, task.id, error="Модель не найдена")
        return None


async def _resolve_import_mappings(
    db: AsyncSession,
    *,
    group_id: UUID,
    native_classes,
    mappings,
    categories,
    ungrouped_classes,
) -> ResolvedImportMapping:
    try:
        mapping_domain.validate_import_mappings(
            mappings=mappings,
            native_classes=native_classes,
            categories=categories,
            ungrouped_classes=ungrouped_classes,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    all_classes = [
        *(item for category in categories for item in category.segment_classes),
        *ungrouped_classes,
    ]
    by_id = {item.id: item for item in all_classes}
    target_by_native: dict[str, SegmentClassRef] = {}

    for item in mappings:
        if item.mode == "existing":
            target_by_native[item.native_key] = by_id[item.segment_class_id]
            continue

        segment_class = await repository.create_segment_class(
            db,
            group_id=group_id,
            name=item.new_class_name,
            hue=item.new_class_hue,
            class_group_id=item.new_class_group_id,
        )
        by_id[segment_class.id] = segment_class
        target_by_native[item.native_key] = segment_class

    try:
        return mapping_domain.build_resolved_import_mapping(
            native_classes=native_classes,
            target_by_native=target_by_native,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


async def _fail_task(
    db: AsyncSession,
    task_id: UUID,
    *,
    error: str,
    message: str | None = None,
) -> None:
    await update_task_status(
        db,
        task_id=task_id,
        status=tasks_constants.statuses.failed,
        stage="Ошибка",
        message=message,
        error=error,
    )


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
