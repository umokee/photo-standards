from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from uuid import UUID

import structlog
from app.db import get_sync_session
from app.exception import ValidationError
from app.observability import bind_context, elapsed_ms, log_event
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from modules.tasks.service import update_task_status
from modules.yolo.training.adapters import bus, repository, storage, yolo
from modules.yolo.training.domain import dataset_split
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


async def run_persisted_training(
    db: AsyncSession,
    *,
    task_id: UUID,
) -> None:
    started_at = time.perf_counter()

    task = await db.get(Task, task_id)
    if task is None:
        return

    payload = task.payload or {}
    try:
        paths = storage.TrainingPaths.from_payload(payload)
    except (KeyError, TypeError, ValueError):
        await update_task_status(
            db,
            task_id=task_id,
            status=tasks_constants.statuses.failed,
            stage="Ошибка",
            error="Некорректный payload задачи обучения",
        )
        return

    try:
        if task.status == tasks_constants.statuses.cancelled:
            await update_task_status(
                db,
                task_id=task_id,
                status=tasks_constants.statuses.cancelled,
                stage="Остановлено",
                message="Обучение остановлено пользователем",
            )
            return

        if task.abort_requested:
            await _handle_interrupted(db, task_id)
            return

        model = await _load_model_or_fail(db, task)
        if model is None:
            return

        bind_context(group_id=model.group_id, model_id=model.id)

        params = storage.TrainingParams.from_payload(payload)
        version = storage.payload_version(payload)
        resume = paths.checkpoint.exists()

        await update_task_status(
            db,
            task_id=task_id,
            status=(
                tasks_constants.statuses.resuming
                if resume
                else tasks_constants.statuses.running
            ),
            stage="Возобновление" if resume else "Подготовка данных",
            message="Продолжаем обучение" if resume else "Сбор данных для обучения",
        )

        try:
            dataset_info = await _prepare_dataset(
                db,
                model=model,
                params=params,
                paths=paths,
                resume=resume,
            )
            result = await _run_training(
                db,
                task_id=task_id,
                paths=paths,
                params=params,
                dataset_info=dataset_info,
                resume=resume,
            )
            await _finalize_success(
                db,
                task_id=task_id,
                model=model,
                paths=paths,
                result=result,
                dataset_info=dataset_info,
                version=version,
                started_at=started_at,
            )
        except yolo.TrainingInterrupted:
            await _handle_interrupted(db, task_id)
        except Exception as exc:
            await _finalize_failure(
                db,
                task_id=task_id,
                model=model,
                paths=paths,
                exc=exc,
                started_at=started_at,
            )
    finally:
        await _cleanup_task_artifacts(db, task_id=task_id, paths=paths)


async def _load_model_or_fail(
    db: AsyncSession,
    task: Task,
) -> MlModel | None:
    if task.entity_id is None:
        await update_task_status(
            db,
            task_id=task.id,
            status=tasks_constants.statuses.failed,
            stage="Ошибка",
            error="В задаче отсутствует ссылка на модель",
        )
        return None

    model = await db.get(MlModel, task.entity_id)
    if model is None:
        await update_task_status(
            db,
            task_id=task.id,
            status=tasks_constants.statuses.failed,
            stage="Ошибка",
            error="Модель не найдена",
        )
        return None
    return model


async def _prepare_dataset(
    db: AsyncSession,
    *,
    model: MlModel,
    params: storage.TrainingParams,
    paths: storage.TrainingPaths,
    resume: bool,
) -> storage.DatasetInfo:
    training_data = await repository.load_training_data(db, group_id=model.group_id)

    try:
        dataset_split.validate_training_data(training_data)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    if resume and model.class_keys is not None:
        if list(model.class_keys) != training_data.class_keys:
            raise ValidationError(
                "Состав классов группы изменился во время обучения. "
                "Возобновление невозможно - запустите обучение заново."
            )

        cached_yaml = paths.dataset_root / "data.yaml"
        if paths.dataset_root.exists() and cached_yaml.exists():
            logger.info(
                "training.dataset_cache_hit",
                extra={
                    "event": "training.dataset_cache_hit",
                    "dataset_root": str(paths.dataset_root),
                },
            )
            return storage.DatasetInfo(
                class_keys=list(model.class_keys),
                class_meta=list(model.class_meta or []),
                yaml_path=cached_yaml,
            )

    try:
        split = dataset_split.plan_dataset_split(
            training_data,
            train_ratio=params.train_ratio,
            val_ratio=params.val_ratio,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    log_event(
        logger,
        "info",
        "training.dataset.split",
        group_id=model.group_id,
        model_id=model.id,
        train_count=len(split.train),
        val_count=len(split.val),
        test_count=len(split.test),
        total_count=split.total,
        train_ratio=params.train_ratio,
        val_ratio=params.val_ratio,
    )

    dataset_info = storage.build_temp_dataset(
        training_data,
        split,
        dataset_root=paths.dataset_root,
    )

    model.num_classes = len(training_data.class_keys)
    model.class_keys = training_data.class_keys
    model.class_meta = training_data.class_meta
    model.total_images = split.total
    model.train_count = len(split.train)
    model.val_count = len(split.val)
    model.test_count = len(split.test)
    await db.commit()

    log_event(
        logger,
        "info",
        "training.dataset.prepared",
        group_id=model.group_id,
        model_id=model.id,
        yaml_path=dataset_info.yaml_path,
        train_count=len(split.train),
        val_count=len(split.val),
        test_count=len(split.test),
    )

    return dataset_info


async def _run_training(
    db: AsyncSession,
    *,
    task_id: UUID,
    paths: storage.TrainingPaths,
    params: storage.TrainingParams,
    dataset_info: storage.DatasetInfo,
    resume: bool,
) -> yolo.TrainingRunResult:
    loop = asyncio.get_running_loop()

    reporter = bus.TrainingProgressReporter(
        db,
        task_id,
        loop,
    )

    config = yolo.TrainingRunConfig(
        yaml_path=dataset_info.yaml_path,
        base_weights_path=paths.base_weights,
        checkpoint_path=paths.checkpoint,
        best_checkpoint_path=paths.best_checkpoint,
        output_checkpoint_path=paths.final_checkpoint,
        output_weights_path=paths.final_weights,
        run_dir=paths.run_dir,
        epochs=params.epochs,
        imgsz=params.imgsz,
        batch_size=params.batch_size,
        resume=resume,
        save_period=1,
        on_status=reporter.on_status,
        on_epoch_end=reporter.on_epoch_end,
        on_model_save=reporter.on_model_save,
        on_heartbeat=reporter.on_heartbeat,
        should_stop=lambda: _is_stop_requested(task_id),
    )

    return await asyncio.to_thread(yolo.run_training_sync, config)


async def _finalize_success(
    db: AsyncSession,
    *,
    task_id: UUID,
    model: MlModel,
    paths: storage.TrainingPaths,
    result: yolo.TrainingRunResult,
    dataset_info: storage.DatasetInfo,
    version: int | None,
    started_at: float,
) -> None:
    storage.persist_training_metrics_artifacts(paths)
    model.version = version
    model.weights_path = paths.final_weights_rel
    model.metrics = result.metrics
    model.trained_at = datetime.now(UTC).replace(tzinfo=None)
    await db.commit()

    await update_task_status(
        db,
        task_id=task_id,
        status=tasks_constants.statuses.succeeded,
        stage="Готово",
        message="Обучение завершено",
        result={
            "model_id": str(model.id),
            "checkpoint_path": paths.final_checkpoint_rel,
            "weights_path": model.weights_path,
            "class_keys": dataset_info.class_keys,
            "class_meta": dataset_info.class_meta,
            "metrics": result.metrics,
        },
    )
    log_event(
        logger,
        "info",
        "training.finished",
        task_id=task_id,
        group_id=model.group_id,
        model_id=model.id,
        epochs=model.epochs,
        imgsz=model.imgsz,
        batch_size=model.batch_size,
        train_count=model.train_count,
        val_count=model.val_count,
        test_count=model.test_count,
        weights_path=model.weights_path,
        duration_ms=elapsed_ms(started_at),
    )


async def _finalize_failure(
    db: AsyncSession,
    *,
    task_id: UUID,
    model: MlModel,
    paths: storage.TrainingPaths,
    exc: Exception,
    started_at: float,
) -> None:
    paths.final_checkpoint.unlink(missing_ok=True)
    paths.final_weights.unlink(missing_ok=True)
    paths.final_results_csv.unlink(missing_ok=True)

    model.version = None
    model.weights_path = None
    model.metrics = None
    model.trained_at = None
    await db.commit()

    await update_task_status(
        db,
        task_id=task_id,
        status=tasks_constants.statuses.failed,
        stage="Ошибка",
        message="Обучение прервано из-за ошибки",
        error=str(exc),
    )
    log_event(
        logger,
        "error",
        "training.failed",
        task_id=task_id,
        group_id=model.group_id,
        model_id=model.id,
        duration_ms=elapsed_ms(started_at),
        error_type=type(exc).__name__,
        exception=exc,
    )


async def _handle_interrupted(
    db: AsyncSession,
    task_id: UUID,
) -> None:
    task = await db.get(Task, task_id)
    if task is None:
        return

    await db.refresh(task)

    if task.status == tasks_constants.statuses.cancelled:
        await update_task_status(
            db,
            task_id=task_id,
            status=tasks_constants.statuses.cancelled,
            stage="Остановлено",
            message="Обучение остановлено пользователем",
        )
        return

    await update_task_status(
        db,
        task_id=task_id,
        status=tasks_constants.statuses.paused,
        stage="Приостановлено",
        message="Обучение приостановлено и будет продолжено позже",
    )


def _is_stop_requested(task_id: UUID) -> bool:
    with get_sync_session() as db:
        task = db.get(Task, task_id)
        if task is None:
            return True

        return bool(
            task.abort_requested or task.status == tasks_constants.statuses.cancelled
        )


async def _cleanup_task_artifacts(
    db: AsyncSession,
    *,
    task_id: UUID,
    paths: storage.TrainingPaths,
) -> None:
    task = await db.get(Task, task_id)
    status = task.status if task is not None else None

    if status == tasks_constants.statuses.paused:
        log_event(
            logger,
            "info",
            "training.artifacts.kept",
            task_id=task_id,
            reason="paused - checkpoint and dataset preserved for resume",
        )
        return

    storage.cleanup_task_artifacts(
        paths=paths,
        preserve_dataset_and_checkpoint=False,
    )
