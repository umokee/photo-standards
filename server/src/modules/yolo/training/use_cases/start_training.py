from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from modules.tasks.service import create_task
from modules.yolo.training.adapters import queue, repository, storage
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(slots=True, frozen=True)
class TrainingStartResult:
    task: Task
    model: MlModel


async def start_training(
    db: AsyncSession,
    *,
    group_id: UUID,
    architecture: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    train_ratio: int,
    val_ratio: int,
) -> TrainingStartResult:
    await repository.ensure_group_exists(db, group_id=group_id)
    await queue.ensure_no_active_training_task(db, group_id=group_id)

    base_weights_path = storage.base_weights_rel(architecture)
    storage.ensure_base_weights_exist(base_weights_path)

    version = await repository.get_next_model_version(db, group_id=group_id)
    model = await repository.create_model(
        db,
        group_id=group_id,
        architecture=architecture,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=100 - train_ratio - val_ratio,
    )

    task = await create_task(
        db,
        type=tasks_constants.types.training,
        status=tasks_constants.statuses.pending,
        queue=tasks_constants.queues.gpu,
        priority=tasks_constants.priorities.training,
        entity_type="ml_model",
        entity_id=model.id,
        group_id=group_id,
        auto_resume=True,
    )

    paths = storage.build_training_paths(
        group_id=group_id,
        task_id=task.id,
        version=version,
        base_weights_path=base_weights_path,
    )

    task.payload = storage.build_training_payload(
        model_id=model.id,
        group_id=group_id,
        version=version,
        architecture=architecture,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        paths=paths,
    )
    task.run_dir = paths.run_dir_rel
    task.checkpoint_path = paths.checkpoint_rel

    task = await queue.enqueue_training(db, task)
    await db.refresh(model)

    return TrainingStartResult(
        task=task,
        model=model,
    )
