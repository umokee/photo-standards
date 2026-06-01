from infra.queue.scheduler import (
    enqueue_inspection as _enqueue_inspection,
)
from infra.queue.scheduler import (
    maybe_resume_paused_training as _maybe_resume_paused_training,
)
from infra.queue.scheduler import (
    request_training_pause_for_inspection as _request_training_pause_for_inspection,
)


async def enqueue_inspection(db, task):
    return await _enqueue_inspection(db, task)


async def request_training_pause_for_inspection(db) -> None:
    await _request_training_pause_for_inspection(db)


async def maybe_resume_paused_training(
    db,
    *,
    exclude_inspection_task_id=None,
) -> None:
    await _maybe_resume_paused_training(
        db,
        exclude_inspection_task_id=exclude_inspection_task_id,
    )
