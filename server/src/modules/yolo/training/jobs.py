from __future__ import annotations

from uuid import UUID

import structlog
from app.db import AsyncSessionLocal
from app.observability import log_event
from modules.tasks.constants import tasks as tasks_constants
from modules.yolo.training.use_cases import run_persisted_training

logger = structlog.get_logger(__name__)


async def execute_training(*, task_id: str) -> None:
    log_event(
        logger,
        "info",
        "training.started",
        task_id=task_id,
        queue=tasks_constants.queues.gpu,
    )
    await process_training_task(task_id)


async def process_training_task(task_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await run_persisted_training.run_persisted_training(
            db,
            task_id=UUID(task_id),
        )
