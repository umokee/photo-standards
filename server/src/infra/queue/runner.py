from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO
from uuid import UUID

import psutil
import structlog
from app.config import settings
from app.db import AsyncSessionLocal
from app.observability import elapsed_ms, log_event
from infra.queue.process_tree import (
    get_process_tree_cpu_percent_of_system,
    get_process_tree_rss_bytes,
    kill_pid_tree,
    prime_process_tree_cpu_percent,
    terminate_pid_tree,
)
from infra.queue.scheduler import maybe_resume_paused_training
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from modules.tasks.service import update_task_status
from sqlalchemy import select

logger = structlog.get_logger(__name__)

SRC_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = SRC_ROOT.parent

StopKind = Literal["cancelled", "failed", "shutdown"]


@dataclass(slots=True)
class ClaimedTask:
    id: UUID
    type: str
    queue: str | None


@dataclass(slots=True)
class ProcessStopReason:
    kind: StopKind
    stage: str
    message: str
    error: str | None = None


@dataclass(slots=True)
class ManagedProcessTask:
    task_id: UUID
    task_type: str
    queue: str | None
    process: subprocess.Popen
    create_time: float | None
    started_at: float
    stdout_path: Path
    stderr_path: Path
    stdout_file: TextIO
    stderr_file: TextIO
    resource_violations: int = 0
    stop_reason: ProcessStopReason | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self, timeout_sec: float) -> None:
        terminate_pid_tree(
            self.process.pid,
            expected_create_time=self.create_time,
            timeout_sec=timeout_sec,
        )

    def kill(self) -> None:
        kill_pid_tree(
            self.process.pid,
            expected_create_time=self.create_time,
        )

    def rss_mb(self) -> int:
        rss_bytes = get_process_tree_rss_bytes(
            self.process.pid,
            expected_create_time=self.create_time,
        )
        return rss_bytes // 1024 // 1024

    def cpu_percent(self) -> float:
        return get_process_tree_cpu_percent_of_system(
            self.process.pid,
            expected_create_time=self.create_time,
        )

    def close_logs(self) -> None:
        self.stdout_file.close()
        self.stderr_file.close()


class TaskQueueRunner:
    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._tasks: dict[UUID, ManagedProcessTask] = {}

    async def start(self) -> None:
        if not settings.TASK_RUNNER_ENABLED:
            log_event(logger, "info", "task.runner.disabled")
            return

        await self._recover_active_tasks_on_startup()

        self._stopping = False
        self._task = asyncio.create_task(
            self._run(),
            name="task_queue_runner",
        )

        log_event(
            logger,
            "info",
            "task.runner.started",
            pid=os.getpid(),
            mode="per_task_subprocess",
            gpu_concurrency=settings.TASK_RUNNER_GPU_CONCURRENCY,
            cpu_concurrency=settings.TASK_RUNNER_CPU_CONCURRENCY,
            inspection_concurrency=settings.TASK_RUNNER_INSPECTION_CONCURRENCY,
        )

    async def stop(self) -> None:
        self._stopping = True

        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        shutdown_reason = ProcessStopReason(
            kind="shutdown",
            stage="Остановлено",
            message="Сервер остановлен, subprocess задачи завершён",
            error="Server shutdown",
        )

        for managed in list(self._tasks.values()):
            managed.stop_reason = shutdown_reason
            await self._terminate_process(managed)
            await self._handle_finished_task(managed)

        self._tasks.clear()

        log_event(logger, "info", "task.runner.stopped")

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    logger,
                    "error",
                    "task.runner.tick_failed",
                    error_type=type(exc).__name__,
                    exception=exc,
                )

            await asyncio.sleep(settings.TASK_RUNNER_POLL_INTERVAL_SEC)

    async def _tick(self) -> None:
        await self._sync_running_tasks()
        await self._start_available_tasks()

    async def _sync_running_tasks(self) -> None:
        for task_id, managed in list(self._tasks.items()):
            return_code = managed.poll()
            if return_code is not None:
                await self._handle_finished_task(managed)
                continue

            stop_reason = await self._build_stop_reason(managed)
            if stop_reason is None:
                continue

            managed.stop_reason = stop_reason
            log_event(
                logger,
                "warning" if stop_reason.kind == "cancelled" else "error",
                "task.runner.process.stop_required",
                task_id=task_id,
                task_type=managed.task_type,
                queue=managed.queue,
                pid=managed.pid,
                reason=stop_reason.message,
                kind=stop_reason.kind,
                duration_ms=elapsed_ms(managed.started_at),
            )
            await self._terminate_process(managed)
            await self._handle_finished_task(managed)

    async def _handle_finished_task(self, managed: ManagedProcessTask) -> None:
        self._tasks.pop(managed.task_id, None)

        return_code = managed.poll()
        managed.close_logs()

        task = await self._get_task(managed.task_id)
        stop_reason = managed.stop_reason

        if stop_reason is not None:
            if stop_reason.kind == "cancelled":
                await self._mark_task_cancelled_if_not_terminal(
                    managed.task_id,
                    stage=stop_reason.stage,
                    message=stop_reason.message,
                )
            else:
                await self._mark_task_failed_if_not_terminal(
                    managed.task_id,
                    stage=stop_reason.stage,
                    message=stop_reason.message,
                    error=stop_reason.error or stop_reason.message,
                )

            log_event(
                logger,
                "warning" if stop_reason.kind == "cancelled" else "error",
                "task.runner.process.stopped",
                task_id=managed.task_id,
                task_type=managed.task_type,
                queue=managed.queue,
                pid=managed.pid,
                return_code=return_code,
                reason=stop_reason.message,
                duration_ms=elapsed_ms(managed.started_at),
            )
            await self._maybe_resume_training_after_task(managed)
            return

        if task is not None and task.status in tasks_constants.statuses.terminal:
            log_event(
                logger,
                "info",
                "task.runner.process.finished",
                task_id=managed.task_id,
                task_type=managed.task_type,
                queue=managed.queue,
                pid=managed.pid,
                return_code=return_code,
                final_status=task.status,
                duration_ms=elapsed_ms(managed.started_at),
            )
            await self._maybe_resume_training_after_task(managed)
            return

        if return_code == 0:
            await self._mark_task_failed_if_not_terminal(
                managed.task_id,
                stage="Ошибка",
                message="Subprocess завершился без финального статуса задачи",
                error="Task process finished without terminal task status",
            )
        else:
            await self._mark_task_failed_if_not_terminal(
                managed.task_id,
                stage="Ошибка",
                message="Subprocess задачи завершился с ошибкой",
                error=f"Task process exited with code {return_code}",
            )

        log_event(
            logger,
            "error",
            "task.runner.process.failed",
            task_id=managed.task_id,
            task_type=managed.task_type,
            queue=managed.queue,
            pid=managed.pid,
            return_code=return_code,
            duration_ms=elapsed_ms(managed.started_at),
        )
        await self._maybe_resume_training_after_task(managed)

    async def _start_available_tasks(self) -> None:
        limits = {
            tasks_constants.queues.gpu: max(1, settings.TASK_RUNNER_GPU_CONCURRENCY),
            tasks_constants.queues.cpu: max(1, settings.TASK_RUNNER_CPU_CONCURRENCY),
        }

        for queue, limit in limits.items():
            while not self._stopping:
                used = self._count_running_for_queue(queue)
                available = max(0, limit - used)

                if available <= 0:
                    break

                task = await self._claim_next_runnable_task(queue)

                if task is None:
                    break

                await self._start_claimed_task(task)

    def _count_running_for_queue(self, queue: str) -> int:
        return sum(1 for task in self._tasks.values() if task.queue == queue)

    def _count_running_for_type(self, task_type: str) -> int:
        return sum(1 for task in self._tasks.values() if task.task_type == task_type)

    def _can_start_task_type(self, task_type: str) -> bool:
        if task_type == tasks_constants.types.training:
            if (
                self._count_running_for_type(task_type)
                >= settings.TASK_RUNNER_TRAINING_CONCURRENCY
            ):
                return False

            return not self._has_running_any_type(
                {
                    tasks_constants.types.inspection,
                    tasks_constants.types.model_import,
                }
            )

        if task_type == tasks_constants.types.inspection:
            if (
                self._count_running_for_type(task_type)
                >= settings.TASK_RUNNER_INSPECTION_CONCURRENCY
            ):
                return False

            return not self._has_running_any_type(
                {
                    tasks_constants.types.training,
                    tasks_constants.types.model_import,
                }
            )

        if task_type == tasks_constants.types.model_import:
            if (
                self._count_running_for_type(task_type)
                >= settings.TASK_RUNNER_MODEL_IMPORT_CONCURRENCY
            ):
                return False

            return not self._has_running_any_type(
                {
                    tasks_constants.types.training,
                    tasks_constants.types.inspection,
                }
            )

        return False

    async def _claim_next_runnable_task(self, queue: str) -> ClaimedTask | None:
        async with AsyncSessionLocal() as db, db.begin():
            result = await db.execute(
                select(Task)
                .where(
                    Task.queue == queue,
                    Task.status == tasks_constants.statuses.queued,
                )
                .order_by(Task.priority.desc(), Task.created_at.asc())
                .with_for_update(skip_locked=True)
                .limit(20)
            )
            queued_tasks = list(result.scalars().all())

            for task in queued_tasks:
                if not self._can_start_task_type(task.type):
                    continue

                task.status = tasks_constants.statuses.running
                task.stage = "Запуск задачи"
                task.message = "Задача взята сервером для запуска subprocess"
                task.heartbeat_at = _now()
                task.external_job_id = "process:starting"

                return ClaimedTask(
                    id=task.id,
                    type=task.type,
                    queue=task.queue,
                )

            return None

    async def _start_claimed_task(self, task: ClaimedTask) -> None:
        try:
            managed = await asyncio.to_thread(_start_task_process, task)
        except Exception as exc:
            log_event(
                logger,
                "error",
                "task.runner.process_start_failed",
                task_id=task.id,
                task_type=task.type,
                queue=task.queue,
                error_type=type(exc).__name__,
                exception=exc,
            )
            await self._mark_task_failed_if_not_terminal(
                task.id,
                stage="Ошибка запуска",
                message="Не удалось запустить subprocess задачи",
                error=str(exc),
            )
            return

        self._tasks[task.id] = managed

        log_event(
            logger,
            "info",
            "task.runner.process.started",
            task_id=task.id,
            task_type=task.type,
            queue=task.queue,
            pid=managed.pid,
            stdout_path=managed.stdout_path,
            stderr_path=managed.stderr_path,
        )

        async with AsyncSessionLocal() as db:
            await update_task_status(
                db,
                task_id=task.id,
                status=tasks_constants.statuses.running,
                stage="Задача запущена",
                message="Задача выполняется в отдельном subprocess",
                external_job_id=_process_job_id(managed.pid),
            )

    async def _build_stop_reason(
        self,
        managed: ManagedProcessTask,
    ) -> ProcessStopReason | None:
        task = await self._get_task(managed.task_id)

        if task is None:
            return ProcessStopReason(
                kind="cancelled",
                stage="Остановлено",
                message="Задача удалена во время выполнения",
            )

        if task.status == tasks_constants.statuses.cancelled:
            return ProcessStopReason(
                kind="cancelled",
                stage="Отменено",
                message="Задача отменена пользователем, subprocess остановлен",
            )

        timeout_reason = self._check_heartbeat_timeout(task)
        if timeout_reason is not None:
            return timeout_reason

        resource_reason = self._check_resource_limits(managed)
        if resource_reason is None:
            managed.resource_violations = 0
            return None

        managed.resource_violations += 1

        log_event(
            logger,
            "warning",
            "task.runner.process_resource_warning",
            task_id=managed.task_id,
            task_type=managed.task_type,
            pid=managed.pid,
            violation=managed.resource_violations,
            grace=settings.TASK_RUNNER_SERVICE_RESOURCE_GRACE_TICKS,
            reason=resource_reason,
        )

        if (
            managed.resource_violations
            < settings.TASK_RUNNER_SERVICE_RESOURCE_GRACE_TICKS
        ):
            return None

        managed.resource_violations = 0
        return ProcessStopReason(
            kind="failed",
            stage="Остановлено по ресурсам",
            message="Subprocess задачи остановлен из-за превышения лимитов ресурсов",
            error=resource_reason,
        )

    def _check_heartbeat_timeout(self, task: Task) -> ProcessStopReason | None:
        timeout_sec = settings.TASK_RUNNER_HEARTBEAT_TIMEOUT_SEC
        if timeout_sec <= 0:
            return None

        last_signal = task.heartbeat_at or task.started_at or task.created_at
        if last_signal is None:
            return None

        age_sec = (_now() - last_signal).total_seconds()

        if age_sec <= timeout_sec:
            return None

        return ProcessStopReason(
            kind="failed",
            stage="Задача зависла",
            message=f"Нет heartbeat {int(age_sec)}с (лимит {int(timeout_sec)}с)",
            error=f"Task heartbeat timeout: {int(age_sec)}s > {int(timeout_sec)}s",
        )

    def _check_resource_limits(self, managed: ManagedProcessTask) -> str | None:
        min_free_mb = settings.TASK_RUNNER_SERVICE_MIN_FREE_MEMORY_MB
        if min_free_mb > 0:
            available_mb = int(psutil.virtual_memory().available) // 1024 // 1024
            if available_mb < min_free_mb:
                return (
                    "свободная память системы ниже резерва: "
                    f"{available_mb} MB < {min_free_mb} MB"
                )

        memory_limit_mb = settings.TASK_RUNNER_SERVICE_MEMORY_LIMIT_MB
        if memory_limit_mb > 0:
            rss_mb = managed.rss_mb()
            if rss_mb > memory_limit_mb:
                return f"subprocess превысил память: {rss_mb} MB > {memory_limit_mb} MB"

        cpu_limit = settings.TASK_RUNNER_SERVICE_CPU_LIMIT_PERCENT
        if cpu_limit > 0:
            cpu_percent = managed.cpu_percent()
            if cpu_percent > cpu_limit:
                return f"subprocess превысил CPU: {cpu_percent:.1f}% > {cpu_limit:.1f}%"

        return None

    async def _terminate_process(self, managed: ManagedProcessTask) -> None:
        await asyncio.to_thread(
            managed.terminate,
            settings.TASK_RUNNER_SERVICE_STOP_TIMEOUT_SEC,
        )

        if managed.poll() is None:
            await asyncio.to_thread(managed.kill)

    async def _maybe_resume_training_after_task(
        self,
        task: ManagedProcessTask,
    ) -> None:
        if task.task_type != tasks_constants.types.inspection:
            return

        async with AsyncSessionLocal() as db:
            with suppress(Exception):
                await maybe_resume_paused_training(
                    db,
                    exclude_inspection_task_id=task.task_id,
                )

    async def _recover_active_tasks_on_startup(self) -> None:
        if not settings.TASK_RUNNER_FAIL_ACTIVE_ON_STARTUP:
            return

        active_statuses = {
            tasks_constants.statuses.running,
            tasks_constants.statuses.pausing,
            tasks_constants.statuses.resuming,
        }

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Task).where(Task.status.in_(active_statuses))
            )
            active_tasks = list(result.scalars().all())

            recovered_any = False

            for task in active_tasks:
                if task.status == tasks_constants.statuses.pausing:
                    await update_task_status(
                        db,
                        task_id=task.id,
                        status=tasks_constants.statuses.paused,
                        stage="Приостановлено",
                        message="Обучение было прервано перезапуском сервера и ожидает возобновления",
                        external_job_id=None,
                    )
                else:
                    await update_task_status(
                        db,
                        task_id=task.id,
                        status=tasks_constants.statuses.failed,
                        stage="Ошибка",
                        message="Задача была остановлена при перезапуске сервера",
                        error="Сервер был перезапущен во время выполнения subprocess задачи",
                        external_job_id=None,
                    )

                recovered_any = True

            if recovered_any:
                with suppress(Exception):
                    await maybe_resume_paused_training(db)

    async def _get_task(self, task_id: UUID) -> Task | None:
        async with AsyncSessionLocal() as db:
            return await db.get(Task, task_id)

    async def _mark_task_cancelled_if_not_terminal(
        self,
        task_id: UUID,
        *,
        stage: str,
        message: str,
    ) -> None:
        async with AsyncSessionLocal() as db:
            task = await db.get(Task, task_id)

            if task is None:
                return

            if task.status in tasks_constants.statuses.terminal:
                return

            await update_task_status(
                db,
                task_id=task_id,
                status=tasks_constants.statuses.cancelled,
                stage=stage,
                message=message,
                external_job_id=None,
            )

    async def _mark_task_failed_if_not_terminal(
        self,
        task_id: UUID,
        *,
        stage: str,
        message: str,
        error: str,
    ) -> None:
        async with AsyncSessionLocal() as db:
            task = await db.get(Task, task_id)

            if task is None:
                return

            if task.status in tasks_constants.statuses.terminal:
                return

            if task.status == tasks_constants.statuses.paused:
                return

            await update_task_status(
                db,
                task_id=task_id,
                status=tasks_constants.statuses.failed,
                stage=stage,
                message=message,
                error=error,
                external_job_id=None,
            )

    def _has_running_type(self, task_type: str) -> bool:
        return any(task.task_type == task_type for task in self._tasks.values())

    def _has_running_any_type(self, task_types: set[str]) -> bool:
        return any(task.task_type in task_types for task in self._tasks.values())


def _start_task_process(task: ClaimedTask) -> ManagedProcessTask:
    logs_dir = settings.worker_logs_path / "processes" / _safe_log_part(task.type)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_stem = _build_process_log_stem(task_id=task.id, task_type=task.type)
    stdout_path = logs_dir / f"{log_stem}.out.log"
    stderr_path = logs_dir / f"{log_stem}.err.log"

    stdout_file = stdout_path.open("a", encoding="utf-8", buffering=1)
    stderr_file = stderr_path.open("a", encoding="utf-8", buffering=1)

    env = os.environ.copy()
    env["PYTHONPATH"] = _build_pythonpath(env.get("PYTHONPATH"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    parent_process = psutil.Process(os.getpid())
    env["PHOTOAPP_PARENT_PID"] = str(parent_process.pid)
    env["PHOTOAPP_PARENT_CREATE_TIME"] = str(parent_process.create_time())

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "infra.queue.task_executor",
                "--task-id",
                str(task.id),
                "--task-type",
                task.type,
            ],
            cwd=str(SERVER_ROOT),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            stdin=subprocess.DEVNULL,
            text=True,
            **_build_popen_platform_kwargs(),
        )

        create_time = _process_create_time(process.pid)
        prime_process_tree_cpu_percent(
            process.pid,
            expected_create_time=create_time,
        )
    except Exception:
        stdout_file.close()
        stderr_file.close()
        raise

    return ManagedProcessTask(
        task_id=task.id,
        task_type=task.type,
        queue=task.queue,
        process=process,
        create_time=create_time,
        started_at=time.perf_counter(),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )


def _process_create_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _process_job_id(pid: int) -> str:
    return f"process:{pid}"


def _build_process_log_stem(
    *,
    task_id: UUID,
    task_type: str,
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    short_id = str(task_id)[:8]
    safe_task_type = _safe_log_part(task_type)
    return f"{timestamp}_{safe_task_type}_{short_id}"


def _build_popen_platform_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        }

    return {
        "start_new_session": True,
    }


def _build_pythonpath(current: str | None) -> str:
    src = str(SRC_ROOT)
    if not current:
        return src

    parts = current.split(os.pathsep)
    if src in parts:
        return current

    return os.pathsep.join([src, current])


def _safe_log_part(value: str) -> str:
    safe_chars = []

    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    safe_value = "".join(safe_chars).strip("_")
    return safe_value or "unknown"


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
