from __future__ import annotations

import argparse
import asyncio
import contextvars
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from uuid import UUID

import psutil
import structlog
from app.asyncio_compat import configure_asyncio_policy
from app.config import settings
from app.db import dispose_engines
from app.import_models import import_models
from app.logging import configure_logging
from app.observability import (
    bind_context,
    build_json_formatter,
    clear_context,
    elapsed_ms,
    log_event,
)
from modules.tasks.constants import tasks as tasks_constants

logger = structlog.get_logger(__name__)

_CURRENT_TASK_LOG_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_log_id",
    default=None,
)
_PARENT_WATCHDOG_INTERVAL_SEC = 2.0


@dataclass(frozen=True, slots=True)
class TaskLogPaths:
    stdout_path: Path
    stderr_path: Path


class _CurrentTaskLogFilter(logging.Filter):
    def __init__(self, task_log_id: str) -> None:
        super().__init__()
        self._task_log_id = task_log_id

    def filter(self, record: logging.LogRecord) -> bool:
        return _CURRENT_TASK_LOG_ID.get() == self._task_log_id


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self._max_level


class TaskLogContext:
    def __init__(self, *, task_id: UUID, task_type: str) -> None:
        self.task_log_id = str(task_id)
        self.task_type = task_type
        self.paths = build_task_log_paths(task_id=task_id, task_type=task_type)

        self._stdout_file = None
        self._stderr_file = None
        self._stdout_handler: logging.Handler | None = None
        self._stderr_handler: logging.Handler | None = None
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> TaskLogPaths:
        self.paths.stdout_path.parent.mkdir(parents=True, exist_ok=True)

        self._stdout_file = self.paths.stdout_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._stderr_file = self.paths.stderr_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

        self._token = _CURRENT_TASK_LOG_ID.set(self.task_log_id)

        formatter = build_json_formatter()

        stdout_filter = _CurrentTaskLogFilter(self.task_log_id)
        stderr_filter = _CurrentTaskLogFilter(self.task_log_id)

        self._stdout_handler = logging.StreamHandler(self._stdout_file)
        self._stdout_handler.setLevel(logging.INFO)
        self._stdout_handler.setFormatter(formatter)
        self._stdout_handler.addFilter(stdout_filter)
        self._stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))

        self._stderr_handler = logging.StreamHandler(self._stderr_file)
        self._stderr_handler.setLevel(logging.WARNING)
        self._stderr_handler.setFormatter(formatter)
        self._stderr_handler.addFilter(stderr_filter)

        root = logging.getLogger()
        root.addHandler(self._stdout_handler)
        root.addHandler(self._stderr_handler)
        bind_context(task_id=self.task_log_id, task_type=self.task_type)

        return self.paths

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        root = logging.getLogger()

        if self._stdout_handler is not None:
            root.removeHandler(self._stdout_handler)
            self._stdout_handler.close()
            self._stdout_handler = None

        if self._stderr_handler is not None:
            root.removeHandler(self._stderr_handler)
            self._stderr_handler.close()
            self._stderr_handler = None

        if self._token is not None:
            _CURRENT_TASK_LOG_ID.reset(self._token)
            self._token = None

        clear_context()

        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

        if self._stdout_file is not None:
            self._stdout_file.close()
            self._stdout_file = None

        return False


def build_task_log_paths(
    *,
    task_id: UUID,
    task_type: str,
) -> TaskLogPaths:
    logs_dir = settings.worker_logs_path / "tasks" / _safe_log_part(task_type)
    log_stem = _build_task_log_stem(task_id=task_id, task_type=task_type)

    return TaskLogPaths(
        stdout_path=logs_dir / f"{log_stem}.out.log",
        stderr_path=logs_dir / f"{log_stem}.err.log",
    )


async def execute_task(
    *,
    task_id: UUID,
    task_type: str,
) -> None:
    started_at = time.perf_counter()
    with TaskLogContext(task_id=task_id, task_type=task_type) as paths:
        log_event(
            logger,
            "info",
            "task.executor.started",
            task_id=task_id,
            task_type=task_type,
            stdout_path=paths.stdout_path,
            stderr_path=paths.stderr_path,
            pid=os.getpid(),
        )
        try:
            await _execute_task(task_id=task_id, task_type=task_type)
        except Exception as exc:
            log_event(
                logger,
                "error",
                "task.executor.failed",
                task_id=task_id,
                task_type=task_type,
                duration_ms=elapsed_ms(started_at),
                error_type=type(exc).__name__,
                exception=exc,
            )
            raise

        log_event(
            logger,
            "info",
            "task.executor.finished",
            task_id=task_id,
            task_type=task_type,
            duration_ms=elapsed_ms(started_at),
            pid=os.getpid(),
        )


async def _execute_task(
    *,
    task_id: UUID,
    task_type: str,
) -> None:
    task_id_text = str(task_id)

    if task_type == tasks_constants.types.training:
        from modules.yolo.training.jobs import execute_training

        await execute_training(task_id=task_id_text)
        return

    if task_type == tasks_constants.types.inspection:
        from modules.yolo.inspection.jobs import execute_inspection

        await execute_inspection(task_id=task_id_text)
        return

    if task_type == tasks_constants.types.model_import:
        from modules.yolo.interop.jobs import execute_model_import

        await execute_model_import(task_id=task_id_text)
        return

    raise ValueError(f"Неизвестный тип задачи: {task_type}")


def current_service_job_id() -> str:
    return current_process_job_id()


def current_process_job_id() -> str:
    return f"process:{os.getpid()}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        asyncio.run(_main_async(args))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        return 1


async def _main_async(args: argparse.Namespace) -> None:
    configure_asyncio_policy()
    configure_logging(debug=settings.DEBUG)
    import_models()
    _ensure_storage_dirs()

    stop_parent_watchdog = _start_parent_watchdog()

    try:
        await execute_task(task_id=args.task_id, task_type=args.task_type)
    finally:
        stop_parent_watchdog()
        await dispose_engines()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one queued task in subprocess")
    parser.add_argument("--task-id", required=True, type=UUID)
    parser.add_argument("--task-type", required=True)
    return parser.parse_args(argv)


def _ensure_storage_dirs() -> None:
    settings.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    settings.standards_storage_path.mkdir(parents=True, exist_ok=True)
    settings.inspections_storage_path.mkdir(parents=True, exist_ok=True)
    settings.models_storage_path.mkdir(parents=True, exist_ok=True)
    settings.logs_storage_path.mkdir(parents=True, exist_ok=True)
    (settings.STORAGE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
    (settings.STORAGE_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)


def _start_parent_watchdog() -> Callable[[], None]:
    parent_pid = _parse_int(os.getenv("PHOTOAPP_PARENT_PID"))
    parent_create_time = _parse_float(os.getenv("PHOTOAPP_PARENT_CREATE_TIME"))

    if parent_pid is None:
        return lambda: None

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_parent_watchdog_loop,
        kwargs={
            "parent_pid": parent_pid,
            "parent_create_time": parent_create_time,
            "stop_event": stop_event,
        },
        name="task-parent-watchdog",
        daemon=True,
    )
    thread.start()

    def stop() -> None:
        stop_event.set()
        thread.join(timeout=1.0)

    return stop


def _parent_watchdog_loop(
    *,
    parent_pid: int,
    parent_create_time: float | None,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(_PARENT_WATCHDOG_INTERVAL_SEC):
        if _parent_process_is_alive(parent_pid, parent_create_time):
            continue

        log_event(
            logger,
            "warning",
            "task.executor.parent_dead",
            parent_pid=parent_pid,
            pid=os.getpid(),
        )
        os._exit(2)


def _parent_process_is_alive(
    parent_pid: int,
    parent_create_time: float | None,
) -> bool:
    try:
        process = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return False

    if parent_create_time is not None:
        with suppress(psutil.Error):
            if abs(process.create_time() - parent_create_time) > 1.0:
                return False

    with suppress(psutil.Error):
        if process.status() == psutil.STATUS_ZOMBIE:
            return False

    return True


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped.isdigit():
        return None

    return int(stripped)


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _build_task_log_stem(
    *,
    task_id: UUID,
    task_type: str,
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    short_id = str(task_id)[:8]
    safe_task_type = _safe_log_part(task_type)
    return f"{timestamp}_{safe_task_type}_{short_id}"


def _safe_log_part(value: str) -> str:
    safe_chars = []

    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    safe_value = "".join(safe_chars).strip("_")
    return safe_value or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
