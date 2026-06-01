from __future__ import annotations

import threading
import time
from uuid import UUID, uuid4

import structlog
from app.exception import NotFoundError
from app.observability import log_event
from modules.cameras.models import Camera
from modules.core.standards.reference_runtime import clear_reference_runtime_cache
from modules.yolo.inspection.adapters.context import InspectionContext
from modules.yolo.inspection.adapters.yolo import clear_model_cache
from modules.yolo.inspection.realtime.constants import (
    BROWSER_SESSION_FRAME_TIMEOUT_SEC,
    SESSION_FAILED_IDLE_TIMEOUT_SEC,
    SESSION_IDLE_TIMEOUT_SEC,
    SESSION_SWEEP_INTERVAL_SEC,
    SESSION_THREAD_JOIN_TIMEOUT_SEC,
)

from .frame_result import FrameResult
from .session import InspectionSession

logger = structlog.get_logger(__name__)


class InspectionStreamer:
    def __init__(self) -> None:
        self._sessions: dict[UUID, InspectionSession] = {}
        self._threads: dict[UUID, threading.Thread] = {}
        self._lock = threading.Lock()
        self._gpu_runtime_released_for_idle = True
        self._janitor_stop = threading.Event()
        self._janitor_thread = threading.Thread(
            target=self._janitor_loop,
            name="realtime-session-janitor",
            daemon=True,
        )
        self._janitor_thread.start()

    @property
    def active_count(self) -> int:
        with self._lock:
            self._remove_stopped_sessions_locked()
            self._release_gpu_runtime_if_idle_locked()
            return sum(1 for thread in self._threads.values() if thread.is_alive())

    def create_session(
        self,
        *,
        context: InspectionContext,
        camera: Camera | None,
        selected_class_ids: list[UUID],
        frame_source,
    ) -> InspectionSession:
        session = InspectionSession(
            session_id=uuid4(),
            context=context,
            camera=camera,
            selected_class_ids=selected_class_ids,
            frame_source=frame_source,
        )
        thread = threading.Thread(
            target=session.run,
            name=f"realtime-session-{session.session_id}",
            daemon=True,
        )

        with self._lock:
            self._sessions[session.session_id] = session
            self._threads[session.session_id] = thread
            self._gpu_runtime_released_for_idle = False

        thread.start()
        log_event(
            logger,
            "info",
            "realtime.session.started",
            session_id=session.session_id,
            camera_id=camera.id if camera is not None else None,
            standard_id=context.standard.id,
            group_id=context.standard.group_id,
            model_id=context.model.id,
            source_type="camera" if camera is not None else "browser",
        )
        return session

    def get_session(self, session_id: UUID) -> InspectionSession | None:
        with self._lock:
            self._release_gpu_runtime_if_idle_locked()
            return self._sessions.get(session_id)

    def push_browser_frame(self, session_id: UUID, data: bytes) -> None:
        session = self.get_session(session_id)
        if session is None:
            raise NotFoundError("Сессия", session_id)

        session.push_browser_frame(data)

    def stop_session(self, session_id: UUID) -> None:
        self._stop_session(session_id, reason="manual")

    def stop_all(self) -> None:
        self._janitor_stop.set()

        with self._lock:
            sessions = list(self._sessions.values())
            threads = list(self._threads.values())
            self._sessions.clear()
            self._threads.clear()

        for session in sessions:
            session.stop()
            log_event(
                logger,
                "info",
                "realtime.session.stopped",
                session_id=session.session_id,
                camera_id=session.camera.id if session.camera is not None else None,
                reason="manager_stop",
            )

        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

        if self._janitor_thread.is_alive():
            self._janitor_thread.join(timeout=2.0)

        with self._lock:
            self._release_gpu_runtime_if_idle_locked()

    def _release_gpu_runtime_if_idle_locked(self) -> None:
        if self._gpu_runtime_released_for_idle:
            return

        if any(thread.is_alive() for thread in self._threads.values()):
            return

        clear_model_cache()
        clear_reference_runtime_cache()
        self._gpu_runtime_released_for_idle = True
        log_event(
            logger,
            "info",
            "realtime.runtime.cache_cleared",
            active_sessions=0,
        )

    def cleanup_idle_sessions(self) -> None:
        now = time.monotonic()
        expired: list[tuple[UUID, str]] = []

        with self._lock:
            for session_id, session in list(self._sessions.items()):
                reason = session.should_stop_for_idle(
                    now=now,
                    idle_timeout_sec=SESSION_IDLE_TIMEOUT_SEC,
                    browser_frame_timeout_sec=BROWSER_SESSION_FRAME_TIMEOUT_SEC,
                    failed_idle_timeout_sec=SESSION_FAILED_IDLE_TIMEOUT_SEC,
                )
                if reason is not None:
                    expired.append((session_id, reason))

        for session_id, reason in expired:
            self._stop_session(session_id, reason=reason)

    def _stop_session(self, session_id: UUID, *, reason: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            thread = self._threads.pop(session_id, None)

        if session is not None:
            session.stop()
            log_event(
                logger,
                "info",
                "realtime.session.stopped",
                session_id=session_id,
                camera_id=session.camera.id if session.camera is not None else None,
                reason=reason,
            )

        if thread is not None and thread.is_alive():
            thread.join(timeout=SESSION_THREAD_JOIN_TIMEOUT_SEC)

        with self._lock:
            self._release_gpu_runtime_if_idle_locked()

    def _janitor_loop(self) -> None:
        while not self._janitor_stop.wait(SESSION_SWEEP_INTERVAL_SEC):
            try:
                self.cleanup_idle_sessions()
            except Exception as exc:
                log_event(
                    logger,
                    "error",
                    "realtime.session.janitor_failed",
                    error_type=type(exc).__name__,
                    exception=exc,
                )

    def _remove_stopped_sessions_locked(self) -> None:
        stopped_ids: list[UUID] = []

        for session_id, session in self._sessions.items():
            thread = self._threads.get(session_id)

            if session.failed:
                continue

            if not session.stopped.is_set():
                continue

            if thread is not None and thread.is_alive():
                continue

            stopped_ids.append(session_id)

        for session_id in stopped_ids:
            self._sessions.pop(session_id, None)
            self._threads.pop(session_id, None)


__all__ = ["FrameResult", "InspectionSession", "InspectionStreamer"]
