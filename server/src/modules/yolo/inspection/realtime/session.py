from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import suppress
from uuid import UUID

import cv2
import numpy as np
import structlog
from app.exception import ValidationError
from app.observability import log_event
from infra.storage.file_storage import resolve_storage_path
from modules.cameras.models import Camera
from modules.core.standards.reference_runtime import warmup_reference_matching
from modules.yolo.inspection.adapters.context import InspectionContext
from modules.yolo.inspection.adapters.yolo import warmup_model
from modules.yolo.inspection.realtime.browser_source import BrowserFrameSource

from .constants import JPEG_QUALITY, MAX_FRAME_AGE_SEC
from .frame_processor import RealtimeFrameProcessor
from .frame_result import FrameResult
from .profiler import FrameProfiler

logger = structlog.get_logger(__name__)


class InspectionSession:
    def __init__(
        self,
        *,
        session_id: UUID,
        context: InspectionContext,
        camera: Camera | None,
        selected_class_ids: list[UUID],
        frame_source,
    ) -> None:
        self.session_id = session_id
        self.context = context
        self.camera = camera
        self.selected_class_ids = selected_class_ids

        self._stopped = threading.Event()
        self._frame_source = frame_source
        self._state = _RealtimeFrameState(frame_source=frame_source)
        self._processor = RealtimeFrameProcessor(
            session_id=session_id,
            context=context,
        )
        self._failed = False
        self._failure_message: str | None = None
        self._models_warmed_up = False

        now = time.monotonic()
        self._created_at = now
        self._last_client_activity_at = now
        self._last_browser_frame_at: float | None = None
        self._last_processed_frame_at: float | None = None
        self._activity_lock = threading.Lock()

        self._stop_callbacks: list[Callable[[], None]] = []
        self._stop_callbacks_lock = threading.Lock()

    @property
    def stopped(self) -> threading.Event:
        return self._stopped

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def failure_message(self) -> str | None:
        return self._failure_message

    @property
    def is_browser_source(self) -> bool:
        return isinstance(self._frame_source, BrowserFrameSource)

    def touch(self) -> None:
        with self._activity_lock:
            self._last_client_activity_at = time.monotonic()

    def touch_browser_frame(self) -> None:
        now = time.monotonic()
        with self._activity_lock:
            self._last_client_activity_at = now
            self._last_browser_frame_at = now

    def add_stop_callback(self, callback: Callable[[], None]) -> None:
        should_call_now = False

        with self._stop_callbacks_lock:
            if self._stopped.is_set():
                should_call_now = True
            else:
                self._stop_callbacks.append(callback)

        if should_call_now:
            _safe_call_stop_callback(callback)

    def should_stop_for_idle(
        self,
        *,
        now: float,
        idle_timeout_sec: float,
        browser_frame_timeout_sec: float,
        failed_idle_timeout_sec: float,
    ) -> str | None:
        with self._activity_lock:
            created_at = self._created_at
            last_client_activity_at = self._last_client_activity_at
            last_browser_frame_at = self._last_browser_frame_at

        if self._stopped.is_set():
            if self._failed:
                if now - last_client_activity_at >= failed_idle_timeout_sec:
                    return "failed_idle_timeout"
                return None

            return "stopped"

        if self.is_browser_source:
            if last_browser_frame_at is None:
                if now - created_at >= browser_frame_timeout_sec:
                    return "browser_frame_start_timeout"
            elif now - last_browser_frame_at >= browser_frame_timeout_sec:
                return "browser_frame_idle_timeout"

        if now - last_client_activity_at >= idle_timeout_sec:
            return "client_idle_timeout"

        return None

    def stop(self) -> None:
        if self._stopped.is_set():
            return

        self._stopped.set()
        self._run_stop_callbacks()

        with suppress(Exception):
            self._frame_source.close_threadsafe(timeout=2.0)

        self._processor.close()
        self._state.clear()
        FrameProfiler.clear_session(str(self.session_id))

    def get_latest_video_frame(self) -> tuple[np.ndarray | None, int]:
        self.touch()
        return self._state.get_latest_video_frame()

    def get_latest_jpeg_since(self, last_version: int) -> tuple[int, bytes] | None:
        self.touch()
        return self._state.get_latest_jpeg_since(last_version)

    def get_latest_jpeg(self) -> bytes | None:
        self.touch()
        result = self.get_latest_jpeg_since(-1)
        return result[1] if result is not None else None

    def get_latest_result(self) -> FrameResult | None:
        self.touch()
        return self._state.get_latest_result()

    def get_frame_for_save(self) -> tuple[np.ndarray | None, FrameResult | None]:
        self.touch()
        return self._state.get_frame_for_save()

    def push_browser_frame(self, data: bytes) -> None:
        if not isinstance(self._frame_source, BrowserFrameSource):
            raise ValidationError("Сессия не принимает кадры браузерной камеры")

        self._frame_source.push_jpeg(data)
        self.touch_browser_frame()

    def push_browser_video_frame(self, frame: np.ndarray) -> None:
        if not isinstance(self._frame_source, BrowserFrameSource):
            raise ValidationError("Сессия не принимает кадры браузерной камеры")

        self._frame_source.push_frame(frame)
        self.touch_browser_frame()

    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:
            self._failed = True
            self._failure_message = str(exc) or type(exc).__name__
            log_event(
                logger,
                "error",
                "realtime.session.failed",
                session_id=self.session_id,
                camera_id=self.camera.id if self.camera is not None else None,
                error_type=type(exc).__name__,
                exception=exc,
            )
        finally:
            self._frame_source.close_threadsafe(timeout=2.0)
            self._processor.close()
            self._run_stop_callbacks()
            FrameProfiler.clear_session(str(self.session_id))
            self._stopped.set()

    def _ensure_models_warmed_up(self) -> None:
        if self._models_warmed_up:
            return

        weights_path = resolve_storage_path(self.context.model.weights_path)
        warmup_model(
            weights_path=weights_path,
            imgsz=self.context.model.imgsz,
        )
        warmup_reference_matching()
        self._models_warmed_up = True

    def _loop(self) -> None:
        last_frame_at: float | None = None

        while not self._stopped.is_set():
            frame_pair = self._frame_source.get_latest_frame()
            if frame_pair is None:
                time.sleep(0.02)
                continue

            frame, frame_at = frame_pair
            frame_age = time.monotonic() - frame_at

            if frame_age > MAX_FRAME_AGE_SEC:
                time.sleep(0.01)
                continue

            if last_frame_at is not None and frame_at <= last_frame_at:
                time.sleep(0.005)
                continue

            last_frame_at = frame_at
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> None:
        self._ensure_models_warmed_up()
        processed = self._processor.process(frame)
        if processed is None:
            return

        self._state.publish(
            rendered=processed.rendered,
            raw_frame=frame,
            result=processed.result,
        )

        with self._activity_lock:
            self._last_processed_frame_at = time.monotonic()

    def _run_stop_callbacks(self) -> None:
        with self._stop_callbacks_lock:
            callbacks = list(self._stop_callbacks)
            self._stop_callbacks.clear()

        for callback in callbacks:
            _safe_call_stop_callback(callback)


class _RealtimeFrameState:
    def __init__(
        self,
        *,
        frame_source,
    ) -> None:
        self._frame_source = frame_source

        self._lock = threading.Lock()

        self._latest_rendered: np.ndarray | None = None
        self._latest_result: FrameResult | None = None
        self._latest_raw_frame: np.ndarray | None = None

        self._latest_source_frame_at: float | None = None
        self._latest_video_version = 0

        self._latest_jpeg: bytes | None = None
        self._latest_jpeg_version = 0

    def publish(
        self,
        *,
        rendered: np.ndarray,
        raw_frame: np.ndarray,
        result: FrameResult,
    ) -> None:
        with self._lock:
            self._latest_rendered = rendered
            self._latest_result = result
            self._latest_raw_frame = raw_frame.copy()
            self._latest_jpeg = None
            self._latest_jpeg_version = 0
            self._latest_video_version += 1

    def clear(self) -> None:
        with self._lock:
            self._latest_rendered = None
            self._latest_result = None
            self._latest_raw_frame = None
            self._latest_source_frame_at = None
            self._latest_video_version = 0
            self._latest_jpeg = None
            self._latest_jpeg_version = 0

    def get_latest_video_frame(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest_rendered is not None:
                return self._latest_rendered.copy(), self._latest_video_version

        frame_pair = self._frame_source.get_latest_frame()
        if frame_pair is None:
            with self._lock:
                return None, self._latest_video_version

        frame, frame_at = frame_pair

        with self._lock:
            if (
                self._latest_source_frame_at is None
                or frame_at > self._latest_source_frame_at
            ):
                self._latest_source_frame_at = frame_at
                self._latest_video_version += 1

            return frame.copy(), self._latest_video_version

    def get_latest_jpeg_since(self, last_version: int) -> tuple[int, bytes] | None:
        frame_to_encode, version = self._get_frame_for_jpeg(last_version)
        if frame_to_encode is None:
            return None

        jpeg = _encode_jpeg(frame_to_encode)
        if jpeg is None:
            return None

        with self._lock:
            if version != self._latest_video_version:
                return None

            self._latest_jpeg = jpeg
            self._latest_jpeg_version = version
            return version, jpeg

    def get_latest_result(self) -> FrameResult | None:
        with self._lock:
            return self._latest_result

    def get_frame_for_save(self) -> tuple[np.ndarray | None, FrameResult | None]:
        with self._lock:
            if self._latest_raw_frame is None:
                return None, None

            return self._latest_raw_frame.copy(), self._latest_result

    def _get_frame_for_jpeg(
        self,
        last_version: int,
    ) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest_rendered is None:
                return None, self._latest_video_version

            version = self._latest_video_version
            if version <= last_version:
                return None, version

            if self._latest_jpeg is not None and self._latest_jpeg_version == version:
                return None, version

            return self._latest_rendered.copy(), version


def _encode_jpeg(frame: np.ndarray) -> bytes | None:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    return encoded.tobytes() if ok else None


def _safe_call_stop_callback(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception as exc:
        log_event(
            logger,
            "warning",
            "realtime.session.stop_callback_failed",
            error_type=type(exc).__name__,
            exception=exc,
        )
