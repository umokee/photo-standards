from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Literal

import numpy as np
from app.observability import log_event, throttled_log

from ..models import Camera
from ..urls import (
    build_camera_stream_url,
    build_camera_stream_url_no_auth,
)
from ._backends import HttpMjpegBackend, VideoBackend, VideoBackendImpl
import structlog

logger = structlog.get_logger(__name__)

CaptureState = Literal["connecting", "online", "reconnecting", "offline"]

_BACKOFF_SCHEDULE = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
_MAX_PERMANENT_FAILURES = 20
_MAX_READ_FAILURES = 10


class CameraCapture:
    def __init__(
        self,
        camera: Camera,
        on_state_change: Callable[[CaptureState, str | None], None] | None = None,
    ) -> None:
        self._camera = _CameraSnapshot(camera)
        self._on_state_change = on_state_change

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_at: float | None = None

        self._state: CaptureState = "connecting"
        self._last_error: str | None = None

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-capture-{self._camera.id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def get_latest_frame(self) -> tuple[np.ndarray, float] | None:
        with self._frame_lock:
            if self._latest_frame is None or self._latest_frame_at is None:
                return None
            return self._latest_frame, self._latest_frame_at

    def _run(self) -> None:
        consecutive_failures = 0

        while not self._stop_event.is_set():
            self._set_state(
                "reconnecting" if consecutive_failures > 0 else "connecting"
            )

            try:
                self._capture_session()
                consecutive_failures = 0
            except _CaptureBroken as exc:
                consecutive_failures += 1
                self._last_error = str(exc)
                log_event(
                    logger,
                    "warning",
                    "camera.capture.broken",
                    camera_id=self._camera.id,
                    attempt=consecutive_failures,
                    reason=str(exc),
                )
            except Exception as exc:
                consecutive_failures += 1
                self._last_error = "Внутренняя ошибка захвата"
                throttled_log(
                    logger,
                    event="camera.stream.error",
                    repeated_event="camera.stream.error.repeated",
                    error=exc,
                    throttle_key=f"camera-capture:{self._camera.id}",
                    interval_sec=10.0,
                    camera_id=self._camera.id,
                    attempt=consecutive_failures,
                )

            if self._stop_event.is_set():
                break

            if consecutive_failures >= _MAX_PERMANENT_FAILURES:
                self._set_state(
                    "offline",
                    "Камера недоступна после многократных попыток",
                )
                log_event(
                    logger,
                    "error",
                    "camera.stream.offline",
                    camera_id=self._camera.id,
                    reason="permanent_capture_failure",
                )
                return

            backoff = _BACKOFF_SCHEDULE[
                min(consecutive_failures - 1, len(_BACKOFF_SCHEDULE) - 1)
            ]
            self._stop_event.wait(timeout=backoff)

        self._set_state("offline")

    def _capture_session(self) -> None:
        backend = self._open_backend()

        try:
            self._set_state("online")
            consecutive_read_failures = 0

            while not self._stop_event.is_set():
                ok, frame = backend.read()

                if not ok or frame is None or frame.size == 0:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= _MAX_READ_FAILURES:
                        raise _CaptureBroken("Поток открыт, но камера не отдаёт кадры")
                    continue

                consecutive_read_failures = 0
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_at = time.monotonic()
                time.sleep(0.001)
        finally:
            backend.release()

    def _open_backend(self) -> VideoBackend:
        camera = self._camera

        if camera.protocol == "http":
            stream_url = build_camera_stream_url_no_auth(camera)
            if not camera.stream_path and not camera.path:
                raise _CaptureBroken("Для HTTP-камеры не указан path или stream_path")

            url = build_camera_stream_url_no_auth(camera)
            auth = (camera.username, camera.password or "") if camera.username else None
            backend = HttpMjpegBackend.open(
                url,
                timeout=float(camera.timeout_sec),
                auth=auth,
            )
            if backend is None:
                raise _CaptureBroken(
                    f"Не удалось открыть HTTP MJPEG-поток: {stream_url}"
                )
            return backend

        if camera.protocol == "rtsp":
            url = build_camera_stream_url(camera)
            safe_url = build_camera_stream_url_no_auth(camera)
            backend = VideoBackendImpl.open_rtsp(url)
            if backend is None:
                raise _CaptureBroken(f"Не удалось открыть RTSP-поток: {safe_url}")
            return backend

        if camera.protocol == "usb":
            if not camera.device_path:
                raise _CaptureBroken("Для USB-камеры не указан путь к устройству")

            device = _resolve_usb_device(camera.device_path)
            backend = VideoBackendImpl.open_usb(device)
            if backend is None:
                raise _CaptureBroken(
                    f"Не удалось открыть USB-устройство: {camera.device_path}"
                )
            return backend

        raise _CaptureBroken(f"Неподдерживаемый протокол: {camera.protocol}")

    def _set_state(self, new_state: CaptureState, error: str | None = None) -> None:
        if error is not None:
            self._last_error = error

        if self._state == new_state:
            return

        self._state = new_state
        callback = self._on_state_change
        if callback is None:
            return

        try:
            callback(new_state, self._last_error)
        except Exception as exc:
            throttled_log(
                logger,
                event="camera.stream.error",
                repeated_event="camera.stream.error.repeated",
                error=exc,
                throttle_key=f"camera-capture-callback:{self._camera.id}",
                interval_sec=10.0,
                camera_id=self._camera.id,
                state=new_state,
            )


class _CaptureBroken(Exception):
    pass


class _CameraSnapshot:
    __slots__ = (
        "id",
        "protocol",
        "host",
        "port",
        "path",
        "stream_path",
        "username",
        "password",
        "device_path",
        "timeout_sec",
    )

    def __init__(self, camera: Camera) -> None:
        self.id = camera.id
        self.protocol = camera.protocol
        self.host = camera.host
        self.port = camera.port
        self.path = camera.path
        self.stream_path = camera.stream_path
        self.username = camera.username
        self.password = camera.password
        self.device_path = camera.device_path
        self.timeout_sec = camera.timeout_sec


def _resolve_usb_device(device_path: str) -> int | str:
    stripped = device_path.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped
