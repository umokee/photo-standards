from __future__ import annotations

import logging
import os
import platform
import threading
import time
from contextlib import suppress

import cv2
import numpy as np

from ._base import VideoBackend

logger = logging.getLogger(__name__)


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

_READ_TIMEOUT_SEC = 1.0
_GRAB_FAILURE_SLEEP = 0.01


class VideoBackendImpl(VideoBackend):
    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._capture_lock = threading.Lock()

        self._grab_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_available = threading.Event()
        self._grab_error: Exception | None = None
        self._opened = True

    @classmethod
    def open_rtsp(cls, url: str) -> VideoBackendImpl | None:
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        return cls._wrap(capture, "rtsp")

    @classmethod
    def open_usb(cls, device: int | str) -> VideoBackendImpl | None:
        normalized_device = _normalize_usb_device(device)

        if isinstance(normalized_device, int) and platform.system() == "Windows":
            capture = cv2.VideoCapture(normalized_device, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(normalized_device)

        return cls._wrap(capture, "usb")

    @classmethod
    def _wrap(
        cls,
        capture: cv2.VideoCapture,
        kind: str,
    ) -> VideoBackendImpl | None:
        if not capture.isOpened():
            with suppress(Exception):
                capture.release()
            return None

        with suppress(Exception):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        backend = cls(capture)
        backend._start_grab_thread(kind)
        return backend

    def is_opened(self) -> bool:
        return self._opened and self._grab_error is None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_opened():
            return False, None

        if not self._frame_available.wait(timeout=_READ_TIMEOUT_SEC):
            return False, None

        with self._capture_lock:
            ok, frame = self._capture.retrieve()
            self._frame_available.clear()

        if not ok or frame is None or frame.size == 0:
            return False, None
        return True, frame

    def release(self) -> None:
        self._opened = False
        self._stop_event.set()
        self._frame_available.set()

        if self._grab_thread is not None and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2.0)

        self._grab_thread = None

        with self._capture_lock, suppress(Exception):
            self._capture.release()

    def _start_grab_thread(self, kind: str) -> None:
        self._grab_thread = threading.Thread(
            target=self._grab_loop,
            name=f"video-grab-{kind}-{id(self)}",
            daemon=True,
        )
        self._grab_thread.start()

    def _grab_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._capture_lock:
                    ok = self._capture.grab()

                if not ok:
                    time.sleep(_GRAB_FAILURE_SLEEP)
                    continue

                self._frame_available.set()

        except Exception as exc:
            self._grab_error = exc
            logger.debug(
                "video_backend.grab_error",
                extra={"error": str(exc)},
            )


def _normalize_usb_device(device: int | str) -> int | str:
    if isinstance(device, int):
        return device

    value = str(device).strip()

    if value.isdigit():
        return int(value)

    return value
