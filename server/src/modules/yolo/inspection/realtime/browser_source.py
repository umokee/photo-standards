from __future__ import annotations

import asyncio
import threading
import time

import cv2
import numpy as np
from app.exception import ValidationError

from .constants import MAX_FRAME_AGE_SEC


class BrowserFrameSource:
    def __init__(self, *, max_frame_age_sec: float = MAX_FRAME_AGE_SEC) -> None:
        self._max_frame_age_sec = max_frame_age_sec
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_at: float | None = None

    def push_frame(self, frame: np.ndarray) -> None:
        if self._closed.is_set():
            raise ValidationError("Сессия реального времени уже остановлена")

        if frame is None or frame.size == 0:
            raise ValidationError("Камера устройства передала пустой кадр")

        with self._lock:
            self._latest_frame = frame.copy()
            self._latest_frame_at = time.monotonic()
            self._ready.set()

    def push_jpeg(self, data: bytes) -> None:
        if not data:
            raise ValidationError("Камера устройства передала пустой кадр")

        raw = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            raise ValidationError("Не удалось прочитать кадр с камеры устройства")

        self.push_frame(frame)

    async def wait_for_frame(self, timeout: float) -> np.ndarray:
        ready = await asyncio.to_thread(self._ready.wait, timeout)
        if not ready:
            raise TimeoutError("Браузерная камера не отправила кадр")

        frame_pair = self.get_latest_frame()
        if frame_pair is None:
            raise TimeoutError("Браузерная камера не отправила свежий кадр")

        frame, _ = frame_pair
        return frame.copy()

    def get_latest_frame(self) -> tuple[np.ndarray, float] | None:
        if self._closed.is_set():
            return None

        with self._lock:
            if self._latest_frame is None or self._latest_frame_at is None:
                return None

            if time.monotonic() - self._latest_frame_at > self._max_frame_age_sec:
                return None

            return self._latest_frame.copy(), self._latest_frame_at

    async def close(self) -> None:
        self._close()

    def close_threadsafe(self, timeout: float = 0) -> None:
        self._close()

    def _close(self) -> None:
        if self._closed.is_set():
            return

        self._closed.set()
        self._ready.set()

        with self._lock:
            self._latest_frame = None
            self._latest_frame_at = None
