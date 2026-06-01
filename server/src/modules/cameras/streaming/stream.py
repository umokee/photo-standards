from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

import cv2
import numpy as np
from app.observability import log_event, throttled_log

from ..models import Camera
from .capture import CameraCapture, CaptureState
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_SUBSCRIBER_FPS = 5
MIN_SUBSCRIBER_FPS = 1
MAX_SUBSCRIBER_FPS = 30

_SUBSCRIBER_QUEUE_MAXSIZE = 4

_JPEG_QUALITY = 75

SubscriberMessage = (
    tuple[Literal["frame"], bytes] | tuple[Literal["status"], dict[str, Any]]
)

OnEmptyCallback = Callable[["CameraStream"], None]
StateChangeCallback = Callable[[CaptureState, str | None], None]


@dataclass(slots=True)
class Subscriber:
    subscriber_id: UUID
    queue: asyncio.Queue[SubscriberMessage]
    fps: int
    _send_task: asyncio.Task[None] | None = field(default=None)


class CameraStream:
    def __init__(
        self,
        camera: Camera,
        loop: asyncio.AbstractEventLoop,
        on_empty: OnEmptyCallback | None = None,
        on_state_change: StateChangeCallback | None = None,
    ) -> None:
        self._camera_id = camera.id
        self._loop = loop
        self._on_empty = on_empty
        self._on_state_change_external = on_state_change

        self._subscribers: dict[UUID, Subscriber] = {}
        self._lock = asyncio.Lock()

        self._current_state: CaptureState = "connecting"
        self._current_error: str | None = None

        self._capture = CameraCapture(
            camera,
            on_state_change=self._on_capture_state_change,
        )

    @property
    def camera_id(self) -> UUID:
        return self._camera_id

    @property
    def state(self) -> CaptureState:
        return self._current_state

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def start(self) -> None:
        self._capture.start()

    async def stop(self) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.values())
            self._subscribers.clear()

        for subscriber in subscribers:
            self._cancel_send_task(subscriber)

        await asyncio.to_thread(self._capture.stop)

    async def subscribe(self, fps: int = DEFAULT_SUBSCRIBER_FPS) -> Subscriber:
        clamped_fps = _clamp_fps(fps)
        subscriber = Subscriber(
            subscriber_id=uuid4(),
            queue=asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAXSIZE),
            fps=clamped_fps,
        )

        async with self._lock:
            self._subscribers[subscriber.subscriber_id] = subscriber

        self._enqueue(
            subscriber,
            (
                "status",
                {
                    "state": self._current_state,
                    "message": self._current_error,
                },
            ),
        )

        subscriber._send_task = asyncio.create_task(self._send_loop(subscriber))
        return subscriber

    async def unsubscribe(self, subscriber_id: UUID) -> None:
        async with self._lock:
            subscriber = self._subscribers.pop(subscriber_id, None)
            became_empty = subscriber is not None and not self._subscribers

        if subscriber is not None:
            self._cancel_send_task(subscriber)

        if became_empty and self._on_empty is not None:
            try:
                self._on_empty(self)
            except Exception as exc:  # noqa: BLE001
                throttled_log(
                    logger,
                    event="camera.stream.error",
                    repeated_event="camera.stream.error.repeated",
                    error=exc,
                    throttle_key=f"camera-stream-empty:{self._camera_id}",
                    interval_sec=10.0,
                    camera_id=self._camera_id,
                )

    def get_latest_frame(self) -> tuple[np.ndarray, float] | None:
        return self._capture.get_latest_frame()

    async def wait_for_online_frame(self, timeout: float) -> np.ndarray:
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            if self._current_state == "online":
                frame_pair = await asyncio.to_thread(self._capture.get_latest_frame)
                if frame_pair is not None:
                    return frame_pair[0].copy()

            if self._current_state == "offline":
                error = self._current_error or "Камера недоступна"
                raise TimeoutError(error)

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                if self._current_error:
                    raise TimeoutError(self._current_error)
                raise TimeoutError("Камера не отдала кадр за отведённое время")

            await asyncio.sleep(0.1)

    async def update_subscriber_fps(self, subscriber_id: UUID, fps: int) -> None:
        clamped_fps = _clamp_fps(fps)
        async with self._lock:
            subscriber = self._subscribers.get(subscriber_id)
            if subscriber is None:
                return
            subscriber.fps = clamped_fps


    async def _send_loop(self, subscriber: Subscriber) -> None:
        try:
            while True:
                interval = 1.0 / max(subscriber.fps, MIN_SUBSCRIBER_FPS)
                await asyncio.sleep(interval)

                async with self._lock:
                    if subscriber.subscriber_id not in self._subscribers:
                        return

                if self._current_state != "online":
                    continue

                frame_pair = await asyncio.to_thread(self._capture.get_latest_frame)
                if frame_pair is None:
                    continue

                frame, _captured_at = frame_pair
                jpeg = await asyncio.to_thread(_encode_jpeg, frame)
                if jpeg is None:
                    continue

                self._enqueue(subscriber, ("frame", jpeg))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            throttled_log(
                logger,
                event="camera.stream.error",
                repeated_event="camera.stream.error.repeated",
                error=exc,
                throttle_key=f"camera-stream-send:{self._camera_id}",
                interval_sec=10.0,
                camera_id=self._camera_id,
                subscriber_id=subscriber.subscriber_id,
            )

    def _enqueue(self, subscriber: Subscriber, message: SubscriberMessage) -> None:
        queue = subscriber.queue
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    def _cancel_send_task(self, subscriber: Subscriber) -> None:
        task = subscriber._send_task
        if task is None or task.done():
            return
        task.cancel()

    def _on_capture_state_change(
        self, new_state: CaptureState, error: str | None
    ) -> None:
        try:
            self._loop.call_soon_threadsafe(self._handle_state_change, new_state, error)
        except RuntimeError:
            pass

    def _handle_state_change(self, new_state: CaptureState, error: str | None) -> None:
        self._current_state = new_state
        self._current_error = error

        message: SubscriberMessage = (
            "status",
            {"state": new_state, "message": error},
        )
        for subscriber in list(self._subscribers.values()):
            self._enqueue(subscriber, message)

        if self._on_state_change_external is not None:
            try:
                self._on_state_change_external(new_state, error)
            except Exception as exc:  # noqa: BLE001
                throttled_log(
                    logger,
                    event="camera.stream.error",
                    repeated_event="camera.stream.error.repeated",
                    error=exc,
                    throttle_key=f"camera-stream-callback:{self._camera_id}",
                    interval_sec=10.0,
                    camera_id=self._camera_id,
                )


def _clamp_fps(fps: int) -> int:
    if fps < MIN_SUBSCRIBER_FPS:
        return MIN_SUBSCRIBER_FPS
    if fps > MAX_SUBSCRIBER_FPS:
        return MAX_SUBSCRIBER_FPS
    return fps


def _encode_jpeg(frame: np.ndarray) -> bytes | None:
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    )
    if not ok:
        return None
    return encoded.tobytes()
