from __future__ import annotations

import logging
import threading
from contextlib import suppress

import cv2
import numpy as np
import requests

from ._base import VideoBackend

logger = logging.getLogger(__name__)

_MIN_JPEG_BYTES = 256
_MAX_CORRUPT_FRAMES = 5
_HTTP_CHUNK_SIZE = 16 * 1024
_MAX_BUFFER_SIZE = 10 * 1024 * 1024
_READ_TIMEOUT_SEC = 1.0

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


class HttpMjpegBackend(VideoBackend):
    def __init__(
        self,
        url: str,
        *,
        timeout: float,
        auth: tuple[str, str] | None = None,
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._auth = auth

        self._session = requests.Session()
        self._response: requests.Response | None = None
        self._opened = False

        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._reader_error: Exception | None = None

        self._latest_jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._frame_available = threading.Event()

        self._corrupt_count = 0

    @classmethod
    def open(
        cls,
        url: str,
        *,
        timeout: float,
        auth: tuple[str, str] | None = None,
    ) -> HttpMjpegBackend | None:
        backend = cls(url, timeout=timeout, auth=auth)
        if backend._connect():
            return backend
        backend.release()
        return None

    def is_opened(self) -> bool:
        return self._opened and self._reader_error is None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_opened():
            return False, None

        if not self._frame_available.wait(timeout=_READ_TIMEOUT_SEC):
            return False, None

        with self._latest_jpeg_lock:
            jpeg_bytes = self._latest_jpeg
            self._latest_jpeg = None
            self._frame_available.clear()

        if jpeg_bytes is None:
            return False, None

        frame = _decode_jpeg(jpeg_bytes)
        if frame is None:
            self._corrupt_count += 1
            if self._corrupt_count >= _MAX_CORRUPT_FRAMES:
                logger.warning(
                    "http_mjpeg.too_many_corrupt_frames",
                    extra={"url": self._url, "count": self._corrupt_count},
                )
                self.release()
            return False, None

        self._corrupt_count = 0
        return True, frame

    def release(self) -> None:
        self._opened = False
        self._reader_stop.set()
        self._frame_available.set()

        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self._reader_thread = None

        if self._response is not None:
            with suppress(Exception):
                self._response.close()
            self._response = None

        with suppress(Exception):
            self._session.close()

    def _connect(self) -> bool:
        try:
            response = self._session.get(
                self._url,
                stream=True,
                timeout=self._timeout,
                auth=self._auth,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.debug(
                "http_mjpeg.connect_failed",
                extra={"url": self._url, "error": str(exc)},
            )
            return False

        content_type = response.headers.get("Content-Type", "")
        if "multipart" not in content_type.lower():
            response.close()
            logger.warning(
                "http_mjpeg.not_multipart",
                extra={"url": self._url, "content_type": content_type},
            )
            return False

        self._response = response
        self._opened = True

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"mjpeg-reader-{id(self)}",
            daemon=True,
        )
        self._reader_thread.start()
        return True

    def _reader_loop(self) -> None:
        if self._response is None:
            return

        buffer = bytearray()
        chunk_iter = self._response.iter_content(chunk_size=_HTTP_CHUNK_SIZE)

        try:
            while not self._reader_stop.is_set():
                while True:
                    soi = buffer.find(_JPEG_SOI)
                    if soi < 0:
                        if len(buffer) > 65536:
                            del buffer[:-32]
                        break

                    eoi = buffer.find(_JPEG_EOI, soi + 2)
                    if eoi < 0:
                        if soi > 0:
                            del buffer[:soi]
                        break

                    jpeg_bytes = bytes(buffer[soi : eoi + 2])
                    del buffer[: eoi + 2]

                    with self._latest_jpeg_lock:
                        self._latest_jpeg = jpeg_bytes
                    self._frame_available.set()

                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                if not chunk:
                    break

                buffer.extend(chunk)
                if len(buffer) > _MAX_BUFFER_SIZE:
                    logger.warning(
                        "http_mjpeg.buffer_overflow",
                        extra={"url": self._url, "size": len(buffer)},
                    )
                    buffer.clear()

        except Exception as exc:
            self._reader_error = exc
            logger.debug(
                "http_mjpeg.reader_error",
                extra={"url": self._url, "error": str(exc)},
            )


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    if len(jpeg_bytes) < _MIN_JPEG_BYTES:
        return None
    if not jpeg_bytes.startswith(_JPEG_SOI):
        return None
    if not jpeg_bytes.endswith(_JPEG_EOI):
        return None

    try:
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None

    if frame is None or frame.size == 0:
        return None
    return frame
