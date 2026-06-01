from __future__ import annotations

import time
from collections.abc import Iterator
from uuid import UUID

from fastapi.responses import StreamingResponse
from modules.yolo.inspection.realtime.streamer import InspectionStreamer
from modules.yolo.inspection.use_cases.realtime_session import (
    get_realtime_session_or_raise,
)

_FRAME_BOUNDARY = b"--frame"
_MJPEG_MEDIA_TYPE = "multipart/x-mixed-replace; boundary=frame"
_WAIT_NEXT_FRAME_SEC = 0.01


def build_realtime_mjpeg_response(
    *,
    streamer: InspectionStreamer,
    session_id: UUID,
) -> StreamingResponse:
    session = get_realtime_session_or_raise(
        streamer=streamer,
        session_id=session_id,
    )

    return StreamingResponse(
        _generate_mjpeg_frames(session),
        media_type=_MJPEG_MEDIA_TYPE,
    )


def _generate_mjpeg_frames(session) -> Iterator[bytes]:
    last_version = 0

    while not session.stopped.is_set():
        result = session.get_latest_jpeg_since(last_version)

        if result is None:
            time.sleep(_WAIT_NEXT_FRAME_SEC)
            continue

        last_version, jpeg = result

        yield _FRAME_BOUNDARY + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
        yield jpeg
        yield b"\r\n"
