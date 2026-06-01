from uuid import UUID

from app.exception import NotFoundError
from modules.yolo.inspection.realtime.streamer import (
    InspectionSession,
    InspectionStreamer,
)


def get_realtime_session_or_raise(
    *,
    streamer: InspectionStreamer,
    session_id: UUID,
) -> InspectionSession:
    session = streamer.get_session(session_id)
    if session is None:
        raise NotFoundError("Сессия", session_id)
    return session


def stop_realtime_session(
    *,
    streamer: InspectionStreamer,
    session_id: UUID,
) -> None:
    streamer.stop_session(session_id)
