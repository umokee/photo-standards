from fastapi import UploadFile
from modules.yolo.inspection.realtime.streamer import InspectionStreamer


async def push_realtime_browser_frame(
    *,
    streamer: InspectionStreamer,
    session_id,
    frame: UploadFile,
) -> None:
    content = await frame.read()
    if not content:
        from app.exception import ValidationError

        raise ValidationError("Пустой кадр камеры устройства")

    streamer.push_browser_frame(session_id, content)
