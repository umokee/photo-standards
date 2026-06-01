from __future__ import annotations

from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from modules.yolo.inspection.constants import inspections
from modules.yolo.inspection.realtime.mjpeg import build_realtime_mjpeg_response
from modules.yolo.inspection.realtime.webrtc import create_realtime_webrtc_answer
from modules.yolo.inspection.use_cases.discard_inspection import discard_inspection
from modules.yolo.inspection.use_cases.get_inspection import get_inspection
from modules.yolo.inspection.use_cases.get_realtime_status import get_realtime_status
from modules.yolo.inspection.use_cases.list_history import list_history
from modules.yolo.inspection.use_cases.push_realtime_browser_frame import (
    push_realtime_browser_frame,
)
from modules.yolo.inspection.use_cases.realtime_session import stop_realtime_session
from modules.yolo.inspection.use_cases.save_inspection import save_inspection
from modules.yolo.inspection.use_cases.save_realtime_result import (
    save_realtime_session_snapshot,
)
from modules.yolo.inspection.use_cases.start_inspection import start_inspection

from . import presenters
from .schemas import (
    InspectionHistoryItemResponse,
    InspectionRealtimeSnapshotSaveRequest,
    InspectionRealtimeStatusResponse,
    InspectionResultResponse,
    InspectionSaveRequest,
    InspectionSaveResponse,
    InspectionStartResponse,
    InspectionWebRTCOfferRequest,
    InspectionWebRTCOfferResponse,
)

router = APIRouter(prefix="/inspection", tags=["yolo-inspection"])


@router.post("/run", response_model=InspectionStartResponse, status_code=202)
async def start_inspection_route(
    db: DbSession,
    request: Request,
    standard_id: UUID = Form(...),  # noqa: B008
    selected_segment_class_ids: list[UUID] = Form(...),  # noqa: B008
    camera_id: UUID | None = Form(None),  # noqa: B008
    mode: str = Form(inspections.modes.default),
    notes: str | None = Form(None),
    image: UploadFile | None = File(None),  # noqa: B008
) -> InspectionStartResponse:
    result = await start_inspection(
        db,
        standard_id=standard_id,
        selected_segment_class_ids=selected_segment_class_ids,
        camera_id=camera_id,
        mode=mode,
        notes=notes,
        image=image,
        stream_manager=request.app.state.camera_stream_manager,
        inspection_streamer=request.app.state.inspection_streamer,
    )
    return presenters.start_result(result)


@router.get(
    "/realtime/sessions/{session_id}/status",
    response_model=InspectionRealtimeStatusResponse,
)
def get_realtime_status_route(
    request: Request,
    session_id: UUID,
) -> InspectionRealtimeStatusResponse:
    return get_realtime_status(
        streamer=request.app.state.inspection_streamer,
        session_id=session_id,
    )


@router.delete("/realtime/sessions/{session_id}", status_code=204)
def stop_realtime_session_route(
    request: Request,
    session_id: UUID,
) -> None:
    stop_realtime_session(
        streamer=request.app.state.inspection_streamer,
        session_id=session_id,
    )


@router.get("/realtime/sessions/{session_id}/stream")
def stream_realtime_session(
    request: Request,
    session_id: UUID,
) -> StreamingResponse:
    return build_realtime_mjpeg_response(
        streamer=request.app.state.inspection_streamer,
        session_id=session_id,
    )


@router.post(
    "/realtime/sessions/{session_id}/snapshot",
    response_model=InspectionSaveResponse,
)
async def save_realtime_snapshot_route(
    request: Request,
    session_id: UUID,
    payload: InspectionRealtimeSnapshotSaveRequest,
    db: DbSession,
) -> InspectionSaveResponse:
    inspection = await save_realtime_session_snapshot(
        db,
        streamer=request.app.state.inspection_streamer,
        session_id=session_id,
        notes=payload.notes,
    )
    return presenters.save_result(inspection, message="Снимок сохранён")


@router.post("/realtime/sessions/{session_id}/frames", status_code=204)
async def push_realtime_browser_frame_route(
    request: Request,
    session_id: UUID,
    frame: UploadFile = File(...),  # noqa: B008
) -> None:
    await push_realtime_browser_frame(
        streamer=request.app.state.inspection_streamer,
        session_id=session_id,
        frame=frame,
    )


@router.post(
    "/realtime/sessions/{session_id}/webrtc/offer",
    response_model=InspectionWebRTCOfferResponse,
)
async def create_realtime_webrtc_offer_answer(
    request: Request,
    session_id: UUID,
    payload: InspectionWebRTCOfferRequest,
) -> InspectionWebRTCOfferResponse:
    return await create_realtime_webrtc_answer(
        streamer=request.app.state.inspection_streamer,
        session_id=session_id,
        payload=payload,
    )


@router.get("/history", response_model=list[InspectionHistoryItemResponse])
async def list_inspection_history_route(
    db: DbSession,
    group_id: UUID | None = Query(None),  # noqa: B008
) -> list[InspectionHistoryItemResponse]:
    items = await list_history(db, group_id=group_id)
    return [presenters.history_item(item) for item in items]


@router.get("/{inspection_id}", response_model=InspectionResultResponse)
async def get_inspection_route(
    db: DbSession,
    inspection_id: UUID,
) -> InspectionResultResponse:
    inspection = await get_inspection(db, inspection_id=inspection_id)
    return presenters.inspection_result(inspection)


@router.post("/save", response_model=InspectionSaveResponse)
async def save_inspection_route(
    db: DbSession,
    payload: InspectionSaveRequest,
) -> InspectionSaveResponse:
    inspection = await save_inspection(
        db,
        task_id=payload.task_id,
        notes=payload.notes,
    )
    return presenters.save_result(inspection, message="Результат проверки сохранён")


@router.delete("/task/{task_id}", status_code=204)
async def discard_inspection_route(
    db: DbSession,
    task_id: UUID,
) -> None:
    await discard_inspection(db, task_id=task_id)
