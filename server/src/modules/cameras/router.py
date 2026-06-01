import asyncio
from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter, Query, Request

from . import service
from .schemas import (
    CameraCreate,
    CameraResponse,
    CameraTestResponse,
    CameraUpdate,
    CameraWebRTCOfferRequest,
    CameraWebRTCOfferResponse,
    UsbCameraDeviceResponse,
)
from .usb_discovery import discover_usb_cameras

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraResponse])
async def list_cameras(
    db: DbSession,
) -> list[CameraResponse]:
    return await service.list_cameras(db)


@router.get("/usb-devices", response_model=list[UsbCameraDeviceResponse])
async def list_usb_camera_devices(
    max_devices: int = Query(default=10, ge=1, le=32),
) -> list[UsbCameraDeviceResponse]:
    devices = await asyncio.to_thread(
        discover_usb_cameras,
        max_devices=max_devices,
    )
    return [
        UsbCameraDeviceResponse(
            name=device.name,
            device_path=device.device_path,
            index=device.index,
            width=device.width,
            height=device.height,
            backend=device.backend,
        )
        for device in devices
    ]


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    db: DbSession,
    camera_id: UUID,
) -> CameraResponse:
    return await service.get_camera(db, camera_id)


@router.post("", response_model=CameraResponse, status_code=201)
async def create_camera(
    db: DbSession,
    data: CameraCreate,
) -> CameraResponse:
    return await service.create_camera(db, data)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    db: DbSession,
    camera_id: UUID,
    data: CameraUpdate,
    request: Request,
) -> CameraResponse:
    return await service.update_camera(
        db,
        camera_id,
        data,
        stream_manager=request.app.state.camera_stream_manager,
    )


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(
    db: DbSession,
    camera_id: UUID,
    request: Request,
) -> None:
    await service.delete_camera(
        db,
        camera_id,
        stream_manager=request.app.state.camera_stream_manager,
    )


@router.post("/{camera_id}/test", response_model=CameraTestResponse)
async def test_camera(
    db: DbSession,
    camera_id: UUID,
    request: Request,
) -> CameraTestResponse:
    result = await service.test_connection(
        db,
        camera_id,
        stream_manager=request.app.state.camera_stream_manager,
    )
    return CameraTestResponse(
        status=result.status,
        message=result.message,
        checked_at=result.checked_at,
        width=result.width,
        height=result.height,
    )


@router.post("/{camera_id}/webrtc/offer", response_model=CameraWebRTCOfferResponse)
async def camera_preview_webrtc_offer(
    request: Request,
    camera_id: UUID,
    payload: CameraWebRTCOfferRequest,
) -> CameraWebRTCOfferResponse:
    return await service.create_camera_preview_answer(
        manager=request.app.state.camera_stream_manager,
        camera_id=camera_id,
        payload=payload,
    )
