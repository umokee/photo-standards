from __future__ import annotations

import asyncio
import socket
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import cv2
import numpy as np
import structlog
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from app.config import settings
from app.exception import NotFoundError, ValidationError
from app.live.webrtc import resolve_mdns_in_sdp, wait_ice_gathering_complete
from app.observability import log_event
from infra.storage.file_storage import resolve_storage_path
from modules.cameras import crud
from modules.cameras.streaming._backends import (
    HttpMjpegBackend,
    VideoBackend,
    VideoBackendImpl,
)
from modules.cameras.streaming.manager import CameraStreamManager
from modules.cameras.streaming.webrtc import CameraPreviewVideoTrack
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Camera
from .schemas import (
    CameraCreate,
    CameraUpdate,
    CameraWebRTCOfferRequest,
    CameraWebRTCOfferResponse,
    validate_camera_connection_fields,
)
from .urls import build_camera_stream_url, build_camera_stream_url_no_auth

CAMERA_SNAPSHOTS_DIR = settings.STORAGE_ROOT / "camera_snapshots"

_PREVIEW_FRAME_TIMEOUT_SEC = 8.0
_BACKGROUND_FAILURES_TO_OFFLINE = 2
_PREVIEW_PEERS: dict[RTCPeerConnection, CameraPreviewVideoTrack] = {}

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class CameraProbeResult:
    status: str
    message: str
    checked_at: datetime
    width: int | None = None
    height: int | None = None
    is_available: bool = False


@dataclass(slots=True)
class CameraSnapshotResult:
    camera: Camera
    image_path: str
    width: int
    height: int
    captured_at: datetime


async def list_cameras(
    db: AsyncSession,
) -> list[Camera]:
    return await crud.list_cameras(db)


async def get_camera(
    db: AsyncSession,
    camera_id: UUID,
) -> Camera:
    camera = await crud.get_camera(db, camera_id)
    if camera is None:
        raise NotFoundError("Камера", camera_id)
    return camera


async def create_camera(
    db: AsyncSession,
    data: CameraCreate,
) -> Camera:
    camera = Camera(**data.model_dump())
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    log_event(logger, "info", "camera.created", **_camera_log_fields(camera))
    return camera


async def update_camera(
    db: AsyncSession,
    camera_id: UUID,
    data: CameraUpdate,
    *,
    stream_manager: CameraStreamManager | None = None,
) -> Camera:
    camera = await get_camera(db, camera_id)

    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(camera, key, value)

    try:
        validate_camera_connection_fields(
            protocol=camera.protocol,
            host=camera.host,
            path=camera.path,
            stream_path=camera.stream_path,
            device_path=camera.device_path,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    await db.commit()
    await db.refresh(camera)

    if stream_manager is not None:
        await stream_manager.restart_camera(camera_id)

    log_event(logger, "info", "camera.updated", **_camera_log_fields(camera))
    return camera


async def delete_camera(
    db: AsyncSession,
    camera_id: UUID,
    *,
    stream_manager: CameraStreamManager | None = None,
) -> None:
    camera = await get_camera(db, camera_id)
    camera_fields = _camera_log_fields(camera)
    await db.delete(camera)
    await db.commit()

    if stream_manager is not None:
        await stream_manager.restart_camera(camera_id)
    log_event(logger, "info", "camera.deleted", **camera_fields)


async def test_connection(
    db: AsyncSession,
    camera_id: UUID,
    *,
    stream_manager: CameraStreamManager,
) -> CameraProbeResult:
    camera = await get_camera(db, camera_id)
    checked_at = _now()
    log_event(
        logger,
        "info",
        "camera.probe.started",
        checked_at=checked_at,
        **_camera_log_fields(camera),
    )

    try:
        if not camera.is_active:
            raise ValidationError("Камера отключена")

        frame = await stream_manager.capture_one_frame(
            camera_id,
            timeout=max(float(camera.timeout_sec), 1.0),
        )
        height, width = frame.shape[:2]

        _apply_probe_success(camera, checked_at)
        await db.commit()
        log_event(
            logger,
            "info",
            "camera.probe.success",
            checked_at=checked_at,
            width=width,
            height=height,
            **_camera_log_fields(camera),
        )

        return CameraProbeResult(
            status="online",
            message="Кадр успешно получен",
            checked_at=checked_at,
            width=width,
            height=height,
            is_available=True,
        )
    except Exception as exc:
        message = _normalize_probe_error(exc)
        _apply_probe_failure(camera, checked_at, message, force_offline=True)
        await db.commit()
        log_event(
            logger,
            "warning",
            "camera.probe.failed",
            checked_at=checked_at,
            error_type=type(exc).__name__,
            reason=message,
            **_camera_log_fields(camera),
        )

        return CameraProbeResult(
            status="offline",
            message=message,
            checked_at=checked_at,
            is_available=False,
        )


async def take_snapshot(
    db: AsyncSession,
    camera_id: UUID,
    *,
    task_id: UUID,
    stream_manager: CameraStreamManager,
) -> CameraSnapshotResult:
    camera = await get_camera(db, camera_id)

    if not camera.is_active:
        raise ValidationError("Камера отключена")

    try:
        frame = await stream_manager.capture_one_frame(camera_id, timeout=10.0)
    except TimeoutError as exc:
        raise ValidationError(f"Не удалось получить снимок с камеры: {exc}") from exc
    except NotFoundError:
        raise
    except Exception as exc:
        raise ValidationError(f"Не удалось получить снимок с камеры: {exc}") from exc

    captured_at = _now()
    height, width = frame.shape[:2]

    relative_path = f"inspections/{task_id}/source.jpg"
    absolute_path = resolve_storage_path(relative_path)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(absolute_path), frame)
    if not ok:
        raise ValidationError("Не удалось сохранить снимок камеры")

    return CameraSnapshotResult(
        camera=camera,
        image_path=relative_path,
        width=width,
        height=height,
        captured_at=captured_at,
    )


async def check_camera_health(
    db: AsyncSession,
    camera: Camera | UUID,
    *,
    failure_streak: int = 0,
) -> CameraProbeResult:
    camera_model = (
        camera if isinstance(camera, Camera) else await get_camera(db, camera)
    )
    checked_at = _now()

    if not camera_model.is_active:
        camera_model.last_status = "unknown"
        camera_model.last_checked_at = checked_at
        camera_model.last_error = "Камера отключена"
        await db.commit()

        return CameraProbeResult(
            status="unknown",
            message="Камера отключена",
            checked_at=checked_at,
            is_available=False,
        )

    try:
        frame = await _capture_background_probe_frame(camera_model)
        height, width = frame.shape[:2]

        _apply_probe_success(camera_model, checked_at)
        await db.commit()

        return CameraProbeResult(
            status="online",
            message="Кадр успешно получен",
            checked_at=checked_at,
            width=width,
            height=height,
            is_available=True,
        )
    except Exception as exc:
        message = _normalize_probe_error(exc)
        force_offline = (
            camera_model.last_status != "online"
            or failure_streak + 1 >= _BACKGROUND_FAILURES_TO_OFFLINE
        )
        _apply_probe_failure(
            camera_model,
            checked_at,
            message,
            force_offline=force_offline,
        )
        await db.commit()

        return CameraProbeResult(
            status=camera_model.last_status or "offline",
            message=message,
            checked_at=checked_at,
            is_available=False,
        )


async def create_camera_preview_answer(
    *,
    manager: CameraStreamManager,
    camera_id: UUID,
    payload: CameraWebRTCOfferRequest,
) -> CameraWebRTCOfferResponse:
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    track = CameraPreviewVideoTrack(
        manager=manager,
        camera_id=camera_id,
        fps=payload.fps,
    )

    _PREVIEW_PEERS[pc] = track

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        logger.info(
            "camera_preview.webrtc.ice_state",
            extra={
                "event": "camera_preview.webrtc.ice_state",
                "camera_id": str(camera_id),
                "state": pc.iceConnectionState,
            },
        )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info(
            "camera_preview.webrtc.connection_state",
            extra={
                "event": "camera_preview.webrtc.connection_state",
                "camera_id": str(camera_id),
                "state": pc.connectionState,
            },
        )

        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _close_preview_peer(pc)

    try:
        await track.start_preview()
        await track.wait_ready(timeout=_PREVIEW_FRAME_TIMEOUT_SEC)
        pc.addTrack(track)

        offer = RTCSessionDescription(
            sdp=resolve_mdns_in_sdp(payload.sdp),
            type=payload.type,
        )

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await wait_ice_gathering_complete(
            pc,
            log_event_prefix="camera_preview",
        )

        if pc.localDescription is None:
            raise ValidationError("Не удалось сформировать WebRTC-ответ")

        return CameraWebRTCOfferResponse(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type,
        )

    except ValidationError:
        await _close_preview_peer(pc)
        raise

    except Exception as exc:
        log_event(
            logger,
            "error",
            "camera.preview.webrtc_offer_failed",
            camera_id=camera_id,
            error_type=type(exc).__name__,
            exception=exc,
        )
        await _close_preview_peer(pc)
        raise ValidationError(_get_preview_error_message(exc)) from exc


async def close_all_camera_preview_peers() -> None:
    if not _PREVIEW_PEERS:
        return

    peers = list(_PREVIEW_PEERS)
    await asyncio.gather(
        *(_close_preview_peer(pc) for pc in peers),
        return_exceptions=True,
    )


async def _capture_background_probe_frame(camera: Camera) -> np.ndarray:
    protocol = camera.protocol
    host = camera.host
    port = _resolve_camera_port(camera)
    timeout_sec = max(float(camera.timeout_sec), 1.0)
    stream_url = build_camera_stream_url(camera)
    safe_url = build_camera_stream_url_no_auth(camera)
    auth = (camera.username, camera.password or "") if camera.username else None
    device_path = camera.device_path

    return await asyncio.to_thread(
        _capture_background_probe_frame_sync,
        protocol=protocol,
        host=host,
        port=port,
        timeout_sec=timeout_sec,
        stream_url=stream_url,
        safe_url=safe_url,
        auth=auth,
        device_path=device_path,
    )


async def _close_preview_peer(pc: RTCPeerConnection) -> None:
    track = _PREVIEW_PEERS.pop(pc, None)

    try:
        if track is not None:
            await track.stop_preview()
    finally:
        await pc.close()


def _camera_needs_health_check(
    camera: Camera,
    *,
    now: datetime,
    online_interval_sec: float,
    offline_interval_sec: float,
) -> bool:
    if not camera.is_active:
        return False

    if camera.last_checked_at is None:
        return True

    interval = (
        online_interval_sec if camera.last_status == "online" else offline_interval_sec
    )

    age_sec = (now - camera.last_checked_at).total_seconds()
    return age_sec >= interval


def _get_preview_error_message(exc: Exception) -> str:
    message = str(exc).strip()

    if message:
        return message

    return "Не удалось подключить WebRTC-предпросмотр камеры"


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _camera_log_fields(camera: Camera) -> dict[str, object]:
    return {
        "camera_id": camera.id,
        "protocol": camera.protocol,
        "host": camera.host,
        "port": camera.port,
        "device_path": camera.device_path,
        "safe_url": build_camera_stream_url_no_auth(camera),
        "is_active": camera.is_active,
    }


def _apply_probe_success(
    camera: Camera,
    checked_at: datetime,
) -> None:
    camera.last_status = "online"
    camera.last_checked_at = checked_at
    camera.last_error = None


def _apply_probe_failure(
    camera: Camera,
    checked_at: datetime,
    message: str,
    *,
    force_offline: bool,
) -> None:
    camera.last_checked_at = checked_at
    camera.last_error = message[:500]

    if force_offline:
        camera.last_status = "offline"


def _normalize_probe_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return str(exc)

    if isinstance(exc, TimeoutError):
        message = str(exc).strip()
        return message or "Камера не отдала кадр за отведённое время"

    message = str(exc).strip()
    if message:
        return message

    return "Не удалось получить кадр с камеры"


def _capture_background_probe_frame_sync(
    *,
    protocol: str,
    host: str | None,
    port: int,
    timeout_sec: float,
    stream_url: str,
    safe_url: str,
    auth: tuple[str, str] | None,
    device_path: str | None,
) -> np.ndarray:
    if protocol == "http":
        preflight_error = _preflight_tcp_endpoint(
            host=host,
            port=port,
            timeout_sec=timeout_sec,
        )
        if preflight_error is not None:
            raise ValidationError(preflight_error)

        backend = HttpMjpegBackend.open(
            safe_url,
            timeout=timeout_sec,
            auth=auth,
        )
        return _read_backend_frame_or_raise(
            backend,
            timeout_sec=timeout_sec,
            failure_message=f"Не удалось открыть HTTP MJPEG-поток: {safe_url}",
        )

    if protocol == "rtsp":
        preflight_error = _preflight_tcp_endpoint(
            host=host,
            port=port,
            timeout_sec=timeout_sec,
        )
        if preflight_error is not None:
            raise ValidationError(preflight_error)

        backend = VideoBackendImpl.open_rtsp(stream_url)
        return _read_backend_frame_or_raise(
            backend,
            timeout_sec=timeout_sec,
            failure_message=f"Не удалось открыть RTSP-поток: {safe_url}",
        )

    if protocol == "usb":
        if not device_path:
            raise ValidationError("Для USB-камеры не указан путь к устройству")

        backend = VideoBackendImpl.open_usb(device_path)
        return _read_backend_frame_or_raise(
            backend,
            timeout_sec=timeout_sec,
            failure_message=f"Не удалось открыть USB-устройство: {device_path}",
        )

    raise ValidationError(f"Протокол камеры не поддерживается: {protocol}")


def _preflight_tcp_endpoint(
    *,
    host: str | None,
    port: int,
    timeout_sec: float,
) -> str | None:
    if not host:
        return "Не указан IPv4-адрес камеры"

    timeout = min(max(timeout_sec, 1.0), 3.0)

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        message = str(exc).strip() or exc.__class__.__name__
        return f"Не удалось подключиться к {host}:{port}: {message}"


def _read_backend_frame_or_raise(
    backend: VideoBackend | None,
    *,
    timeout_sec: float,
    failure_message: str,
) -> np.ndarray:
    if backend is None:
        raise ValidationError(failure_message)

    try:
        deadline = time.monotonic() + max(timeout_sec, 1.0)

        while time.monotonic() < deadline:
            ok, frame = backend.read()
            if ok and frame is not None and frame.size > 0:
                return frame

        raise ValidationError(failure_message)
    finally:
        with suppress(Exception):
            backend.release()


def _resolve_camera_port(camera: Camera) -> int:
    if camera.port is not None:
        return camera.port

    if camera.protocol == "http":
        return 80

    if camera.protocol == "rtsp":
        return 554

    return 0
