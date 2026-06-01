from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from uuid import UUID

import structlog
from aiortc import (
    MediaStreamError,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from app.live.webrtc import resolve_mdns_in_sdp, wait_ice_gathering_complete
from app.observability import log_event, throttled_log
from modules.cameras.streaming.webrtc import InspectionVideoTrack
from modules.yolo.inspection.api.schemas import (
    InspectionWebRTCOfferRequest,
    InspectionWebRTCOfferResponse,
)
from modules.yolo.inspection.realtime.streamer import InspectionStreamer
from modules.yolo.inspection.use_cases.realtime_session import (
    get_realtime_session_or_raise,
)

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class RealtimePeer:
    pc: RTCPeerConnection
    session_id: UUID


_REALTIME_PEERS: dict[RTCPeerConnection, RealtimePeer] = {}


async def create_realtime_webrtc_answer(
    *,
    streamer: InspectionStreamer,
    session_id: UUID,
    payload: InspectionWebRTCOfferRequest,
) -> InspectionWebRTCOfferResponse:
    session = get_realtime_session_or_raise(
        streamer=streamer,
        session_id=session_id,
    )

    loop = asyncio.get_running_loop()
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    _REALTIME_PEERS[pc] = RealtimePeer(pc=pc, session_id=session_id)

    session.add_stop_callback(
        lambda: loop.call_soon_threadsafe(
            lambda: asyncio.create_task(_close_realtime_peer(pc))
        )
    )

    if payload.receive_video:
        pc.addTrack(InspectionVideoTrack(session, fps=payload.fps))

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind != "video":
            return

        task = asyncio.create_task(
            _consume_browser_camera_track(
                session=session,
                track=track,
                session_id=session_id,
                fps=payload.fps,
            )
        )

        @track.on("ended")
        async def on_ended() -> None:
            task.cancel()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        log_event(
            logger,
            "info",
            "realtime.webrtc.ice_connection_state",
            session_id=session_id,
            state=pc.iceConnectionState,
        )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        log_event(
            logger,
            "info",
            "realtime.webrtc.connection_state",
            session_id=session_id,
            state=pc.connectionState,
        )

        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _close_realtime_peer(pc)

    offer = RTCSessionDescription(
        sdp=resolve_mdns_in_sdp(payload.sdp),
        type=payload.type,
    )

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await wait_ice_gathering_complete(pc)

    if pc.localDescription is None:
        await _close_realtime_peer(pc)
        raise RuntimeError("Не удалось создать WebRTC answer")

    log_event(
        logger,
        "info",
        "realtime.webrtc.answer_created",
        session_id=session_id,
        ice_gathering_state=pc.iceGatheringState,
        answer_has_candidates="a=candidate:" in pc.localDescription.sdp,
        receive_video=payload.receive_video,
    )

    return InspectionWebRTCOfferResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
    )


async def close_all_realtime_webrtc_peers() -> None:
    if not _REALTIME_PEERS:
        return

    peers = list(_REALTIME_PEERS)

    await asyncio.gather(
        *(_close_realtime_peer(pc) for pc in peers),
        return_exceptions=True,
    )


async def _close_realtime_peer(pc: RTCPeerConnection) -> None:
    peer = _REALTIME_PEERS.pop(pc, None)

    try:
        await pc.close()
    finally:
        if peer is not None:
            log_event(
                logger,
                "info",
                "realtime.webrtc.peer_closed",
                session_id=peer.session_id,
            )


async def _consume_browser_camera_track(
    *,
    session,
    track,
    session_id: UUID,
    fps: int,
) -> None:
    target_fps = max(1, min(fps, 15))
    min_interval_sec = 1.0 / target_fps
    last_pushed_at = 0.0

    try:
        while True:
            video_frame = await track.recv()

            now = time.monotonic()
            if now - last_pushed_at < min_interval_sec:
                continue

            frame = video_frame.to_ndarray(format="bgr24")
            session.push_browser_video_frame(frame)
            last_pushed_at = now

    except MediaStreamError:
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        throttled_log(
            logger,
            event="realtime.error",
            repeated_event="realtime.error.repeated",
            error=exc,
            throttle_key=f"realtime-webrtc-track:{session_id}",
            interval_sec=10.0,
            session_id=session_id,
        )
