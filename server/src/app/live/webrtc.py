from __future__ import annotations

import asyncio
import re

from aiortc import RTCPeerConnection
from app.observability import log_event
import structlog

logger = structlog.get_logger(__name__)


def resolve_mdns_in_sdp(sdp: str) -> str:
    return re.sub(
        r"([0-9a-f-]{36})\.local",
        "127.0.0.1",
        sdp,
    )


async def wait_ice_gathering_complete(
    pc: RTCPeerConnection,
    *,
    timeout: float = 5.0,
    log_event_prefix: str = "webrtc",
) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange() -> None:
        log_event(
            logger,
            "info",
            f"{log_event_prefix}.ice_gathering_state",
            state=pc.iceGatheringState,
        )

        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        log_event(
            logger,
            "warning",
            f"{log_event_prefix}.ice_gathering_timeout",
            state=pc.iceGatheringState,
        )
