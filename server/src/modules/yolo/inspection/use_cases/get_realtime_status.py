from uuid import UUID

from modules.yolo.inspection.domain.debug_payload import build_realtime_debug_payload
from modules.yolo.inspection.api.schemas import InspectionRealtimeStatusResponse
from modules.yolo.inspection.constants import inspections as inspections_constants
from modules.yolo.inspection.realtime.streamer import InspectionStreamer

from .realtime_session import get_realtime_session_or_raise


def get_realtime_status(
    *,
    streamer: InspectionStreamer,
    session_id: UUID,
) -> InspectionRealtimeStatusResponse:
    session = get_realtime_session_or_raise(
        streamer=streamer,
        session_id=session_id,
    )
    result = session.get_latest_result()

    if session.failed:
        if result is None:
            return InspectionRealtimeStatusResponse(
                state="failed",
                matched=0,
                total=0,
                missing=[],
                status=inspections_constants.statuses.failed,
                passed=False,
                alignment_status=session.failure_message or "session_failed",
                alignment_inlier_count=None,
                alignment_raw_match_count=None,
                captured_at=None,
                details=[],
                debug_payload=None,
            )

        matched, total, missing, _, _ = _status_from_realtime_result(result)
        return InspectionRealtimeStatusResponse(
            state="failed",
            matched=matched,
            total=total,
            missing=missing,
            status=inspections_constants.statuses.failed,
            passed=False,
            alignment_status=session.failure_message or result.alignment_status,
            alignment_inlier_count=result.alignment_inlier_count,
            alignment_raw_match_count=result.alignment_raw_match_count,
            captured_at=result.captured_at,
            details=result.details,
            debug_payload=build_realtime_debug_payload(result.details),
        )

    if result is None:
        return InspectionRealtimeStatusResponse(
            state="warming_up",
            matched=0,
            total=0,
            missing=[],
            status=inspections_constants.statuses.failed,
            passed=False,
            alignment_status="pending",
            alignment_inlier_count=None,
            alignment_raw_match_count=None,
            captured_at=None,
            details=[],
            debug_payload=None,
        )

    matched, total, missing, status, passed = _status_from_realtime_result(result)

    return InspectionRealtimeStatusResponse(
        state="online",
        matched=matched,
        total=total,
        missing=missing,
        status=status,
        passed=passed,
        alignment_status=result.alignment_status,
        alignment_inlier_count=result.alignment_inlier_count,
        alignment_raw_match_count=result.alignment_raw_match_count,
        captured_at=result.captured_at,
        details=result.details,
        debug_payload=build_realtime_debug_payload(result.details),
    )


def _status_from_realtime_result(
    result,
) -> tuple[int, int, list[str], str, bool]:
    details = result.details or []

    expected_details = [
        detail
        for detail in details
        if str(detail.get("status") or "") in {"ok", "missing"}
    ]

    total = max(
        int(result.total or 0),
        len(expected_details),
    )

    matched = sum(
        1 for detail in expected_details if str(detail.get("status") or "") == "ok"
    )

    missing = [
        str(detail.get("name") or "Объект")
        for detail in expected_details
        if str(detail.get("status") or "") == "missing"
    ]

    has_extra = any(str(detail.get("status") or "") == "extra" for detail in details)

    passed = total > 0 and matched == total and not missing and not has_extra

    status = (
        inspections_constants.statuses.passed
        if passed
        else inspections_constants.statuses.failed
    )

    return matched, total, missing, status, passed
