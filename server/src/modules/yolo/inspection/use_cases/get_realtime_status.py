from uuid import UUID

from modules.yolo.inspection.domain.debug_payload import build_realtime_debug_payload
from modules.yolo.inspection.api.schemas import InspectionRealtimeStatusResponse
from modules.yolo.inspection.constants import inspections as inspections_constants
from modules.yolo.inspection.realtime.streamer import InspectionStreamer

from .realtime_session import get_realtime_session_or_raise


def _build_debug_payload(result):
    return build_realtime_debug_payload(
        result.details,
        alignment_debug=getattr(result, "alignment_debug", None),
        verification_mode=getattr(result, "verification_mode", "realtime"),
    )


def _extract_pose_pipeline(debug_payload):
    if not isinstance(debug_payload, dict):
        return None
    pose_pipeline = debug_payload.get("pose_pipeline")
    return pose_pipeline if isinstance(pose_pipeline, dict) else None


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
                pose_pipeline=None,
            )

        matched, total, missing, _, _ = _status_from_realtime_result(result)
        failed_debug_payload = _build_debug_payload(result)
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
            debug_payload=failed_debug_payload,
            pose_pipeline=_extract_pose_pipeline(failed_debug_payload),
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
    online_debug_payload = _build_debug_payload(result)

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
        debug_payload=online_debug_payload,
        pose_pipeline=_extract_pose_pipeline(online_debug_payload),
    )


def _status_from_realtime_result(
    result,
) -> tuple[int, int, list[str], str, bool]:
    details = result.details or []

    expected_details = [
        detail
        for detail in details
        if detail.get("annotation_id") is not None
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
        if str(detail.get("status") or "") in {"missing", "unmatched"}
    ]

    has_extra = any(str(detail.get("status") or "") == "extra" for detail in details)

    passed = total > 0 and matched == total and not missing and not has_extra

    status = (
        inspections_constants.statuses.passed
        if passed
        else inspections_constants.statuses.failed
    )

    return matched, total, missing, status, passed
