from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from modules.yolo.inspection.constants import inspections as inspections_constants


@dataclass(slots=True)
class FrameResult:
    matched: int
    total: int
    missing: list[str]
    status: str
    passed: bool
    alignment_status: str
    captured_at: datetime
    details: list[dict[str, Any]]

    alignment_db_status: str | None = None
    alignment_inlier_count: int | None = None
    alignment_raw_match_count: int | None = None
    homography: list | None = None
    alignment_debug: dict[str, Any] | None = None
    verification_mode: str = "alignment"
    pose_method: str | None = None


def _status_of(detail: dict[str, Any]) -> str:
    return str(detail.get("status") or "missing")


def _summarize_details(
    details: list[dict[str, Any]],
    *,
    fallback_total: int | None = None,
) -> tuple[int, int, list[dict[str, Any]], bool]:
    expected_details = [
        detail for detail in details if _status_of(detail) in {"ok", "missing"}
    ]

    expected_total = len(expected_details)

    if fallback_total is not None:
        total = max(int(fallback_total), expected_total)
    else:
        total = expected_total

    matched = sum(1 for detail in expected_details if _status_of(detail) == "ok")

    missing_details = [
        detail for detail in expected_details if _status_of(detail) == "missing"
    ]

    has_extra = any(_status_of(detail) == "extra" for detail in details)

    passed = (
        total > 0 and matched == total and len(missing_details) == 0 and not has_extra
    )

    return total, matched, missing_details, passed


def rebuild_frame_result_from_details(
    result: FrameResult,
    details: list[dict[str, Any]],
    *,
    fallback_total: int | None = None,
) -> FrameResult:
    total, matched, missing_details, passed = _summarize_details(
        details,
        fallback_total=fallback_total,
    )

    status = (
        inspections_constants.statuses.passed
        if passed
        else inspections_constants.statuses.failed
    )

    return FrameResult(
        matched=matched,
        total=total,
        missing=[str(detail.get("name") or "Объект") for detail in missing_details],
        status=status,
        passed=passed,
        alignment_status=result.alignment_status,
        alignment_db_status=result.alignment_db_status,
        alignment_inlier_count=result.alignment_inlier_count,
        alignment_raw_match_count=result.alignment_raw_match_count,
        homography=result.homography,
        alignment_debug=result.alignment_debug,
        verification_mode=result.verification_mode,
        pose_method=result.pose_method,
        captured_at=result.captured_at,
        details=details,
    )

