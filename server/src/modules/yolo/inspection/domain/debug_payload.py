from __future__ import annotations

from collections import Counter
from typing import Any

from app.config import settings
from modules.yolo.inspection.domain.runtime_fusion_contract import (
    APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM,
    APPLIED_GEOMETRY_SOURCE_TRUSTED_YOLO_MASK,
    APPLIED_GEOMETRY_SOURCE_UNRESOLVED_BASELINE_FAILURE,
    RUNTIME_FUSION_SUMMARY_MODE,
    YOLO_ANCHOR_STATUS_NO_ANCHOR,
    YOLO_ANCHOR_STATUS_TRUSTED,
)


def build_inspection_debug_payload(
    *,
    details: list[dict[str, Any]],
    raw_class_counts: dict[str, int] | None,
    imgsz: int,
    weights_path: str | None,
    alignment_debug: dict[str, Any] | None,
    verification_mode: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "raw_counts": raw_class_counts or {},
        "imgsz": imgsz,
        "weights_path": weights_path,
        "alignment": alignment_debug,
        "verification_mode": verification_mode,
    }

    _add_runtime_anchor_fusion_summary(payload, details)
    return payload


def build_realtime_debug_payload(
    details: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not settings.INSPECTION_DEBUG_PAYLOAD:
        return None

    payload: dict[str, Any] = {}
    _add_runtime_anchor_fusion_summary(payload, details)
    return payload or None


def build_runtime_anchor_fusion_summary(
    details: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    details = list(details or [])
    status_counts = Counter(_string_value(detail.get("status")) for detail in details)
    status_counts.pop("", None)

    yolo_anchor_status_counts: Counter[str] = Counter()
    applied_geometry_source_counts: Counter[str] = Counter()
    runtime_anchor_reject_counts: Counter[str] = Counter()
    runtime_anchor_best_rejected_reason_counts: Counter[str] = Counter()
    rejected_samples: list[dict[str, Any]] = []

    candidate_count = 0
    trusted_candidate_count = 0
    rejected_candidate_count = 0

    for detail in details:
        debug = detail.get("debug")
        if not isinstance(debug, dict):
            continue

        yolo_status = _string_value(debug.get("yolo_anchor_status"))
        if yolo_status:
            yolo_anchor_status_counts[yolo_status] += 1

        applied_source = _string_value(debug.get("applied_geometry_source"))
        if applied_source:
            applied_geometry_source_counts[applied_source] += 1

        candidate_count += _int_value(debug.get("runtime_anchor_candidate_count"))
        trusted_candidate_count += _int_value(
            debug.get("runtime_anchor_trusted_candidate_count")
        )
        rejected_candidate_count += _int_value(
            debug.get("runtime_anchor_rejected_candidate_count")
        )

        reject_counts = debug.get("runtime_anchor_reject_counts")
        if isinstance(reject_counts, dict):
            for reason, count in reject_counts.items():
                reason_text = _string_value(reason)
                if reason_text:
                    runtime_anchor_reject_counts[reason_text] += _int_value(count)

        best_rejected_reason = _string_value(
            debug.get("runtime_anchor_best_rejected_reason")
        )
        if best_rejected_reason:
            runtime_anchor_best_rejected_reason_counts[best_rejected_reason] += 1

        if _should_include_rejected_anchor_sample(debug):
            rejected_samples.append(_runtime_anchor_rejected_sample(detail, debug))

    return {
        "mode": RUNTIME_FUSION_SUMMARY_MODE,
        "profile": settings.INSPECTION_ALIGNMENT_PROFILE,
        "status_counts": dict(status_counts),
        "yolo_anchor_status_counts": dict(yolo_anchor_status_counts),
        "applied_geometry_source_counts": dict(applied_geometry_source_counts),
        "runtime_anchor_candidate_count": candidate_count,
        "runtime_anchor_trusted_candidate_count": trusted_candidate_count,
        "runtime_anchor_rejected_candidate_count": rejected_candidate_count,
        "runtime_anchor_reject_counts": dict(runtime_anchor_reject_counts),
        "runtime_anchor_best_rejected_reason_counts": dict(
            runtime_anchor_best_rejected_reason_counts
        ),
        "trusted_anchor_match_count": yolo_anchor_status_counts.get(
            YOLO_ANCHOR_STATUS_TRUSTED,
            0,
        ),
        "no_anchor_match_count": yolo_anchor_status_counts.get(
            YOLO_ANCHOR_STATUS_NO_ANCHOR,
            0,
        ),
        "trusted_yolo_mask_applied_count": applied_geometry_source_counts.get(
            APPLIED_GEOMETRY_SOURCE_TRUSTED_YOLO_MASK,
            0,
        ),
        "trusted_anchor_transform_applied_count": applied_geometry_source_counts.get(
            APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM,
            0,
        ),
        "unresolved_baseline_failure_count": applied_geometry_source_counts.get(
            APPLIED_GEOMETRY_SOURCE_UNRESOLVED_BASELINE_FAILURE,
            0,
        ),
        "top_runtime_anchor_rejected_samples": rejected_samples[:10],
        "next_step_hint": _runtime_anchor_next_step_hint(
            yolo_anchor_status_counts=yolo_anchor_status_counts,
            applied_geometry_source_counts=applied_geometry_source_counts,
            reject_counts=runtime_anchor_reject_counts,
            candidate_count=candidate_count,
        ),
    }


def _add_runtime_anchor_fusion_summary(
    payload: dict[str, Any],
    details: list[dict[str, Any]] | None,
) -> None:
    if not settings.INSPECTION_RUNTIME_ANCHOR_FUSION_SUMMARY:
        return

    payload["runtime_anchor_fusion"] = build_runtime_anchor_fusion_summary(details)


def _should_include_rejected_anchor_sample(debug: dict[str, Any]) -> bool:
    if _int_value(debug.get("runtime_anchor_rejected_candidate_count")) > 0:
        return True

    return bool(_string_value(debug.get("runtime_anchor_best_rejected_reason")))


def _runtime_anchor_rejected_sample(
    detail: dict[str, Any],
    debug: dict[str, Any],
) -> dict[str, Any]:
    trace = debug.get("runtime_anchor_trace")
    top_candidates: list[Any] = []
    if isinstance(trace, dict):
        candidates = trace.get("top_candidates")
        if isinstance(candidates, list):
            top_candidates = candidates[:3]

    return {
        "name": detail.get("name"),
        "class_key": detail.get("class_key"),
        "status": detail.get("status"),
        "yolo_anchor_status": debug.get("yolo_anchor_status"),
        "applied_geometry_source": debug.get("applied_geometry_source"),
        "best_rejected_reason": debug.get("runtime_anchor_best_rejected_reason"),
        "candidate_count": debug.get("runtime_anchor_candidate_count"),
        "rejected_candidate_count": debug.get(
            "runtime_anchor_rejected_candidate_count"
        ),
        "top_candidates": top_candidates,
    }


def _runtime_anchor_next_step_hint(
    *,
    yolo_anchor_status_counts: Counter[str],
    applied_geometry_source_counts: Counter[str],
    reject_counts: Counter[str],
    candidate_count: int,
) -> str:
    if not yolo_anchor_status_counts and candidate_count == 0:
        return "runtime anchor fusion debug is absent for this result"

    if candidate_count == 0:
        return "YOLO produced no anchor candidates for selected expected segments"

    if applied_geometry_source_counts.get(APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM, 0) > 0:
        return "trusted anchor transform is already applied; inspect unresolved failures next"

    if applied_geometry_source_counts.get(APPLIED_GEOMETRY_SOURCE_TRUSTED_YOLO_MASK, 0) > 0:
        return "trusted YOLO masks are applied; transform rescue is the next reserve"

    if reject_counts:
        reason = reject_counts.most_common(1)[0][0]
        return f"most rejected anchors are blocked by {reason}"

    if yolo_anchor_status_counts.get(YOLO_ANCHOR_STATUS_NO_ANCHOR, 0) > 0:
        return "most segments have no trusted YOLO anchor; check detector coverage or slot policy"

    return "runtime anchor fusion is stable; no immediate action required"


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0

    return 0
