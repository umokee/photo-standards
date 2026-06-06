from __future__ import annotations

import math
from typing import Any

from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area,
    bbox_center_distance_factor,
    bbox_containment,
    bbox_iou,
    polygon_iou,
)
from modules.yolo.inspection.domain.matcher_structs import (
    DetectionCandidate,
    ExpectedSlot,
    ProjectedExpected,
)
from modules.yolo.inspection.domain.runtime_fusion_contract import (
    APPLIED_GEOMETRY_SOURCE_BASELINE,
    APPLIED_GEOMETRY_SOURCE_TRUSTED_YOLO_MASK,
    NO_ANCHOR_NOTE,
    NO_ANCHOR_SEMANTICS,
    RUNTIME_ANCHOR_REJECT_BAD_AREA_RATIO,
    RUNTIME_ANCHOR_REJECT_CENTER_OUTSIDE_SLOT,
    RUNTIME_ANCHOR_REJECT_LOW_CONFIDENCE,
    RUNTIME_ANCHOR_REJECT_LOW_DETECTION_CONTAINMENT,
    RUNTIME_ANCHOR_REJECT_LOW_SLOT_COVERAGE,
    RUNTIME_ANCHOR_REJECT_LOW_SLOT_IOU,
    RUNTIME_ANCHOR_REJECT_UNKNOWN,
    RUNTIME_FUSION_MODE_YOLO_ANCHOR_RESCUE,
    RUNTIME_REASON_NO_TRUSTED_YOLO_ANCHOR,
    RUNTIME_REASON_TRUSTED_YOLO_ANCHOR,
    RUNTIME_REASON_YOLO_ANCHOR_CANDIDATE_REJECTED,
    YOLO_ANCHOR_STATUS_NO_ANCHOR,
    YOLO_ANCHOR_STATUS_REJECTED,
    YOLO_ANCHOR_STATUS_TRUSTED,
)

_RUNTIME_YOLO_ANCHOR_MIN_CONFIDENCE = 0.90
_RUNTIME_YOLO_ANCHOR_MAX_CENTER_FACTOR = 0.28
_RUNTIME_YOLO_ANCHOR_MIN_SLOT_IOU = 0.32
_RUNTIME_YOLO_ANCHOR_MIN_DETECTION_CONTAINMENT = 0.45
_RUNTIME_YOLO_ANCHOR_MIN_SLOT_COVERAGE = 0.50
_RUNTIME_YOLO_ANCHOR_MIN_AREA_RATIO = 0.72
_RUNTIME_YOLO_ANCHOR_MAX_AREA_RATIO = 1.60


def try_runtime_yolo_anchor_candidate(
    expected_item: ProjectedExpected,
    detection_item: DetectionCandidate,
    *,
    slot: ExpectedSlot | None,
    global_debug: dict[str, Any],
    slot_debug: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, Any]] | None:
    """Return a trusted YOLO anchor candidate for the runtime fusion path.

    This is intentionally stricter than the legacy slot matcher.  A detection
    becomes an anchor only when the expected LightGlue slot and the factual YOLO
    mask/bbox agree with each other.  If this check fails, the result is
    ``no_anchor``; it must not be interpreted as a final missing decision by
    itself.
    """

    details = runtime_yolo_anchor_candidate_details(
        expected_item,
        detection_item,
        slot=slot,
        global_debug=global_debug,
        slot_debug=slot_debug,
    )
    if details is None:
        return None

    score, direct_iou, debug = details
    if not debug.get("passed"):
        return None

    return score, direct_iou, debug


def runtime_yolo_anchor_candidate_details(
    expected_item: ProjectedExpected,
    detection_item: DetectionCandidate,
    *,
    slot: ExpectedSlot | None,
    global_debug: dict[str, Any],
    slot_debug: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, Any]] | None:
    if slot is None:
        return None

    confidence = float(detection_item.detection.confidence or 0.0)
    expected_bbox = slot.projected_bbox
    detection_bbox = detection_item.bbox

    slot_iou = bbox_iou(expected_bbox, detection_bbox)
    detection_containment = bbox_containment(detection_bbox, expected_bbox)
    slot_coverage = bbox_containment(expected_bbox, detection_bbox)
    center_factor = bbox_center_distance_factor(detection_bbox, expected_bbox)

    expected_area = max(1.0, bbox_area(expected_bbox))
    detection_area = max(1.0, bbox_area(detection_bbox))
    slot_area_ratio = float(detection_area / expected_area)

    rejected_reason = _runtime_yolo_anchor_reject_reason(
        confidence=confidence,
        center_factor=center_factor,
        slot_iou=slot_iou,
        detection_containment=detection_containment,
        slot_coverage=slot_coverage,
        slot_area_ratio=slot_area_ratio,
    )
    accepted = rejected_reason is None
    direct_iou = polygon_iou(expected_item.polygon, detection_item.polygon)
    score = (
        confidence * 1.5
        + slot_iou * 2.0
        + slot_coverage
        + detection_containment * 0.75
        - center_factor * 2.0
    )

    debug = {
        "reason": "trusted_yolo_anchor" if accepted else "yolo_anchor_rejected",
        "reason_code": (
            RUNTIME_REASON_TRUSTED_YOLO_ANCHOR
            if accepted
            else RUNTIME_REASON_YOLO_ANCHOR_CANDIDATE_REJECTED
        ),
        "projection": (
            "trusted_yolo_anchor_rescue"
            if accepted
            else "expected_slot"
        ),
        "candidate_source": (
            "trusted_yolo_anchor"
            if accepted
            else "rejected_yolo_anchor_candidate"
        ),
        "passed": accepted,
        "runtime_fusion_mode": RUNTIME_FUSION_MODE_YOLO_ANCHOR_RESCUE,
        "runtime_yolo_anchor_trusted": accepted,
        "runtime_anchor_reject_reason": rejected_reason,
        "yolo_anchor_status": (
            YOLO_ANCHOR_STATUS_TRUSTED if accepted else YOLO_ANCHOR_STATUS_REJECTED
        ),
        "applied_geometry_source": (
            APPLIED_GEOMETRY_SOURCE_TRUSTED_YOLO_MASK
            if accepted
            else APPLIED_GEOMETRY_SOURCE_BASELINE
        ),
        "no_anchor_semantics": NO_ANCHOR_SEMANTICS,
        "score": _round_debug(score),
        "iou": _round_debug(direct_iou),
        "global_score": global_debug.get("score"),
        "global_iou": global_debug.get("iou"),
        "global_reject_reason": global_debug.get("reject_reason"),
        "runtime_anchor_confidence": _round_debug(confidence),
        "runtime_anchor_min_confidence": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MIN_CONFIDENCE
        ),
        "runtime_anchor_center_factor": _round_debug(center_factor),
        "runtime_anchor_max_center_factor": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MAX_CENTER_FACTOR
        ),
        "runtime_anchor_slot_iou": _round_debug(slot_iou),
        "runtime_anchor_min_slot_iou": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MIN_SLOT_IOU
        ),
        "runtime_anchor_detection_containment": _round_debug(
            detection_containment
        ),
        "runtime_anchor_min_detection_containment": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MIN_DETECTION_CONTAINMENT
        ),
        "runtime_anchor_slot_coverage": _round_debug(slot_coverage),
        "runtime_anchor_min_slot_coverage": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MIN_SLOT_COVERAGE
        ),
        "runtime_anchor_slot_area_ratio": _round_debug(slot_area_ratio),
        "runtime_anchor_min_slot_area_ratio": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MIN_AREA_RATIO
        ),
        "runtime_anchor_max_slot_area_ratio": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MAX_AREA_RATIO
        ),
        "slot": slot_debug,
    }

    return float(score), float(direct_iou), debug


def _runtime_yolo_anchor_reject_reason(
    *,
    confidence: float,
    center_factor: float,
    slot_iou: float,
    detection_containment: float,
    slot_coverage: float,
    slot_area_ratio: float,
) -> str | None:
    if confidence < _RUNTIME_YOLO_ANCHOR_MIN_CONFIDENCE:
        return RUNTIME_ANCHOR_REJECT_LOW_CONFIDENCE
    if center_factor > _RUNTIME_YOLO_ANCHOR_MAX_CENTER_FACTOR:
        return RUNTIME_ANCHOR_REJECT_CENTER_OUTSIDE_SLOT
    if slot_iou < _RUNTIME_YOLO_ANCHOR_MIN_SLOT_IOU:
        return RUNTIME_ANCHOR_REJECT_LOW_SLOT_IOU
    if detection_containment < _RUNTIME_YOLO_ANCHOR_MIN_DETECTION_CONTAINMENT:
        return RUNTIME_ANCHOR_REJECT_LOW_DETECTION_CONTAINMENT
    if slot_coverage < _RUNTIME_YOLO_ANCHOR_MIN_SLOT_COVERAGE:
        return RUNTIME_ANCHOR_REJECT_LOW_SLOT_COVERAGE
    if (
        slot_area_ratio < _RUNTIME_YOLO_ANCHOR_MIN_AREA_RATIO
        or slot_area_ratio > _RUNTIME_YOLO_ANCHOR_MAX_AREA_RATIO
    ):
        return RUNTIME_ANCHOR_REJECT_BAD_AREA_RATIO
    return None


def build_runtime_yolo_anchor_trace(
    expected_item: ProjectedExpected,
    slot: ExpectedSlot | None,
) -> dict[str, Any]:
    return {
        "mode": RUNTIME_FUSION_MODE_YOLO_ANCHOR_RESCUE,
        "expected_index": expected_item.index,
        "expected_name": expected_item.item.name,
        "class_key": expected_item.item.class_key,
        "slot_available": slot is not None,
        "candidate_count": 0,
        "trusted_candidate_count": 0,
        "rejected_candidate_count": 0,
        "reject_counts": {},
        "best_rejected_reason": None,
        "best_rejected_score": None,
        "best_candidate": None,
        "top_candidates": [],
        "thresholds": _runtime_yolo_anchor_thresholds_debug(),
    }


def record_runtime_yolo_anchor_trace_candidate(
    trace: dict[str, Any] | None,
    *,
    detection_index: int,
    debug: dict[str, Any],
) -> None:
    if trace is None:
        return

    passed = bool(debug.get("passed"))
    reject_reason = debug.get("runtime_anchor_reject_reason")
    score = _debug_float(debug.get("score"), default=0.0)

    trace["candidate_count"] = int(trace.get("candidate_count") or 0) + 1
    if passed:
        trace["trusted_candidate_count"] = (
            int(trace.get("trusted_candidate_count") or 0) + 1
        )
    else:
        trace["rejected_candidate_count"] = (
            int(trace.get("rejected_candidate_count") or 0) + 1
        )
        reason = str(reject_reason or RUNTIME_ANCHOR_REJECT_UNKNOWN)
        reject_counts = trace.setdefault("reject_counts", {})
        reject_counts[reason] = int(reject_counts.get(reason) or 0) + 1
        if (
            trace.get("best_rejected_score") is None
            or score > float(trace.get("best_rejected_score") or 0.0)
        ):
            trace["best_rejected_score"] = _round_debug(score)
            trace["best_rejected_reason"] = reason

    candidate = _runtime_yolo_anchor_candidate_summary(
        detection_index=detection_index,
        debug=debug,
    )
    if (
        trace.get("best_candidate") is None
        or score > float(trace["best_candidate"].get("score") or 0.0)
    ):
        trace["best_candidate"] = candidate

    top_candidates = list(trace.get("top_candidates") or [])
    top_candidates.append(candidate)
    top_candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    trace["top_candidates"] = top_candidates[:3]


def merge_runtime_yolo_anchor_trace_debug(
    debug: dict[str, Any] | None,
    trace: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if debug is None or trace is None:
        return debug

    payload = _runtime_yolo_anchor_trace_payload(trace)
    return {
        **debug,
        "runtime_anchor_trace": payload,
        "runtime_anchor_candidate_count": payload["candidate_count"],
        "runtime_anchor_trusted_candidate_count": payload["trusted_candidate_count"],
        "runtime_anchor_rejected_candidate_count": payload["rejected_candidate_count"],
        "runtime_anchor_reject_counts": payload["reject_counts"],
        "runtime_anchor_best_rejected_reason": payload["best_rejected_reason"],
    }


def _runtime_yolo_anchor_trace_payload(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": trace.get("mode"),
        "expected_index": trace.get("expected_index"),
        "expected_name": trace.get("expected_name"),
        "class_key": trace.get("class_key"),
        "slot_available": bool(trace.get("slot_available")),
        "candidate_count": int(trace.get("candidate_count") or 0),
        "trusted_candidate_count": int(trace.get("trusted_candidate_count") or 0),
        "rejected_candidate_count": int(trace.get("rejected_candidate_count") or 0),
        "reject_counts": dict(trace.get("reject_counts") or {}),
        "best_rejected_reason": trace.get("best_rejected_reason"),
        "best_rejected_score": trace.get("best_rejected_score"),
        "best_candidate": trace.get("best_candidate"),
        "top_candidates": list(trace.get("top_candidates") or []),
        "thresholds": dict(trace.get("thresholds") or {}),
    }


def _runtime_yolo_anchor_candidate_summary(
    *,
    detection_index: int,
    debug: dict[str, Any],
) -> dict[str, Any]:
    return {
        "detection_index": detection_index,
        "passed": bool(debug.get("passed")),
        "reject_reason": debug.get("runtime_anchor_reject_reason"),
        "score": debug.get("score"),
        "confidence": debug.get("runtime_anchor_confidence"),
        "slot_iou": debug.get("runtime_anchor_slot_iou"),
        "center_factor": debug.get("runtime_anchor_center_factor"),
        "detection_containment": debug.get("runtime_anchor_detection_containment"),
        "slot_coverage": debug.get("runtime_anchor_slot_coverage"),
        "slot_area_ratio": debug.get("runtime_anchor_slot_area_ratio"),
        "direct_iou": debug.get("iou"),
    }


def _runtime_yolo_anchor_thresholds_debug() -> dict[str, float | None]:
    return {
        "min_confidence": _round_debug(_RUNTIME_YOLO_ANCHOR_MIN_CONFIDENCE),
        "max_center_factor": _round_debug(_RUNTIME_YOLO_ANCHOR_MAX_CENTER_FACTOR),
        "min_slot_iou": _round_debug(_RUNTIME_YOLO_ANCHOR_MIN_SLOT_IOU),
        "min_detection_containment": _round_debug(
            _RUNTIME_YOLO_ANCHOR_MIN_DETECTION_CONTAINMENT
        ),
        "min_slot_coverage": _round_debug(_RUNTIME_YOLO_ANCHOR_MIN_SLOT_COVERAGE),
        "min_slot_area_ratio": _round_debug(_RUNTIME_YOLO_ANCHOR_MIN_AREA_RATIO),
        "max_slot_area_ratio": _round_debug(_RUNTIME_YOLO_ANCHOR_MAX_AREA_RATIO),
    }


def merge_runtime_yolo_no_anchor_debug(
    debug: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = {
        **debug,
        "reason": "YOLO не дал надёжный якорь в ожидаемой области",
        "reason_code": RUNTIME_REASON_NO_TRUSTED_YOLO_ANCHOR,
        "runtime_fusion_mode": RUNTIME_FUSION_MODE_YOLO_ANCHOR_RESCUE,
        "yolo_anchor_status": YOLO_ANCHOR_STATUS_NO_ANCHOR,
        "applied_geometry_source": APPLIED_GEOMETRY_SOURCE_BASELINE,
        "no_anchor_semantics": NO_ANCHOR_SEMANTICS,
        "note": NO_ANCHOR_NOTE,
    }
    return merge_runtime_yolo_anchor_trace_debug(merged, trace) or merged


def _debug_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _round_debug(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)
