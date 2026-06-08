from __future__ import annotations

from collections import Counter
from typing import Any

from app.config import settings


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

    payload["pose_pipeline"] = build_pose_pipeline_summary(
        details=details,
        raw_class_counts=raw_class_counts,
        alignment_debug=alignment_debug,
        verification_mode=verification_mode,
    )
    return payload


def build_realtime_debug_payload(
    details: list[dict[str, Any]] | None,
    *,
    raw_class_counts: dict[str, int] | None = None,
    alignment_debug: dict[str, Any] | None = None,
    verification_mode: str | None = None,
) -> dict[str, Any] | None:
    if not settings.INSPECTION_DEBUG_PAYLOAD:
        return None

    pose_pipeline = build_pose_pipeline_summary(
        details=details,
        raw_class_counts=raw_class_counts,
        alignment_debug=alignment_debug,
        verification_mode=verification_mode or "realtime",
    )
    return {
        "alignment": alignment_debug,
        "raw_counts": raw_class_counts or {},
        "verification_mode": verification_mode or "realtime",
        "pose_pipeline": pose_pipeline,
    }


def build_pose_pipeline_summary(
    *,
    details: list[dict[str, Any]] | None,
    raw_class_counts: dict[str, int] | None,
    alignment_debug: dict[str, Any] | None,
    verification_mode: str | None,
) -> dict[str, Any]:
    details = list(details or [])
    alignment_debug = alignment_debug if isinstance(alignment_debug, dict) else {}

    mode = str(getattr(settings, "INSPECTION_YOLO_ANCHOR_POSE_MODE", "auto"))
    raw_counts = raw_class_counts or {}
    yolo_detection_count = int(sum(_int_value(value) for value in raw_counts.values()))
    method = _string_value(alignment_debug.get("method"))
    status = _string_value(alignment_debug.get("status"))
    reason = _string_value(alignment_debug.get("reason"))
    stage = _string_value(alignment_debug.get("stage"))

    has_scene_unconfirmed = _has_scene_unconfirmed_details(details)
    final_source = _final_pose_source(
        verification_mode=verification_mode,
        method=method,
        status=status,
        yolo_detection_count=yolo_detection_count,
        has_scene_unconfirmed=has_scene_unconfirmed,
    )
    final_reason = _final_pose_reason(
        final_source=final_source,
        method=method,
        status=status,
        stage=stage,
        reason=reason,
        yolo_detection_count=yolo_detection_count,
    )

    return {
        "mode": mode,
        "verification_mode": verification_mode,
        "final_pose_source": final_source,
        "final_pose_reason": final_reason,
        "yolo_detection_count": yolo_detection_count,
        "detail_status_counts": dict(_detail_status_counts(details)),
        "feature_alignment": _feature_alignment_summary(alignment_debug),
        "yolo_anchor_pose": _yolo_anchor_pose_summary(alignment_debug),
        "pose_arbiter": _pose_arbiter_summary(alignment_debug),
        "next_step_hint": _next_step_hint(
            final_source=final_source,
            yolo_detection_count=yolo_detection_count,
            status=status,
            method=method,
        ),
    }


def _final_pose_source(
    *,
    verification_mode: str | None,
    method: str,
    status: str,
    yolo_detection_count: int,
    has_scene_unconfirmed: bool,
) -> str:
    if verification_mode == "yolo_count":
        return "yolo_count"
    if has_scene_unconfirmed:
        return "scene_unconfirmed"
    if method == "yolo_anchor_pose" and status == "success":
        return "yolo_anchor_pose"
    if status == "success":
        return "feature_slot_fallback"
    if yolo_detection_count <= 0:
        return "none_yolo_empty_feature_unconfirmed"
    if method:
        return "unconfirmed"
    return "unknown"


def _final_pose_reason(
    *,
    final_source: str,
    method: str,
    status: str,
    stage: str,
    reason: str,
    yolo_detection_count: int,
) -> str:
    if final_source == "yolo_anchor_pose":
        return "visible YOLO objects produced an accepted scene pose"
    if final_source == "feature_slot_fallback":
        return "YOLO-anchor pose was unavailable/not selected; feature/slot fallback produced a pose"
    if final_source == "yolo_count":
        return "alignment is disabled by verification mode"
    if final_source == "scene_unconfirmed":
        return "no confirmed pose; missing polygons were intentionally hidden by fail-safe policy"
    if final_source == "none_yolo_empty_feature_unconfirmed":
        return "YOLO produced no detections and feature/slot fallback did not confirm a pose"
    parts = [part for part in (method, status, stage, reason) if part]
    if parts:
        return "; ".join(parts)
    if yolo_detection_count <= 0:
        return "YOLO produced no detections"
    return "pose source is not available in debug payload"


def _pose_arbiter_summary(alignment_debug: dict[str, Any]) -> dict[str, Any]:
    extra = alignment_debug.get("extra_debug")
    if isinstance(extra, dict):
        pose_arbiter = extra.get("pose_arbiter")
        if isinstance(pose_arbiter, dict):
            return pose_arbiter
    return {}

def _feature_alignment_summary(alignment_debug: dict[str, Any]) -> dict[str, Any]:
    method = _string_value(alignment_debug.get("method"))
    is_yolo_anchor = method == "yolo_anchor_pose"
    return {
        "method": None if is_yolo_anchor else method or None,
        "status": alignment_debug.get("status"),
        "stage": alignment_debug.get("stage"),
        "reason": alignment_debug.get("reason"),
        "raw_match_count": alignment_debug.get("raw_match_count"),
        "inlier_count": alignment_debug.get("inlier_count"),
        "median_error": alignment_debug.get("median_error"),
        "reference_feature_count": alignment_debug.get("reference_feature_count"),
        "frame_feature_count": alignment_debug.get("frame_feature_count"),
        "masked_alignment_used": alignment_debug.get("masked_alignment_used"),
    }


def _yolo_anchor_pose_summary(alignment_debug: dict[str, Any]) -> dict[str, Any]:
    extra = alignment_debug.get("extra_debug")
    if isinstance(extra, dict):
        pose = extra.get("yolo_anchor_pose")
        if isinstance(pose, dict):
            return pose

    method = _string_value(alignment_debug.get("method"))
    if method != "yolo_anchor_pose":
        return {"attempted": False, "accepted": False, "reject_reason": "not_selected"}

    return {
        "attempted": True,
        "accepted": alignment_debug.get("status") == "success",
        "reject_reason": None if alignment_debug.get("status") == "success" else alignment_debug.get("reason"),
        "raw_point_count": alignment_debug.get("raw_match_count"),
        "inlier_point_count": alignment_debug.get("inlier_count"),
        "median_center_error_px": alignment_debug.get("median_error"),
        "reference_anchor_count": alignment_debug.get("reference_feature_count"),
        "frame_detection_count": alignment_debug.get("frame_feature_count"),
    }


def _next_step_hint(
    *,
    final_source: str,
    yolo_detection_count: int,
    status: str,
    method: str,
) -> str:
    if final_source == "yolo_anchor_pose":
        return "V4 pose is active; inspect missing/presence decisions inside projected slots"
    if final_source == "feature_slot_fallback" and yolo_detection_count <= 0:
        return "YOLO is empty; this result is using the slot/feature fallback as intended"
    if final_source == "feature_slot_fallback":
        return "YOLO detections exist but V4 pose was not selected; inspect yolo_anchor_pose reject thresholds"
    if final_source in {"none_yolo_empty_feature_unconfirmed", "scene_unconfirmed"}:
        return "No confirmed pose; do not draw confident missing polygons, ask for recapture or inspect fallback thresholds"
    if status and status != "success":
        return f"pose is unconfirmed by {method or 'unknown'}: {status}"
    return "inspect final_pose_source and alignment debug"


def _has_scene_unconfirmed_details(details: list[dict[str, Any]]) -> bool:
    for detail in details:
        debug = detail.get("debug")
        if not isinstance(debug, dict):
            continue
        if debug.get("reason_code") == "scene_pose_unconfirmed":
            return True
    return False


def _detail_status_counts(details: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for detail in details:
        status = _string_value(detail.get("status"))
        if status:
            counts[status] += 1
    return counts


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
