from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np
from app.config import settings
from modules.yolo.inspection.constants import inspections as inspections_constants
from modules.yolo.inspection.domain.debug_payload import build_inspection_debug_payload
from modules.yolo.inspection.domain.types import SegmentMatch
from modules.yolo.training.models import MlModel

if TYPE_CHECKING:
    from modules.yolo.inspection.realtime.frame_result import (
        FrameResult as RealtimeFrameResult,
    )

    from .context import InspectionContext


_DB_ALIGNMENT_STATUSES = {
    "success",
    "insufficient_matches",
    "insufficient_inliers",
    "homography_failed",
}


def build_inspection_payload(
    *,
    context: InspectionContext,
    camera_id: UUID | None,
    mode: str,
    notes: str | None,
    filename: str | None,
    content_type: str | None,
    image_path: str,
    selected_segment_class_ids: list[UUID],
) -> dict:
    return {
        "standard_id": str(context.standard.id),
        "group_id": str(context.standard.group_id),
        "camera_id": str(camera_id) if camera_id else None,
        "mode": mode,
        "notes": notes,
        "filename": filename,
        "content_type": content_type,
        "image_path": image_path,
        "model_id": str(context.model.id),
        "weights_path": context.model.weights_path,
        "imgsz": context.model.imgsz,
        "selected_segment_class_ids": [
            str(class_id) for class_id in selected_segment_class_ids
        ],
        "verification_mode": settings.INSPECTION_VERIFICATION_MODE,
    }


def build_result_payload(
    *,
    task_id: UUID | None,
    context: InspectionContext,
    payload: dict,
    matches: list[SegmentMatch],
    inspection_status: str,
    total: int,
    matched: int,
    missing: list[str],
    alignment_status: str | None,
    alignment_inlier_count: int | None,
    alignment_raw_match_count: int | None,
    homography: np.ndarray | list | None,
    raw_class_counts: dict[str, int] | None,
    result_image_path: str | None = None,
    alignment_debug: dict[str, Any] | None = None,
    verification_mode: str | None = None,
) -> dict:
    resolved_verification_mode = (
        verification_mode
        or payload.get("verification_mode")
        or settings.INSPECTION_VERIFICATION_MODE
    )
    normalized_homography = _homography_to_json(homography)
    details = [build_result_item(match) for match in matches]
    debug_payload = build_inspection_debug_payload(
        details=details,
        raw_class_counts=raw_class_counts,
        imgsz=context.model.imgsz,
        weights_path=context.model.weights_path,
        alignment_debug=alignment_debug,
        verification_mode=resolved_verification_mode,
    )

    return {
        "task_id": str(task_id) if task_id is not None else None,
        "inspection_id": None,
        "status": inspection_status,
        "inspection_status": inspection_status,
        "passed": inspection_status == inspections_constants.statuses.passed,
        "matched": matched,
        "total": total,
        "missing": missing,
        "alignment_status": (
            None
            if resolved_verification_mode == "yolo_count"
            else normalize_alignment_status_for_db(alignment_status)
        ),
        "alignment_inlier_count": (
            None
            if resolved_verification_mode == "yolo_count"
            else alignment_inlier_count
        ),
        "alignment_raw_match_count": (
            None
            if resolved_verification_mode == "yolo_count"
            else alignment_raw_match_count
        ),
        "homography": (
            None
            if resolved_verification_mode == "yolo_count"
            else normalized_homography
        ),
        "details": details,
        "standard_id": str(context.standard.id),
        "model_id": str(context.model.id),
        "camera_id": payload.get("camera_id"),
        "mode": payload["mode"],
        "image_path": payload["image_path"],
        "result_image_path": result_image_path,
        "model_name": build_model_name(context.model),
        "debug_payload": debug_payload,
        "pose_pipeline": _extract_pose_pipeline(debug_payload),
    }


def build_result_payload_from_frame_result(
    *,
    task_id: UUID | None,
    context: InspectionContext,
    payload: dict,
    frame_result: Any,
    result_image_path: str | None = None,
) -> dict:
    return build_result_payload(
        task_id=task_id,
        context=context,
        payload=payload,
        matches=frame_result.matches,
        inspection_status=frame_result.inspection_status,
        total=frame_result.total,
        matched=frame_result.matched,
        missing=frame_result.missing,
        alignment_status=(
            None
            if frame_result.verification_mode == "yolo_count"
            else frame_result.alignment.status.value
        ),
        alignment_inlier_count=(
            None
            if frame_result.verification_mode == "yolo_count"
            else frame_result.alignment.inlier_count
        ),
        alignment_raw_match_count=(
            None
            if frame_result.verification_mode == "yolo_count"
            else frame_result.alignment.raw_match_count
        ),
        homography=(
            None
            if frame_result.verification_mode == "yolo_count"
            else frame_result.alignment.homography
        ),
        raw_class_counts=frame_result.raw_class_counts,
        result_image_path=result_image_path,
        alignment_debug=frame_result.alignment.to_debug_payload(),
        verification_mode=frame_result.verification_mode,
    )


def build_realtime_save_payload(
    *,
    context: InspectionContext,
    camera_id: UUID | None,
    image_path: str,
    result: RealtimeFrameResult,
    result_image_path: str | None = None,
) -> dict:
    verification_mode = getattr(
        result,
        "verification_mode",
        settings.INSPECTION_VERIFICATION_MODE,
    )
    debug_payload = build_inspection_debug_payload(
        details=result.details,
        raw_class_counts=None,
        imgsz=context.model.imgsz,
        weights_path=context.model.weights_path,
        alignment_debug=result.alignment_debug,
        verification_mode=verification_mode,
    )

    return {
        "task_id": None,
        "inspection_id": None,
        "standard_id": str(context.standard.id),
        "model_id": str(context.model.id),
        "camera_id": str(camera_id) if camera_id else None,
        "mode": inspections_constants.modes.realtime,
        "image_path": image_path,
        "result_image_path": result_image_path,
        "status": result.status,
        "inspection_status": result.status,
        "passed": bool(result.passed),
        "matched": result.matched,
        "total": result.total,
        "missing": result.missing,
        "alignment_status": (
            None
            if verification_mode == "yolo_count"
            else normalize_alignment_status_for_db(
                result.alignment_db_status or result.alignment_status
            )
        ),
        "alignment_inlier_count": (
            None
            if verification_mode == "yolo_count"
            else result.alignment_inlier_count
        ),
        "alignment_raw_match_count": (
            None
            if verification_mode == "yolo_count"
            else result.alignment_raw_match_count
        ),
        "homography": (
            None
            if verification_mode == "yolo_count"
            else _homography_to_json(result.homography)
        ),
        "details": result.details,
        "model_name": build_model_name(context.model),
        "debug_payload": debug_payload,
        "pose_pipeline": _extract_pose_pipeline(debug_payload),
    }


def _extract_pose_pipeline(debug_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(debug_payload, dict):
        return None

    pose_pipeline = debug_payload.get("pose_pipeline")
    if isinstance(pose_pipeline, dict):
        return pose_pipeline

    return None


def build_result_item(match: SegmentMatch) -> dict:
    return {
        "annotation_id": str(match.annotation_id) if match.annotation_id else None,
        "segment_class_id": (
            str(match.segment_class_id) if match.segment_class_id else None
        ),
        "class_key": match.class_key,
        "name": match.name,
        "hue": match.hue,
        "status": match.status,
        "iou": match.iou,
        "confidence": match.confidence,
        "expected_polygon": match.expected_polygon,
        "detected_polygon": match.detected_polygon,
        "detected_bbox": match.detected_bbox,
        "detected_class_in_zone": match.detected_class_in_zone,
        "debug": match.debug,
    }


def build_model_name(model: MlModel) -> str:
    if model.version is None:
        return model.architecture

    return f"{model.architecture} v{model.version}"


def build_native_to_internal_key_map(model: MlModel) -> dict[str, str]:
    mapping: dict[str, str] = {}
    class_meta = list(model.class_meta or [])

    if class_meta:
        for item in class_meta:
            internal_key = _str_or_none(item.get("id")) or _str_or_none(item.get("key"))
            if internal_key is None:
                continue

            _register_native_mapping_aliases(
                mapping,
                source_key=internal_key,
                target_key=internal_key,
            )

            native_key = _str_or_none(item.get("native_key"))
            if native_key is None:
                continue

            _register_native_mapping_aliases(
                mapping,
                source_key=native_key,
                target_key=internal_key,
            )

        return mapping

    for internal_key in model.class_keys or []:
        normalized_key = _str_or_none(internal_key)
        if normalized_key is None:
            continue

        _register_native_mapping_aliases(
            mapping,
            source_key=normalized_key,
            target_key=normalized_key,
        )

    return mapping


def normalize_alignment_status_for_db(alignment_status: str | None) -> str | None:
    if alignment_status in _DB_ALIGNMENT_STATUSES:
        return alignment_status
    if alignment_status in {"skipped", "yolo_count"}:
        return None
    return "homography_failed"


def _homography_to_json(
    homography: np.ndarray | list | None,
) -> list[list[float]] | None:
    if homography is None:
        return None

    if isinstance(homography, np.ndarray):
        return homography.tolist()

    return homography


def _register_native_mapping_aliases(
    mapping: dict[str, str],
    *,
    source_key: str,
    target_key: str,
) -> None:
    mapping[source_key] = target_key

    if "__" in source_key:
        mapping[source_key.split("__", 1)[1]] = target_key
        return

    if "_" in source_key:
        suffix = source_key.rsplit("_", 1)[-1]
        if suffix and suffix.isdigit():
            mapping[source_key.rsplit("_", 1)[0]] = target_key


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None
