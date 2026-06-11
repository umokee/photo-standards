from __future__ import annotations

from typing import Any
from uuid import UUID

from app.exception import ValidationError
from modules.core.segments.constants import segments
from modules.yolo.inspection.domain.types import SegmentMatch
from modules.yolo.inspection.models import InspectionResult, InspectionSegmentResult


def build_inspection_entities(
    *,
    data: dict[str, Any],
    notes: str | None,
    inspection_id: UUID | None = None,
) -> tuple[InspectionResult, list[InspectionSegmentResult]]:
    inspection_status = data.get("inspection_status") or data.get("status")

    if not inspection_status:
        raise ValidationError("В результате задачи отсутствует статус проверки")

    inspection_data: dict[str, Any] = {
        "standard_id": UUID(data["standard_id"]),
        "model_id": UUID(data["model_id"]),
        "camera_id": UUID(data["camera_id"]) if data.get("camera_id") else None,
        "image_path": data["image_path"],
        "result_image_path": data.get("result_image_path"),
        "status": inspection_status,
        "mode": data["mode"],
        "total_segments": data["total"],
        "matched_segments": data["matched"],
        "alignment_status": data.get("alignment_status"),
        "alignment_inlier_count": data.get("alignment_inlier_count"),
        "alignment_raw_match_count": data.get("alignment_raw_match_count"),
        "homography": data.get("homography"),
        "notes": notes,
        "debug_payload": data.get("debug_payload"),
    }
    if inspection_id is not None:
        inspection_data["id"] = inspection_id

    inspection = InspectionResult(**inspection_data)

    segment_results = build_segment_results(data.get("details", []))

    return inspection, segment_results


def build_segment_results(
    details: list[dict[str, Any]],
) -> list[InspectionSegmentResult]:
    return [
        InspectionSegmentResult(
            segment_annotation_id=(
                UUID(detail["annotation_id"]) if detail.get("annotation_id") else None
            ),
            segment_class_id=(
                UUID(detail["segment_class_id"])
                if detail.get("segment_class_id")
                else None
            ),
            class_key=detail["class_key"],
            name=detail["name"],
            hue=detail.get("hue")
            if detail.get("hue") is not None
            else segments.hue.default,
            status=detail["status"],
            iou=detail.get("iou"),
            confidence=detail.get("confidence"),
            expected_polygon=detail.get("expected_polygon"),
            detected_polygon=detail.get("detected_polygon"),
            detected_bbox=detail.get("detected_bbox"),
        )
        for detail in details
    ]


def build_matches_from_details(details: list[dict[str, Any]]) -> list[SegmentMatch]:
    matches: list[SegmentMatch] = []

    for detail in details:
        if not isinstance(detail, dict):
            continue

        matches.append(
            SegmentMatch(
                annotation_id=_uuid_or_none(detail.get("annotation_id")),
                segment_class_id=_uuid_or_none(detail.get("segment_class_id")),
                class_key=str(detail.get("class_key") or ""),
                name=str(detail.get("name") or "Объект"),
                hue=detail.get("hue")
                if detail.get("hue") is not None
                else segments.hue.default,
                status=str(detail.get("status") or "missing"),
                iou=detail.get("iou"),
                confidence=detail.get("confidence"),
                expected_polygon=detail.get("expected_polygon"),
                detected_polygon=detail.get("detected_polygon"),
                detected_bbox=detail.get("detected_bbox"),
                detected_class_in_zone=(
                    str(detail.get("detected_class_in_zone"))
                    if detail.get("detected_class_in_zone") is not None
                    else None
                ),
                debug=detail.get("debug") if isinstance(detail.get("debug"), dict) else None,
            )
        )

    return matches


def _uuid_or_none(value: Any) -> UUID | None:
    if value is None:
        return None

    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None
