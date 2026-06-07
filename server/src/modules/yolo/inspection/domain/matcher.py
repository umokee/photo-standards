from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from modules.yolo.inspection.domain.alignment import LocalProjectionData, project_polygon
from modules.yolo.inspection.domain.types import ExpectedSegment, SegmentMatch, YoloDetection

# This matcher is intentionally small again.
# Pose/alignment is produced before this file:
#   - primary: YOLO-anchor pose when YOLO found visible objects;
#   - fallback: feature alignment from SuperPoint/LightGlue homography.
# This file only projects reference slots, matches detections to projected slots,
# and marks missing/extra/unmatched safely.

_MIN_POLYGON_POINTS = 3
_MIN_MATCH_IOU = 0.10
_MAX_CENTER_DISTANCE_FACTOR = 0.60
_MIN_VISIBLE_FRACTION = 0.04

BBox = tuple[float, float, float, float]


def build_expected_segments(
    reference_image: Any,
    selected_class_ids: set[Any],
) -> list[ExpectedSegment]:
    items: list[ExpectedSegment] = []

    for annotation in reference_image.annotations:
        if annotation.segment_class_id is None or annotation.segment_class is None:
            continue
        if annotation.segment_class_id not in selected_class_ids:
            continue

        segment_class = annotation.segment_class
        for polygon in annotation.points or []:
            if len(polygon) < _MIN_POLYGON_POINTS:
                continue
            items.append(
                ExpectedSegment(
                    annotation_id=annotation.id,
                    segment_class_id=segment_class.id,
                    class_key=str(segment_class.id),
                    name=segment_class.name,
                    hue=segment_class.hue,
                    reference_polygon=_clean_polygon(polygon),
                )
            )

    return items


def build_missing_matches(
    expected: list[ExpectedSegment],
    homography: np.ndarray | None = None,
    *,
    frame_size: tuple[int, int] | None = None,
    projection_data: LocalProjectionData | None = None,
    hide_unconfirmed_projection: bool = False,
) -> list[SegmentMatch]:
    matches: list[SegmentMatch] = []
    transform = _resolve_homography(homography, projection_data)

    for item in expected:
        projected, reason = _project_expected_polygon(
            item,
            transform,
            frame_size=_resolve_frame_size(frame_size, projection_data),
        )
        unconfirmed = hide_unconfirmed_projection or projected is None
        matches.append(
            _expected_match(
                item,
                status="missing",
                expected_polygon=None if unconfirmed else projected,
                debug=_missing_debug(
                    projection="feature_lightglue_homography" if projected else "unconfirmed_hidden",
                    safety="unsafe_hidden" if unconfirmed else "confirmed",
                    reason=reason or "no_matching_detection",
                    reason_code=(
                        "scene_pose_unconfirmed" if unconfirmed else "no_matching_detection"
                    ),
                ),
            )
        )

    return matches


def match_segments(
    expected: list[ExpectedSegment],
    detections: list[YoloDetection],
    homography: np.ndarray | None,
    *,
    frame_size: tuple[int, int] | None = None,
    projection_data: LocalProjectionData | None = None,
) -> list[SegmentMatch]:
    transform = _resolve_homography(homography, projection_data)
    resolved_frame_size = _resolve_frame_size(frame_size, projection_data)

    projected: list[_ProjectedSlot] = []
    hidden_expected: list[tuple[ExpectedSegment, str]] = []
    for index, item in enumerate(expected):
        polygon, reason = _project_expected_polygon(
            item,
            transform,
            frame_size=resolved_frame_size,
        )
        if polygon is None:
            hidden_expected.append((item, reason or "scene_pose_unconfirmed"))
            continue
        bbox = _bbox_from_polygon(polygon)
        if bbox is None:
            hidden_expected.append((item, "invalid_projected_polygon"))
            continue
        projected.append(_ProjectedSlot(index=index, item=item, polygon=polygon, bbox=bbox))

    if not projected:
        return build_missing_matches(
            expected,
            transform,
            frame_size=resolved_frame_size,
            projection_data=projection_data,
            hide_unconfirmed_projection=True,
        )

    detection_items = [
        _DetectionSlot(index=index, detection=detection, polygon=_detection_polygon(detection))
        for index, detection in enumerate(detections)
    ]
    detection_items = [item for item in detection_items if item.bbox is not None]

    candidates: list[tuple[float, float, int, int, dict[str, Any]]] = []
    for slot in projected:
        for det in detection_items:
            if slot.item.class_key != det.detection.class_key:
                continue
            score, iou, center_factor = _score_slot_detection(slot, det)
            if _candidate_is_acceptable(iou=iou, center_factor=center_factor):
                candidates.append(
                    (
                        score,
                        iou,
                        slot.index,
                        det.index,
                        {
                            "match_source": "projected_slot",
                            "iou": _round(iou),
                            "center_distance_factor": _round(center_factor),
                        },
                    )
                )

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_expected: set[int] = set()
    used_detections: set[int] = set()
    chosen: list[tuple[int, int, float, dict[str, Any]]] = []
    for _score, iou, expected_index, detection_index, debug in candidates:
        if expected_index in used_expected or detection_index in used_detections:
            continue
        used_expected.add(expected_index)
        used_detections.add(detection_index)
        chosen.append((expected_index, detection_index, iou, debug))

    projected_by_index = {slot.index: slot for slot in projected}
    detection_by_index = {item.index: item for item in detection_items}
    matches: list[SegmentMatch] = []

    for expected_index, detection_index, iou, debug in chosen:
        slot = projected_by_index[expected_index]
        det = detection_by_index[detection_index]
        matches.append(
            _expected_match(
                slot.item,
                status="ok",
                iou=iou,
                confidence=det.detection.confidence,
                expected_polygon=slot.polygon,
                detected_polygon=det.polygon,
                detected_bbox=det.detection.bbox,
                debug=debug,
            )
        )

    for slot in projected:
        if slot.index in used_expected:
            continue
        matches.append(
            _expected_match(
                slot.item,
                status="missing",
                expected_polygon=slot.polygon,
                debug=_missing_debug(
                    projection="feature_lightglue_homography",
                    safety="confirmed",
                    reason="no_matching_detection",
                    reason_code="no_matching_detection",
                ),
            )
        )

    for item, reason in hidden_expected:
        matches.append(
            _expected_match(
                item,
                status="missing",
                expected_polygon=None,
                debug=_missing_debug(
                    projection="unconfirmed_hidden",
                    safety="unsafe_hidden",
                    reason=reason,
                    reason_code="scene_pose_unconfirmed",
                ),
            )
        )

    for det in detection_items:
        if det.index in used_detections:
            continue
        containing_slot = _containing_slot(det, projected)
        if containing_slot is not None:
            match = _detection_match(
                det.detection,
                name=containing_slot.item.name,
                hue=containing_slot.item.hue,
                status="unmatched",
                detected_polygon=det.polygon,
                debug={
                    "reason": "detection_inside_projected_slot_but_not_matched",
                    "expected_class_key": containing_slot.item.class_key,
                },
            )
            match.detected_class_in_zone = det.detection.class_key
            matches.append(match)
        else:
            matches.append(
                _detection_match(
                    det.detection,
                    name="Лишнее",
                    hue=None,
                    status="extra",
                    detected_polygon=det.polygon,
                    debug={"reason": "outside_projected_slots"},
                )
            )

    return matches


def match_segments_by_count(
    expected: list[ExpectedSegment],
    detections: list[YoloDetection],
) -> list[SegmentMatch]:
    expected_by_class: dict[str, list[ExpectedSegment]] = {}
    for item in expected:
        expected_by_class.setdefault(item.class_key, []).append(item)

    detections_by_class: dict[str, list[YoloDetection]] = {}
    for detection in detections:
        detections_by_class.setdefault(detection.class_key, []).append(detection)

    matches: list[SegmentMatch] = []
    for class_key in sorted(set(expected_by_class) | set(detections_by_class)):
        expected_items = expected_by_class.get(class_key, [])
        detected_items = detections_by_class.get(class_key, [])
        matched_count = min(len(expected_items), len(detected_items))
        fallback_name = expected_items[0].name if expected_items else "Лишнее"
        fallback_hue = expected_items[0].hue if expected_items else None

        for index in range(matched_count):
            expected_item = expected_items[index]
            detection = detected_items[index]
            matches.append(
                _expected_match(
                    expected_item,
                    status="ok",
                    confidence=detection.confidence,
                    detected_polygon=_detection_polygon(detection),
                    detected_bbox=detection.bbox,
                )
            )

        for expected_item in expected_items[matched_count:]:
            matches.append(_expected_match(expected_item, status="missing"))

        for detection in detected_items[matched_count:]:
            matches.append(
                _detection_match(
                    detection,
                    name=fallback_name,
                    hue=fallback_hue,
                    status="extra",
                    detected_polygon=_detection_polygon(detection),
                )
            )

    return matches


def summarize(matches: list[SegmentMatch]) -> tuple[int, int, list[str]]:
    expected_matches = [match for match in matches if match.annotation_id is not None]
    total = len(expected_matches)
    matched = sum(1 for match in expected_matches if match.status == "ok")
    missing_names = [
        match.name for match in expected_matches if match.status in {"missing", "unmatched"}
    ]
    return total, matched, missing_names


def all_ok(matches: list[SegmentMatch], *, expected_total: int | None = None) -> bool:
    expected_matches = [match for match in matches if match.annotation_id is not None]
    total = expected_total if expected_total is not None else len(expected_matches)
    if len(expected_matches) != total:
        return False
    if any(match.status != "ok" for match in expected_matches):
        return False
    return not any(match.status in {"extra", "unmatched"} for match in matches)


class _ProjectedSlot:
    __slots__ = ("index", "item", "polygon", "bbox")

    def __init__(self, *, index: int, item: ExpectedSegment, polygon: list[list[float]], bbox: BBox) -> None:
        self.index = index
        self.item = item
        self.polygon = polygon
        self.bbox = bbox


class _DetectionSlot:
    __slots__ = ("index", "detection", "polygon", "bbox")

    def __init__(self, *, index: int, detection: YoloDetection, polygon: list[list[float]] | None) -> None:
        self.index = index
        self.detection = detection
        self.polygon = polygon
        self.bbox = _bbox_from_polygon(polygon) if polygon else _bbox_from_detection(detection)


def _expected_match(
    item: ExpectedSegment,
    *,
    status: str,
    iou: float | None = None,
    confidence: float | None = None,
    expected_polygon: list[list[float]] | None = None,
    detected_polygon: list[list[float]] | None = None,
    detected_bbox: dict[str, float] | None = None,
    debug: dict[str, Any] | None = None,
) -> SegmentMatch:
    return SegmentMatch(
        annotation_id=item.annotation_id,
        segment_class_id=item.segment_class_id,
        class_key=item.class_key,
        name=item.name,
        hue=item.hue,
        status=status,
        iou=iou,
        confidence=confidence,
        expected_polygon=expected_polygon,
        detected_polygon=detected_polygon,
        detected_bbox=detected_bbox,
        debug=debug,
    )


def _detection_match(
    detection: YoloDetection,
    *,
    name: str,
    hue: int | None,
    status: str,
    detected_polygon: list[list[float]] | None,
    debug: dict[str, Any] | None = None,
) -> SegmentMatch:
    return SegmentMatch(
        annotation_id=None,
        segment_class_id=None,
        class_key=detection.class_key,
        name=name,
        hue=hue,
        status=status,
        iou=None,
        confidence=detection.confidence,
        expected_polygon=None,
        detected_polygon=detected_polygon,
        detected_bbox=detection.bbox,
        debug=debug,
    )


def _missing_debug(
    *,
    projection: str,
    safety: str,
    reason: str,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "missing_polygon_projection": projection,
        "projection": projection,
        "missing_polygon_projection_safety": safety,
        "reason": reason,
        "reason_code": reason_code,
    }


def _resolve_homography(
    homography: np.ndarray | None,
    projection_data: LocalProjectionData | None,
) -> np.ndarray | None:
    if homography is not None:
        return homography
    if projection_data is not None:
        return projection_data.global_homography
    return None


def _resolve_frame_size(
    frame_size: tuple[int, int] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[int, int] | None:
    if frame_size is not None:
        return frame_size
    if projection_data is not None:
        return projection_data.frame_size
    return None


def _project_expected_polygon(
    item: ExpectedSegment,
    homography: np.ndarray | None,
    *,
    frame_size: tuple[int, int] | None,
) -> tuple[list[list[float]] | None, str | None]:
    if homography is None:
        return None, "scene_pose_unconfirmed"
    try:
        projected = project_polygon(item.reference_polygon, homography)
    except cv2.error:
        return None, "projection_failed"
    if len(projected) < _MIN_POLYGON_POINTS or not _polygon_is_finite(projected):
        return None, "invalid_projected_polygon"
    bbox = _bbox_from_polygon(projected)
    if bbox is None:
        return None, "invalid_projected_bbox"
    if frame_size is not None and not _bbox_visible(bbox, frame_size):
        return None, "projected_polygon_outside_frame"
    return _clean_polygon(projected), None


def _detection_polygon(detection: YoloDetection) -> list[list[float]] | None:
    if detection.polygon and len(detection.polygon) >= _MIN_POLYGON_POINTS:
        return _clean_polygon(detection.polygon)
    bbox = _bbox_from_detection(detection)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _score_slot_detection(slot: _ProjectedSlot, det: _DetectionSlot) -> tuple[float, float, float]:
    if det.bbox is None:
        return -1.0, 0.0, 999.0
    iou = _bbox_iou(slot.bbox, det.bbox)
    center_factor = _center_distance_factor(slot.bbox, det.bbox)
    score = iou - 0.18 * center_factor + 0.001 * float(det.detection.confidence or 0.0)
    return score, iou, center_factor


def _candidate_is_acceptable(*, iou: float, center_factor: float) -> bool:
    return iou >= _MIN_MATCH_IOU or center_factor <= _MAX_CENTER_DISTANCE_FACTOR


def _containing_slot(det: _DetectionSlot, slots: list[_ProjectedSlot]) -> _ProjectedSlot | None:
    if det.bbox is None:
        return None
    center = _bbox_center(det.bbox)
    best: tuple[float, _ProjectedSlot] | None = None
    for slot in slots:
        if not _point_in_bbox(center, slot.bbox):
            continue
        iou = _bbox_iou(slot.bbox, det.bbox)
        if best is None or iou > best[0]:
            best = (iou, slot)
    return best[1] if best is not None else None


def _bbox_from_detection(detection: YoloDetection) -> BBox | None:
    bbox = detection.bbox or {}
    candidates = [
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]
    for keys in candidates:
        if all(key in bbox for key in keys):
            return _normalize_bbox(tuple(float(bbox[key]) for key in keys))
    if all(key in bbox for key in ("x", "y", "width", "height")):
        x = float(bbox["x"])
        y = float(bbox["y"])
        return _normalize_bbox((x, y, x + float(bbox["width"]), y + float(bbox["height"])))
    return None


def _bbox_from_polygon(polygon: list[list[float]] | None) -> BBox | None:
    if polygon is None or len(polygon) < _MIN_POLYGON_POINTS:
        return None
    arr = np.asarray(polygon, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.isfinite(arr[:, :2]).all():
        return None
    x1 = float(np.min(arr[:, 0]))
    y1 = float(np.min(arr[:, 1]))
    x2 = float(np.max(arr[:, 0]))
    y2 = float(np.max(arr[:, 1]))
    return _normalize_bbox((x1, y1, x2, y2))


def _normalize_bbox(values: tuple[float, float, float, float]) -> BBox | None:
    x1, y1, x2, y2 = values
    if not all(np.isfinite(value) for value in values):
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left <= 1e-6 or bottom - top <= 1e-6:
        return None
    return left, top, right, bottom


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _center_distance_factor(a: BBox, b: BBox) -> float:
    ac = _bbox_center(a)
    bc = _bbox_center(b)
    diag = max(_bbox_diag(a), 1.0)
    return float(np.hypot(ac[0] - bc[0], ac[1] - bc[1]) / diag)


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1))


def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_visible(bbox: BBox, frame_size: tuple[int, int]) -> bool:
    height, width = frame_size
    frame = (0.0, 0.0, float(width), float(height))
    visible_area = _intersection_area(bbox, frame)
    bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return visible_area / bbox_area >= _MIN_VISIBLE_FRACTION


def _intersection_area(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _polygon_is_finite(polygon: list[list[float]]) -> bool:
    arr = np.asarray(polygon, dtype=np.float64)
    return bool(arr.ndim == 2 and arr.shape[0] >= _MIN_POLYGON_POINTS and np.isfinite(arr[:, :2]).all())


def _clean_polygon(polygon: list[list[float]]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in polygon if len(point) >= 2]


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)
