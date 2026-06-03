from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from app.config import settings
from modules.core.standards.reference_constants import IOU_MATCH_THRESHOLD
from modules.yolo.inspection.domain.alignment import (
    LocalProjectionData,
    project_polygon,
    project_polygon_adaptive,
)
from modules.yolo.inspection.domain.types import (
    ExpectedSegment,
    SegmentMatch,
    YoloDetection,
)
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.validation import make_valid

BBox = tuple[float, float, float, float]

_MIN_MATCH_SCORE = 0.42
_MAX_CENTER_DISTANCE_FACTOR = 0.85
_MIN_AREA_RATIO = 0.20
_MAX_AREA_RATIO = 5.00
_MAX_OPTIMAL_ASSIGNMENT_CANDIDATES = 72
_MAX_OPTIMAL_ASSIGNMENT_ITEMS = 18
_DUPLICATE_MATCHED_BBOX_IOU = 0.55
_DUPLICATE_MATCHED_CONTAINMENT = 0.80

_MISSING_POLYGON_MAX_LOCAL_POINTS = 96
_MISSING_POLYGON_MIN_CONTAINMENT = 0.18
_MISSING_POLYGON_MIN_AREA_SCORE = 0.16
_MISSING_POLYGON_MIN_AFFINE_SCALE = 0.45
_MISSING_POLYGON_MAX_AFFINE_SCALE = 2.80
_MISSING_POLYGON_EDGE_CROP_PADDING = 8
_MISSING_POLYGON_EDGE_MIN_PIXELS = 8
_MISSING_POLYGON_EDGE_MIN_MOVED_POINTS = 3
_MISSING_POLYGON_EDGE_MIN_MOVED_FRACTION = 0.10
_MISSING_POLYGON_EDGE_MIN_CONTAINMENT = 0.12
_MISSING_POLYGON_EDGE_MIN_AREA_SCORE = 0.38
_MISSING_POLYGON_EDGE_MAX_CENTER_DRIFT_FACTOR = 2.00
_MISSING_POLYGON_EDGE_MAX_DENSE_POINTS = 220
_MISSING_POLYGON_EDGE_MIN_GRADIENT_ALIGNMENT = 0.22
_MISSING_POLYGON_EDGE_TANGENT_BAND_FRACTION = 0.24
_MISSING_POLYGON_EDGE_SIMPLIFY_EPSILON = 1.25
_MISSING_POLYGON_EDGE_MIN_INWARD_SHIFT = 2.0
_MISSING_POLYGON_EDGE_AREA_RATIO_FACTOR = 0.80


@dataclass(slots=True)
class _ProjectedExpected:
    index: int
    item: ExpectedSegment
    polygon: list[list[float]]
    bbox: BBox


@dataclass(slots=True)
class _DetectionCandidate:
    index: int
    detection: YoloDetection
    polygon: list[list[float]]
    bbox: BBox


@dataclass(slots=True)
class _ExpectedSlot:
    projected_bbox: BBox
    search_bbox: BBox
    reference_bbox: BBox | None
    feature_support: int
    feature_total: int


@dataclass(slots=True)
class _MissingPolygonRefinement:
    polygon: list[list[float]]
    bbox: BBox
    feature_support: int
    feature_total: int
    candidate_count: int
    inlier_count: int
    median_error: float
    containment: float
    area_score: float
    edge_snapped: bool = False
    edge_moved_points: int = 0
    edge_mean_shift: float = 0.0
    edge_max_shift: float = 0.0


@dataclass(slots=True)
class _EdgeSnapResult:
    polygon: list[list[float]]
    bbox: BBox
    moved_points: int
    mean_shift: float
    max_shift: float


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
            if len(polygon) < 3:
                continue

            items.append(
                ExpectedSegment(
                    annotation_id=annotation.id,
                    segment_class_id=segment_class.id,
                    class_key=str(segment_class.id),
                    name=segment_class.name,
                    hue=segment_class.hue,
                    reference_polygon=polygon,
                )
            )

    return items


def build_missing_matches(
    expected: list[ExpectedSegment],
    homography: np.ndarray | None = None,
    *,
    projection_data: LocalProjectionData | None = None,
) -> list[SegmentMatch]:
    matches: list[SegmentMatch] = []

    for item in expected:
        expected_polygon = None

        if homography is not None:
            projected = _safe_project(
                item.reference_polygon,
                homography,
                projection_data=projection_data,
            )
            expected_polygon = projected if len(projected) >= 3 else None

        matches.append(
            SegmentMatch(
                annotation_id=item.annotation_id,
                segment_class_id=item.segment_class_id,
                class_key=item.class_key,
                name=item.name,
                hue=item.hue,
                status="missing",
                iou=None,
                confidence=None,
                expected_polygon=expected_polygon,
                detected_polygon=None,
                detected_bbox=None,
                debug={"reason": "no_matching_detection"},
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
    projected_expected: list[_ProjectedExpected] = []
    unprojected_expected: list[ExpectedSegment] = []

    for index, item in enumerate(expected):
        projected = _safe_project(
            item.reference_polygon,
            homography,
            projection_data=projection_data,
        )
        if len(projected) < 3:
            unprojected_expected.append(item)
            continue

        bbox = _bbox_from_polygon(projected)
        if bbox is None:
            unprojected_expected.append(item)
            continue

        if frame_size is not None and not _is_visible_in_frame(bbox, frame_size):
            unprojected_expected.append(item)
            continue

        projected_expected.append(
            _ProjectedExpected(
                index=index,
                item=item,
                polygon=projected,
                bbox=bbox,
            )
        )

    if not projected_expected:
        return build_missing_matches(
            expected,
            homography,
            projection_data=projection_data,
        )

    detection_candidates: list[_DetectionCandidate] = []
    for index, detection in enumerate(detections):
        polygon = _detection_polygon(detection)
        bbox = _bbox_from_detection(detection, polygon)
        if polygon is None or bbox is None:
            continue

        detection_candidates.append(
            _DetectionCandidate(
                index=index,
                detection=detection,
                polygon=polygon,
                bbox=bbox,
            )
        )

    candidate_pairs: list[tuple[float, float, int, int]] = []
    candidate_debug: dict[tuple[int, int], dict[str, Any]] = {}
    projected_by_index = {item.index: item for item in projected_expected}
    detection_by_index = {item.index: item for item in detection_candidates}
    slot_by_index = _build_expected_slots(
        projected_expected,
        projection_data=projection_data,
    )

    for expected_item in projected_expected:
        for detection_item in detection_candidates:
            if expected_item.item.class_key != detection_item.detection.class_key:
                continue

            global_debug = _match_score_details(
                expected_item.polygon,
                detection_item.polygon,
                expected_item.bbox,
                detection_item.bbox,
            )
            slot_candidate = _try_slot_candidate(
                expected_item,
                detection_item,
                slot=slot_by_index.get(expected_item.index),
                global_debug=global_debug,
            )
            if slot_candidate is None:
                continue

            slot_score, slot_iou, slot_debug = slot_candidate
            candidate_debug[(expected_item.index, detection_item.index)] = slot_debug
            candidate_pairs.append(
                (slot_score, slot_iou, expected_item.index, detection_item.index)
            )

    chosen_pairs = _choose_candidate_pairs(candidate_pairs)
    matched_expected_indices = {expected_index for expected_index, _, _ in chosen_pairs}
    matched_detection_indices = {detection_index for _, detection_index, _ in chosen_pairs}

    matches: list[SegmentMatch] = []

    for expected_index, detection_index, iou in chosen_pairs:
        expected_item = projected_by_index[expected_index]
        detection_item = detection_by_index[detection_index]
        detection = detection_item.detection

        expected_polygon = expected_item.polygon
        detected_polygon = _display_detected_polygon(detection_item)

        matches.append(
            SegmentMatch(
                annotation_id=expected_item.item.annotation_id,
                segment_class_id=expected_item.item.segment_class_id,
                class_key=expected_item.item.class_key,
                name=expected_item.item.name,
                hue=expected_item.item.hue,
                status="ok",
                iou=round(float(iou), 4),
                confidence=detection.confidence,
                expected_polygon=expected_polygon,
                detected_polygon=detected_polygon,
                detected_bbox=detection.bbox,
                debug=candidate_debug.get((expected_index, detection_index)),
            )
        )

    for expected_item in projected_expected:
        if expected_item.index in matched_expected_indices:
            continue

        slot = slot_by_index.get(expected_item.index)
        missing_debug = _missing_slot_debug(expected_item, slot)
        missing_polygon = expected_item.polygon

        missing_refinement = _try_missing_polygon_refinement(
            expected_item,
            slot=slot,
            projection_data=projection_data,
        )
        if missing_refinement is not None:
            missing_polygon = missing_refinement.polygon
            missing_debug = _merge_missing_refinement_debug(
                missing_debug,
                missing_refinement,
            )

        matches.append(
            SegmentMatch(
                annotation_id=expected_item.item.annotation_id,
                segment_class_id=expected_item.item.segment_class_id,
                class_key=expected_item.item.class_key,
                name=expected_item.item.name,
                hue=expected_item.item.hue,
                status="missing",
                iou=None,
                confidence=None,
                expected_polygon=missing_polygon,
                detected_polygon=None,
                detected_bbox=None,
                debug=missing_debug,
            )
        )

    for item in unprojected_expected:
        matches.append(
            SegmentMatch(
                annotation_id=item.annotation_id,
                segment_class_id=item.segment_class_id,
                class_key=item.class_key,
                name=item.name,
                hue=item.hue,
                status="missing",
                iou=None,
                confidence=None,
                expected_polygon=None,
                detected_polygon=None,
                detected_bbox=None,
                debug={"reason": "projection_failed"},
            )
        )

    expected_index_to_match: dict[int, SegmentMatch] = {}
    for expected_item in projected_expected:
        if expected_item.index in matched_expected_indices:
            continue
        for match in matches:
            if (
                match.status == "missing"
                and match.annotation_id == expected_item.item.annotation_id
                and match.segment_class_id == expected_item.item.segment_class_id
            ):
                expected_index_to_match[expected_item.index] = match
                break

    unmatched_expected_indices: set[int] = set()
    occupied_detection_indices = set(matched_detection_indices)

    for detection_item in detection_candidates:
        if detection_item.index in occupied_detection_indices:
            continue

        action, expected_index, debug = _classify_unmatched_detection(
            detection_item,
            projected_expected,
            matched_expected_indices | unmatched_expected_indices,
            slot_by_index=slot_by_index,
            occupied_detection_indices=occupied_detection_indices,
            detection_by_index=detection_by_index,
        )

        detection = detection_item.detection

        if action == "discard":
            if expected_index is not None:
                target_match = expected_index_to_match.get(expected_index)
                if (
                    target_match is not None
                    and target_match.detected_class_in_zone is None
                ):
                    target_match.detected_class_in_zone = detection.class_key
            continue

        if action == "unmatched":
            expected_item = (
                projected_by_index.get(expected_index)
                if expected_index is not None
                else None
            )
            target_match = (
                expected_index_to_match.get(expected_index)
                if expected_index is not None
                else None
            )
            if target_match is not None and target_match.detected_class_in_zone is None:
                target_match.detected_class_in_zone = detection.class_key

            matches.append(
                SegmentMatch(
                    annotation_id=None,
                    segment_class_id=None,
                    class_key=detection.class_key,
                    name=expected_item.item.name if expected_item else "Не сопоставлено",
                    hue=expected_item.item.hue if expected_item else None,
                    status="unmatched",
                    iou=None,
                    confidence=detection.confidence,
                    expected_polygon=None,
                    detected_polygon=detection_item.polygon,
                    detected_bbox=detection.bbox,
                    debug=debug,
                )
            )
            if expected_index is not None:
                unmatched_expected_indices.add(expected_index)
            occupied_detection_indices.add(detection_item.index)
            continue

        matches.append(
            SegmentMatch(
                annotation_id=None,
                segment_class_id=None,
                class_key=detection.class_key,
                name="Лишнее",
                hue=None,
                status="extra",
                iou=None,
                confidence=detection.confidence,
                expected_polygon=None,
                detected_polygon=detection_item.polygon,
                detected_bbox=detection.bbox,
                debug=debug,
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

    class_keys = sorted(set(expected_by_class) | set(detections_by_class))
    matches: list[SegmentMatch] = []

    for class_key in class_keys:
        expected_items = expected_by_class.get(class_key, [])
        detected_items = detections_by_class.get(class_key, [])
        matched_count = min(len(expected_items), len(detected_items))
        fallback_name = expected_items[0].name if expected_items else "Лишнее"
        fallback_hue = expected_items[0].hue if expected_items else None

        for index in range(matched_count):
            expected_item = expected_items[index]
            detection = detected_items[index]
            detected_polygon = _detection_polygon(detection)

            matches.append(
                SegmentMatch(
                    annotation_id=expected_item.annotation_id,
                    segment_class_id=expected_item.segment_class_id,
                    class_key=class_key,
                    name=expected_item.name,
                    hue=expected_item.hue,
                    status="ok",
                    iou=None,
                    confidence=detection.confidence,
                    expected_polygon=None,
                    detected_polygon=detected_polygon,
                    detected_bbox=detection.bbox,
                )
            )

        for expected_item in expected_items[matched_count:]:
            matches.append(
                SegmentMatch(
                    annotation_id=expected_item.annotation_id,
                    segment_class_id=expected_item.segment_class_id,
                    class_key=class_key,
                    name=expected_item.name,
                    hue=expected_item.hue,
                    status="missing",
                    iou=None,
                    confidence=None,
                    expected_polygon=None,
                    detected_polygon=None,
                    detected_bbox=None,
                )
            )

        for detection in detected_items[matched_count:]:
            detected_polygon = _detection_polygon(detection)
            matches.append(
                SegmentMatch(
                    annotation_id=None,
                    segment_class_id=None,
                    class_key=class_key,
                    name=fallback_name,
                    hue=fallback_hue,
                    status="extra",
                    iou=None,
                    confidence=detection.confidence,
                    expected_polygon=None,
                    detected_polygon=detected_polygon,
                    detected_bbox=detection.bbox,
                )
            )

    return matches


def summarize(matches: list[SegmentMatch]) -> tuple[int, int, list[str]]:
    expected_matches = [match for match in matches if match.status in ("ok", "missing")]
    total = len(expected_matches)
    matched = sum(1 for match in expected_matches if match.status == "ok")
    missing_names = [match.name for match in expected_matches if match.status == "missing"]
    return total, matched, missing_names


def all_ok(matches: list[SegmentMatch], *, expected_total: int | None = None) -> bool:
    expected_matches = [match for match in matches if match.status in ("ok", "missing")]
    total = expected_total if expected_total is not None else len(expected_matches)
    if total <= 0:
        return False
    if any(match.status in ("extra", "unmatched") for match in matches):
        return False
    return sum(1 for match in expected_matches if match.status == "ok") == total


def _safe_project(
    polygon: list[list[float]],
    homography: np.ndarray | None,
    *,
    projection_data: LocalProjectionData | None = None,
) -> list[list[float]]:
    try:
        if projection_data is not None:
            projected = project_polygon_adaptive(
                polygon,
                data=projection_data,
            )
        elif homography is not None:
            projected = project_polygon(polygon, homography)
        else:
            return []
    except Exception:
        return []

    if not projected or len(projected) < 3:
        return []

    if not all(len(point) >= 2 for point in projected):
        return []

    return [[float(point[0]), float(point[1])] for point in projected]


def _detection_polygon(detection: YoloDetection) -> list[list[float]] | None:
    if detection.polygon is not None and len(detection.polygon) >= 3:
        return [[float(x), float(y)] for x, y in detection.polygon]
    bbox = _bbox_from_detection(detection, None)
    if bbox is not None:
        return _polygon_from_bbox(bbox)
    return None


def _bbox_from_detection(
    detection: YoloDetection,
    polygon: list[list[float]] | None,
) -> BBox | None:
    bbox = detection.bbox or {}
    try:
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        w = float(bbox.get("w", 0))
        h = float(bbox.get("h", 0))
    except (TypeError, ValueError):
        x = y = w = h = 0.0

    if w > 0 and h > 0:
        return (x, y, x + w, y + h)

    if polygon is not None:
        return _bbox_from_polygon(polygon)

    return None


def _bbox_from_polygon(polygon: list[list[float]]) -> BBox | None:
    if len(polygon) < 3:
        return None

    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


def _bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1))


def _polygon_from_bbox(bbox: BBox) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _polygon_iou(
    polygon_a: list[list[float]],
    polygon_b: list[list[float]],
) -> float:
    try:
        shape_a = make_valid(Polygon(polygon_a))
        shape_b = make_valid(Polygon(polygon_b))
    except (ValueError, GEOSException):
        return 0.0

    if shape_a.is_empty or shape_b.is_empty:
        return 0.0

    try:
        intersection = shape_a.intersection(shape_b).area
        union = shape_a.union(shape_b).area
    except GEOSException:
        return 0.0

    if union <= 0:
        return 0.0
    return float(intersection / union)


def _build_expected_slots(
    projected_expected: list[_ProjectedExpected],
    *,
    projection_data: LocalProjectionData | None,
) -> dict[int, _ExpectedSlot]:
    slots: dict[int, _ExpectedSlot] = {}

    for expected_item in projected_expected:
        reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
        search_bbox = _expand_bbox(
            expected_item.bbox,
            factor=settings.INSPECTION_SLOT_SEARCH_EXPANSION,
            frame_size=(projection_data.frame_size if projection_data is not None else None),
        )
        feature_support, feature_total = _slot_feature_support(
            expected_item,
            search_bbox=search_bbox,
            projection_data=projection_data,
        )

        slots[expected_item.index] = _ExpectedSlot(
            projected_bbox=expected_item.bbox,
            search_bbox=search_bbox,
            reference_bbox=reference_bbox,
            feature_support=feature_support,
            feature_total=feature_total,
        )

    return slots


def _slot_candidate_details(
    expected_item: _ProjectedExpected,
    detection_item: _DetectionCandidate,
    *,
    slot: _ExpectedSlot | None,
    global_debug: dict[str, Any],
) -> tuple[float, float, dict[str, Any]] | None:
    """Return slot score details for same-class YOLO detection.

    A slot candidate exists only when YOLO detection is actually inside the
    expanded expected slot. The exact projected polygon IoU is kept as a debug
    metric only and must not be used as proof that a slot candidate is absent.
    """

    if slot is None:
        return None

    if (
        detection_item.detection.confidence
        < settings.INSPECTION_SLOT_MIN_YOLO_CONFIDENCE
    ):
        return None

    containment = _bbox_containment(detection_item.bbox, slot.search_bbox)
    if containment < settings.INSPECTION_SLOT_MIN_DETECTION_CONTAINMENT:
        return None

    expected_center = _bbox_center(slot.projected_bbox)
    detection_center = _bbox_center(detection_item.bbox)
    search_diag = max(1.0, _bbox_diag(slot.search_bbox))
    center_distance = float(
        np.hypot(
            expected_center[0] - detection_center[0],
            expected_center[1] - detection_center[1],
        )
    )
    center_score = max(0.0, 1.0 - center_distance / max(1.0, search_diag * 0.55))

    expected_area = max(1.0, _bbox_area(slot.projected_bbox))
    detection_area = max(1.0, _bbox_area(detection_item.bbox))
    area_ratio = detection_area / expected_area
    area_score = min(area_ratio, 1.0 / area_ratio)
    area_score = max(0.0, min(1.0, area_score))

    slot_iou = _bbox_iou(slot.search_bbox, detection_item.bbox)
    projected_iou = _polygon_iou(expected_item.polygon, detection_item.polygon)
    feature_score = min(
        1.0,
        slot.feature_support / max(1, settings.INSPECTION_SLOT_MIN_FEATURE_SUPPORT),
    )

    confidence_score = max(0.0, min(1.0, float(detection_item.detection.confidence)))

    score = (
        center_score * 0.30
        + containment * 0.26
        + area_score * 0.14
        + min(1.0, slot_iou * 5.0) * 0.08
        + min(1.0, projected_iou * 3.0) * 0.05
        + feature_score * 0.05
        + confidence_score * 0.12
    )
    passed = score >= settings.INSPECTION_SLOT_MIN_SCORE

    debug = {
        "reason": "slot_matched" if passed else "slot_candidate_rejected",
        "projection": "expected_slot",
        "candidate_source": "yolo_slot",
        "score": _round_debug(score),
        "threshold": _round_debug(settings.INSPECTION_SLOT_MIN_SCORE),
        "passed": passed,
        "reject_reason": None if passed else "slot_score_below_threshold",
        "iou": _round_debug(projected_iou),
        "global_score": global_debug.get("score"),
        "global_iou": global_debug.get("iou"),
        "global_reject_reason": global_debug.get("reject_reason"),
        "slot_center_score": _round_debug(center_score),
        "slot_detection_containment": _round_debug(containment),
        "slot_bbox_iou": _round_debug(slot_iou),
        "slot_area_score": _round_debug(area_score),
        "slot_feature_score": _round_debug(feature_score),
        "slot_yolo_confidence": _round_debug(detection_item.detection.confidence),
        "slot_yolo_min_confidence": _round_debug(
            settings.INSPECTION_SLOT_MIN_YOLO_CONFIDENCE
        ),
        "slot": _slot_debug_payload(slot),
    }

    return float(score), float(projected_iou), debug


def _try_slot_candidate(
    expected_item: _ProjectedExpected,
    detection_item: _DetectionCandidate,
    *,
    slot: _ExpectedSlot | None,
    global_debug: dict[str, Any],
) -> tuple[float, float, dict[str, Any]] | None:
    """Match detection inside an expanded expected slot instead of exact projection.

    The projected polygon is only a navigation hint. A same-class YOLO object that
    lands inside the expected slot should be accepted even when polygon IoU is low,
    which is common with 3D perspective drift and thin/round details.
    """

    details = _slot_candidate_details(
        expected_item,
        detection_item,
        slot=slot,
        global_debug=global_debug,
    )
    if details is None:
        return None

    score, projected_iou, debug = details
    if not debug.get("passed"):
        return None

    return score, projected_iou, debug


def _missing_slot_debug(
    expected_item: _ProjectedExpected,
    slot: _ExpectedSlot | None,
) -> dict[str, Any]:
    if slot is None:
        return {"reason": "slot_no_evidence", "projection": "expected_slot"}

    debug: dict[str, Any] = {
        "reason": "Модель YOLO не обнаружила деталь в ожидаемой области",
        "reason_code": "slot_no_yolo_confirmation",
        "projection": "expected_slot",
        "candidate_source": "none",
        "slot": _slot_debug_payload(slot),
    }

    if slot.feature_support >= settings.INSPECTION_SLOT_MIN_FEATURE_SUPPORT:
        debug.update(
            {
                "reason": "Ожидаемая область найдена, но YOLO не обнаружила деталь",
                "reason_code": "feature_slot_unconfirmed_without_yolo",
                "candidate_source": "feature_support_only",
                "slot_feature_support": slot.feature_support,
                "slot_feature_total": slot.feature_total,
                "slot_feature_min_support": (
                    settings.INSPECTION_SLOT_MIN_FEATURE_SUPPORT
                ),
                "note": (
                    "Reference/frame features support the expected slot, but YOLO "
                    "did not confirm an object of the required class."
                ),
            }
        )

    return debug


def _try_missing_polygon_refinement(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
) -> _MissingPolygonRefinement | None:
    if not settings.INSPECTION_MISSING_POLYGON_REFINEMENT:
        return None
    if slot is None or projection_data is None:
        return None

    min_support = max(3, settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT)
    if slot.feature_support < min_support:
        return None

    local_reference, local_frame = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
    )
    if len(local_reference) < min_support:
        return None

    ransac_threshold = max(
        1.0,
        min(25.0, settings.INSPECTION_MISSING_POLYGON_MAX_REPROJECTION_ERROR),
    )

    try:
        affine, inliers = cv2.estimateAffinePartial2D(
            local_reference,
            local_frame,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=1200,
            confidence=0.98,
            refineIters=10,
        )
    except cv2.error:
        return None

    if affine is None or inliers is None:
        return None
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        return None
    if not _missing_affine_scale_in_range(affine):
        return None

    inlier_mask = np.asarray(inliers, dtype=np.uint8).reshape(-1) > 0
    inlier_count = int(np.count_nonzero(inlier_mask))
    if inlier_count < min_support:
        return None

    median_error = _affine_reprojection_median_error(
        affine,
        source=local_reference[inlier_mask],
        target=local_frame[inlier_mask],
    )
    if median_error is None:
        return None
    if median_error > settings.INSPECTION_MISSING_POLYGON_MAX_REPROJECTION_ERROR:
        return None

    polygon = _project_polygon_by_affine(
        expected_item.item.reference_polygon,
        np.asarray(affine, dtype=np.float32),
    )
    if len(polygon) < 3:
        return None

    bbox = _bbox_from_polygon(polygon)
    if bbox is None:
        return None

    if (
        projection_data.frame_size is not None
        and not _is_visible_in_frame(
            bbox,
            projection_data.frame_size,
            min_visible_fraction=0.05,
        )
    ):
        return None

    containment = _bbox_containment(bbox, slot.search_bbox)
    if containment < _MISSING_POLYGON_MIN_CONTAINMENT:
        return None

    area_score = _bbox_area_similarity(bbox, expected_item.bbox)
    if area_score < _MISSING_POLYGON_MIN_AREA_SCORE:
        return None

    edge_snapped = False
    edge_moved_points = 0
    edge_mean_shift = 0.0
    edge_max_shift = 0.0

    edge_result = _try_snap_missing_polygon_to_edges(
        polygon,
        frame=projection_data.frame,
        frame_size=projection_data.frame_size,
        slot=slot,
        base_bbox=bbox,
    )
    if edge_result is not None:
        edge_containment = _bbox_containment(edge_result.bbox, slot.search_bbox)
        edge_area_score = _bbox_area_similarity(edge_result.bbox, expected_item.bbox)

        polygon = edge_result.polygon
        bbox = edge_result.bbox
        containment = edge_containment
        area_score = edge_area_score
        edge_snapped = True
        edge_moved_points = edge_result.moved_points
        edge_mean_shift = edge_result.mean_shift
        edge_max_shift = edge_result.max_shift

    return _MissingPolygonRefinement(
        polygon=polygon,
        bbox=bbox,
        feature_support=slot.feature_support,
        feature_total=slot.feature_total,
        candidate_count=int(len(local_reference)),
        inlier_count=inlier_count,
        median_error=float(median_error),
        containment=float(containment),
        area_score=float(area_score),
        edge_snapped=edge_snapped,
        edge_moved_points=edge_moved_points,
        edge_mean_shift=float(edge_mean_shift),
        edge_max_shift=float(edge_max_shift),
    )


def _merge_missing_refinement_debug(
    missing_debug: dict[str, Any],
    refinement: _MissingPolygonRefinement,
) -> dict[str, Any]:
    return {
        **missing_debug,
        "missing_polygon_refined": True,
        "missing_polygon_projection": (
            "local_feature_affine_active_contour"
            if refinement.edge_snapped
            else "local_feature_affine"
        ),
        "missing_polygon_bbox": _bbox_debug(refinement.bbox),
        "missing_polygon_feature_support": refinement.feature_support,
        "missing_polygon_feature_total": refinement.feature_total,
        "missing_polygon_candidate_count": refinement.candidate_count,
        "missing_polygon_inliers": refinement.inlier_count,
        "missing_polygon_median_error": _round_debug(refinement.median_error),
        "missing_polygon_containment": _round_debug(refinement.containment),
        "missing_polygon_area_score": _round_debug(refinement.area_score),
        "missing_polygon_edge_snapped": refinement.edge_snapped,
        "missing_polygon_edge_moved_points": refinement.edge_moved_points,
        "missing_polygon_edge_mean_shift": _round_debug(refinement.edge_mean_shift),
        "missing_polygon_edge_max_shift": _round_debug(refinement.edge_max_shift),
        "note": (
            "Missing polygon was refined by local feature matches and an active "
            "edge-aware contour; YOLO still did not confirm an object of the "
            "required class."
        ),
    }


def _missing_polygon_refinement_points(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot,
    projection_data: LocalProjectionData,
) -> tuple[np.ndarray, np.ndarray]:
    reference_points = _as_match_points(projection_data.reference_points)
    frame_points = _as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return _empty_match_points(), _empty_match_points()
    if len(reference_points) != len(frame_points):
        return _empty_match_points(), _empty_match_points()

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return _empty_match_points(), _empty_match_points()

    reference_window = _expand_bbox(
        reference_bbox,
        factor=settings.INSPECTION_SLOT_FEATURE_SEARCH_EXPANSION,
    )

    selected_reference: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []
    selected_inner_reference: list[np.ndarray] = []
    selected_inner_frame: list[np.ndarray] = []
    reference_polygon = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )

    for reference_point, frame_point in zip(reference_points, frame_points, strict=True):
        if not _point_in_bbox(reference_point, reference_window):
            continue
        if not _point_in_bbox(frame_point, slot.search_bbox):
            continue

        selected_reference.append(reference_point)
        selected_frame.append(frame_point)

        if _point_in_np_polygon(reference_point, reference_polygon):
            selected_inner_reference.append(reference_point)
            selected_inner_frame.append(frame_point)

    min_support = max(3, settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT)
    if len(selected_inner_reference) >= min_support:
        selected_reference = selected_inner_reference
        selected_frame = selected_inner_frame

    if not selected_reference:
        return _empty_match_points(), _empty_match_points()

    local_reference = np.asarray(selected_reference, dtype=np.float32)
    local_frame = np.asarray(selected_frame, dtype=np.float32)
    if len(local_reference) <= _MISSING_POLYGON_MAX_LOCAL_POINTS:
        return local_reference, local_frame

    reference_center = np.asarray(_bbox_center(reference_bbox), dtype=np.float32)
    frame_center = np.asarray(_bbox_center(slot.projected_bbox), dtype=np.float32)
    reference_distance = np.linalg.norm(local_reference - reference_center, axis=1)
    frame_distance = np.linalg.norm(local_frame - frame_center, axis=1)
    indices = np.argsort(reference_distance + frame_distance)[
        :_MISSING_POLYGON_MAX_LOCAL_POINTS
    ]
    return local_reference[indices], local_frame[indices]



def _try_snap_missing_polygon_to_edges(
    polygon: list[list[float]],
    *,
    frame: np.ndarray | None,
    frame_size: tuple[int, int] | None,
    slot: _ExpectedSlot,
    base_bbox: BBox,
) -> _EdgeSnapResult | None:
    if not settings.INSPECTION_MISSING_POLYGON_EDGE_REFINEMENT:
        return None
    if frame is None or frame.size == 0:
        return None

    radius = int(settings.INSPECTION_MISSING_POLYGON_EDGE_SNAP_RADIUS)
    if radius <= 0:
        return None

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 3:
        return None
    points = points[:, :2]

    inferred_size = (int(frame.shape[1]), int(frame.shape[0]))
    effective_frame_size = frame_size or inferred_size
    if effective_frame_size[0] <= 0 or effective_frame_size[1] <= 0:
        return None

    dense_points = _densify_closed_polygon(
        points,
        step=float(settings.INSPECTION_MISSING_POLYGON_EDGE_DENSIFY_STEP),
        max_points=_MISSING_POLYGON_EDGE_MAX_DENSE_POINTS,
    )
    if dense_points is None or len(dense_points) < 3:
        return None

    contour_vectors = _closed_contour_vectors(dense_points)
    if contour_vectors is None:
        return None
    tangents, normals = contour_vectors
    normals = _orient_normals_outward(dense_points, normals)

    dense_bbox = _bbox_from_np_points(dense_points)
    if dense_bbox is None:
        return None

    edge_payload = _missing_polygon_edge_payload(
        frame,
        polygon=[[float(x), float(y)] for x, y in dense_points.tolist()],
        bbox=dense_bbox,
        radius=radius,
    )
    if edge_payload is None:
        return None

    edges, gradient_x, gradient_y, crop_origin = edge_payload
    snapped_points, displacements = _snap_points_to_edge_map(
        dense_points,
        tangents=tangents,
        normals=normals,
        edges=edges,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        crop_origin=crop_origin,
        radius=radius,
    )
    if snapped_points is None or displacements is None:
        return None

    smoothed_points, smoothed_displacements = _smooth_edge_displacements(
        base_points=dense_points,
        snapped_points=snapped_points,
        radius=radius,
    )
    if smoothed_points is None or smoothed_displacements is None:
        return None
    if not _edge_refinement_preserves_shape(dense_points, smoothed_points):
        return None

    moved_mask = smoothed_displacements > 0.25
    moved_points = int(np.count_nonzero(moved_mask))
    min_moved_points = max(
        _MISSING_POLYGON_EDGE_MIN_MOVED_POINTS,
        int(round(len(dense_points) * _MISSING_POLYGON_EDGE_MIN_MOVED_FRACTION)),
    )
    if moved_points < min_moved_points:
        return None

    output_points = _simplify_edge_polygon(smoothed_points)
    if output_points is None or len(output_points) < 3:
        return None
    if not _edge_refinement_preserves_shape(dense_points, output_points):
        return None

    polygon_points = [[float(x), float(y)] for x, y in output_points.tolist()]
    if not _polygon_has_usable_area(polygon_points):
        return None

    bbox = _bbox_from_polygon(polygon_points)
    if bbox is None:
        return None
    if not _is_visible_in_frame(
        bbox,
        effective_frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    containment = _bbox_containment(bbox, slot.search_bbox)
    if containment < _MISSING_POLYGON_EDGE_MIN_CONTAINMENT:
        return None

    area_score = _bbox_area_similarity(bbox, base_bbox)
    if area_score < _MISSING_POLYGON_EDGE_MIN_AREA_SCORE:
        return None

    center_drift = _center_distance(base_bbox, bbox)
    max_center_drift = max(6.0, radius * _MISSING_POLYGON_EDGE_MAX_CENTER_DRIFT_FACTOR)
    if center_drift > max_center_drift:
        return None

    moved_displacements = smoothed_displacements[moved_mask]
    return _EdgeSnapResult(
        polygon=polygon_points,
        bbox=bbox,
        moved_points=moved_points,
        mean_shift=float(np.mean(moved_displacements)),
        max_shift=float(np.max(moved_displacements)),
    )


def _missing_polygon_edge_payload(
    frame: np.ndarray,
    *,
    polygon: list[list[float]],
    bbox: BBox,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]] | None:
    x1, y1, x2, y2 = bbox
    padding = radius + _MISSING_POLYGON_EDGE_CROP_PADDING
    frame_height, frame_width = frame.shape[:2]

    crop_x1 = max(0, int(np.floor(x1 - padding)))
    crop_y1 = max(0, int(np.floor(y1 - padding)))
    crop_x2 = min(frame_width, int(np.ceil(x2 + padding)))
    crop_y2 = min(frame_height, int(np.ceil(y2 + padding)))
    if crop_x2 <= crop_x1 + 2 or crop_y2 <= crop_y1 + 2:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.astype(np.uint8, copy=False)

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(blurred))
    low_threshold = int(max(20.0, min(110.0, median * 0.66)))
    high_threshold = int(max(60.0, min(220.0, median * 1.33)))
    if high_threshold <= low_threshold:
        high_threshold = min(255, low_threshold + 45)

    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    if int(np.count_nonzero(edges)) < _MISSING_POLYGON_EDGE_MIN_PIXELS:
        return None

    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    band = np.zeros(edges.shape, dtype=np.uint8)
    relative_polygon = np.asarray(
        [[x - crop_x1, y - crop_y1] for x, y in polygon],
        dtype=np.float32,
    )
    if len(relative_polygon) < 3:
        return None

    cv2.polylines(
        band,
        [np.round(relative_polygon).astype(np.int32)],
        isClosed=True,
        color=255,
        thickness=max(1, radius // 3),
    )
    kernel_size = max(3, radius * 2 + 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    band = cv2.dilate(band, kernel, iterations=1)
    edges = cv2.bitwise_and(edges, band)

    if int(np.count_nonzero(edges)) < _MISSING_POLYGON_EDGE_MIN_PIXELS:
        return None

    return edges, gradient_x, gradient_y, (crop_x1, crop_y1)


def _snap_points_to_edge_map(
    points: np.ndarray,
    *,
    tangents: np.ndarray,
    normals: np.ndarray,
    edges: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    crop_origin: tuple[int, int],
    radius: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    crop_x, crop_y = crop_origin
    height, width = edges.shape[:2]
    blend = float(settings.INSPECTION_MISSING_POLYGON_EDGE_BLEND)
    if blend <= 0:
        return None, None

    snapped = points.copy()
    displacements = np.zeros((len(points),), dtype=np.float32)
    tangent_band = max(1.25, radius * _MISSING_POLYGON_EDGE_TANGENT_BAND_FRACTION)
    max_inward_shift = _edge_max_inward_shift(radius)
    max_outward_shift = max(1.0, float(radius))

    for index, point in enumerate(points):
        local_x = float(point[0] - crop_x)
        local_y = float(point[1] - crop_y)
        if not np.isfinite(local_x) or not np.isfinite(local_y):
            continue

        tangent = tangents[index]
        normal = normals[index]
        if not np.isfinite(tangent).all() or not np.isfinite(normal).all():
            continue

        px = int(round(local_x))
        py = int(round(local_y))
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(width - 1, px + radius)
        y2 = min(height - 1, py + radius)
        if x2 < x1 or y2 < y1:
            continue

        window = edges[y1 : y2 + 1, x1 : x2 + 1]
        ys, xs = np.nonzero(window)
        if len(xs) == 0:
            continue

        edge_x = xs.astype(np.float32) + float(x1)
        edge_y = ys.astype(np.float32) + float(y1)
        offsets = np.stack((edge_x - local_x, edge_y - local_y), axis=1)
        signed_normal_distance = offsets @ normal
        normal_distance = np.abs(signed_normal_distance)
        tangent_distance = np.abs(offsets @ tangent)

        gx = gradient_x[ys + y1, xs + x1].astype(np.float32)
        gy = gradient_y[ys + y1, xs + x1].astype(np.float32)
        gradient_norm = np.hypot(gx, gy)
        gradient_alignment = np.zeros_like(gradient_norm, dtype=np.float32)
        valid_gradient = gradient_norm > 1e-3
        gradient_alignment[valid_gradient] = np.abs(
            (gx[valid_gradient] * normal[0] + gy[valid_gradient] * normal[1])
            / gradient_norm[valid_gradient]
        )

        valid = (
            (signed_normal_distance >= -max_inward_shift)
            & (signed_normal_distance <= max_outward_shift)
            & (normal_distance <= radius)
            & (tangent_distance <= tangent_band)
            & (gradient_alignment >= _MISSING_POLYGON_EDGE_MIN_GRADIENT_ALIGNMENT)
        )
        if not bool(np.any(valid)):
            continue

        inward_penalty = np.maximum(-signed_normal_distance, 0.0)
        scores = (
            normal_distance
            + tangent_distance * 2.35
            + inward_penalty * 1.65
            + (1.0 - gradient_alignment) * radius * 0.80
        )
        scores = np.where(valid, scores, np.inf)
        best_index = int(np.argmin(scores))
        if not np.isfinite(scores[best_index]):
            continue

        target = np.asarray(
            [edge_x[best_index] + crop_x, edge_y[best_index] + crop_y],
            dtype=np.float32,
        )
        delta = (target - points[index]) * blend
        inward_delta = float(delta @ normal)
        if inward_delta < -max_inward_shift:
            delta += normal * (-max_inward_shift - inward_delta)

        snapped[index] = points[index] + delta
        displacements[index] = float(np.linalg.norm(snapped[index] - points[index]))

    if not np.isfinite(snapped).all():
        return None, None

    return snapped, displacements


def _densify_closed_polygon(
    points: np.ndarray,
    *,
    step: float,
    max_points: int,
) -> np.ndarray | None:
    if len(points) < 3:
        return None

    clean_points = np.asarray(points, dtype=np.float32)[:, :2]
    if not np.isfinite(clean_points).all():
        return None

    lengths: list[float] = []
    perimeter = 0.0
    for index, start in enumerate(clean_points):
        end = clean_points[(index + 1) % len(clean_points)]
        length = float(np.linalg.norm(end - start))
        lengths.append(length)
        perimeter += length

    if perimeter <= 1.0:
        return None

    effective_step = max(2.0, float(step))
    if max_points > 0:
        effective_step = max(effective_step, perimeter / max_points)

    dense: list[np.ndarray] = []
    for index, start in enumerate(clean_points):
        end = clean_points[(index + 1) % len(clean_points)]
        length = lengths[index]
        segments = max(1, int(np.ceil(length / effective_step)))
        for segment_index in range(segments):
            ratio = float(segment_index) / float(segments)
            dense.append(start + (end - start) * ratio)

    if len(dense) < 3:
        return None

    return np.asarray(dense, dtype=np.float32)


def _closed_contour_vectors(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None

    previous_points = np.roll(points, shift=1, axis=0)
    next_points = np.roll(points, shift=-1, axis=0)
    tangents = next_points - previous_points
    tangent_norms = np.linalg.norm(tangents, axis=1)
    valid = tangent_norms > 1e-6
    if not bool(np.any(valid)):
        return None

    tangents[valid] = tangents[valid] / tangent_norms[valid, None]
    tangents[~valid] = np.asarray([1.0, 0.0], dtype=np.float32)
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1).astype(np.float32)
    return tangents.astype(np.float32), normals


def _orient_normals_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if len(points) != len(normals) or len(points) == 0:
        return normals.astype(np.float32)

    center = np.mean(points, axis=0)
    oriented = normals.astype(np.float32, copy=True)
    radial = points - center
    dot = np.sum(oriented * radial, axis=1)
    flip = dot < 0
    oriented[flip] *= -1.0
    return oriented


def _edge_max_inward_shift(radius: int) -> float:
    fraction = float(settings.INSPECTION_MISSING_POLYGON_EDGE_MAX_INWARD_SHIFT_FRACTION)
    return max(
        _MISSING_POLYGON_EDGE_MIN_INWARD_SHIFT,
        min(float(radius), float(radius) * fraction),
    )


def _edge_refinement_preserves_shape(
    base_points: np.ndarray,
    refined_points: np.ndarray,
) -> bool:
    if len(base_points) < 3 or len(refined_points) < 3:
        return False
    if not np.isfinite(base_points).all() or not np.isfinite(refined_points).all():
        return False

    min_width_ratio = float(settings.INSPECTION_MISSING_POLYGON_EDGE_MIN_WIDTH_RATIO)

    base_width = _oriented_min_side(base_points)
    refined_width = _oriented_min_side(refined_points)
    if base_width is not None and refined_width is not None and base_width > 2.0:
        if refined_width / base_width < min_width_ratio:
            return False

    base_area = _contour_area(base_points)
    refined_area = _contour_area(refined_points)
    if base_area is not None and refined_area is not None and base_area > 2.0:
        min_area_ratio = max(
            0.20,
            min_width_ratio * _MISSING_POLYGON_EDGE_AREA_RATIO_FACTOR,
        )
        if refined_area / base_area < min_area_ratio:
            return False

    return True


def _oriented_min_side(points: np.ndarray) -> float | None:
    if len(points) < 3:
        return None

    try:
        _center, size, _angle = cv2.minAreaRect(points.astype(np.float32))
    except cv2.error:
        return None

    width, height = float(size[0]), float(size[1])
    if width <= 0.0 or height <= 0.0:
        return None
    return min(width, height)


def _contour_area(points: np.ndarray) -> float | None:
    if len(points) < 3:
        return None

    try:
        area = float(
            abs(cv2.contourArea(points.reshape(-1, 1, 2).astype(np.float32)))
        )
    except cv2.error:
        return None

    if area <= 0.0:
        return None
    return area


def _smooth_edge_displacements(
    *,
    base_points: np.ndarray,
    snapped_points: np.ndarray,
    radius: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(base_points) != len(snapped_points) or len(base_points) < 3:
        return None, None

    delta = snapped_points - base_points
    if not np.isfinite(delta).all():
        return None, None

    smoothing = float(settings.INSPECTION_MISSING_POLYGON_EDGE_SMOOTHING)
    iterations = int(settings.INSPECTION_MISSING_POLYGON_EDGE_SMOOTH_ITERATIONS)
    if smoothing > 0 and iterations > 0:
        smoothing = min(0.45, max(0.0, smoothing))
        for _ in range(iterations):
            neighbor_delta = (np.roll(delta, 1, axis=0) + np.roll(delta, -1, axis=0)) * 0.5
            delta = delta * (1.0 - smoothing) + neighbor_delta * smoothing

    max_shift = max(1.0, float(radius))
    lengths = np.linalg.norm(delta, axis=1)
    too_far = lengths > max_shift
    if bool(np.any(too_far)):
        delta[too_far] *= (max_shift / lengths[too_far])[:, None]

    points = base_points + delta
    if not np.isfinite(points).all():
        return None, None

    displacements = np.linalg.norm(points - base_points, axis=1).astype(np.float32)
    return points.astype(np.float32), displacements


def _simplify_edge_polygon(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None

    contour = points.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(
        contour,
        epsilon=_MISSING_POLYGON_EDGE_SIMPLIFY_EPSILON,
        closed=True,
    ).reshape(-1, 2)

    if len(simplified) < 3:
        return points.astype(np.float32)

    return simplified.astype(np.float32)


def _bbox_from_np_points(points: np.ndarray) -> BBox | None:
    if len(points) == 0 or not np.isfinite(points).all():
        return None

    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)


def _polygon_has_usable_area(polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    try:
        shape = make_valid(Polygon(polygon))
    except (ValueError, GEOSException):
        return False
    return not shape.is_empty and float(shape.area) > 1.0


def _center_distance(a: BBox, b: BBox) -> float:
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    return float(np.hypot(acx - bcx, acy - bcy))

def _project_polygon_by_affine(
    polygon: list[list[float]],
    affine: np.ndarray,
) -> list[list[float]]:
    if len(polygon) < 3:
        return []

    source = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    try:
        projected = cv2.transform(source, affine.astype(np.float32)).reshape(-1, 2)
    except cv2.error:
        return []

    if not np.isfinite(projected).all():
        return []

    return [[float(x), float(y)] for x, y in projected.tolist()]


def _affine_reprojection_median_error(
    affine: np.ndarray,
    *,
    source: np.ndarray,
    target: np.ndarray,
) -> float | None:
    if len(source) == 0 or len(target) == 0:
        return None

    try:
        projected = cv2.transform(
            source.reshape(-1, 1, 2),
            affine.astype(np.float32),
        ).reshape(-1, 2)
    except cv2.error:
        return None

    if not np.isfinite(projected).all():
        return None

    errors = np.linalg.norm(projected - target, axis=1)
    if len(errors) == 0 or not np.isfinite(errors).all():
        return None

    return float(np.median(errors))


def _missing_affine_scale_in_range(affine: np.ndarray) -> bool:
    transform = np.asarray(affine, dtype=np.float32)
    if transform.shape != (2, 3):
        return False

    scale_x = float(np.linalg.norm(transform[:, 0]))
    scale_y = float(np.linalg.norm(transform[:, 1]))
    return (
        _MISSING_POLYGON_MIN_AFFINE_SCALE
        <= scale_x
        <= _MISSING_POLYGON_MAX_AFFINE_SCALE
        and _MISSING_POLYGON_MIN_AFFINE_SCALE
        <= scale_y
        <= _MISSING_POLYGON_MAX_AFFINE_SCALE
    )


def _bbox_area_similarity(a: BBox, b: BBox) -> float:
    area_a = max(1.0, _bbox_area(a))
    area_b = max(1.0, _bbox_area(b))
    ratio = area_a / area_b
    return float(max(0.0, min(1.0, min(ratio, 1.0 / ratio))))


def _empty_match_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _slot_feature_support(
    expected_item: _ProjectedExpected,
    *,
    search_bbox: BBox,
    projection_data: LocalProjectionData | None,
) -> tuple[int, int]:
    if projection_data is None:
        return 0, 0

    reference_points = _as_match_points(projection_data.reference_points)
    frame_points = _as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return 0, 0
    if len(reference_points) != len(frame_points):
        return 0, 0

    reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return 0, 0

    reference_window = _expand_bbox(
        reference_bbox,
        factor=settings.INSPECTION_SLOT_FEATURE_SEARCH_EXPANSION,
    )

    total = 0
    support = 0
    for reference_point, frame_point in zip(reference_points, frame_points, strict=True):
        if not _point_in_bbox(reference_point, reference_window):
            continue
        total += 1
        if _point_in_bbox(frame_point, search_bbox):
            support += 1

    return support, total


def _slot_debug_payload(slot: _ExpectedSlot | None) -> dict[str, Any] | None:
    if slot is None:
        return None
    return {
        "projected_bbox": _bbox_debug(slot.projected_bbox),
        "search_bbox": _bbox_debug(slot.search_bbox),
        "reference_bbox": _bbox_debug(slot.reference_bbox),
        "feature_support": slot.feature_support,
        "feature_total": slot.feature_total,
        "search_expansion": settings.INSPECTION_SLOT_SEARCH_EXPANSION,
        "feature_search_expansion": settings.INSPECTION_SLOT_FEATURE_SEARCH_EXPANSION,
    }


def _bbox_debug(bbox: BBox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "x": _round_debug(x1),
        "y": _round_debug(y1),
        "w": _round_debug(max(0.0, x2 - x1)),
        "h": _round_debug(max(0.0, y2 - y1)),
    }


def _as_match_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 3 and array.shape[1:] == (1, 2):
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    if len(array) == 0:
        return None
    return array[:, :2]


def _expand_bbox(
    bbox: BBox,
    *,
    factor: float,
    frame_size: tuple[int, int] | None = None,
) -> BBox:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx, cy = _bbox_center(bbox)
    new_w = width * factor
    new_h = height * factor
    result = (
        cx - new_w * 0.5,
        cy - new_h * 0.5,
        cx + new_w * 0.5,
        cy + new_h * 0.5,
    )

    if frame_size is None:
        return result

    fw, fh = frame_size
    return (
        max(0.0, result[0]),
        max(0.0, result[1]),
        min(float(fw), result[2]),
        min(float(fh), result[3]),
    )


def _bbox_containment(inner: BBox, outer: BBox) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    inner_area = _bbox_area(inner)
    if inner_area <= 0:
        return 0.0
    return float(inter_area / inner_area)


def _point_in_bbox(point: np.ndarray, bbox: BBox) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _point_in_np_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
        return False
    if not np.isfinite(point).all() or not np.isfinite(polygon).all():
        return False

    try:
        return (
            cv2.pointPolygonTest(
                polygon[:, :2].astype(np.float32),
                (float(point[0]), float(point[1])),
                False,
            )
            >= 0
        )
    except cv2.error:
        return False


def _display_detected_polygon(
    detection_item: _DetectionCandidate,
) -> list[list[float]] | None:
    return detection_item.polygon

def _match_score_details(
    expected_polygon: list[list[float]],
    detection_polygon: list[list[float]] | None,
    expected_bbox: BBox,
    detection_bbox: BBox,
) -> dict[str, Any]:
    bbox_iou = _bbox_iou(expected_bbox, detection_bbox)

    if detection_polygon is None or len(detection_polygon) < 3:
        polygon_iou = 0.0
    else:
        polygon_iou = _polygon_iou(expected_polygon, detection_polygon)

    iou = max(bbox_iou, polygon_iou)
    threshold = _match_threshold(expected_bbox)
    debug: dict[str, Any] = {
        "bbox_iou": _round_debug(bbox_iou),
        "polygon_iou": _round_debug(polygon_iou),
        "iou": _round_debug(iou),
        "score": 0.0,
        "threshold": _round_debug(threshold),
        "passed": False,
    }

    if iou < IOU_MATCH_THRESHOLD:
        debug["reject_reason"] = "iou_below_min"
        return debug

    expected_area = _bbox_area(expected_bbox)
    detection_area = _bbox_area(detection_bbox)
    if expected_area <= 0 or detection_area <= 0:
        debug["reject_reason"] = "empty_bbox"
        return debug

    area_ratio = detection_area / expected_area
    debug["area_ratio"] = _round_debug(area_ratio)
    if area_ratio < _MIN_AREA_RATIO or area_ratio > _MAX_AREA_RATIO:
        debug["reject_reason"] = "area_ratio_out_of_range"
        return debug

    expected_center = _bbox_center(expected_bbox)
    detection_center = _bbox_center(detection_bbox)
    center_distance = float(
        np.hypot(
            expected_center[0] - detection_center[0],
            expected_center[1] - detection_center[1],
        )
    )
    expected_diag = max(1.0, _bbox_diag(expected_bbox))
    center_limit = expected_diag * _MAX_CENTER_DISTANCE_FACTOR
    center_score = max(0.0, 1.0 - center_distance / expected_diag)
    debug["center_distance"] = _round_debug(center_distance)
    debug["center_limit"] = _round_debug(center_limit)

    if center_distance > center_limit:
        debug["reject_reason"] = "center_too_far"
        return debug

    area_score = min(area_ratio, 1.0 / area_ratio)
    score = (
        polygon_iou * 0.50
        + bbox_iou * 0.25
        + center_score * 0.20
        + area_score * 0.05
    )
    debug["score"] = _round_debug(score)

    if score < threshold:
        debug["reject_reason"] = "score_below_threshold"
        return debug

    debug["passed"] = True
    debug["reject_reason"] = None
    return debug


def _round_debug(value: float) -> float:
    return round(float(value), 4)


def _match_threshold(expected_bbox: BBox) -> float:
    area = _bbox_area(expected_bbox)
    if area <= 32 * 32:
        return min(_MIN_MATCH_SCORE, 0.34)
    if area >= 180 * 180:
        return max(_MIN_MATCH_SCORE, 0.46)
    return _MIN_MATCH_SCORE


def _choose_candidate_pairs(
    candidate_pairs: list[tuple[float, float, int, int]],
) -> list[tuple[int, int, float]]:
    if not candidate_pairs:
        return []

    expected_indices = sorted({item[2] for item in candidate_pairs})
    detection_indices = sorted({item[3] for item in candidate_pairs})

    if (
        len(candidate_pairs) > _MAX_OPTIMAL_ASSIGNMENT_CANDIDATES
        or len(expected_indices) > _MAX_OPTIMAL_ASSIGNMENT_ITEMS
        or len(detection_indices) > _MAX_OPTIMAL_ASSIGNMENT_ITEMS
    ):
        return _choose_candidate_pairs_greedy(candidate_pairs)

    expected_pos = {value: index for index, value in enumerate(expected_indices)}
    detection_pos = {value: index for index, value in enumerate(detection_indices)}

    by_expected: list[list[tuple[int, float, float, int]]] = [
        [] for _ in expected_indices
    ]
    for score, iou, expected_index, detection_index in candidate_pairs:
        by_expected[expected_pos[expected_index]].append(
            (detection_pos[detection_index], score, iou, detection_index)
        )

    states: dict[int, tuple[float, list[tuple[int, int, float]]]] = {0: (0.0, [])}
    for expected_index, options in zip(expected_indices, by_expected, strict=True):
        next_states = dict(states)

        for used_mask, (total_score, chosen) in states.items():
            for detection_bit, score, iou, detection_index in options:
                bit = 1 << detection_bit
                if used_mask & bit:
                    continue

                next_mask = used_mask | bit
                candidate_score = total_score + score
                candidate_chosen = [*chosen, (expected_index, detection_index, iou)]
                previous = next_states.get(next_mask)

                if previous is None or _assignment_better(
                    candidate_score,
                    candidate_chosen,
                    previous[0],
                    previous[1],
                ):
                    next_states[next_mask] = (candidate_score, candidate_chosen)

        states = next_states

    return max(
        (value[1] for value in states.values()),
        key=lambda chosen: (
            len(chosen),
            sum(iou for _, _, iou in chosen),
        ),
    )


def _assignment_better(
    candidate_score: float,
    candidate_chosen: list[tuple[int, int, float]],
    previous_score: float,
    previous_chosen: list[tuple[int, int, float]],
) -> bool:
    if len(candidate_chosen) != len(previous_chosen):
        return len(candidate_chosen) > len(previous_chosen)
    return candidate_score > previous_score


def _choose_candidate_pairs_greedy(
    candidate_pairs: list[tuple[float, float, int, int]],
) -> list[tuple[int, int, float]]:
    candidate_pairs = sorted(candidate_pairs, key=lambda item: item[0], reverse=True)
    matched_expected_indices: set[int] = set()
    occupied_detection_indices: set[int] = set()
    chosen_pairs: list[tuple[int, int, float]] = []

    for _score, iou, expected_index, detection_index in candidate_pairs:
        if expected_index in matched_expected_indices:
            continue
        if detection_index in matched_detection_indices:
            continue

        matched_expected_indices.add(expected_index)
        matched_detection_indices.add(detection_index)
        chosen_pairs.append((expected_index, detection_index, iou))

    return chosen_pairs


def _is_visible_in_frame(
    bbox: BBox,
    frame_size: tuple[int, int],
    *,
    min_visible_fraction: float = 0.20,
) -> bool:
    x1, y1, x2, y2 = bbox
    fw, fh = frame_size

    visible_x1 = max(0.0, x1)
    visible_y1 = max(0.0, y1)
    visible_x2 = min(float(fw), x2)
    visible_y2 = min(float(fh), y2)

    if visible_x2 <= visible_x1 or visible_y2 <= visible_y1:
        return False

    visible_area = (visible_x2 - visible_x1) * (visible_y2 - visible_y1)
    full_area = (x2 - x1) * (y2 - y1)
    if full_area <= 0:
        return False

    return visible_area / full_area >= min_visible_fraction


def _classify_unmatched_detection(
    detection: _DetectionCandidate,
    projected_expected: list[_ProjectedExpected],
    occupied_expected_indices: set[int],
    *,
    slot_by_index: dict[int, _ExpectedSlot],
    occupied_detection_indices: set[int],
    detection_by_index: dict[int, _DetectionCandidate],
    iou_threshold: float = 0.10,
) -> tuple[str, int | None, dict[str, Any] | None]:
    if detection.detection.confidence < settings.YOLO_EXTRA_CONF_THRESHOLD:
        return (
            "discard",
            None,
            {
                "reason": "confidence_below_extra_threshold",
                "confidence": _round_debug(detection.detection.confidence),
                "threshold": _round_debug(settings.YOLO_EXTRA_CONF_THRESHOLD),
            },
        )

    duplicate_debug = _occupied_detection_duplicate_debug(
        detection,
        occupied_detection_indices=occupied_detection_indices,
        detection_by_index=detection_by_index,
    )
    if duplicate_debug is not None:
        return "discard", None, duplicate_debug

    best_same_class: tuple[float, float, _ProjectedExpected, dict[str, Any]] | None = (
        None
    )

    for expected_item in projected_expected:
        if expected_item.index in occupied_expected_indices:
            continue
        if expected_item.item.class_key != detection.detection.class_key:
            continue

        global_debug = _match_score_details(
            expected_item.polygon,
            detection.polygon,
            expected_item.bbox,
            detection.bbox,
        )
        slot_details = _slot_candidate_details(
            expected_item,
            detection,
            slot=slot_by_index.get(expected_item.index),
            global_debug=global_debug,
        )
        if slot_details is None:
            continue

        score, _projected_iou, debug = slot_details
        containment = float(debug.get("slot_detection_containment") or 0.0)
        candidate = (score, containment, expected_item, debug)

        if best_same_class is None or (score, containment) > (
            best_same_class[0],
            best_same_class[1],
        ):
            best_same_class = candidate

    if best_same_class is not None:
        _score, _containment, expected_item, debug = best_same_class
        debug = {
            **debug,
            "reason": "same_class_slot_detection_not_matched",
            "expected_name": expected_item.item.name,
        }
        # The exact projected polygon IoU may legitimately be zero after
        # perspective drift. Do not expose it as a panel-level match percentage
        # for an unmatched slot result.
        debug.pop("iou", None)
        return "unmatched", expected_item.index, debug

    for expected_item in projected_expected:
        iou = _bbox_iou(expected_item.bbox, detection.bbox)

        if iou < iou_threshold:
            continue

        return (
            "discard",
            expected_item.index,
            {
                "reason": "inside_expected_zone",
                "bbox_iou": _round_debug(iou),
                "expected_name": expected_item.item.name,
            },
        )

    return (
        "extra",
        None,
        {
            "reason": "no_unmatched_expected_slot_of_same_class",
            "confidence": _round_debug(detection.detection.confidence),
        },
    )


def _occupied_detection_duplicate_debug(
    detection: _DetectionCandidate,
    *,
    occupied_detection_indices: set[int],
    detection_by_index: dict[int, _DetectionCandidate],
) -> dict[str, Any] | None:
    for occupied_index in occupied_detection_indices:
        occupied_detection = detection_by_index.get(occupied_index)
        if occupied_detection is None:
            continue

        bbox_iou = _bbox_iou(detection.bbox, occupied_detection.bbox)
        containment = max(
            _bbox_containment(detection.bbox, occupied_detection.bbox),
            _bbox_containment(occupied_detection.bbox, detection.bbox),
        )

        if (
            bbox_iou < _DUPLICATE_MATCHED_BBOX_IOU
            and containment < _DUPLICATE_MATCHED_CONTAINMENT
        ):
            continue

        return {
            "reason": "duplicate_of_accepted_detection",
            "accepted_detection_index": occupied_index,
            "accepted_class_key": occupied_detection.detection.class_key,
            "bbox_iou": _round_debug(bbox_iou),
            "containment": _round_debug(containment),
        }

    return None
