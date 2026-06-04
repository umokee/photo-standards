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
    inlier_ratio: float
    median_error: float
    reference_spread: float
    frame_spread: float
    quadrant_count: int
    center_drift_factor: float
    containment: float
    area_score: float
    edge_snapped: bool = False
    edge_moved_points: int = 0
    edge_mean_shift: float = 0.0
    edge_max_shift: float = 0.0


@dataclass(slots=True)
class _MissingFallbackProjection:
    polygon: list[list[float]] | None
    bbox: BBox | None
    hidden: bool
    debug: dict[str, Any]


@dataclass(slots=True)
class _MissingTranslationRescue:
    polygon: list[list[float]]
    bbox: BBox
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    shift_x: float
    shift_y: float
    shift_factor: float
    source_counts: dict[str, int] | None = None
    single_anchor: bool = False
    anchor_dispersion: float = 0.0
    anchor_local_inlier_ratio: float = 0.0


@dataclass(slots=True)
class _TrustedAnchor:
    expected_index: int
    source: str
    residual: np.ndarray
    global_bbox: BBox
    local_bbox: BBox
    weight: float
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    shift_factor: float


@dataclass(slots=True)
class _AnchorReleaseConsensus:
    shift: np.ndarray
    inlier_mask: np.ndarray
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    dispersion: float


@dataclass(slots=True)
class _AnchorOverlapDecision:
    allowed: bool
    reason: str | None
    overlap_count: int
    unresolved_count: int
    strong_anchor_count: int
    max_overlap: float
    max_resolved_overlap: float


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
    frame_size: tuple[int, int] | None = None,
    projection_data: LocalProjectionData | None = None,
    hide_unconfirmed_projection: bool = False,
) -> list[SegmentMatch]:
    matches: list[SegmentMatch] = []

    for item in expected:
        expected_polygon = None

        debug: dict[str, Any] = {"reason": "no_matching_detection"}

        if hide_unconfirmed_projection:
            debug = {
                "reason": "projection_hidden_as_unsafe",
                "reason_code": "missing_projection_low_confidence",
                "missing_polygon_projection": "unsafe_hidden",
                "missing_polygon_projection_safety": "unsafe_hidden",
            }
        elif homography is not None:
            projected = _safe_project(
                item.reference_polygon,
                homography,
                projection_data=projection_data,
            )
            if len(projected) >= 3:
                bbox = _bbox_from_polygon(projected)
                effective_frame_size = frame_size
                if effective_frame_size is None and projection_data is not None:
                    effective_frame_size = projection_data.frame_size
                visible = (
                    bbox is not None
                    and (
                        effective_frame_size is None
                        or _is_visible_in_frame(
                            bbox,
                            effective_frame_size,
                            min_visible_fraction=0.05,
                        )
                    )
                )
                if visible:
                    expected_polygon = projected
                else:
                    debug = {
                        "reason": "projection_hidden_as_unsafe",
                        "reason_code": "missing_projection_low_confidence",
                        "missing_polygon_projection": "unsafe_hidden",
                        "missing_polygon_projection_safety": "unsafe_hidden",
                    }

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
                debug=debug,
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
            frame_size=frame_size,
            projection_data=projection_data,
            hide_unconfirmed_projection=True,
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
    trusted_anchors = _build_trusted_missing_anchors(
        projected_expected,
        slot_by_index=slot_by_index,
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
            all_expected=projected_expected,
        )
        if missing_refinement is not None:
            translation_override = _try_missing_refinement_translation_override(
                expected_item,
                refinement=missing_refinement,
                slot=slot,
                projection_data=projection_data,
                all_expected=projected_expected,
            )
            if translation_override is not None:
                rescue, projection_name = translation_override
                missing_polygon = rescue.polygon
                missing_debug = _merge_missing_translation_rescue_debug(
                    missing_debug,
                    rescue,
                    source_projection=projection_name,
                    replaced_projection=(
                        "context_feature_affine_edge_guarded"
                        if missing_refinement.edge_snapped
                        else "context_feature_affine"
                    ),
                    replaced_median_error=missing_refinement.median_error,
                    replaced_inlier_ratio=missing_refinement.inlier_ratio,
                    replaced_area_score=missing_refinement.area_score,
                )
            else:
                missing_polygon = missing_refinement.polygon
                missing_debug = _merge_missing_refinement_debug(
                    missing_debug,
                    missing_refinement,
                )
        else:
            fallback = _resolve_missing_fallback_projection(
                expected_item,
                slot=slot,
                projection_data=projection_data,
                all_expected=projected_expected,
                all_slots=slot_by_index,
                trusted_anchors=trusted_anchors,
            )
            missing_polygon = fallback.polygon
            if fallback.debug:
                missing_debug = {**missing_debug, **fallback.debug}
            if fallback.hidden:
                missing_debug = _merge_unsafe_missing_projection_debug(missing_debug)

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
            all_expected=projected_expected,
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
                "candidate_source": "context_feature_support_only",
                "slot_feature_source": "context_ring",
                "slot_feature_support": slot.feature_support,
                "slot_feature_total": slot.feature_total,
                "slot_feature_min_support": (
                    settings.INSPECTION_SLOT_MIN_FEATURE_SUPPORT
                ),
                "note": (
                    "Surrounding context features support the expected slot, but "
                    "YOLO did not confirm an object of the required class."
                ),
            }
        )

    return debug


def _try_missing_polygon_refinement(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[_ProjectedExpected] | None = None,
) -> _MissingPolygonRefinement | None:
    if not settings.INSPECTION_MISSING_POLYGON_REFINEMENT:
        return None
    if slot is None or projection_data is None:
        return None

    min_support = max(3, settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT)
    if slot.feature_support < min_support:
        return None

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return None

    local_reference, local_frame = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
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

    candidate_count = int(len(local_reference))
    inlier_ratio = inlier_count / max(1, candidate_count)
    if inlier_ratio < settings.INSPECTION_MISSING_POLYGON_CONTEXT_MIN_INLIER_RATIO:
        return None
    is_multi_object_context = all_expected is not None and len(all_expected) > 1
    if is_multi_object_context and inlier_ratio < 0.50:
        return None

    inlier_reference = local_reference[inlier_mask]
    inlier_frame = local_frame[inlier_mask]
    reference_spread = _missing_context_spread_score(inlier_reference, reference_bbox)
    frame_spread = _missing_context_spread_score(inlier_frame, slot.projected_bbox)
    min_spread = settings.INSPECTION_MISSING_POLYGON_CONTEXT_MIN_SPREAD
    if min(reference_spread, frame_spread) < min_spread:
        return None

    quadrant_count = _missing_context_quadrant_count(inlier_reference, reference_bbox)
    min_quadrants = settings.INSPECTION_MISSING_POLYGON_CONTEXT_MIN_QUADRANTS
    if is_multi_object_context:
        min_quadrants = max(min_quadrants, 3)
    if quadrant_count < min_quadrants:
        return None

    median_error = _affine_reprojection_median_error(
        affine,
        source=inlier_reference,
        target=inlier_frame,
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
    min_area_score = max(
        _MISSING_POLYGON_MIN_AREA_SCORE,
        settings.INSPECTION_MISSING_POLYGON_CONTEXT_MIN_AREA_SCORE,
    )
    if area_score < min_area_score:
        return None

    center_drift_factor = _bbox_center_distance_factor(bbox, expected_item.bbox)
    if (
        center_drift_factor
        > settings.INSPECTION_MISSING_POLYGON_CONTEXT_MAX_CENTER_DRIFT_FACTOR
    ):
        return None

    if is_multi_object_context:
        global_projection = _missing_global_projection(
            expected_item,
            projection_data=projection_data,
        )
        if global_projection is None:
            return None
        _, global_bbox = global_projection
        global_area_score = _bbox_area_similarity(bbox, global_bbox)
        global_center_factor = _bbox_center_distance_factor(bbox, global_bbox)
        if global_area_score < 0.42 or global_center_factor > 0.58:
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
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        reference_spread=float(reference_spread),
        frame_spread=float(frame_spread),
        quadrant_count=int(quadrant_count),
        center_drift_factor=float(center_drift_factor),
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
            "context_feature_affine_edge_guarded"
            if refinement.edge_snapped
            else "context_feature_affine"
        ),
        "missing_polygon_bbox": _bbox_debug(refinement.bbox),
        "missing_polygon_feature_support": refinement.feature_support,
        "missing_polygon_feature_total": refinement.feature_total,
        "missing_polygon_candidate_count": refinement.candidate_count,
        "missing_polygon_inliers": refinement.inlier_count,
        "missing_polygon_inlier_ratio": _round_debug(refinement.inlier_ratio),
        "missing_polygon_median_error": _round_debug(refinement.median_error),
        "missing_polygon_reference_spread": _round_debug(refinement.reference_spread),
        "missing_polygon_frame_spread": _round_debug(refinement.frame_spread),
        "missing_polygon_context_quadrants": refinement.quadrant_count,
        "missing_polygon_center_drift_factor": _round_debug(
            refinement.center_drift_factor
        ),
        "missing_polygon_containment": _round_debug(refinement.containment),
        "missing_polygon_area_score": _round_debug(refinement.area_score),
        "missing_polygon_edge_snapped": refinement.edge_snapped,
        "missing_polygon_edge_moved_points": refinement.edge_moved_points,
        "missing_polygon_edge_mean_shift": _round_debug(refinement.edge_mean_shift),
        "missing_polygon_edge_max_shift": _round_debug(refinement.edge_max_shift),
        "note": (
            "Missing polygon was positioned from surrounding context features; "
            "YOLO still did not confirm an object of the required class."
        ),
    }



def _try_missing_refinement_translation_override(
    expected_item: _ProjectedExpected,
    *,
    refinement: _MissingPolygonRefinement,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[_ProjectedExpected] | None = None,
) -> tuple[_MissingTranslationRescue, str] | None:
    if slot is None or projection_data is None:
        return None
    if not _missing_refinement_should_try_translation_override(refinement):
        return None

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return None

    global_polygon, global_bbox = global_projection
    rescue = _try_missing_translation_rescue(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
    )
    projection_name = "context_feature_affine_translation_rescue"

    if rescue is None:
        rescue = _try_missing_scene_translation_rescue(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            global_polygon=global_polygon,
            global_bbox=global_bbox,
            all_expected=all_expected,
        )
        projection_name = "context_feature_affine_scene_translation_rescue"

    if rescue is None:
        return None
    if not _missing_translation_override_is_safer(
        refinement,
        rescue,
        expected_item=expected_item,
        all_expected=all_expected,
    ):
        return None

    return rescue, projection_name


def _missing_refinement_should_try_translation_override(
    refinement: _MissingPolygonRefinement,
) -> bool:
    # A local affine is allowed to rotate/scale/shear the expected polygon.  It is
    # useful when the surrounding geometry is strong, but on thin or hook-like
    # details a noisy affine often gives a safe-looking polygon with poor IoU.
    # In these weak cases we try a translation-only rescue that preserves the
    # global shape and only corrects the scene shift.
    if refinement.edge_snapped:
        return False
    if refinement.candidate_count <= 0:
        return False

    weak_ratio = refinement.inlier_ratio < 0.58
    noisy_error = refinement.median_error > 5.75
    weak_area = refinement.area_score < 0.78
    wide_drift = refinement.center_drift_factor > 0.30
    weak_spread = min(refinement.reference_spread, refinement.frame_spread) < 0.24
    weak_quadrants = refinement.quadrant_count < 3

    return any(
        (
            weak_ratio,
            noisy_error,
            weak_area,
            wide_drift,
            weak_spread,
            weak_quadrants,
        )
    )


def _missing_translation_override_is_safer(
    refinement: _MissingPolygonRefinement,
    rescue: _MissingTranslationRescue,
    *,
    expected_item: _ProjectedExpected,
    all_expected: list[_ProjectedExpected] | None,
) -> bool:
    rescue_area_score = _bbox_area_similarity(rescue.bbox, expected_item.bbox)
    if rescue_area_score < 0.32:
        return False
    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=rescue.bbox,
        all_expected=all_expected,
    ):
        return False

    # Do not replace a clearly stable affine.  Otherwise prefer translation when
    # it is at least as coherent by residuals or when the affine has clear signs
    # of deformation.  This keeps Dangerous=0 behaviour while reducing visually
    # poor affine warps.
    affine_is_deformed = (
        refinement.area_score < 0.72
        or refinement.center_drift_factor > 0.42
        or refinement.inlier_ratio < 0.48
    )
    rescue_error_better = rescue.median_error <= max(
        3.25,
        refinement.median_error * 0.82,
    )
    rescue_ratio_better = rescue.inlier_ratio >= refinement.inlier_ratio + 0.08

    return affine_is_deformed or rescue_error_better or rescue_ratio_better


def _merge_missing_translation_rescue_debug(
    missing_debug: dict[str, Any],
    rescue: _MissingTranslationRescue,
    *,
    source_projection: str,
    replaced_projection: str,
    replaced_median_error: float,
    replaced_inlier_ratio: float,
    replaced_area_score: float,
) -> dict[str, Any]:
    return {
        **missing_debug,
        "missing_polygon_refined": True,
        "missing_polygon_projection": source_projection,
        "missing_polygon_projection_safety": "translation_override",
        "missing_polygon_bbox": _bbox_debug(rescue.bbox),
        "missing_polygon_candidate_count": rescue.candidate_count,
        "missing_polygon_inliers": rescue.inlier_count,
        "missing_polygon_inlier_ratio": _round_debug(rescue.inlier_ratio),
        "missing_polygon_median_error": _round_debug(rescue.median_error),
        "missing_polygon_translation_shift_x": _round_debug(rescue.shift_x),
        "missing_polygon_translation_shift_y": _round_debug(rescue.shift_y),
        "missing_polygon_translation_shift_factor": _round_debug(
            rescue.shift_factor
        ),
        "missing_polygon_replaced_projection": replaced_projection,
        "missing_polygon_replaced_median_error": _round_debug(
            replaced_median_error
        ),
        "missing_polygon_replaced_inlier_ratio": _round_debug(
            replaced_inlier_ratio
        ),
        "missing_polygon_replaced_area_score": _round_debug(
            replaced_area_score
        ),
        "note": (
            "Local affine was safe but geometrically weak, so the expected "
            "missing zone was rendered with a conservative translation-only "
            "rescue. YOLO still did not confirm an object of the required class."
        ),
    }

def _merge_unsafe_missing_projection_debug(
    missing_debug: dict[str, Any],
) -> dict[str, Any]:
    return {
        **missing_debug,
        "missing_polygon_refined": False,
        "missing_polygon_projection": "unsafe_hidden",
        "missing_polygon_projection_safety": "unsafe_hidden",
        "reason_code": "missing_projection_low_confidence",
        "note": (
            "Expected missing zone was not rendered because surrounding context "
            "did not produce a safe local transform. YOLO still did not confirm "
            "an object of the required class."
        ),
    }


def _resolve_missing_fallback_projection(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[_ProjectedExpected] | None = None,
    all_slots: dict[int, _ExpectedSlot] | None = None,
    trusted_anchors: list[_TrustedAnchor] | None = None,
) -> _MissingFallbackProjection:
    fallback_polygon: list[list[float]] | None = expected_item.polygon
    fallback_bbox: BBox | None = expected_item.bbox
    debug: dict[str, Any] = {}
    context_rescue_used = False
    fallback_source = "expected_slot"

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is not None:
        global_polygon, global_bbox = global_projection
        rescue = _try_missing_translation_rescue(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            global_polygon=global_polygon,
            global_bbox=global_bbox,
            all_expected=all_expected,
        )
        if rescue is not None:
            fallback_polygon = rescue.polygon
            fallback_bbox = rescue.bbox
            context_rescue_used = True
            fallback_source = "context_translation_rescue"
            debug.update(
                {
                    "projection": "expected_slot_context_translation_rescue",
                    "missing_polygon_projection": "expected_slot_context_translation_rescue",
                    "missing_polygon_projection_safety": "context_translation_rescue",
                    "missing_polygon_candidate_count": rescue.candidate_count,
                    "missing_polygon_inliers": rescue.inlier_count,
                    "missing_polygon_inlier_ratio": _round_debug(rescue.inlier_ratio),
                    "missing_polygon_median_error": _round_debug(rescue.median_error),
                    "missing_polygon_translation_shift_x": _round_debug(rescue.shift_x),
                    "missing_polygon_translation_shift_y": _round_debug(rescue.shift_y),
                    "missing_polygon_translation_shift_factor": _round_debug(
                        rescue.shift_factor
                    ),
                    "note": (
                        "Context affine was not safe, but surrounding context "
                        "still produced a conservative translation correction "
                        "from global alignment."
                    ),
                }
            )
        else:
            scene_rescue = _try_missing_scene_translation_rescue(
                expected_item,
                slot=slot,
                projection_data=projection_data,
                global_polygon=global_polygon,
                global_bbox=global_bbox,
                all_expected=all_expected,
            )
            if scene_rescue is not None:
                fallback_polygon = scene_rescue.polygon
                fallback_bbox = scene_rescue.bbox
                context_rescue_used = True
                fallback_source = "scene_translation_rescue"
                debug.update(
                    {
                        "projection": "expected_slot_scene_translation_rescue",
                        "missing_polygon_projection": "expected_slot_scene_translation_rescue",
                        "missing_polygon_projection_safety": "scene_translation_rescue",
                        "missing_polygon_candidate_count": scene_rescue.candidate_count,
                        "missing_polygon_inliers": scene_rescue.inlier_count,
                        "missing_polygon_inlier_ratio": _round_debug(
                            scene_rescue.inlier_ratio
                        ),
                        "missing_polygon_median_error": _round_debug(
                            scene_rescue.median_error
                        ),
                        "missing_polygon_translation_shift_x": _round_debug(
                            scene_rescue.shift_x
                        ),
                        "missing_polygon_translation_shift_y": _round_debug(
                            scene_rescue.shift_y
                        ),
                        "missing_polygon_translation_shift_factor": _round_debug(
                            scene_rescue.shift_factor
                        ),
                        "note": (
                            "Local context was too weak for this missing slot, "
                            "but nearby stable scene context produced a safe "
                            "translation correction from global alignment."
                        ),
                    }
                )
            else:
                anchor_release_debug: dict[str, Any] = {}
                anchor_release = _try_missing_anchor_release(
                    expected_item,
                    slot=slot,
                    projection_data=projection_data,
                    global_polygon=global_polygon,
                    global_bbox=global_bbox,
                    fallback_bbox=fallback_bbox,
                    all_expected=all_expected,
                    all_slots=all_slots,
                    trusted_anchors=trusted_anchors,
                    reject_debug=anchor_release_debug,
                )
                if anchor_release is not None:
                    fallback_polygon = anchor_release.polygon
                    fallback_bbox = anchor_release.bbox
                    context_rescue_used = True
                    fallback_source = "anchor_release"
                    debug.update(
                        {
                            "projection": "expected_slot_anchor_release",
                            "missing_polygon_projection": "expected_slot_anchor_release",
                            "missing_polygon_projection_safety": "trusted_anchor_release",
                            "missing_polygon_candidate_count": anchor_release.candidate_count,
                            "missing_polygon_inliers": anchor_release.inlier_count,
                            "missing_polygon_inlier_ratio": _round_debug(
                                anchor_release.inlier_ratio
                            ),
                            "missing_polygon_median_error": _round_debug(
                                anchor_release.median_error
                            ),
                            "missing_polygon_translation_shift_x": _round_debug(
                                anchor_release.shift_x
                            ),
                            "missing_polygon_translation_shift_y": _round_debug(
                                anchor_release.shift_y
                            ),
                            "missing_polygon_translation_shift_factor": _round_debug(
                                anchor_release.shift_factor
                            ),
                            "missing_polygon_anchor_sources": (
                                anchor_release.source_counts or {}
                            ),
                            "missing_polygon_anchor_single": (
                                anchor_release.single_anchor
                            ),
                            "missing_polygon_anchor_dispersion": _round_debug(
                                anchor_release.anchor_dispersion
                            ),
                            "missing_polygon_anchor_local_inlier_ratio": _round_debug(
                                anchor_release.anchor_local_inlier_ratio
                            ),
                            "note": (
                                "Multi-object expected slot had no direct global "
                                "consensus, but nearby trusted translation-rescue "
                                "anchors agreed on a safe scene correction. The "
                                "missing zone was rendered from the global polygon "
                                "with that anchor-only translation."
                            ),
                        }
                    )
                else:
                    debug.update(anchor_release_debug)
                    consensus_rescue = _try_missing_multi_consensus_rescue(
                        expected_item,
                        projection_data=projection_data,
                        global_polygon=global_polygon,
                        global_bbox=global_bbox,
                        fallback_bbox=fallback_bbox,
                        all_expected=all_expected,
                    )
                    if consensus_rescue is not None:
                        fallback_polygon = consensus_rescue.polygon
                        fallback_bbox = consensus_rescue.bbox
                        context_rescue_used = True
                        fallback_source = "multi_consensus_rescue"
                        debug.update(
                            {
                                "projection": "expected_slot_multi_consensus_rescue",
                                "missing_polygon_projection": "expected_slot_multi_consensus_rescue",
                                "missing_polygon_projection_safety": "multi_slot_consensus",
                                "missing_polygon_candidate_count": consensus_rescue.candidate_count,
                                "missing_polygon_inliers": consensus_rescue.inlier_count,
                                "missing_polygon_inlier_ratio": _round_debug(
                                    consensus_rescue.inlier_ratio
                                ),
                                "missing_polygon_median_error": _round_debug(
                                    consensus_rescue.median_error
                                ),
                                "missing_polygon_translation_shift_x": _round_debug(
                                    consensus_rescue.shift_x
                                ),
                                "missing_polygon_translation_shift_y": _round_debug(
                                    consensus_rescue.shift_y
                                ),
                                "missing_polygon_translation_shift_factor": _round_debug(
                                    consensus_rescue.shift_factor
                                ),
                                "note": (
                                    "Local context was not safe for this missing slot, "
                                    "but neighboring expected slots agreed on a common "
                                    "scene translation. The missing zone was rendered "
                                    "from that conservative multi-slot consensus."
                                ),
                            }
                        )
                    else:
                        local_global_area_score = _bbox_area_similarity(
                            expected_item.bbox,
                            global_bbox,
                        )
                        local_global_center_factor = _bbox_center_distance_factor(
                            expected_item.bbox,
                            global_bbox,
                        )
                        local_projection_disagrees = (
                            local_global_area_score
                            < settings.INSPECTION_MISSING_FALLBACK_MIN_LOCAL_GLOBAL_AREA_SCORE
                            or local_global_center_factor
                            > settings.INSPECTION_MISSING_FALLBACK_MAX_LOCAL_GLOBAL_CENTER_FACTOR
                        )

                        if local_projection_disagrees:
                            fallback_polygon = global_polygon
                            fallback_bbox = global_bbox
                            fallback_source = "global_fallback"
                            debug.update(
                                {
                                    "projection": "expected_slot_global_fallback",
                                    "missing_polygon_projection": "expected_slot_global_fallback",
                                    "missing_polygon_projection_safety": "global_fallback",
                                    "missing_polygon_local_global_area_score": _round_debug(
                                        local_global_area_score
                                    ),
                                    "missing_polygon_local_global_center_factor": _round_debug(
                                        local_global_center_factor
                                    ),
                                    "note": (
                                        "Adaptive local projection disagreed with the global "
                                        "alignment, so missing expected zone was rendered from "
                                        "global alignment instead of local object-near matches."
                                    ),
                                }
                            )


    if (
        fallback_source == "expected_slot"
        and _missing_expected_slot_has_guarded_local_release(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            fallback_polygon=fallback_polygon,
            fallback_bbox=fallback_bbox,
            all_expected=all_expected,
        )
    ):
        fallback_source = "expected_slot_local_guarded"
        # Stage-2 local release is intentionally conservative: it still draws
        # the already projected expected slot, but only after the dedicated
        # local/global and neighbor-overlap guards above have passed.  Mark it
        # as guarded so the generic rich-context hide rule does not immediately
        # cancel the release again.
        context_rescue_used = True
        debug.update(
            {
                "projection": "expected_slot_local_guarded",
                "missing_polygon_projection": "expected_slot_local_guarded",
                "missing_polygon_projection_safety": "local_slot_guard",
                "note": (
                    "Multi-object expected slot did not pass strict global "
                    "consensus, but stage-2 local/global geometry and neighbor "
                    "overlap guards were sufficient to render the conservative "
                    "expected zone."
                ),
            }
        )

    hidden_reason = _missing_fallback_projection_unsafe_reason(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        fallback_bbox=fallback_bbox,
        context_rescue_used=context_rescue_used,
        fallback_source=fallback_source,
        all_expected=all_expected,
    )
    if hidden_reason is not None:
        return _MissingFallbackProjection(
            polygon=None,
            bbox=None,
            hidden=True,
            debug={
                **debug,
                "missing_polygon_hidden_reason": hidden_reason,
            },
        )

    return _MissingFallbackProjection(
        polygon=fallback_polygon,
        bbox=fallback_bbox,
        hidden=False,
        debug=debug,
    )


def _try_missing_translation_rescue(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    all_expected: list[_ProjectedExpected] | None = None,
) -> _MissingTranslationRescue | None:
    if not settings.INSPECTION_MISSING_RESCUE_TRANSLATION_ENABLED:
        return None
    if slot is None or projection_data is None:
        return None
    if projection_data.global_homography is None:
        return None

    min_support = max(
        settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT,
        settings.INSPECTION_MISSING_RESCUE_TRANSLATION_MIN_SUPPORT,
        3,
    )

    local_reference, local_frame = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    if len(local_reference) < min_support or len(local_reference) != len(local_frame):
        return None

    projected_reference = _project_points_with_homography(
        local_reference,
        projection_data.global_homography,
    )
    if projected_reference is None or len(projected_reference) != len(local_frame):
        return None

    residuals = local_frame.astype(np.float32) - projected_reference.astype(np.float32)
    if residuals.ndim != 2 or residuals.shape[1] < 2 or not np.isfinite(residuals).all():
        return None

    median_residual = np.median(residuals[:, :2], axis=0).astype(np.float32)
    if not np.isfinite(median_residual).all():
        return None

    residual_errors = np.linalg.norm(residuals[:, :2] - median_residual[None, :], axis=1)
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        return None

    threshold = float(settings.INSPECTION_MISSING_RESCUE_TRANSLATION_MAX_RESIDUAL_ERROR)
    inlier_mask = residual_errors <= threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(residual_errors))
    if inlier_count < min_support:
        return None

    inlier_ratio = inlier_count / max(1, candidate_count)
    if inlier_ratio < settings.INSPECTION_MISSING_RESCUE_TRANSLATION_MIN_INLIER_RATIO:
        return None

    inlier_reference = local_reference[inlier_mask]
    reference_bbox = slot.reference_bbox or _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        spread = _missing_context_spread_score(inlier_reference, reference_bbox)
        # Translation-only rescue may work with weaker geometry than affine, but
        # points collapsed into a line/corner are still too risky.
        if spread < max(0.07, settings.INSPECTION_MISSING_POLYGON_CONTEXT_MIN_SPREAD * 0.55):
            return None

    median_error = float(np.median(residual_errors[inlier_mask]))
    if median_error > threshold:
        return None

    shift_x = float(median_residual[0])
    shift_y = float(median_residual[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / max(1.0, _bbox_diag(global_bbox))
    if shift_factor > settings.INSPECTION_MISSING_RESCUE_TRANSLATION_MAX_SHIFT_FACTOR:
        return None

    polygon = _translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        return None

    bbox = _bbox_from_polygon(polygon)
    if bbox is None:
        return None
    if projection_data.frame_size is not None and not _is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    # Translation rescue must remain close to the conservative global expected
    # zone and must not change object scale/shape.
    if _bbox_area_similarity(bbox, global_bbox) < 0.92:
        return None
    if _bbox_center_distance_factor(bbox, global_bbox) > max(
        0.05,
        settings.INSPECTION_MISSING_RESCUE_TRANSLATION_MAX_SHIFT_FACTOR,
    ):
        return None

    search_containment = _bbox_containment(bbox, slot.search_bbox)
    if search_containment < 0.05:
        return None

    return _MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
    )


def _try_missing_scene_translation_rescue(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    all_expected: list[_ProjectedExpected] | None = None,
) -> _MissingTranslationRescue | None:
    if not settings.INSPECTION_MISSING_SCENE_RESCUE_TRANSLATION_ENABLED:
        return None
    if projection_data is None or projection_data.global_homography is None:
        return None

    reference_points = _as_match_points(projection_data.reference_points)
    frame_points = _as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return None
    if len(reference_points) != len(frame_points):
        return None

    reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return None

    min_support = max(
        settings.INSPECTION_MISSING_SCENE_RESCUE_MIN_SUPPORT,
        settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT,
        4,
    )

    reference_window = _expand_bbox(
        reference_bbox,
        factor=settings.INSPECTION_MISSING_SCENE_RESCUE_CONTEXT_EXPANSION,
    )
    frame_window = _expand_bbox(
        global_bbox,
        factor=settings.INSPECTION_MISSING_SCENE_RESCUE_CONTEXT_EXPANSION,
        frame_size=projection_data.frame_size,
    )
    reference_exclusion_zones, frame_exclusion_zones = _all_expected_context_exclusion_zones(
        expected_item,
        all_expected=all_expected,
        projection_data=projection_data,
    )

    selected_reference: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []
    projected_reference = _project_points_with_homography(
        reference_points,
        projection_data.global_homography,
    )
    if projected_reference is None or len(projected_reference) != len(frame_points):
        return None

    residual_prefilter = max(
        settings.INSPECTION_MISSING_SCENE_RESCUE_MAX_RESIDUAL_ERROR * 5.0,
        _bbox_diag(global_bbox) * 0.85,
        32.0,
    )

    for reference_point, frame_point, projected_point in zip(
        reference_points,
        frame_points,
        projected_reference,
        strict=True,
    ):
        if not _point_in_bbox(reference_point, reference_window):
            continue
        if _point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if not _point_in_bbox(frame_point, frame_window):
            continue
        if _point_in_any_bbox(frame_point, frame_exclusion_zones):
            continue
        if float(np.linalg.norm(frame_point[:2] - projected_point[:2])) > residual_prefilter:
            continue

        selected_reference.append(reference_point)
        selected_frame.append(frame_point)

    if len(selected_reference) < min_support:
        return None

    local_reference = np.asarray(selected_reference, dtype=np.float32)
    local_frame = np.asarray(selected_frame, dtype=np.float32)

    max_points = max(_MISSING_POLYGON_MAX_LOCAL_POINTS, 160)
    if len(local_reference) > max_points:
        reference_center = np.asarray(_bbox_center(reference_bbox), dtype=np.float32)
        projected_center = np.asarray(_bbox_center(global_bbox), dtype=np.float32)
        reference_distance = np.linalg.norm(local_reference - reference_center, axis=1)
        frame_distance = np.linalg.norm(local_frame - projected_center, axis=1)
        indices = np.argsort(reference_distance + frame_distance)[:max_points]
        local_reference = local_reference[indices]
        local_frame = local_frame[indices]

    projected_local = _project_points_with_homography(
        local_reference,
        projection_data.global_homography,
    )
    if projected_local is None or len(projected_local) != len(local_frame):
        return None

    residuals = local_frame.astype(np.float32) - projected_local.astype(np.float32)
    if residuals.ndim != 2 or residuals.shape[1] < 2 or not np.isfinite(residuals).all():
        return None

    median_residual = np.median(residuals[:, :2], axis=0).astype(np.float32)
    if not np.isfinite(median_residual).all():
        return None

    residual_errors = np.linalg.norm(residuals[:, :2] - median_residual[None, :], axis=1)
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        return None

    threshold = float(settings.INSPECTION_MISSING_SCENE_RESCUE_MAX_RESIDUAL_ERROR)
    inlier_mask = residual_errors <= threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(residual_errors))
    if inlier_count < min_support:
        return None

    inlier_ratio = inlier_count / max(1, candidate_count)
    if inlier_ratio < settings.INSPECTION_MISSING_SCENE_RESCUE_MIN_INLIER_RATIO:
        return None

    inlier_reference = local_reference[inlier_mask]
    spread = _missing_context_spread_score(inlier_reference, reference_bbox)
    if spread < settings.INSPECTION_MISSING_SCENE_RESCUE_MIN_SPREAD:
        return None

    inlier_residuals = residuals[inlier_mask, :2]
    median_residual = np.median(inlier_residuals, axis=0).astype(np.float32)
    residual_errors = np.linalg.norm(inlier_residuals - median_residual[None, :], axis=1)
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        return None

    median_error = float(np.median(residual_errors))
    if median_error > threshold:
        return None

    shift_x = float(median_residual[0])
    shift_y = float(median_residual[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / max(1.0, _bbox_diag(global_bbox))
    if shift_factor > settings.INSPECTION_MISSING_SCENE_RESCUE_MAX_SHIFT_FACTOR:
        return None

    polygon = _translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        return None

    bbox = _bbox_from_polygon(polygon)
    if bbox is None:
        return None
    if projection_data.frame_size is not None and not _is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    if _bbox_area_similarity(bbox, global_bbox) < 0.92:
        return None
    if _bbox_center_distance_factor(bbox, global_bbox) > max(
        0.05,
        settings.INSPECTION_MISSING_SCENE_RESCUE_MAX_SHIFT_FACTOR,
    ):
        return None
    if slot is not None and _bbox_containment(bbox, slot.search_bbox) < 0.03:
        return None
    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=bbox,
        all_expected=all_expected,
    ):
        return None

    return _MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
    )



def _try_missing_anchor_release(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    fallback_bbox: BBox | None,
    all_expected: list[_ProjectedExpected] | None = None,
    all_slots: dict[int, _ExpectedSlot] | None = None,
    trusted_anchors: list[_TrustedAnchor] | None = None,
    reject_debug: dict[str, Any] | None = None,
) -> _MissingTranslationRescue | None:
    """Release a hidden slot only through a local consensus of trusted anchors.

    The current slot still cannot release itself.  The correction is built from
    already validated translation and high-quality affine anchors from other
    expected objects.  V3 searches for a compact local residual cluster instead
    of averaging the whole scene at once, so multi-object scenes with several
    independent shifts can release the current hidden slot without weakening the
    single-anchor guard.
    """

    _update_anchor_release_debug(
        reject_debug,
        attempted=True,
        total_anchors=len(trusted_anchors or []),
    )

    if not settings.INSPECTION_MISSING_ANCHOR_RELEASE_ENABLED:
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_disabled")
        return None
    if slot is None or projection_data is None or projection_data.global_homography is None:
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_unavailable_inputs")
        return None
    if not all_expected or len(all_expected) <= 1 or not all_slots:
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_not_multi_object")
        return None
    if not trusted_anchors:
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_no_trusted_anchors")
        return None

    min_anchors = max(1, int(settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_ANCHORS))
    max_anchor_error = float(settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_ANCHOR_ERROR)
    current_center = np.asarray(_bbox_center(global_bbox), dtype=np.float32)
    current_diag = max(1.0, _bbox_diag(global_bbox))
    max_neighbor_distance = (
        settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_NEIGHBOR_DISTANCE_FACTOR
        * current_diag
    )

    all_candidates: list[tuple[_TrustedAnchor, np.ndarray, float, float]] = []
    rejected_self = 0
    rejected_bad_residual = 0
    rejected_shift = 0
    rejected_overlap = 0
    candidate_sources: dict[str, int] = {}

    for anchor in trusted_anchors:
        if anchor.expected_index == expected_item.index:
            rejected_self += 1
            continue

        residual = np.asarray(anchor.residual, dtype=np.float32)
        if residual.shape != (2,) or not np.isfinite(residual).all():
            rejected_bad_residual += 1
            continue

        residual_shift_factor = float(np.linalg.norm(residual) / current_diag)
        if (
            residual_shift_factor
            > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_SHIFT_FACTOR
        ):
            rejected_shift += 1
            continue

        overlap_with_current = max(
            _bbox_iou(anchor.local_bbox, global_bbox),
            _bbox_containment(anchor.local_bbox, global_bbox),
            _bbox_containment(global_bbox, anchor.local_bbox),
        )
        if (
            overlap_with_current
            > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_OTHER_OVERLAP
        ):
            rejected_overlap += 1
            continue

        anchor_center = np.asarray(_bbox_center(anchor.global_bbox), dtype=np.float32)
        distance = float(np.linalg.norm(anchor_center - current_center))
        distance_weight = 1.0 / max(1.0, distance)
        source_weight = _trusted_anchor_source_weight(anchor.source)
        weight = max(0.01, float(anchor.weight) * distance_weight * source_weight)
        all_candidates.append((anchor, residual, weight, distance))
        candidate_sources[anchor.source] = candidate_sources.get(anchor.source, 0) + 1

    _update_anchor_release_debug(
        reject_debug,
        candidate_count=len(all_candidates),
        rejected_self=rejected_self,
        rejected_bad_residual=rejected_bad_residual,
        rejected_shift=rejected_shift,
        rejected_overlap=rejected_overlap,
        candidate_sources=candidate_sources,
    )

    if not all_candidates:
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_no_candidates")
        return None

    neighbor_candidates = [
        (anchor, residual, weight)
        for anchor, residual, weight, distance in all_candidates
        if distance <= max_neighbor_distance
    ]
    _update_anchor_release_debug(
        reject_debug,
        neighbor_candidate_count=len(neighbor_candidates),
        max_neighbor_distance=_round_debug(max_neighbor_distance),
    )
    if len(neighbor_candidates) >= min_anchors:
        candidates = neighbor_candidates
    else:
        candidates = [
            (anchor, residual, weight)
            for anchor, residual, weight, _ in all_candidates
        ]

    _update_anchor_release_debug(
        reject_debug,
        selected_candidate_count=len(candidates),
        min_anchors=min_anchors,
    )

    if len(candidates) < min_anchors:
        if not _anchor_release_allows_single_anchor(
            candidates,
            slot=slot,
            global_bbox=global_bbox,
            fallback_bbox=fallback_bbox,
        ):
            _set_anchor_release_reject(
                reject_debug,
                "anchor_release_rejected_single_anchor_guard",
                candidates=len(candidates),
                min_anchors=min_anchors,
            )
            return None
        min_anchors = 1

    consensus = _select_anchor_release_consensus(
        candidates,
        min_anchors=min_anchors,
        max_anchor_error=max_anchor_error,
        global_bbox=global_bbox,
    )
    if consensus is None:
        _set_anchor_release_reject(
            reject_debug,
            "anchor_release_rejected_high_dispersion",
            candidates=len(candidates),
            min_anchors=min_anchors,
        )
        return None

    _update_anchor_release_debug(
        reject_debug,
        inliers=consensus.inlier_count,
        inlier_ratio=_round_debug(consensus.inlier_ratio),
        median_error=_round_debug(consensus.median_error),
        dispersion=_round_debug(consensus.dispersion),
    )

    inlier_anchors = [
        candidate[0]
        for candidate, is_inlier in zip(candidates, consensus.inlier_mask, strict=True)
        if bool(is_inlier)
    ]
    source_counts = _trusted_anchor_source_counts(inlier_anchors)
    single_anchor = consensus.inlier_count == 1
    median_error = consensus.median_error
    if single_anchor:
        anchor_median_error = max(anchor.median_error for anchor in inlier_anchors)
        median_error = max(median_error, float(anchor_median_error))
    if median_error > max_anchor_error:
        _set_anchor_release_reject(
            reject_debug,
            "anchor_release_rejected_high_dispersion",
            median_error=_round_debug(median_error),
            max_anchor_error=_round_debug(max_anchor_error),
        )
        return None

    shift = consensus.shift
    shift_x = float(shift[0])
    shift_y = float(shift[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / current_diag
    _update_anchor_release_debug(
        reject_debug,
        shift_x=_round_debug(shift_x),
        shift_y=_round_debug(shift_y),
        shift_factor=_round_debug(shift_factor),
    )
    if shift_factor > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_SHIFT_FACTOR:
        _set_anchor_release_reject(
            reject_debug,
            "anchor_release_rejected_large_shift",
            shift_factor=_round_debug(shift_factor),
        )
        return None

    polygon = _translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_bad_polygon")
        return None

    bbox = _bbox_from_polygon(polygon)
    if bbox is None:
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_bad_bbox")
        return None

    if projection_data.frame_size is not None and not _is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_VISIBLE_FRACTION,
    ):
        _set_anchor_release_reject(reject_debug, "anchor_release_rejected_bad_overlap")
        return None

    if fallback_bbox is not None:
        local_area_score = _bbox_area_similarity(bbox, fallback_bbox)
        local_center_factor = _bbox_center_distance_factor(bbox, fallback_bbox)
        _update_anchor_release_debug(
            reject_debug,
            local_area_score=_round_debug(local_area_score),
            local_center_factor=_round_debug(local_center_factor),
        )
        if (
            local_area_score
            < settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_LOCAL_AREA_SCORE
        ):
            _set_anchor_release_reject(
                reject_debug,
                "anchor_release_rejected_bad_overlap",
                local_area_score=_round_debug(local_area_score),
            )
            return None
        if (
            local_center_factor
            > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_LOCAL_CENTER_FACTOR
        ):
            _set_anchor_release_reject(
                reject_debug,
                "anchor_release_rejected_bad_overlap",
                local_center_factor=_round_debug(local_center_factor),
            )
            return None

        if single_anchor:
            single_min_area = max(
                settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_LOCAL_AREA_SCORE,
                settings.INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MIN_LOCAL_AREA_SCORE,
            )
            single_max_center = min(
                settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_LOCAL_CENTER_FACTOR,
                settings.INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MAX_LOCAL_CENTER_FACTOR,
            )
            if local_area_score < single_min_area:
                _set_anchor_release_reject(
                    reject_debug,
                    "anchor_release_rejected_single_anchor_guard",
                    local_area_score=_round_debug(local_area_score),
                )
                return None
            if local_center_factor > single_max_center:
                _set_anchor_release_reject(
                    reject_debug,
                    "anchor_release_rejected_single_anchor_guard",
                    local_center_factor=_round_debug(local_center_factor),
                )
                return None

    if (
        single_anchor
        and shift_factor > settings.INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MAX_SHIFT_FACTOR
    ):
        _set_anchor_release_reject(
            reject_debug,
            "anchor_release_rejected_single_anchor_guard",
            shift_factor=_round_debug(shift_factor),
        )
        return None

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        reference_area_score = _bbox_area_similarity(bbox, reference_bbox)
        _update_anchor_release_debug(
            reject_debug,
            reference_area_score=_round_debug(reference_area_score),
        )
        if reference_area_score < 0.12:
            _set_anchor_release_reject(
                reject_debug,
                "anchor_release_rejected_bad_overlap",
                reference_area_score=_round_debug(reference_area_score),
            )
            return None

    overlap_decision = _anchor_release_overlap_decision(
        expected_item,
        bbox=bbox,
        global_bbox=global_bbox,
        fallback_bbox=fallback_bbox,
        all_expected=all_expected,
        trusted_anchors=trusted_anchors,
        consensus=consensus,
        single_anchor=single_anchor,
        shift_factor=shift_factor,
    )
    _update_anchor_release_debug(
        reject_debug,
        overlap_count=overlap_decision.overlap_count,
        overlap_unresolved_count=overlap_decision.unresolved_count,
        overlap_strong_anchor_count=overlap_decision.strong_anchor_count,
        overlap_max=_round_debug(overlap_decision.max_overlap),
        overlap_resolved_max=_round_debug(overlap_decision.max_resolved_overlap),
    )
    if not overlap_decision.allowed:
        _set_anchor_release_reject(
            reject_debug,
            overlap_decision.reason or "anchor_release_rejected_bad_overlap",
        )
        return None

    return _MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=consensus.candidate_count,
        inlier_count=consensus.inlier_count,
        inlier_ratio=float(consensus.inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
        source_counts=source_counts,
        single_anchor=single_anchor,
        anchor_dispersion=float(consensus.dispersion),
        anchor_local_inlier_ratio=float(consensus.inlier_ratio),
    )


def _select_anchor_release_consensus(
    candidates: list[tuple[_TrustedAnchor, np.ndarray, float]],
    *,
    min_anchors: int,
    max_anchor_error: float,
    global_bbox: BBox,
) -> _AnchorReleaseConsensus | None:
    residual_array = np.asarray([item[1] for item in candidates], dtype=np.float32)
    weights_array = np.asarray([item[2] for item in candidates], dtype=np.float32)
    if residual_array.ndim != 2 or residual_array.shape[1] != 2:
        return None
    if not np.isfinite(residual_array).all():
        return None
    if not np.isfinite(weights_array).all() or float(np.sum(weights_array)) <= 0.0:
        return None

    min_local_ratio = float(
        settings.INSPECTION_MISSING_ANCHOR_RELEASE_LOCAL_MIN_INLIER_RATIO
    )
    best: _AnchorReleaseConsensus | None = None
    best_score = float("-inf")

    for seed in residual_array:
        seed_errors = np.linalg.norm(residual_array - seed[None, :], axis=1)
        seed_mask = seed_errors <= max_anchor_error
        consensus = _build_anchor_release_consensus(
            residual_array,
            weights_array,
            initial_mask=seed_mask,
            min_anchors=min_anchors,
            max_anchor_error=max_anchor_error,
            global_bbox=global_bbox,
        )
        if consensus is None:
            continue
        if consensus.inlier_count > 1 and consensus.inlier_ratio < min_local_ratio:
            continue

        inlier_weight_sum = float(np.sum(weights_array[consensus.inlier_mask]))
        score = (
            consensus.inlier_count * 100.0
            + consensus.inlier_ratio * 10.0
            + inlier_weight_sum
            - consensus.median_error
            - consensus.dispersion * 10.0
        )
        if score > best_score:
            best = consensus
            best_score = score

    return best


def _build_anchor_release_consensus(
    residual_array: np.ndarray,
    weights_array: np.ndarray,
    *,
    initial_mask: np.ndarray,
    min_anchors: int,
    max_anchor_error: float,
    global_bbox: BBox,
) -> _AnchorReleaseConsensus | None:
    if initial_mask.dtype != np.bool_:
        initial_mask = initial_mask.astype(bool)
    if int(np.count_nonzero(initial_mask)) < min_anchors:
        return None

    initial_weights = weights_array[initial_mask]
    if float(np.sum(initial_weights)) <= 0.0:
        return None

    initial_shift = np.average(
        residual_array[initial_mask],
        axis=0,
        weights=initial_weights,
    ).astype(np.float32)
    if not np.isfinite(initial_shift).all():
        return None

    errors = np.linalg.norm(residual_array - initial_shift[None, :], axis=1)
    if len(errors) == 0 or not np.isfinite(errors).all():
        return None

    inlier_mask = errors <= max_anchor_error
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(errors))
    if inlier_count < min_anchors:
        return None

    inlier_weights = weights_array[inlier_mask]
    if float(np.sum(inlier_weights)) <= 0.0:
        return None

    shift = np.average(
        residual_array[inlier_mask],
        axis=0,
        weights=inlier_weights,
    ).astype(np.float32)
    if not np.isfinite(shift).all():
        return None

    inlier_errors = np.linalg.norm(residual_array[inlier_mask] - shift[None, :], axis=1)
    if len(inlier_errors) == 0 or not np.isfinite(inlier_errors).all():
        return None

    median_error = float(np.median(inlier_errors))
    dispersion = float(
        np.sqrt(np.average(np.square(inlier_errors), weights=inlier_weights))
        / max(1.0, _bbox_diag(global_bbox))
    )
    if dispersion > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_DISPERSION_FACTOR:
        return None

    return _AnchorReleaseConsensus(
        shift=shift,
        inlier_mask=inlier_mask,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / max(1, candidate_count),
        median_error=median_error,
        dispersion=dispersion,
    )


def _update_anchor_release_debug(
    reject_debug: dict[str, Any] | None,
    **fields: Any,
) -> None:
    if reject_debug is None:
        return
    if not settings.INSPECTION_MISSING_ANCHOR_RELEASE_DEBUG:
        return

    for key, value in fields.items():
        reject_debug[f"missing_polygon_anchor_release_{key}"] = value


def _set_anchor_release_reject(
    reject_debug: dict[str, Any] | None,
    reason: str,
    **fields: Any,
) -> None:
    if reject_debug is None:
        return
    if not settings.INSPECTION_MISSING_ANCHOR_RELEASE_DEBUG:
        return

    reject_debug["missing_polygon_anchor_release_reject_reason"] = reason
    _update_anchor_release_debug(reject_debug, **fields)


def _build_trusted_missing_anchors(
    all_expected: list[_ProjectedExpected],
    *,
    slot_by_index: dict[int, _ExpectedSlot],
    projection_data: LocalProjectionData | None,
) -> list[_TrustedAnchor]:
    if not settings.INSPECTION_MISSING_ANCHOR_RELEASE_ENABLED:
        return []
    if projection_data is None or projection_data.global_homography is None:
        return []
    if len(all_expected) <= 1:
        return []

    anchors: list[_TrustedAnchor] = []

    for expected_item in all_expected:
        slot = slot_by_index.get(expected_item.index)
        if slot is None:
            continue

        global_projection = _missing_global_projection(
            expected_item,
            projection_data=projection_data,
        )
        if global_projection is None:
            continue
        global_polygon, global_bbox = global_projection

        rescue = _try_missing_translation_rescue(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            global_polygon=global_polygon,
            global_bbox=global_bbox,
            all_expected=all_expected,
        )
        if rescue is not None:
            anchor = _trusted_anchor_from_translation_rescue(
                expected_item,
                rescue=rescue,
                global_bbox=global_bbox,
            )
            if anchor is not None:
                anchors.append(anchor)
                continue

        refinement = _try_missing_polygon_refinement(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            all_expected=all_expected,
        )
        if refinement is not None:
            anchor = _trusted_anchor_from_affine_refinement(
                expected_item,
                refinement=refinement,
                global_bbox=global_bbox,
            )
            if anchor is not None:
                anchors.append(anchor)
                continue

        anchor = _trusted_anchor_from_local_global_slot(
            expected_item,
            slot=slot,
            global_bbox=global_bbox,
        )
        if anchor is not None:
            anchors.append(anchor)

    return anchors


def _trusted_anchor_from_translation_rescue(
    expected_item: _ProjectedExpected,
    *,
    rescue: _MissingTranslationRescue,
    global_bbox: BBox,
) -> _TrustedAnchor | None:
    max_anchor_error = float(settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_ANCHOR_ERROR)
    min_anchor_ratio = max(
        settings.INSPECTION_MISSING_RESCUE_TRANSLATION_MIN_INLIER_RATIO,
        settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_INLIER_RATIO,
    )

    if rescue.inlier_count < settings.INSPECTION_MISSING_RESCUE_TRANSLATION_MIN_SUPPORT:
        return None
    if rescue.inlier_ratio < min_anchor_ratio:
        return None
    if rescue.median_error > max_anchor_error:
        return None
    if rescue.shift_factor > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_SHIFT_FACTOR:
        return None

    residual = np.asarray([rescue.shift_x, rescue.shift_y], dtype=np.float32)
    if not np.isfinite(residual).all():
        return None

    support_weight = float(rescue.inlier_count) / max(1.0, float(rescue.candidate_count))
    error_weight = 1.0 / max(1.0, float(rescue.median_error))

    return _TrustedAnchor(
        expected_index=expected_item.index,
        source="translation_rescue",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=rescue.bbox,
        weight=max(0.01, support_weight * error_weight),
        candidate_count=rescue.candidate_count,
        inlier_count=rescue.inlier_count,
        inlier_ratio=rescue.inlier_ratio,
        median_error=rescue.median_error,
        shift_factor=rescue.shift_factor,
    )


def _trusted_anchor_from_affine_refinement(
    expected_item: _ProjectedExpected,
    *,
    refinement: _MissingPolygonRefinement,
    global_bbox: BBox,
) -> _TrustedAnchor | None:
    max_anchor_error = float(settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_ANCHOR_ERROR)
    min_support = max(6, settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT)
    min_ratio = max(0.55, settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_INLIER_RATIO)

    if refinement.inlier_count < min_support:
        return None
    if refinement.inlier_ratio < min_ratio:
        return None
    if refinement.median_error > max_anchor_error:
        return None
    if refinement.area_score < 0.46:
        return None
    if refinement.center_drift_factor > 0.70:
        return None

    residual = np.asarray(
        [
            _bbox_center(refinement.bbox)[0] - _bbox_center(global_bbox)[0],
            _bbox_center(refinement.bbox)[1] - _bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(residual).all():
        return None

    shift_factor = float(np.linalg.norm(residual) / max(1.0, _bbox_diag(global_bbox)))
    if shift_factor > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_SHIFT_FACTOR:
        return None

    local_global_area_score = _bbox_area_similarity(refinement.bbox, global_bbox)
    if local_global_area_score < 0.30:
        return None

    support_weight = refinement.inlier_count / max(1.0, float(refinement.candidate_count))
    geometry_weight = refinement.area_score * max(0.20, local_global_area_score)
    error_weight = 1.0 / max(1.0, float(refinement.median_error))

    return _TrustedAnchor(
        expected_index=expected_item.index,
        source="context_feature_affine",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=refinement.bbox,
        weight=max(0.01, support_weight * geometry_weight * error_weight),
        candidate_count=refinement.candidate_count,
        inlier_count=refinement.inlier_count,
        inlier_ratio=refinement.inlier_ratio,
        median_error=refinement.median_error,
        shift_factor=shift_factor,
    )


def _trusted_anchor_from_local_global_slot(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot,
    global_bbox: BBox,
) -> _TrustedAnchor | None:
    min_support = max(4, settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT)
    if slot.feature_support < min_support:
        return None

    feature_ratio = slot.feature_support / max(1, slot.feature_total)
    if feature_ratio < 0.10:
        return None

    local_global_area_score = _bbox_area_similarity(expected_item.bbox, global_bbox)
    local_global_center_factor = _bbox_center_distance_factor(
        expected_item.bbox,
        global_bbox,
    )
    if local_global_area_score < 0.55 or local_global_center_factor > 0.65:
        return None

    residual = np.asarray(
        [
            _bbox_center(expected_item.bbox)[0] - _bbox_center(global_bbox)[0],
            _bbox_center(expected_item.bbox)[1] - _bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(residual).all():
        return None

    shift_factor = float(np.linalg.norm(residual) / max(1.0, _bbox_diag(global_bbox)))
    if shift_factor > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_SHIFT_FACTOR:
        return None

    error_weight = 1.0 / max(
        1.0,
        local_global_center_factor * _bbox_diag(global_bbox),
    )

    return _TrustedAnchor(
        expected_index=expected_item.index,
        source="local_global_slot_hint",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=expected_item.bbox,
        weight=max(0.01, feature_ratio * local_global_area_score * error_weight),
        candidate_count=max(1, slot.feature_total),
        inlier_count=slot.feature_support,
        inlier_ratio=feature_ratio,
        median_error=float(local_global_center_factor * _bbox_diag(global_bbox)),
        shift_factor=shift_factor,
    )


def _trusted_anchor_source_weight(source: str) -> float:
    if source == "translation_rescue":
        return 1.0
    if source == "context_feature_affine":
        return 0.78
    return 0.42


def _anchor_release_allows_single_anchor(
    candidates: list[tuple[_TrustedAnchor, np.ndarray, float]],
    *,
    slot: _ExpectedSlot,
    global_bbox: BBox,
    fallback_bbox: BBox | None,
) -> bool:
    if not settings.INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_ENABLED:
        return False
    if len(candidates) != 1:
        return False

    anchor, residual, _ = candidates[0]
    if anchor.source == "local_global_slot_hint":
        return False
    if anchor.inlier_count < max(6, settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT):
        return False
    if anchor.inlier_ratio < max(0.58, settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_INLIER_RATIO):
        return False
    if anchor.median_error > settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_ANCHOR_ERROR:
        return False

    shift_factor = float(np.linalg.norm(residual) / max(1.0, _bbox_diag(global_bbox)))
    if shift_factor > settings.INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MAX_SHIFT_FACTOR:
        return False

    polygon_bbox = _translate_bbox(global_bbox, dx=float(residual[0]), dy=float(residual[1]))
    local_bbox = fallback_bbox or slot.projected_bbox
    local_area_score = _bbox_area_similarity(polygon_bbox, local_bbox)
    local_center_factor = _bbox_center_distance_factor(polygon_bbox, local_bbox)

    if local_area_score < settings.INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MIN_LOCAL_AREA_SCORE:
        return False
    if local_center_factor > settings.INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MAX_LOCAL_CENTER_FACTOR:
        return False

    return True


def _trusted_anchor_source_counts(anchors: list[_TrustedAnchor]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for anchor in anchors:
        counts[anchor.source] = counts.get(anchor.source, 0) + 1
    return counts


def _translate_bbox(bbox: BBox, *, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _try_missing_multi_consensus_rescue(
    expected_item: _ProjectedExpected,
    *,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    fallback_bbox: BBox | None,
    all_expected: list[_ProjectedExpected] | None = None,
) -> _MissingTranslationRescue | None:
    """Use neighboring expected slots to recover a hidden multi-object slot.

    This is deliberately translation-only: if several other projected expected
    zones share the same local-vs-global offset, we can apply that offset to the
    current global polygon without letting one object's local features deform the
    missing shape or snap it to a distractor.
    """

    if not settings.INSPECTION_MISSING_MULTI_CONSENSUS_RESCUE_ENABLED:
        return None
    if projection_data is None or projection_data.global_homography is None:
        return None
    if not all_expected or len(all_expected) <= 1:
        return None

    min_neighbors = max(
        1,
        int(settings.INSPECTION_MISSING_MULTI_CONSENSUS_MIN_NEIGHBORS),
    )
    min_candidates = max(min_neighbors, 2)

    current_center = np.asarray(_bbox_center(global_bbox), dtype=np.float32)
    residuals: list[np.ndarray] = []
    weights: list[float] = []

    for other in all_expected:
        if other.index == expected_item.index:
            continue

        other_global = _missing_global_projection(
            other,
            projection_data=projection_data,
        )
        if other_global is None:
            continue
        _, other_global_bbox = other_global

        area_score = _bbox_area_similarity(other.bbox, other_global_bbox)
        if area_score < settings.INSPECTION_MISSING_MULTI_CONSENSUS_MIN_AREA_SCORE:
            continue

        center_factor = _bbox_center_distance_factor(other.bbox, other_global_bbox)
        if center_factor > settings.INSPECTION_MISSING_MULTI_CONSENSUS_MAX_CENTER_FACTOR:
            continue

        # A neighbor that already occupies the current target area is more likely
        # to be a distractor/duplicate than a reliable scene-motion witness.
        overlap_with_current = max(
            _bbox_iou(other.bbox, global_bbox),
            _bbox_containment(other.bbox, global_bbox),
            _bbox_containment(global_bbox, other.bbox),
        )
        if overlap_with_current > settings.INSPECTION_MISSING_SCENE_RESCUE_MAX_OTHER_OVERLAP:
            continue

        gx, gy = _bbox_center(other_global_bbox)
        lx, ly = _bbox_center(other.bbox)
        residual = np.asarray([lx - gx, ly - gy], dtype=np.float32)
        if not np.isfinite(residual).all():
            continue

        shift_factor = float(np.linalg.norm(residual) / max(1.0, _bbox_diag(global_bbox)))
        if shift_factor > settings.INSPECTION_MISSING_MULTI_CONSENSUS_MAX_SHIFT_FACTOR:
            continue

        other_center = np.asarray((gx, gy), dtype=np.float32)
        distance = float(np.linalg.norm(other_center - current_center))
        weight = 1.0 / max(1.0, distance)
        weight *= max(0.05, area_score)
        residuals.append(residual)
        weights.append(weight)

    if len(residuals) < min_candidates:
        return None

    residual_array = np.asarray(residuals, dtype=np.float32)
    if residual_array.ndim != 2 or residual_array.shape[1] != 2:
        return None
    if not np.isfinite(residual_array).all():
        return None

    # Start with a weighted average for the broad scene shift, then keep only
    # neighbors that agree with it tightly enough.
    weights_array = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(weights_array).all() or float(np.sum(weights_array)) <= 0.0:
        return None

    initial_shift = np.average(residual_array, axis=0, weights=weights_array).astype(np.float32)
    residual_errors = np.linalg.norm(residual_array - initial_shift[None, :], axis=1)
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        return None

    threshold = float(settings.INSPECTION_MISSING_MULTI_CONSENSUS_MAX_RESIDUAL_ERROR)
    inlier_mask = residual_errors <= threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(residual_errors))
    if inlier_count < min_neighbors:
        return None

    inlier_ratio = inlier_count / max(1, candidate_count)
    if inlier_ratio < settings.INSPECTION_MISSING_MULTI_CONSENSUS_MIN_INLIER_RATIO:
        return None

    inlier_residuals = residual_array[inlier_mask]
    median_shift = np.median(inlier_residuals, axis=0).astype(np.float32)
    if not np.isfinite(median_shift).all():
        return None

    inlier_errors = np.linalg.norm(inlier_residuals - median_shift[None, :], axis=1)
    if len(inlier_errors) == 0 or not np.isfinite(inlier_errors).all():
        return None

    median_error = float(np.median(inlier_errors))
    if median_error > threshold:
        return None

    shift_x = float(median_shift[0])
    shift_y = float(median_shift[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / max(1.0, _bbox_diag(global_bbox))
    if shift_factor > settings.INSPECTION_MISSING_MULTI_CONSENSUS_MAX_SHIFT_FACTOR:
        return None

    polygon = _translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        return None

    bbox = _bbox_from_polygon(polygon)
    if bbox is None:
        return None
    if projection_data.frame_size is not None and not _is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    # Consensus rescue must preserve the global-projected object geometry.
    if _bbox_area_similarity(bbox, global_bbox) < 0.92:
        return None

    # If the local expected slot is available, the consensus result must still
    # remain compatible with it. Otherwise we risk inventing a new position that
    # neither the local slot nor the global slot supported.
    if fallback_bbox is not None:
        fallback_area_score = _bbox_area_similarity(bbox, fallback_bbox)
        fallback_center_factor = _bbox_center_distance_factor(bbox, fallback_bbox)
        if fallback_area_score < settings.INSPECTION_MISSING_MULTI_CONSENSUS_MIN_AREA_SCORE:
            return None
        if (
            fallback_center_factor
            > settings.INSPECTION_MISSING_MULTI_CONSENSUS_MAX_FALLBACK_CENTER_FACTOR
        ):
            return None

    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=bbox,
        all_expected=all_expected,
    ):
        return None

    return _MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
    )


def _missing_global_projection(
    expected_item: _ProjectedExpected,
    *,
    projection_data: LocalProjectionData | None,
) -> tuple[list[list[float]], BBox] | None:
    if projection_data is None or projection_data.global_homography is None:
        return None

    try:
        polygon = project_polygon(
            expected_item.item.reference_polygon,
            projection_data.global_homography,
        )
    except Exception:
        return None

    if len(polygon) < 3:
        return None

    polygon = [[float(point[0]), float(point[1])] for point in polygon]
    bbox = _bbox_from_polygon(polygon)
    if bbox is None:
        return None

    if projection_data.frame_size is not None and not _is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        global_area_score = _bbox_area_similarity(bbox, reference_bbox)
        if (
            global_area_score
            < settings.INSPECTION_MISSING_FALLBACK_MIN_GLOBAL_AREA_SCORE
        ):
            return None

    return polygon, bbox


def _missing_expected_slot_has_global_consensus(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot,
    projection_data: LocalProjectionData,
    fallback_bbox: BBox,
    all_expected: list[_ProjectedExpected] | None,
    strong_support: int,
) -> bool:
    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return False

    _, global_bbox = global_projection
    area_score = _bbox_area_similarity(fallback_bbox, global_bbox)
    center_factor = _bbox_center_distance_factor(fallback_bbox, global_bbox)

    min_area_score = settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_MIN_GLOBAL_AREA_SCORE
    max_center_factor = settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_MAX_GLOBAL_CENTER_FACTOR

    # If there were many nearby features but neither local affine nor
    # translation rescue survived, require a much stricter agreement with the
    # global projection before showing the polygon in a multi-object scene.
    if slot.feature_support >= strong_support:
        min_area_score = max(min_area_score, 0.78)
        max_center_factor = min(max_center_factor, 0.18)

    if area_score < min_area_score or center_factor > max_center_factor:
        return False

    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=fallback_bbox,
        all_expected=all_expected,
    ):
        return False

    return True


def _missing_expected_slot_has_guarded_local_release(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    fallback_polygon: list[list[float]] | None,
    fallback_bbox: BBox | None,
    all_expected: list[_ProjectedExpected] | None,
) -> bool:
    """Allow a conservative local expected slot in multi-object scenes.

    Strict global consensus is intentionally cautious, but it can hide a locally
    stable expected slot when the adaptive projection and global projection have
    a moderate disagreement.  This guard keeps the polygon translation-free and
    only releases it when the local slot is supported by context points, remains
    reference-like, and does not overlap another expected object.
    """

    if not settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_GUARDED_RELEASE_ENABLED:
        return False
    if slot is None or projection_data is None or fallback_bbox is None:
        return False
    if not all_expected or len(all_expected) <= 1:
        return False

    # Local guarded release is only a narrow final permission. Hard caps keep a
    # relaxed .env from turning it into an unsafe replacement for anchor/YOLO
    # confirmation.
    max_overlap = min(
        settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_OTHER_OVERLAP,
        0.28,
    )
    for other in all_expected:
        if other.index == expected_item.index:
            continue
        overlap = max(
            _bbox_iou(fallback_bbox, other.bbox),
            _bbox_containment(fallback_bbox, other.bbox),
            _bbox_containment(other.bbox, fallback_bbox),
        )
        if overlap > max_overlap:
            return False

    min_support = max(
        0,
        int(settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_SUPPORT),
    )
    if slot.feature_support < min_support:
        return False

    if slot.feature_total > 0:
        support_ratio = slot.feature_support / max(1, slot.feature_total)
        if (
            support_ratio
            < settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_SUPPORT_RATIO
        ):
            return False
    elif min_support > 0:
        return False

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        reference_area_score = _bbox_area_similarity(fallback_bbox, reference_bbox)
        if (
            reference_area_score
            < settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_REFERENCE_AREA_SCORE
        ):
            return False

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return False

    global_polygon, global_bbox = global_projection
    area_score = _bbox_area_similarity(fallback_bbox, global_bbox)
    center_factor = _bbox_center_distance_factor(fallback_bbox, global_bbox)
    max_center_factor = min(
        settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_GLOBAL_CENTER_FACTOR,
        0.34,
    )
    if (
        area_score
        < settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_GLOBAL_AREA_SCORE
    ):
        return False
    if center_factor > max_center_factor:
        return False

    if fallback_polygon is None or len(fallback_polygon) < 3:
        return False

    global_polygon_iou = _polygon_iou(fallback_polygon, global_polygon)
    min_global_polygon_iou = max(
        settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_GLOBAL_POLYGON_IOU,
        0.20,
    )
    if global_polygon_iou < min_global_polygon_iou:
        return False

    axis_angle, major_ratio = _polygon_axis_delta(fallback_polygon, global_polygon)
    max_axis_angle = min(
        settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_GLOBAL_AXIS_ANGLE,
        18.0,
    )
    min_major_ratio = max(
        settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_GLOBAL_MAJOR_RATIO,
        0.65,
    )
    max_major_ratio = min(
        settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_GLOBAL_MAJOR_RATIO,
        1.55,
    )
    if axis_angle is None or axis_angle > max_axis_angle:
        return False
    if (
        major_ratio is None
        or major_ratio < min_major_ratio
        or major_ratio > max_major_ratio
    ):
        return False

    if projection_data.frame_size is not None and not _is_visible_in_frame(
        fallback_bbox,
        projection_data.frame_size,
        min_visible_fraction=max(
            settings.INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_VISIBLE_FRACTION,
            0.12,
        ),
    ):
        return False

    return True


def _missing_fallback_projection_unsafe_reason(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    fallback_bbox: BBox | None,
    context_rescue_used: bool = False,
    fallback_source: str = "expected_slot",
    all_expected: list[_ProjectedExpected] | None = None,
) -> str | None:
    if fallback_bbox is None:
        return "no_visible_fallback_polygon"
    if slot is None or projection_data is None:
        return None

    is_multi_object_context = all_expected is not None and len(all_expected) > 1
    min_support = max(3, settings.INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT)
    strong_support = max(min_support * 3, 12)

    if is_multi_object_context and fallback_source == "expected_slot":
        if not _missing_expected_slot_has_global_consensus(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            fallback_bbox=fallback_bbox,
            all_expected=all_expected,
            strong_support=strong_support,
        ):
            return "multi_expected_slot_without_global_consensus"

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = _bbox_from_polygon(expected_item.item.reference_polygon)

    if reference_bbox is not None:
        # A fallback polygon that became much larger/smaller than the annotated
        # object is usually a broken global projection. Do not present it as an
        # exact expected place for a missing detail.
        reference_area_score = _bbox_area_similarity(fallback_bbox, reference_bbox)
        if reference_area_score < 0.12:
            return "fallback_area_not_reference_like"

    if slot.feature_support < strong_support:
        return None

    if context_rescue_used:
        return None

    local_reference, _ = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )

    # If a lot of surrounding points exist but no safe context affine survived the
    # guards, the conservative result is to hide the exact polygon instead of
    # showing a confident-looking wrong zone.
    if len(local_reference) >= strong_support:
        return "rich_context_but_no_safe_transform"

    return None


def _missing_fallback_projection_is_unsafe(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    fallback_bbox: BBox | None,
    context_rescue_used: bool = False,
    fallback_source: str = "expected_slot",
    all_expected: list[_ProjectedExpected] | None = None,
) -> bool:
    return (
        _missing_fallback_projection_unsafe_reason(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            fallback_bbox=fallback_bbox,
            context_rescue_used=context_rescue_used,
            fallback_source=fallback_source,
            all_expected=all_expected,
        )
        is not None
    )


def _missing_polygon_refinement_points(
    expected_item: _ProjectedExpected,
    *,
    slot: _ExpectedSlot,
    projection_data: LocalProjectionData,
    all_expected: list[_ProjectedExpected] | None = None,
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

    context_expansion = settings.INSPECTION_MISSING_POLYGON_CONTEXT_EXPANSION
    reference_window = _expand_bbox(reference_bbox, factor=context_expansion)
    frame_window = _expand_bbox(
        slot.projected_bbox,
        factor=context_expansion,
        frame_size=projection_data.frame_size,
    )

    reference_polygon = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    projected_polygon = np.asarray(expected_item.polygon, dtype=np.float32)
    reference_margin = _missing_context_exclusion_margin(reference_bbox)
    frame_margin = _missing_context_exclusion_margin(expected_item.bbox)
    exclude_frame_object = settings.INSPECTION_MISSING_POLYGON_CONTEXT_FRAME_EXCLUSION
    reference_exclusion_zones, frame_exclusion_zones = _multi_object_context_exclusion_zones(
        expected_item,
        all_expected=all_expected,
        projection_data=projection_data,
    )

    selected_reference: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []

    for reference_point, frame_point in zip(reference_points, frame_points, strict=True):
        if not _point_in_bbox(reference_point, reference_window):
            continue
        if not _point_outside_np_polygon_margin(
            reference_point,
            reference_polygon,
            margin=reference_margin,
        ):
            continue
        if _point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if not _point_in_bbox(frame_point, frame_window):
            continue
        if exclude_frame_object and not _point_outside_np_polygon_margin(
            frame_point,
            projected_polygon,
            margin=frame_margin,
        ):
            continue
        if _point_in_any_bbox(frame_point, frame_exclusion_zones):
            continue

        selected_reference.append(reference_point)
        selected_frame.append(frame_point)

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


def _project_points_with_homography(
    points: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray | None:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or len(array) == 0:
        return None
    if not np.isfinite(array[:, :2]).all():
        return None
    try:
        projected = cv2.perspectiveTransform(
            array[:, :2].reshape(-1, 1, 2),
            homography.astype(np.float32),
        ).reshape(-1, 2)
    except cv2.error:
        return None
    if not np.isfinite(projected).all():
        return None
    return projected.astype(np.float32)


def _translate_polygon(
    polygon: list[list[float]],
    *,
    dx: float,
    dy: float,
) -> list[list[float]]:
    if len(polygon) < 3:
        return []
    translated: list[list[float]] = []
    for point in polygon:
        if len(point) < 2:
            return []
        x = float(point[0]) + float(dx)
        y = float(point[1]) + float(dy)
        if not np.isfinite([x, y]).all():
            return []
        translated.append([x, y])
    return translated


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


def _bbox_center_distance_factor(a: BBox, b: BBox) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    distance = float(np.hypot(ax - bx, ay - by))
    return distance / max(1.0, _bbox_diag(b))


def _polygon_axis_delta(
    polygon_a: list[list[float]],
    polygon_b: list[list[float]],
) -> tuple[float | None, float | None]:
    metrics_a = _polygon_axis_metrics(polygon_a)
    metrics_b = _polygon_axis_metrics(polygon_b)
    if metrics_a is None or metrics_b is None:
        return None, None

    angle_a, major_a = metrics_a
    angle_b, major_b = metrics_b
    angle_delta = abs(angle_a - angle_b) % 180.0
    if angle_delta > 90.0:
        angle_delta = 180.0 - angle_delta

    if major_a <= 0.0 or major_b <= 0.0:
        return float(angle_delta), None

    return float(angle_delta), float(major_a / major_b)


def _polygon_axis_metrics(
    polygon: list[list[float]],
) -> tuple[float, float] | None:
    if len(polygon) < 3:
        return None

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        return None

    points = points[:, :2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    if len(centered) < 2:
        return None

    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if vt.shape[0] == 0:
        return None

    major_axis = vt[0]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
    projection = centered @ major_axis
    major_length = float(np.max(projection) - np.min(projection))
    if major_length <= 0.0 or not np.isfinite(major_length):
        return None

    return angle, major_length


def _missing_context_spread_score(points: np.ndarray, bbox: BBox) -> float:
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        return 0.0

    x1, y1, x2, y2 = bbox
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))
    point_w = float(np.max(array[:, 0]) - np.min(array[:, 0]))
    point_h = float(np.max(array[:, 1]) - np.min(array[:, 1]))
    x_spread = max(0.0, min(1.0, point_w / bbox_w))
    y_spread = max(0.0, min(1.0, point_h / bbox_h))

    # Use the geometric mean so a thin but legitimate context can still pass,
    # while points collapsed into a single corner or line are rejected.
    return float(np.sqrt(x_spread * y_spread))


def _missing_context_quadrant_count(points: np.ndarray, bbox: BBox) -> int:
    if len(points) == 0:
        return 0
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        return 0

    cx, cy = _bbox_center(bbox)
    quadrants: set[tuple[int, int]] = set()
    for x, y in array[:, :2]:
        quadrants.add((1 if float(x) >= cx else 0, 1 if float(y) >= cy else 0))
    return len(quadrants)


def _empty_match_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _slot_feature_support(
    expected_item: _ProjectedExpected,
    *,
    search_bbox: BBox,
    projection_data: LocalProjectionData | None,
    all_expected: list[_ProjectedExpected] | None = None,
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
    reference_polygon = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    reference_margin = _missing_context_exclusion_margin(reference_bbox)
    reference_exclusion_zones, frame_exclusion_zones = _multi_object_context_exclusion_zones(
        expected_item,
        all_expected=all_expected,
        projection_data=projection_data,
    )

    total = 0
    support = 0
    for reference_point, frame_point in zip(reference_points, frame_points, strict=True):
        if not _point_in_bbox(reference_point, reference_window):
            continue
        if not _point_outside_np_polygon_margin(
            reference_point,
            reference_polygon,
            margin=reference_margin,
        ):
            continue
        if _point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if _point_in_any_bbox(frame_point, frame_exclusion_zones):
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
        "feature_source": "context_ring",
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


def _point_in_any_bbox(point: np.ndarray, bboxes: list[BBox]) -> bool:
    return any(_point_in_bbox(point, bbox) for bbox in bboxes)


def _all_expected_context_exclusion_zones(
    expected_item: _ProjectedExpected,
    *,
    all_expected: list[_ProjectedExpected] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[list[BBox], list[BBox]]:
    expected_items = all_expected if all_expected else [expected_item]
    expansion = float(
        settings.INSPECTION_MISSING_POLYGON_MULTI_CONTEXT_EXCLUSION_EXPANSION
    )
    reference_zones: list[BBox] = []
    frame_zones: list[BBox] = []
    frame_size = projection_data.frame_size if projection_data is not None else None

    for item in expected_items:
        reference_bbox = _bbox_from_polygon(item.item.reference_polygon)
        if reference_bbox is not None:
            reference_zones.append(_expand_bbox(reference_bbox, factor=expansion))

        frame_zones.append(
            _expand_bbox(
                item.bbox,
                factor=expansion,
                frame_size=frame_size,
            )
        )

    return reference_zones, frame_zones


def _anchor_release_overlap_decision(
    expected_item: _ProjectedExpected,
    *,
    bbox: BBox,
    global_bbox: BBox,
    fallback_bbox: BBox | None,
    all_expected: list[_ProjectedExpected] | None,
    trusted_anchors: list[_TrustedAnchor] | None,
    consensus: _AnchorReleaseConsensus,
    single_anchor: bool,
    shift_factor: float,
) -> _AnchorOverlapDecision:
    """Decide whether overlap with neighboring expected slots is still unsafe.

    The old guard compared the released bbox with every projected expected slot.
    In multi-object scenes those slots can be stale: a neighboring object can have
    already produced a trusted local correction, while its original expected bbox
    still intersects the current released slot.  This arbitration keeps the hard
    safety check against trusted resolved neighbors, but allows overlaps with
    stale/unresolved slots only when the anchor consensus is very compact and the
    released center is still owned by the current slot.
    """

    if not all_expected or len(all_expected) <= 1:
        return _AnchorOverlapDecision(
            allowed=True,
            reason=None,
            overlap_count=0,
            unresolved_count=0,
            strong_anchor_count=0,
            max_overlap=0.0,
            max_resolved_overlap=0.0,
        )

    overlap_limit = float(settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_OTHER_OVERLAP)
    strong_anchor_by_index = _best_strong_anchor_by_expected_index(trusted_anchors or [])
    target_owner_bbox = fallback_bbox or global_bbox
    target_center = np.asarray(_bbox_center(target_owner_bbox), dtype=np.float32)
    released_center = np.asarray(_bbox_center(bbox), dtype=np.float32)
    current_diag = max(1.0, _bbox_diag(global_bbox))

    overlap_count = 0
    unresolved_count = 0
    strong_anchor_count = 0
    max_overlap = 0.0
    max_resolved_overlap = 0.0

    for other in all_expected:
        if other.index == expected_item.index:
            continue

        overlap = max(
            _bbox_iou(bbox, other.bbox),
            _bbox_containment(bbox, other.bbox),
            _bbox_containment(other.bbox, bbox),
        )
        max_overlap = max(max_overlap, float(overlap))
        if overlap <= overlap_limit:
            continue

        overlap_count += 1
        strong_anchor = strong_anchor_by_index.get(other.index)
        if strong_anchor is not None:
            strong_anchor_count += 1
            resolved_overlap = max(
                _bbox_iou(bbox, strong_anchor.local_bbox),
                _bbox_containment(bbox, strong_anchor.local_bbox),
                _bbox_containment(strong_anchor.local_bbox, bbox),
            )
            max_resolved_overlap = max(max_resolved_overlap, float(resolved_overlap))
            if resolved_overlap > overlap_limit:
                return _AnchorOverlapDecision(
                    allowed=False,
                    reason="anchor_release_rejected_overlap_with_strong_anchor",
                    overlap_count=overlap_count,
                    unresolved_count=unresolved_count,
                    strong_anchor_count=strong_anchor_count,
                    max_overlap=max_overlap,
                    max_resolved_overlap=max_resolved_overlap,
                )
            # The overlap is only with the stale projected slot, not with the
            # neighbor's resolved trusted bbox, so it is not unsafe by itself.
            continue

        unresolved_count += 1
        other_center = np.asarray(_bbox_center(other.bbox), dtype=np.float32)
        target_distance = float(np.linalg.norm(released_center - target_center))
        other_distance = float(np.linalg.norm(released_center - other_center))
        ownership_margin = float(
            settings.INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_CENTER_MARGIN
        )
        if other_distance * ownership_margin < target_distance:
            return _AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_owned_by_other_slot",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )

    if overlap_count == 0:
        return _AnchorOverlapDecision(
            allowed=True,
            reason=None,
            overlap_count=0,
            unresolved_count=0,
            strong_anchor_count=0,
            max_overlap=max_overlap,
            max_resolved_overlap=max_resolved_overlap,
        )

    if not settings.INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_ARBITRATION_ENABLED:
        return _AnchorOverlapDecision(
            allowed=False,
            reason="anchor_release_rejected_bad_overlap",
            overlap_count=overlap_count,
            unresolved_count=unresolved_count,
            strong_anchor_count=strong_anchor_count,
            max_overlap=max_overlap,
            max_resolved_overlap=max_resolved_overlap,
        )

    if unresolved_count > 0:
        if single_anchor:
            return _AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_single_anchor_guard",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            consensus.inlier_count
            < settings.INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MIN_INLIERS
        ):
            return _AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_weak_consensus",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            consensus.inlier_ratio
            < settings.INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MIN_INLIER_RATIO
        ):
            return _AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_weak_consensus",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            consensus.dispersion
            > settings.INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MAX_DISPERSION_FACTOR
        ):
            return _AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_high_dispersion",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            shift_factor
            > settings.INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MAX_SHIFT_FACTOR
        ):
            return _AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_large_shift",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )

        target_drift = float(np.linalg.norm(released_center - target_center)) / current_diag
        max_target_drift = (
            settings.INSPECTION_MISSING_ANCHOR_RELEASE_MAX_LOCAL_CENTER_FACTOR
        )
        if target_drift > max_target_drift:
            return _AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_target_drift",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )

    return _AnchorOverlapDecision(
        allowed=True,
        reason=None,
        overlap_count=overlap_count,
        unresolved_count=unresolved_count,
        strong_anchor_count=strong_anchor_count,
        max_overlap=max_overlap,
        max_resolved_overlap=max_resolved_overlap,
    )


def _best_strong_anchor_by_expected_index(
    trusted_anchors: list[_TrustedAnchor],
) -> dict[int, _TrustedAnchor]:
    result: dict[int, _TrustedAnchor] = {}

    for anchor in trusted_anchors:
        if anchor.source not in {"translation_rescue", "context_feature_affine"}:
            continue
        if anchor.inlier_count < 2:
            continue
        min_ratio = settings.INSPECTION_MISSING_ANCHOR_RELEASE_MIN_INLIER_RATIO
        if anchor.inlier_ratio < min_ratio:
            continue
        current = result.get(anchor.expected_index)
        if current is None or anchor.weight > current.weight:
            result[anchor.expected_index] = anchor

    return result


def _missing_rescue_overlaps_other_expected(
    expected_item: _ProjectedExpected,
    *,
    bbox: BBox,
    all_expected: list[_ProjectedExpected] | None,
    max_overlap: float | None = None,
) -> bool:
    if not all_expected or len(all_expected) <= 1:
        return False

    overlap_limit = (
        settings.INSPECTION_MISSING_SCENE_RESCUE_MAX_OTHER_OVERLAP
        if max_overlap is None
        else max_overlap
    )
    for other in all_expected:
        if other.index == expected_item.index:
            continue
        overlap = max(
            _bbox_iou(bbox, other.bbox),
            _bbox_containment(bbox, other.bbox),
            _bbox_containment(other.bbox, bbox),
        )
        if overlap > overlap_limit:
            return True
    return False


def _multi_object_context_exclusion_zones(
    expected_item: _ProjectedExpected,
    *,
    all_expected: list[_ProjectedExpected] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[list[BBox], list[BBox]]:
    if not all_expected or len(all_expected) <= 1:
        return [], []

    expansion = float(
        settings.INSPECTION_MISSING_POLYGON_MULTI_CONTEXT_EXCLUSION_EXPANSION
    )
    reference_zones: list[BBox] = []
    frame_zones: list[BBox] = []
    frame_size = projection_data.frame_size if projection_data is not None else None

    for other in all_expected:
        if other.index == expected_item.index:
            continue

        reference_bbox = _bbox_from_polygon(other.item.reference_polygon)
        if reference_bbox is not None:
            reference_zones.append(_expand_bbox(reference_bbox, factor=expansion))

        frame_zones.append(
            _expand_bbox(
                other.bbox,
                factor=expansion,
                frame_size=frame_size,
            )
        )

    return reference_zones, frame_zones


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


def _point_outside_np_polygon_margin(
    point: np.ndarray,
    polygon: np.ndarray,
    *,
    margin: float,
) -> bool:
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
        return True
    if not np.isfinite(point).all() or not np.isfinite(polygon).all():
        return True

    try:
        signed_distance = cv2.pointPolygonTest(
            polygon[:, :2].astype(np.float32),
            (float(point[0]), float(point[1])),
            True,
        )
    except cv2.error:
        return True

    return float(signed_distance) < -max(0.0, float(margin))


def _missing_context_exclusion_margin(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    min_side = max(1.0, min(float(x2 - x1), float(y2 - y1)))
    configured = float(settings.INSPECTION_MISSING_POLYGON_CONTEXT_EXCLUSION_MARGIN)
    return max(0.0, min(configured, min_side * 0.25))


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
