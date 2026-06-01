from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
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
    projected_by_index = {item.index: item for item in projected_expected}
    detection_by_index = {item.index: item for item in detection_candidates}

    for expected_item in projected_expected:
        for detection_item in detection_candidates:
            if expected_item.item.class_key != detection_item.detection.class_key:
                continue

            iou = _match_iou(
                expected_item.polygon,
                detection_item.polygon,
                expected_item.bbox,
                detection_item.bbox,
            )

            if iou < IOU_MATCH_THRESHOLD:
                continue

            candidate_pairs.append(
                (iou, iou, expected_item.index, detection_item.index)
            )

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)

    matched_expected_indices: set[int] = set()
    matched_detection_indices: set[int] = set()
    chosen_pairs: list[tuple[int, int, float]] = []

    for _score, iou, expected_index, detection_index in candidate_pairs:
        if expected_index in matched_expected_indices:
            continue
        if detection_index in matched_detection_indices:
            continue

        matched_expected_indices.add(expected_index)
        matched_detection_indices.add(detection_index)
        chosen_pairs.append((expected_index, detection_index, iou))

    matches: list[SegmentMatch] = []

    for expected_index, detection_index, iou in chosen_pairs:
        expected_item = projected_by_index[expected_index]
        detection_item = detection_by_index[detection_index]
        detection = detection_item.detection

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
                expected_polygon=expected_item.polygon,
                detected_polygon=detection_item.polygon,
                detected_bbox=detection.bbox,
            )
        )

    for expected_item in projected_expected:
        if expected_item.index in matched_expected_indices:
            continue

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
                expected_polygon=expected_item.polygon,
                detected_polygon=None,
                detected_bbox=None,
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

    for detection_item in detection_candidates:
        if detection_item.index in matched_detection_indices:
            continue

        action, expected_index = _classify_unmatched_detection(
            detection_item,
            projected_expected,
            matched_expected_indices,
        )

        if action == "discard":
            if expected_index is not None:
                target_match = expected_index_to_match.get(expected_index)
                if (
                    target_match is not None
                    and target_match.detected_class_in_zone is None
                ):
                    target_match.detected_class_in_zone = (
                        detection_item.detection.class_key
                    )
            continue

        detection = detection_item.detection
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
    missing_names = [
        match.name for match in expected_matches if match.status == "missing"
    ]
    return total, matched, missing_names


def all_ok(matches: list[SegmentMatch], *, expected_total: int | None = None) -> bool:
    expected_matches = [match for match in matches if match.status in ("ok", "missing")]
    total = expected_total if expected_total is not None else len(expected_matches)
    if total <= 0:
        return False
    if any(match.status == "extra" for match in matches):
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


def _match_iou(
    expected_polygon: list[list[float]],
    detection_polygon: list[list[float]] | None,
    expected_bbox: BBox,
    detection_bbox: BBox,
) -> float:
    bbox_iou = _bbox_iou(expected_bbox, detection_bbox)

    if detection_polygon is None or len(detection_polygon) < 3:
        return bbox_iou

    polygon_iou = _polygon_iou(expected_polygon, detection_polygon)

    return max(bbox_iou, polygon_iou)


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
    matched_expected_indices: set[int],
    *,
    iou_threshold: float = 0.10,
) -> tuple[str, int | None]:
    del matched_expected_indices

    for expected_item in projected_expected:
        iou = _bbox_iou(expected_item.bbox, detection.bbox)

        if iou < iou_threshold:
            continue

        return ("discard", expected_item.index)

    return ("extra", None)
