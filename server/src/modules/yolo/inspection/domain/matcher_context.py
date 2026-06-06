from __future__ import annotations

import numpy as np
from modules.yolo.inspection.domain.alignment import LocalProjectionData
from modules.yolo.inspection.domain.matcher_geometry import (
    as_match_points,
    bbox_center,
    bbox_containment,
    bbox_from_polygon,
    bbox_iou,
    empty_match_points,
    expand_bbox,
    point_in_any_bbox,
    point_in_bbox,
    point_outside_np_polygon_margin,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    ExpectedSlot,
    ProjectedExpected,
)
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds

_MISSING_POLYGON_MAX_LOCAL_POINTS = 96


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

    return float(np.sqrt(x_spread * y_spread))


def _missing_context_quadrant_count(points: np.ndarray, bbox: BBox) -> int:
    if len(points) == 0:
        return 0
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        return 0

    cx, cy = bbox_center(bbox)
    quadrants: set[tuple[int, int]] = set()
    for x, y in array[:, :2]:
        quadrants.add((1 if float(x) >= cx else 0, 1 if float(y) >= cy else 0))
    return len(quadrants)


def _slot_feature_support(
    expected_item: ProjectedExpected,
    *,
    search_bbox: BBox,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None = None,
) -> tuple[int, int]:
    if projection_data is None:
        return 0, 0

    reference_points = as_match_points(projection_data.reference_points)
    frame_points = as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return 0, 0
    if len(reference_points) != len(frame_points):
        return 0, 0

    reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return 0, 0

    reference_window = expand_bbox(
        reference_bbox,
        factor=_thresholds.slot_feature_search_expansion,
    )
    reference_polygon = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    reference_margin = _missing_context_exclusion_margin(reference_bbox)
    reference_exclusion_zones, frame_exclusion_zones = (
        _multi_object_context_exclusion_zones(
            expected_item,
            all_expected=all_expected,
            projection_data=projection_data,
        )
    )

    total = 0
    support = 0
    for reference_point, frame_point in zip(
        reference_points, frame_points, strict=True
    ):
        if not point_in_bbox(reference_point, reference_window):
            continue
        if not point_outside_np_polygon_margin(
            reference_point,
            reference_polygon,
            margin=reference_margin,
        ):
            continue
        if point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if point_in_any_bbox(frame_point, frame_exclusion_zones):
            continue
        total += 1
        if point_in_bbox(frame_point, search_bbox):
            support += 1

    return support, total


def _all_expected_context_exclusion_zones(
    expected_item: ProjectedExpected,
    *,
    all_expected: list[ProjectedExpected] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[list[BBox], list[BBox]]:
    expected_items = all_expected if all_expected else [expected_item]
    expansion = float(_thresholds.missing_polygon_multi_context_exclusion_expansion)
    reference_zones: list[BBox] = []
    frame_zones: list[BBox] = []
    frame_size = projection_data.frame_size if projection_data is not None else None

    for item in expected_items:
        reference_bbox = bbox_from_polygon(item.item.reference_polygon)
        if reference_bbox is not None:
            reference_zones.append(expand_bbox(reference_bbox, factor=expansion))

        frame_zones.append(
            expand_bbox(
                item.bbox,
                factor=expansion,
                frame_size=frame_size,
            )
        )

    return reference_zones, frame_zones


def _missing_rescue_overlaps_other_expected(
    expected_item: ProjectedExpected,
    *,
    bbox: BBox,
    all_expected: list[ProjectedExpected] | None,
    max_overlap: float | None = None,
) -> bool:
    if not all_expected or len(all_expected) <= 1:
        return False

    overlap_limit = (
        _thresholds.missing_scene_rescue_max_other_overlap
        if max_overlap is None
        else max_overlap
    )
    for other in all_expected:
        if other.index == expected_item.index:
            continue
        overlap = max(
            bbox_iou(bbox, other.bbox),
            bbox_containment(bbox, other.bbox),
            bbox_containment(other.bbox, bbox),
        )
        if overlap > overlap_limit:
            return True
    return False


def _multi_object_context_exclusion_zones(
    expected_item: ProjectedExpected,
    *,
    all_expected: list[ProjectedExpected] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[list[BBox], list[BBox]]:
    if not all_expected or len(all_expected) <= 1:
        return [], []

    expansion = float(_thresholds.missing_polygon_multi_context_exclusion_expansion)
    reference_zones: list[BBox] = []
    frame_zones: list[BBox] = []
    frame_size = projection_data.frame_size if projection_data is not None else None

    for other in all_expected:
        if other.index == expected_item.index:
            continue

        reference_bbox = bbox_from_polygon(other.item.reference_polygon)
        if reference_bbox is not None:
            reference_zones.append(expand_bbox(reference_bbox, factor=expansion))

        frame_zones.append(
            expand_bbox(
                other.bbox,
                factor=expansion,
                frame_size=frame_size,
            )
        )

    return reference_zones, frame_zones


def _missing_context_exclusion_margin(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    min_side = max(1.0, min(float(x2 - x1), float(y2 - y1)))
    configured = float(_thresholds.missing_polygon_context_exclusion_margin)
    return max(0.0, min(configured, min_side * 0.25))


def _missing_polygon_refinement_points(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    projection_data: LocalProjectionData,
    all_expected: list[ProjectedExpected] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reference_points = as_match_points(projection_data.reference_points)
    frame_points = as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return empty_match_points(), empty_match_points()
    if len(reference_points) != len(frame_points):
        return empty_match_points(), empty_match_points()

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return empty_match_points(), empty_match_points()

    context_expansion = _thresholds.missing_polygon_context_expansion
    reference_window = expand_bbox(reference_bbox, factor=context_expansion)
    frame_window = expand_bbox(
        slot.projected_bbox,
        factor=context_expansion,
        frame_size=projection_data.frame_size,
    )

    reference_polygon = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    reference_margin = _missing_context_exclusion_margin(reference_bbox)
    reference_exclusion_zones, frame_exclusion_zones = (
        _multi_object_context_exclusion_zones(
            expected_item,
            all_expected=all_expected,
            projection_data=projection_data,
        )
    )

    selected_reference: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []

    for reference_point, frame_point in zip(
        reference_points, frame_points, strict=True
    ):
        if not point_in_bbox(reference_point, reference_window):
            continue
        if not point_outside_np_polygon_margin(
            reference_point,
            reference_polygon,
            margin=reference_margin,
        ):
            continue
        if point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if not point_in_bbox(frame_point, frame_window):
            continue
        if point_in_any_bbox(frame_point, frame_exclusion_zones):
            continue

        selected_reference.append(reference_point)
        selected_frame.append(frame_point)

    if not selected_reference:
        return empty_match_points(), empty_match_points()

    local_reference = np.asarray(selected_reference, dtype=np.float32)
    local_frame = np.asarray(selected_frame, dtype=np.float32)
    if len(local_reference) <= _MISSING_POLYGON_MAX_LOCAL_POINTS:
        return local_reference, local_frame

    reference_center = np.asarray(bbox_center(reference_bbox), dtype=np.float32)
    frame_center = np.asarray(bbox_center(slot.projected_bbox), dtype=np.float32)
    reference_distance = np.linalg.norm(local_reference - reference_center, axis=1)
    frame_distance = np.linalg.norm(local_frame - frame_center, axis=1)
    indices = np.argsort(reference_distance + frame_distance)[
        :_MISSING_POLYGON_MAX_LOCAL_POINTS
    ]
    return local_reference[indices], local_frame[indices]
