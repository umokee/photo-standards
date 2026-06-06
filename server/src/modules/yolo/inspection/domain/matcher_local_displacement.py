from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from modules.yolo.inspection.domain.alignment import LocalProjectionData
from modules.yolo.inspection.domain.matcher_context import (
    _missing_context_quadrant_count,
    _missing_context_spread_score,
    _missing_polygon_refinement_points,
    _missing_rescue_overlaps_other_expected,
)
from modules.yolo.inspection.domain.matcher_debug import _round_debug
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_center_distance_factor,
    bbox_containment,
    bbox_diag,
    bbox_from_polygon,
    is_visible_in_frame,
    polygon_axis_delta,
    polygon_has_usable_area,
)
from modules.yolo.inspection.domain.matcher_structs import (
    ExpectedSlot,
    MissingLocalDisplacement,
    ProjectedExpected,
)
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds


def _try_missing_local_displacement_field(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None = None,
) -> MissingLocalDisplacement | None:
    if slot is None or projection_data is None:
        return None
    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return None

    min_support = max(3, int(_thresholds.missing_local_displacement_min_support))
    local_reference, local_frame = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    local_point_count = int(len(local_reference))
    if local_point_count < min_support:
        return None

    base_affine = _reference_to_expected_affine(expected_item)
    if base_affine is None:
        return None
    base_reference_projected = _project_points_by_affine(
        local_reference,
        base_affine,
    )
    if (
        base_reference_projected is None
        or len(base_reference_projected) != local_point_count
    ):
        return None

    residuals = (local_frame - base_reference_projected).astype(np.float32)
    if not np.isfinite(residuals).all():
        return None

    local_diag = max(1.0, bbox_diag(slot.projected_bbox))
    max_shift = max(
        1.0,
        local_diag * float(_thresholds.missing_local_displacement_max_shift_factor),
    )
    residual_lengths = np.linalg.norm(residuals, axis=1)
    finite_mask = np.isfinite(residual_lengths) & (residual_lengths <= max_shift)
    if int(np.count_nonzero(finite_mask)) < min_support:
        return None

    local_reference = local_reference[finite_mask]
    residuals = residuals[finite_mask]
    kept_mask = _local_displacement_residual_consensus_mask(
        local_reference,
        residuals,
        max_residual_error=float(
            _thresholds.missing_local_displacement_max_residual_error
        ),
    )
    if kept_mask is None or int(np.count_nonzero(kept_mask)) < min_support:
        return None

    kept_reference = local_reference[kept_mask]
    kept_residuals = residuals[kept_mask]
    kept_count = int(len(kept_reference))
    reference_spread = _missing_context_spread_score(kept_reference, reference_bbox)
    if reference_spread < _thresholds.missing_local_displacement_min_spread:
        return None

    quadrant_count = _missing_context_quadrant_count(kept_reference, reference_bbox)
    if quadrant_count < _thresholds.missing_local_displacement_min_quadrants:
        return None

    base_points = np.asarray(expected_item.polygon, dtype=np.float32)
    reference_vertices = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    if (
        len(base_points) < 3
        or len(reference_vertices) != len(base_points)
        or reference_vertices.ndim != 2
        or reference_vertices.shape[1] < 2
    ):
        return None

    nearest_count = max(
        3,
        min(int(_thresholds.missing_local_displacement_nearest_points), kept_count),
    )
    vertex_residuals = _interpolate_local_displacements(
        reference_vertices[:, :2],
        kept_reference,
        kept_residuals,
        nearest_count=nearest_count,
    )
    if vertex_residuals is None:
        return None

    displaced_points = base_points[:, :2] + vertex_residuals
    if not np.isfinite(displaced_points).all():
        return None

    vertex_shifts = np.linalg.norm(displaced_points - base_points[:, :2], axis=1)
    max_vertex_shift = float(np.max(vertex_shifts)) if len(vertex_shifts) else 0.0
    max_vertex_shift_factor = max_vertex_shift / local_diag
    if (
        max_vertex_shift_factor
        > _thresholds.missing_local_displacement_max_shift_factor
    ):
        return None

    polygon = [[float(x), float(y)] for x, y in displaced_points.tolist()]
    if len(polygon) < 3 or not polygon_has_usable_area(polygon):
        return None

    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return None
    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    containment = bbox_containment(bbox, slot.search_bbox)
    if containment < _thresholds.missing_local_displacement_min_search_containment:
        return None

    area_score = bbox_area_similarity(bbox, expected_item.bbox)
    if area_score < _thresholds.missing_local_displacement_min_local_area_score:
        return None

    center_drift_factor = bbox_center_distance_factor(bbox, expected_item.bbox)
    if (
        center_drift_factor
        > _thresholds.missing_local_displacement_max_local_center_factor
    ):
        return None

    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=bbox,
        all_expected=all_expected,
        max_overlap=float(_thresholds.missing_local_displacement_max_other_overlap),
    ):
        return None

    axis_angle_delta, major_length_ratio = polygon_axis_delta(
        polygon, expected_item.polygon
    )
    if axis_angle_delta is not None and axis_angle_delta > 22.0:
        return None
    if major_length_ratio is not None and not (0.62 <= major_length_ratio <= 1.62):
        return None

    return MissingLocalDisplacement(
        polygon=polygon,
        bbox=bbox,
        area_score=float(area_score),
        center_drift_factor=float(center_drift_factor),
    )


def _reference_to_expected_affine(
    expected_item: ProjectedExpected,
) -> np.ndarray | None:
    source = np.asarray(expected_item.item.reference_polygon, dtype=np.float32)
    target = np.asarray(expected_item.polygon, dtype=np.float32)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] < 2
        or target.shape[1] < 2
        or len(source) != len(target)
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        return None
    try:
        affine, _inliers = cv2.estimateAffinePartial2D(
            source[:, :2],
            target[:, :2],
            method=cv2.LMEDS,
            refineIters=10,
        )
    except cv2.error:
        return None
    if affine is None or affine.shape != (2, 3) or not np.isfinite(affine).all():
        return None
    return affine.astype(np.float32)


def _project_points_by_affine(
    points: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray | None:
    if len(points) == 0:
        return None
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] < 2 or not np.isfinite(source).all():
        return None
    try:
        projected = cv2.transform(
            source[:, :2].reshape(-1, 1, 2),
            affine.astype(np.float32),
        ).reshape(-1, 2)
    except cv2.error:
        return None
    if not np.isfinite(projected).all():
        return None
    return projected.astype(np.float32)


def _local_displacement_residual_consensus_mask(
    reference_points: np.ndarray,
    residuals: np.ndarray,
    *,
    max_residual_error: float,
) -> np.ndarray | None:
    count = int(len(reference_points))
    if count == 0 or count != len(residuals):
        return None
    if count < 3:
        return np.ones(count, dtype=bool)

    neighbor_count = max(3, min(7, count - 1))
    keep = np.zeros(count, dtype=bool)
    for idx in range(count):
        distances = np.linalg.norm(reference_points - reference_points[idx], axis=1)
        order = np.argsort(distances)
        neighbors = [int(i) for i in order if int(i) != idx][:neighbor_count]
        if not neighbors:
            continue
        local_median = np.median(residuals[neighbors], axis=0)
        error = float(np.linalg.norm(residuals[idx] - local_median))
        if error <= max_residual_error:
            keep[idx] = True

    if int(np.count_nonzero(keep)) < 3:
        return None
    return keep


def _interpolate_local_displacements(
    target_reference_points: np.ndarray,
    source_reference_points: np.ndarray,
    source_residuals: np.ndarray,
    *,
    nearest_count: int,
) -> np.ndarray | None:
    if (
        len(target_reference_points) == 0
        or len(source_reference_points) == 0
        or len(source_reference_points) != len(source_residuals)
    ):
        return None

    interpolated: list[np.ndarray] = []
    power = 1.65
    eps = 1e-3
    for target in target_reference_points:
        distances = np.linalg.norm(source_reference_points - target, axis=1)
        order = np.argsort(distances)[:nearest_count]
        if len(order) == 0:
            return None
        nearest_distances = distances[order]
        nearest_residuals = source_residuals[order]
        if float(np.min(nearest_distances)) <= eps:
            interpolated.append(nearest_residuals[int(np.argmin(nearest_distances))])
            continue
        weights = 1.0 / np.power(nearest_distances + eps, power)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            return None
        residual = np.sum(nearest_residuals * weights[:, None], axis=0) / weight_sum
        interpolated.append(residual.astype(np.float32))

    result = np.asarray(interpolated, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        return None
    return result


def _merge_missing_local_displacement_debug(
    missing_debug: dict[str, Any],
    displacement: MissingLocalDisplacement,
) -> dict[str, Any]:
    return {
        **missing_debug,
        "projection": "expected_slot_local_displacement",
        "missing_polygon_projection": "expected_slot_local_displacement",
        "missing_polygon_projection_safety": "sparse_local_displacement",
        "missing_polygon_area_score": _round_debug(displacement.area_score),
        "missing_polygon_center_drift_factor": _round_debug(
            displacement.center_drift_factor
        ),
    }
