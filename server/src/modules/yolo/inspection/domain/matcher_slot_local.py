from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from modules.core.standards.reference_features import compute_features
from modules.core.standards.reference_runtime import match_feature_arrays
from modules.yolo.inspection.domain.alignment import LocalProjectionData
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_center_distance_factor,
    bbox_from_polygon,
    bbox_iou,
    is_visible_in_frame,
    project_points_with_homography,
    translate_polygon,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    ExpectedSlot,
    ProjectedExpected,
)
from modules.yolo.inspection.domain.matcher_translation_solver import (
    solve_translation_from_projected_points,
)

_SLOT_LOCAL_MAX_KEYPOINTS = 512
_SLOT_LOCAL_GRID = (6, 6)
_SLOT_LOCAL_MAX_ATTEMPTS_PER_FRAME = 48
_SLOT_LOCAL_MIN_RAW_MATCHES = 12
_SLOT_LOCAL_MIN_INLIERS = 6
_SLOT_LOCAL_MIN_INLIER_RATIO = 0.55
_SLOT_LOCAL_MAX_MEDIAN_ERROR = 4.25
_SLOT_LOCAL_MIN_AREA_SCORE = 0.62
_SLOT_LOCAL_MAX_CENTER_FACTOR = 0.38
_SLOT_LOCAL_MAX_OTHER_OVERLAP = 0.62
_SLOT_LOCAL_TRANSLATION_MIN_INLIERS = 8
_SLOT_LOCAL_TRANSLATION_MIN_INLIER_RATIO = 0.58
_SLOT_LOCAL_TRANSLATION_MAX_MEDIAN_ERROR = 4.75
_SLOT_LOCAL_TRANSLATION_MAX_SHIFT_FACTOR = 0.45
_SLOT_LOCAL_MARGIN_FACTOR = 2.00
_SLOT_LOCAL_MIN_MARGIN = 48.0
_SLOT_LOCAL_MAX_MARGIN = 260.0
_SLOT_LOCAL_WIDE_MARGIN_FACTOR = 3.40
_SLOT_LOCAL_WIDE_MAX_MARGIN = 420.0


@dataclass(slots=True)
class SlotLocalProjection:
    polygon: list[list[float]]
    bbox: BBox
    raw_match_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    area_score: float
    center_factor: float
    max_other_overlap: float
    reference_keypoint_count: int
    frame_keypoint_count: int
    mode: str

    def to_debug(self) -> dict[str, Any]:
        return {
            "missing_polygon_projection": "slot_local_lightglue",
            "missing_polygon_projection_safety": "confirmed",
            "reason_code": "slot_local_lightglue_confirmed",
            "slot_local_lightglue_attempted": True,
            "slot_local_lightglue_accepted": True,
            "slot_local_lightglue_mode": self.mode,
            "slot_local_lightglue_reference_keypoints": self.reference_keypoint_count,
            "slot_local_lightglue_frame_keypoints": self.frame_keypoint_count,
            "slot_local_lightglue_max_keypoints": _SLOT_LOCAL_MAX_KEYPOINTS,
            "slot_local_lightglue_grid_rows": _SLOT_LOCAL_GRID[0],
            "slot_local_lightglue_grid_cols": _SLOT_LOCAL_GRID[1],
            "slot_local_lightglue_raw_matches": self.raw_match_count,
            "slot_local_lightglue_inliers": self.inlier_count,
            "slot_local_lightglue_inlier_ratio": _round_debug(self.inlier_ratio),
            "slot_local_lightglue_median_error": _round_debug(self.median_error),
            "slot_local_lightglue_area_score": _round_debug(self.area_score),
            "slot_local_lightglue_center_factor": _round_debug(self.center_factor),
            "slot_local_lightglue_max_other_overlap": _round_debug(
                self.max_other_overlap
            ),
        }


class SlotLocalBudget:
    def __init__(self, limit: int = _SLOT_LOCAL_MAX_ATTEMPTS_PER_FRAME) -> None:
        self.limit = max(0, int(limit))
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def slot_local_budget() -> SlotLocalBudget:
    return SlotLocalBudget()


def try_slot_local_lightglue_projection(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected],
    budget: SlotLocalBudget | None,
) -> tuple[SlotLocalProjection | None, dict[str, Any]]:
    debug: dict[str, Any] = {"slot_local_lightglue_attempted": False}

    if projection_data is None:
        debug["slot_local_lightglue_reject_reason"] = "no_projection_data"
        return None, debug
    if projection_data.reference_frame is None or projection_data.frame is None:
        debug["slot_local_lightglue_reject_reason"] = "missing_images"
        return None, debug
    if budget is not None and not budget.take():
        debug["slot_local_lightglue_reject_reason"] = "budget_exhausted"
        return None, debug

    debug["slot_local_lightglue_attempted"] = True

    reference_bbox = _reference_bbox(expected_item, slot=slot)
    if reference_bbox is None:
        debug["slot_local_lightglue_reject_reason"] = "no_reference_bbox"
        return None, debug

    frame_bbox = expected_item.bbox
    reference_crop = _crop_with_margin(
        projection_data.reference_frame,
        reference_bbox,
        margin=_slot_margin(reference_bbox),
    )
    if reference_crop is None:
        debug["slot_local_lightglue_reject_reason"] = "empty_reference_crop"
        return None, debug

    frame_crop_specs = _target_frame_crops(
        projection_data.frame,
        frame_bbox,
    )
    if not frame_crop_specs:
        debug["slot_local_lightglue_reject_reason"] = "empty_frame_crop"
        return None, debug

    ref_image, ref_offset = reference_crop

    best_reject_reason = "no_matches"
    best_raw_match_count = 0
    best_ref_keypoints = 0
    best_frame_keypoints = 0

    for crop_mode, frame_image, frame_offset in frame_crop_specs:
        variants = (
            (
                f"{crop_mode}:context_ring",
                _erase_polygon_in_crop(
                    ref_image,
                    expected_item.item.reference_polygon,
                    ref_offset,
                ),
                _erase_polygon_in_crop(
                    frame_image,
                    expected_item.polygon,
                    frame_offset,
                ),
            ),
            (
                f"{crop_mode}:reference_erased",
                _erase_polygon_in_crop(
                    ref_image,
                    expected_item.item.reference_polygon,
                    ref_offset,
                ),
                frame_image,
            ),
            (f"{crop_mode}:full_crop", ref_image, frame_image),
        )

        for mode, ref_variant, frame_variant in variants:
            try:
                ref_features = compute_features(
                    ref_variant,
                    max_side=None,
                    max_keypoints=_SLOT_LOCAL_MAX_KEYPOINTS,
                    selection_grid=_SLOT_LOCAL_GRID,
                )
                frame_features = compute_features(
                    frame_variant,
                    max_side=None,
                    max_keypoints=_SLOT_LOCAL_MAX_KEYPOINTS,
                    selection_grid=_SLOT_LOCAL_GRID,
                )
                best_ref_keypoints = max(best_ref_keypoints, int(ref_features.count))
                best_frame_keypoints = max(best_frame_keypoints, int(frame_features.count))
                matched = match_feature_arrays(
                    reference_keypoints=ref_features.keypoints,
                    reference_descriptors=ref_features.descriptors,
                    reference_size=(ref_features.image_width, ref_features.image_height),
                    frame_keypoints=frame_features.keypoints,
                    frame_descriptors=frame_features.descriptors,
                    frame_size=(frame_features.image_width, frame_features.image_height),
                    max_keypoints=_SLOT_LOCAL_MAX_KEYPOINTS,
                )
            except Exception as exc:
                debug["slot_local_lightglue_reject_reason"] = "exception"
                debug["slot_local_lightglue_error_type"] = type(exc).__name__
                return None, debug

            if matched is None:
                best_reject_reason = "no_matches"
                continue

            ref_points, frame_points = matched
            ref_points = _add_offset(ref_points, ref_offset)
            frame_points = _add_offset(frame_points, frame_offset)
            raw_match_count = int(len(ref_points))
            best_raw_match_count = max(best_raw_match_count, raw_match_count)
            if raw_match_count < _SLOT_LOCAL_MIN_RAW_MATCHES:
                best_reject_reason = "too_few_raw_matches"
                continue

            attempt_debug: dict[str, Any] = {}
            projection = _solve_slot_affine(
                expected_item,
                reference_points=ref_points,
                frame_points=frame_points,
                projection_data=projection_data,
                all_expected=all_expected,
                raw_match_count=raw_match_count,
                reference_keypoint_count=int(ref_features.count),
                frame_keypoint_count=int(frame_features.count),
                mode=mode,
                attempt_debug=attempt_debug,
            )
            if projection is not None:
                return projection, projection.to_debug()

            translation_debug: dict[str, Any] = {}
            projection = _solve_slot_translation(
                expected_item,
                reference_points=ref_points,
                frame_points=frame_points,
                projection_data=projection_data,
                all_expected=all_expected,
                raw_match_count=raw_match_count,
                reference_keypoint_count=int(ref_features.count),
                frame_keypoint_count=int(frame_features.count),
                mode=f"{mode}:translation",
                attempt_debug=translation_debug,
            )
            if projection is not None:
                return projection, projection.to_debug()

            best_payload = _better_reject_payload(
                attempt_debug,
                translation_debug,
                fallback_reason="geometry_rejected",
            )
            best_reject_reason = str(
                best_payload.get("slot_local_lightglue_reject_reason")
                or best_reject_reason
            )
            best_raw_match_count = max(
                best_raw_match_count,
                _int_debug(best_payload.get("slot_local_lightglue_raw_matches")),
            )
            debug.update(best_payload)

    debug["slot_local_lightglue_reject_reason"] = best_reject_reason
    debug["slot_local_lightglue_raw_matches"] = max(
        best_raw_match_count,
        _int_debug(debug.get("slot_local_lightglue_raw_matches")),
    )
    debug["slot_local_lightglue_reference_keypoints"] = max(
        best_ref_keypoints,
        _int_debug(debug.get("slot_local_lightglue_reference_keypoints")),
    )
    debug["slot_local_lightglue_frame_keypoints"] = max(
        best_frame_keypoints,
        _int_debug(debug.get("slot_local_lightglue_frame_keypoints")),
    )
    debug["slot_local_lightglue_max_keypoints"] = _SLOT_LOCAL_MAX_KEYPOINTS
    debug["slot_local_lightglue_grid_rows"] = _SLOT_LOCAL_GRID[0]
    debug["slot_local_lightglue_grid_cols"] = _SLOT_LOCAL_GRID[1]
    return None, debug


def _solve_slot_affine(
    expected_item: ProjectedExpected,
    *,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    projection_data: LocalProjectionData,
    all_expected: list[ProjectedExpected],
    raw_match_count: int,
    reference_keypoint_count: int,
    frame_keypoint_count: int,
    mode: str,
    attempt_debug: dict[str, Any] | None = None,
) -> SlotLocalProjection | None:
    affine, inliers = cv2.estimateAffinePartial2D(
        reference_points,
        frame_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=900,
        confidence=0.995,
        refineIters=10,
    )
    if affine is None or inliers is None:
        _update_reject_debug(
            attempt_debug,
            reason="affine_estimate_failed",
            raw_matches=raw_match_count,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        _update_reject_debug(
            attempt_debug,
            reason="affine_invalid_matrix",
            raw_matches=raw_match_count,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < _SLOT_LOCAL_MIN_INLIERS:
        _update_reject_debug(
            attempt_debug,
            reason="affine_too_few_inliers",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_count / max(raw_match_count, 1),
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    inlier_ratio = inlier_count / max(raw_match_count, 1)
    if inlier_ratio < _SLOT_LOCAL_MIN_INLIER_RATIO:
        _update_reject_debug(
            attempt_debug,
            reason="affine_low_inlier_ratio",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    median_error = _median_affine_error(
        affine,
        reference_points[inlier_mask],
        frame_points[inlier_mask],
    )
    if median_error > _SLOT_LOCAL_MAX_MEDIAN_ERROR:
        _update_reject_debug(
            attempt_debug,
            reason="affine_median_error_bad",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    source = np.asarray(expected_item.item.reference_polygon, dtype=np.float32).reshape(
        -1, 1, 2
    )
    projected = cv2.transform(source, affine.astype(np.float32)).reshape(-1, 2)
    if not np.isfinite(projected).all() or len(projected) < 3:
        _update_reject_debug(
            attempt_debug,
            reason="affine_invalid_polygon",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    polygon = [[float(x), float(y)] for x, y in projected]
    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        _update_reject_debug(
            attempt_debug,
            reason="affine_invalid_bbox",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None
    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.04,
    ):
        _update_reject_debug(
            attempt_debug,
            reason="affine_outside_frame",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    area_score = bbox_area_similarity(bbox, expected_item.bbox)
    if area_score < _SLOT_LOCAL_MIN_AREA_SCORE:
        _update_reject_debug(
            attempt_debug,
            reason="affine_area_bad",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            area_score=area_score,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    center_factor = bbox_center_distance_factor(bbox, expected_item.bbox)
    if center_factor > _SLOT_LOCAL_MAX_CENTER_FACTOR:
        _update_reject_debug(
            attempt_debug,
            reason="affine_center_bad",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            area_score=area_score,
            center_factor=center_factor,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    max_other_overlap = _max_other_overlap(expected_item, bbox, all_expected)
    if max_other_overlap > _SLOT_LOCAL_MAX_OTHER_OVERLAP:
        _update_reject_debug(
            attempt_debug,
            reason="affine_overlap_bad",
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            area_score=area_score,
            center_factor=center_factor,
            max_other_overlap=max_other_overlap,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    mode_reject = _mode_guard_reject_reason(
        mode=mode,
        area_score=area_score,
        center_factor=center_factor,
        inlier_ratio=inlier_ratio,
        median_error=median_error,
        max_other_overlap=max_other_overlap,
    )
    if mode_reject is not None:
        _update_reject_debug(
            attempt_debug,
            reason=mode_reject,
            raw_matches=raw_match_count,
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            area_score=area_score,
            center_factor=center_factor,
            max_other_overlap=max_other_overlap,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    return SlotLocalProjection(
        polygon=polygon,
        bbox=bbox,
        raw_match_count=raw_match_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        median_error=median_error,
        area_score=area_score,
        center_factor=center_factor,
        max_other_overlap=max_other_overlap,
        reference_keypoint_count=reference_keypoint_count,
        frame_keypoint_count=frame_keypoint_count,
        mode=mode,
    )


def _solve_slot_translation(
    expected_item: ProjectedExpected,
    *,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    projection_data: LocalProjectionData,
    all_expected: list[ProjectedExpected],
    raw_match_count: int,
    reference_keypoint_count: int,
    frame_keypoint_count: int,
    mode: str,
    attempt_debug: dict[str, Any] | None = None,
) -> SlotLocalProjection | None:
    if projection_data.global_homography is None:
        _update_reject_debug(
            attempt_debug,
            reason="translation_no_homography",
            raw_matches=raw_match_count,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    projected_reference = project_points_with_homography(
        reference_points,
        projection_data.global_homography,
    )
    if projected_reference is None or len(projected_reference) != len(frame_points):
        _update_reject_debug(
            attempt_debug,
            reason="translation_projection_failed",
            raw_matches=raw_match_count,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    source = np.asarray(expected_item.item.reference_polygon, dtype=np.float32).reshape(
        -1, 2
    )
    projected_polygon = project_points_with_homography(
        source,
        projection_data.global_homography,
    )
    if projected_polygon is None or len(projected_polygon) < 3:
        _update_reject_debug(
            attempt_debug,
            reason="translation_polygon_projection_failed",
            raw_matches=raw_match_count,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    global_polygon = [[float(x), float(y)] for x, y in projected_polygon]
    global_bbox = bbox_from_polygon(global_polygon)
    if global_bbox is None:
        _update_reject_debug(
            attempt_debug,
            reason="translation_invalid_global_bbox",
            raw_matches=raw_match_count,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    solve, solve_reject = solve_translation_from_projected_points(
        local_frame=frame_points,
        projected_reference=projected_reference,
        min_support=_SLOT_LOCAL_TRANSLATION_MIN_INLIERS,
        threshold=_SLOT_LOCAL_TRANSLATION_MAX_MEDIAN_ERROR,
        global_bbox=global_bbox,
    )
    if solve is None:
        _update_reject_debug(
            attempt_debug,
            reason=f"translation_{solve_reject or 'solve_failed'}",
            raw_matches=raw_match_count,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    if solve.inlier_ratio < _SLOT_LOCAL_TRANSLATION_MIN_INLIER_RATIO:
        _update_reject_debug(
            attempt_debug,
            reason="translation_low_inlier_ratio",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None
    if solve.median_error > _SLOT_LOCAL_TRANSLATION_MAX_MEDIAN_ERROR:
        _update_reject_debug(
            attempt_debug,
            reason="translation_median_error_bad",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None
    if solve.shift_factor > _SLOT_LOCAL_TRANSLATION_MAX_SHIFT_FACTOR:
        _update_reject_debug(
            attempt_debug,
            reason="translation_shift_too_large",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    polygon = translate_polygon(
        global_polygon,
        dx=solve.shift_x,
        dy=solve.shift_y,
    )
    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        _update_reject_debug(
            attempt_debug,
            reason="translation_invalid_bbox",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None
    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.04,
    ):
        _update_reject_debug(
            attempt_debug,
            reason="translation_outside_frame",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    area_score = bbox_area_similarity(bbox, expected_item.bbox)
    if area_score < _SLOT_LOCAL_MIN_AREA_SCORE:
        _update_reject_debug(
            attempt_debug,
            reason="translation_area_bad",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            area_score=area_score,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    center_factor = bbox_center_distance_factor(bbox, expected_item.bbox)
    if center_factor > _SLOT_LOCAL_MAX_CENTER_FACTOR:
        _update_reject_debug(
            attempt_debug,
            reason="translation_center_bad",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            area_score=area_score,
            center_factor=center_factor,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    max_other_overlap = _max_other_overlap(expected_item, bbox, all_expected)
    if max_other_overlap > _SLOT_LOCAL_MAX_OTHER_OVERLAP:
        _update_reject_debug(
            attempt_debug,
            reason="translation_overlap_bad",
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            area_score=area_score,
            center_factor=center_factor,
            max_other_overlap=max_other_overlap,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    mode_reject = _mode_guard_reject_reason(
        mode=mode,
        area_score=area_score,
        center_factor=center_factor,
        inlier_ratio=solve.inlier_ratio,
        median_error=solve.median_error,
        max_other_overlap=max_other_overlap,
    )
    if mode_reject is not None:
        _update_reject_debug(
            attempt_debug,
            reason=mode_reject,
            raw_matches=raw_match_count,
            inliers=solve.inlier_count,
            inlier_ratio=solve.inlier_ratio,
            median_error=solve.median_error,
            area_score=area_score,
            center_factor=center_factor,
            max_other_overlap=max_other_overlap,
            reference_keypoints=reference_keypoint_count,
            frame_keypoints=frame_keypoint_count,
        )
        return None

    return SlotLocalProjection(
        polygon=polygon,
        bbox=bbox,
        raw_match_count=raw_match_count,
        inlier_count=solve.inlier_count,
        inlier_ratio=solve.inlier_ratio,
        median_error=solve.median_error,
        area_score=area_score,
        center_factor=center_factor,
        max_other_overlap=max_other_overlap,
        reference_keypoint_count=reference_keypoint_count,
        frame_keypoint_count=frame_keypoint_count,
        mode=mode,
    )


def _mode_guard_reject_reason(
    *,
    mode: str,
    area_score: float,
    center_factor: float,
    inlier_ratio: float,
    median_error: float,
    max_other_overlap: float,
) -> str | None:
    # Wider crops and full-crop matching are useful for recovery, but they see many
    # repeated distractor shapes. Keep them as high-precision confirmations only.
    if ":full_crop" in mode:
        if area_score < 0.78:
            return "full_crop_area_bad"
        if center_factor > 0.20:
            return "full_crop_center_bad"
        if inlier_ratio < 0.66:
            return "full_crop_low_inlier_ratio"
        if median_error > 3.75:
            return "full_crop_median_error_bad"
        if max_other_overlap > 0.35:
            return "full_crop_overlap_bad"

    if mode.startswith("wide:"):
        if area_score < 0.70:
            return "wide_crop_area_bad"
        if center_factor > 0.30:
            return "wide_crop_center_bad"
        if inlier_ratio < 0.60:
            return "wide_crop_low_inlier_ratio"
        if median_error > 4.25:
            return "wide_crop_median_error_bad"
        if max_other_overlap > 0.50:
            return "wide_crop_overlap_bad"

    return None


def _reference_bbox(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
) -> BBox | None:
    if slot is not None and slot.reference_bbox is not None:
        return slot.reference_bbox
    return bbox_from_polygon(expected_item.item.reference_polygon)


def _target_frame_crops(
    frame: np.ndarray,
    bbox: BBox,
) -> list[tuple[str, np.ndarray, tuple[float, float]]]:
    crops: list[tuple[str, np.ndarray, tuple[float, float]]] = []

    normal_margin = _slot_margin(bbox)
    normal_crop = _crop_with_margin(frame, bbox, margin=normal_margin)
    if normal_crop is not None:
        image, offset = normal_crop
        crops.append(("slot", image, offset))

    wide_margin = _slot_margin(
        bbox,
        factor=_SLOT_LOCAL_WIDE_MARGIN_FACTOR,
        max_margin=_SLOT_LOCAL_WIDE_MAX_MARGIN,
    )
    if wide_margin > normal_margin + 16.0:
        wide_crop = _crop_with_margin(frame, bbox, margin=wide_margin)
        if wide_crop is not None:
            image, offset = wide_crop
            if not _same_crop_offset_size(crops, image, offset):
                crops.append(("wide", image, offset))

    return crops


def _same_crop_offset_size(
    crops: list[tuple[str, np.ndarray, tuple[float, float]]],
    image: np.ndarray,
    offset: tuple[float, float],
) -> bool:
    for _, existing_image, existing_offset in crops:
        if existing_offset == offset and existing_image.shape[:2] == image.shape[:2]:
            return True
    return False


def _slot_margin(
    bbox: BBox,
    *,
    factor: float = _SLOT_LOCAL_MARGIN_FACTOR,
    max_margin: float = _SLOT_LOCAL_MAX_MARGIN,
) -> float:
    x1, y1, x2, y2 = bbox
    size = max(1.0, x2 - x1, y2 - y1)
    return max(_SLOT_LOCAL_MIN_MARGIN, min(max_margin, size * factor))


def _expand_bbox_pixels(bbox: BBox, margin: float) -> BBox:
    x1, y1, x2, y2 = bbox
    pad = max(0.0, float(margin))
    return (
        float(x1) - pad,
        float(y1) - pad,
        float(x2) + pad,
        float(y2) + pad,
    )


def _crop_with_margin(
    image: np.ndarray,
    bbox: BBox,
    *,
    margin: float,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    height, width = image.shape[:2]
    expanded = _expand_bbox_pixels(bbox, margin)
    x1, y1, x2, y2 = expanded
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(width, int(np.ceil(x2)))
    bottom = min(height, int(np.ceil(y2)))
    if right - left < 32 or bottom - top < 32:
        return None
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop, (float(left), float(top))


def _erase_polygon_in_crop(
    image: np.ndarray,
    polygon: list[list[float]],
    offset: tuple[float, float],
) -> np.ndarray:
    result = image.copy()
    if len(polygon) < 3 or result.size == 0:
        return result

    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] -= float(offset[0])
    points[:, 1] -= float(offset[1])
    if not np.isfinite(points).all():
        return result

    height, width = result.shape[:2]
    if width <= 0 or height <= 0:
        return result

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    fill = _median_color(result)
    cv2.fillPoly(result, [np.round(points).astype(np.int32)], fill)
    return result


def _median_color(image: np.ndarray) -> tuple[int, ...]:
    if image.ndim == 2:
        return (int(np.median(image)),)
    flat = image.reshape(-1, image.shape[-1])
    median = np.median(flat, axis=0)
    return tuple(int(max(0, min(255, round(float(value))))) for value in median)


def _add_offset(points: np.ndarray, offset: tuple[float, float]) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    result[:, 0] += float(offset[0])
    result[:, 1] += float(offset[1])
    return result


def _median_affine_error(
    affine: np.ndarray,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
) -> float:
    if len(reference_points) == 0:
        return float("inf")
    projected = cv2.transform(
        reference_points.reshape(-1, 1, 2),
        affine.astype(np.float32),
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - frame_points, axis=1)
    if len(errors) == 0:
        return float("inf")
    return float(np.median(errors))


def _max_other_overlap(
    expected_item: ProjectedExpected,
    bbox: BBox,
    all_expected: list[ProjectedExpected],
) -> float:
    result = 0.0
    for other in all_expected:
        if other.index == expected_item.index:
            continue
        result = max(result, float(bbox_iou(bbox, other.bbox)))
    return result


def _update_reject_debug(
    target: dict[str, Any] | None,
    *,
    reason: str,
    raw_matches: int,
    inliers: int = 0,
    inlier_ratio: float | None = None,
    median_error: float | None = None,
    area_score: float | None = None,
    center_factor: float | None = None,
    max_other_overlap: float | None = None,
    reference_keypoints: int = 0,
    frame_keypoints: int = 0,
) -> None:
    if target is None:
        return
    target.update(
        {
            "slot_local_lightglue_reject_reason": reason,
            "slot_local_lightglue_raw_matches": int(raw_matches),
            "slot_local_lightglue_inliers": int(inliers),
            "slot_local_lightglue_reference_keypoints": int(reference_keypoints),
            "slot_local_lightglue_frame_keypoints": int(frame_keypoints),
            "slot_local_lightglue_max_keypoints": _SLOT_LOCAL_MAX_KEYPOINTS,
            "slot_local_lightglue_grid_rows": _SLOT_LOCAL_GRID[0],
            "slot_local_lightglue_grid_cols": _SLOT_LOCAL_GRID[1],
        }
    )
    if inlier_ratio is not None:
        target["slot_local_lightglue_inlier_ratio"] = _round_debug(inlier_ratio)
    if median_error is not None:
        target["slot_local_lightglue_median_error"] = _round_debug(median_error)
    if area_score is not None:
        target["slot_local_lightglue_area_score"] = _round_debug(area_score)
    if center_factor is not None:
        target["slot_local_lightglue_center_factor"] = _round_debug(center_factor)
    if max_other_overlap is not None:
        target["slot_local_lightglue_max_other_overlap"] = _round_debug(
            max_other_overlap
        )


def _better_reject_payload(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    fallback_reason: str,
) -> dict[str, Any]:
    if not first and not second:
        return {"slot_local_lightglue_reject_reason": fallback_reason}
    if not first:
        return second
    if not second:
        return first

    first_score = (
        _int_debug(first.get("slot_local_lightglue_inliers")),
        _int_debug(first.get("slot_local_lightglue_raw_matches")),
    )
    second_score = (
        _int_debug(second.get("slot_local_lightglue_inliers")),
        _int_debug(second.get("slot_local_lightglue_raw_matches")),
    )
    return second if second_score > first_score else first


def _int_debug(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _round_debug(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)
