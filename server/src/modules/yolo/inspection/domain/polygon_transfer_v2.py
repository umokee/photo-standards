from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from modules.yolo.inspection.domain.alignment import LocalProjectionData
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_center_distance_factor,
    bbox_diag,
    bbox_from_polygon,
    bbox_iou,
    is_visible_in_frame,
    translate_polygon,
)
from modules.yolo.inspection.domain.matcher_slot_local import (
    SlotLocalProjection,
    slot_local_budget,
    try_slot_local_lightglue_projection,
)
from modules.yolo.inspection.domain.matcher_structs import ProjectedExpected
from modules.yolo.inspection.domain.types import ExpectedSegment, SegmentMatch

PolygonPoints = list[list[float]]
BBox = tuple[float, float, float, float]

_V2_MIN_SCENE_INLIERS = 18
_V2_MIN_SCENE_INLIER_RATIO = 0.45
_V2_MAX_SCENE_MEDIAN_ERROR = 8.0
_V2_ECC_MAX_ITERATIONS = 80
_V2_ECC_EPS = 1e-4
_V2_MIN_RING_FRACTION = 0.06
_V2_MIN_ECC_SCORE = 0.32
_V2_STRONG_ECC_SCORE = 0.58
_V2_MIN_PHASE_RESPONSE = 0.025
_V2_STRONG_PHASE_RESPONSE = 0.075
_V2_SMALL_SHIFT_FACTOR = 0.22
_V2_LARGE_SHIFT_FACTOR = 0.58
_V2_MAX_SHIFT_FACTOR = 0.82
_V2_MAX_CENTER_FACTOR = 0.46
_V2_MAX_PHASE_ECC_DELTA_FACTOR = 0.40
_V2_MIN_VISIBLE_FRACTION = 0.06
_V2_MAX_OTHER_OVERLAP = 0.56
_V2_MIN_POINT_PAIRS = 4
_V2_MIN_POINT_INLIERS = 3
_V2_MAX_POINT_RESIDUAL_ERROR = 10.0


@dataclass(slots=True)
class _SceneRegistration:
    matrix: np.ndarray
    mode: str
    raw_match_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float | None
    reference_points: np.ndarray | None = None
    frame_points: np.ndarray | None = None


@dataclass(slots=True)
class _ProjectedSeed:
    index: int
    item: ExpectedSegment
    polygon: PolygonPoints
    bbox: BBox


@dataclass(slots=True)
class _LocalRefinement:
    polygon: PolygonPoints
    bbox: BBox
    shift_x: float
    shift_y: float
    shift_factor: float
    ecc_score: float | None
    phase_response: float | None
    ring_fraction: float
    max_other_overlap: float
    center_factor: float
    other_center_factor: float | None
    source: str
    candidate_score: float | None = None
    candidate_selection_score: float | None = None
    crop_mode: str | None = None
    start_mode: str | None = None
    candidate_count: int = 0


@dataclass(slots=True)
class _LocalCandidate:
    polygon: PolygonPoints
    bbox: BBox
    shift_x: float
    shift_y: float
    shift_factor: float
    ecc_score: float | None
    phase_response: float | None
    ring_fraction: float
    max_other_overlap: float
    center_factor: float
    other_center_factor: float | None
    source: str
    score: float | None
    selection_score: float | None
    crop_mode: str
    start_mode: str
    point_pair_count: int = 0
    point_inlier_count: int = 0
    point_residual_error: float | None = None


class _TransferV2Budget:
    def __init__(self, limit: int = 64) -> None:
        self.limit = max(0, int(limit))
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def transfer_missing_segments_v2(
    expected: list[ExpectedSegment],
    *,
    projection_data: LocalProjectionData | None,
    frame_size: tuple[int, int] | None,
) -> list[SegmentMatch]:
    """Experimental registration-first polygon transfer.

    This pipeline intentionally does not use YOLO detections and does not reuse the
    legacy missing-object fallback stack. It tries to solve a smaller problem:
    register the whole scene, project expected slots, then locally refine each slot
    with object-interior masked out on both images.
    """

    registration, registration_debug = _estimate_scene_registration(
        projection_data,
        frame_size=frame_size,
    )
    if registration is None:
        direct_matches = _direct_local_matches_from_raw_pairs(
            expected,
            projection_data=projection_data,
            frame_size=frame_size,
            registration_debug=registration_debug,
        )
        if direct_matches is not None:
            return direct_matches
        return [
            _v2_match(
                item,
                status="missing",
                polygon=None,
                debug={
                    **registration_debug,
                    "missing_polygon_projection": "v2_unconfirmed",
                    "missing_polygon_projection_safety": "unsafe_hidden",
                    "missing_polygon_hidden_reason": "v2_scene_registration_failed",
                    "reason_code": "v2_scene_registration_failed",
                },
            )
            for item in expected
        ]

    seeds = _project_expected_segments(
        expected,
        registration=registration,
        frame_size=frame_size,
    )
    seed_by_index = {seed.index: seed for seed in seeds}
    if len(seeds) != len(expected):
        # Keep per-object matches explicit; unprojected objects stay unconfirmed.
        missing_indices = set(range(len(expected))) - set(seed_by_index)
    else:
        missing_indices = set()

    warped_reference = _warp_reference_to_frame(
        projection_data,
        registration=registration,
        frame_size=frame_size,
    )
    budget = _TransferV2Budget(limit=max(16, len(expected) * 3))
    slot_budget = slot_local_budget()
    all_seed_polygons = [seed.polygon for seed in seeds]
    slot_projected_expected = [_projected_expected_from_seed(seed) for seed in seeds]
    slot_projected_by_index = {item.index: item for item in slot_projected_expected}
    matches: list[SegmentMatch] = []

    for index, item in enumerate(expected):
        if index in missing_indices:
            matches.append(
                _v2_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug={
                        **registration_debug,
                        "missing_polygon_projection": "v2_unconfirmed",
                        "missing_polygon_projection_safety": "unsafe_hidden",
                        "missing_polygon_hidden_reason": "v2_seed_projection_failed",
                        "reason_code": "v2_seed_projection_failed",
                    },
                )
            )
            continue

        seed = seed_by_index[index]
        base_debug: dict[str, Any] = {
            **registration_debug,
            "v2_pipeline": "registration_first_context_only",
            "v2_seed_available": True,
            "v2_seed_bbox": _bbox_debug(seed.bbox),
            "v2_seed_projection": registration.mode,
        }

        refinement, refine_debug = _refine_seed_with_context_registration(
            seed,
            projection_data=projection_data,
            registration=registration,
            warped_reference=warped_reference,
            frame_size=frame_size,
            all_seed_polygons=all_seed_polygons,
            budget=budget,
        )
        debug = {**base_debug, **refine_debug}

        slot_refinement: _LocalRefinement | None = None
        if _should_try_v2_slot_local(refinement):
            slot_refinement, slot_debug = _try_v2_slot_local_refinement(
                seed,
                projection_data=projection_data,
                all_expected=slot_projected_expected,
                projected_expected=slot_projected_by_index.get(seed.index),
                budget=slot_budget,
            )
            debug = {**debug, **slot_debug}

        selected_refinement = _select_confirmed_v2_refinement(
            context_refinement=refinement,
            slot_refinement=slot_refinement,
        )

        if selected_refinement is None:
            matches.append(
                _v2_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug={
                        **debug,
                        "missing_polygon_projection": "v2_unconfirmed",
                        "missing_polygon_projection_safety": "unsafe_hidden",
                        "missing_polygon_hidden_reason": debug.get("v2_reject_reason")
                        or debug.get("v2_slot_local_rescue_reject_reason")
                        or "v2_local_context_registration_failed",
                        "reason_code": "v2_local_context_registration_failed",
                    },
                )
            )
            continue

        confirmed_reason = (
            "slot_local_lightglue_confirmed"
            if selected_refinement.source == "slot_local_lightglue"
            else "v2_context_registration_confirmed"
        )
        matches.append(
            _v2_match(
                item,
                status="missing",
                polygon=selected_refinement.polygon,
                debug={
                    **debug,
                    "missing_polygon_projection": selected_refinement.source,
                    "missing_polygon_projection_safety": "confirmed",
                    "reason_code": confirmed_reason,
                    "v2_shift_x": _round_debug(selected_refinement.shift_x),
                    "v2_shift_y": _round_debug(selected_refinement.shift_y),
                    "v2_shift_factor": _round_debug(selected_refinement.shift_factor),
                    "v2_ecc_score": _round_debug(selected_refinement.ecc_score),
                    "v2_phase_response": _round_debug(selected_refinement.phase_response),
                    "v2_ring_fraction": _round_debug(selected_refinement.ring_fraction),
                    "v2_max_other_overlap": _round_debug(selected_refinement.max_other_overlap),
                    "v2_candidate_score": _round_debug(selected_refinement.candidate_score),
                    "v2_candidate_selection_score": _round_debug(
                        selected_refinement.candidate_selection_score
                    ),
                    "v2_candidate_crop_mode": selected_refinement.crop_mode,
                    "v2_candidate_start_mode": selected_refinement.start_mode,
                    "v2_candidate_count": selected_refinement.candidate_count,
                    "v2_selected_source": selected_refinement.source,
                },
            )
        )

    return matches



def _projected_expected_from_seed(seed: _ProjectedSeed) -> ProjectedExpected:
    return ProjectedExpected(
        index=seed.index,
        item=seed.item,
        polygon=seed.polygon,
        bbox=seed.bbox,
    )


@dataclass(slots=True)
class _DirectLocalCandidate:
    index: int
    polygon: PolygonPoints
    bbox: BBox
    source: str
    raw_match_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    area_score: float
    reference_context_pairs: int
    reference_window: BBox
    scale_x: float
    scale_y: float
    frame_span_factor: float
    reference_span_factor: float


_DIRECT_LOCAL_MIN_RAW_MATCHES = 8
_DIRECT_LOCAL_MIN_INLIERS = 5
_DIRECT_LOCAL_MIN_INLIER_RATIO = 0.42
_DIRECT_LOCAL_MAX_MEDIAN_ERROR = 8.0
_DIRECT_LOCAL_MIN_AREA_SCORE = 0.42
_DIRECT_LOCAL_MIN_SCALE = 0.45
_DIRECT_LOCAL_MAX_SCALE = 2.35
_DIRECT_LOCAL_MIN_CONTEXT_MARGIN = 42.0
_DIRECT_LOCAL_MAX_CONTEXT_MARGIN = 260.0
_DIRECT_LOCAL_CONTEXT_MARGIN_FACTOR = 2.35
_DIRECT_LOCAL_MAX_PAIR_DISTANCE_FACTOR = 4.20


def _direct_local_matches_from_raw_pairs(
    expected: list[ExpectedSegment],
    *,
    projection_data: LocalProjectionData | None,
    frame_size: tuple[int, int] | None,
    registration_debug: dict[str, Any],
) -> list[SegmentMatch] | None:
    """Recover per-object projections when whole-scene registration fails.

    A global homography is the wrong first primitive for scenes with repeated
    missing objects: object-interior matches can agree with the wrong copy and
    make the scene model fail, after which the previous V2 hid every slot.  This
    branch uses the raw LightGlue correspondences directly and solves a small
    affine model around each reference slot's context ring.  It does not need a
    global seed and therefore keeps the useful part of the original slot idea.
    """

    if projection_data is None:
        return None
    reference_points = _as_points(projection_data.reference_points)
    frame_points = _as_points(projection_data.frame_points)
    if reference_points is None or frame_points is None or len(reference_points) != len(frame_points):
        return None
    if len(reference_points) < _DIRECT_LOCAL_MIN_RAW_MATCHES:
        return None

    all_reference_polygons = [item.reference_polygon for item in expected]
    candidates: dict[int, _DirectLocalCandidate] = {}
    rejects: dict[int, dict[str, Any]] = {}

    for index, item in enumerate(expected):
        candidate, debug = _direct_local_candidate_from_raw_pairs(
            index,
            item,
            projection_data=projection_data,
            reference_points=reference_points,
            frame_points=frame_points,
            frame_size=frame_size,
            all_reference_polygons=all_reference_polygons,
        )
        if candidate is not None:
            candidates[index] = candidate
        else:
            rejects[index] = debug

    if not candidates:
        return None

    accepted_indices = _direct_local_non_overlapping_indices(candidates)
    matches: list[SegmentMatch] = []
    for index, item in enumerate(expected):
        candidate = candidates.get(index)
        if candidate is not None and index in accepted_indices:
            matches.append(
                _v2_match(
                    item,
                    status="missing",
                    polygon=candidate.polygon,
                    debug={
                        **registration_debug,
                        "missing_polygon_projection": candidate.source,
                        "missing_polygon_projection_safety": "confirmed",
                        "reason_code": "v3_slot_first_raw_match_confirmed",
                        "v3_slot_first_attempted": True,
                        "v3_slot_first_accepted": True,
                        "v3_slot_first_raw_matches": candidate.raw_match_count,
                        "v3_slot_first_inliers": candidate.inlier_count,
                        "v3_slot_first_inlier_ratio": _round_debug(candidate.inlier_ratio),
                        "v3_slot_first_median_error": _round_debug(candidate.median_error),
                        "v3_slot_first_area_score": _round_debug(candidate.area_score),
                        "v3_slot_first_reference_context_pairs": candidate.reference_context_pairs,
                        "v3_slot_first_reference_window": _bbox_debug(candidate.reference_window),
                        "v3_slot_first_scale_x": _round_debug(candidate.scale_x),
                        "v3_slot_first_scale_y": _round_debug(candidate.scale_y),
                        "v3_slot_first_reference_span_factor": _round_debug(
                            candidate.reference_span_factor
                        ),
                        "v3_slot_first_frame_span_factor": _round_debug(
                            candidate.frame_span_factor
                        ),
                    },
                )
            )
            continue

        reject_debug = rejects.get(index) or {
            "v3_slot_first_reject_reason": "overlaps_stronger_slot_first_candidate"
        }
        matches.append(
            _v2_match(
                item,
                status="missing",
                polygon=None,
                debug={
                    **registration_debug,
                    **reject_debug,
                    "missing_polygon_projection": "v2_unconfirmed",
                    "missing_polygon_projection_safety": "unsafe_hidden",
                    "missing_polygon_hidden_reason": reject_debug.get(
                        "v3_slot_first_reject_reason",
                        "v3_slot_first_rejected",
                    ),
                    "reason_code": "v3_slot_first_rejected_after_scene_failure",
                },
            )
        )
    return matches


def _direct_local_candidate_from_raw_pairs(
    index: int,
    item: ExpectedSegment,
    *,
    projection_data: LocalProjectionData,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    frame_size: tuple[int, int] | None,
    all_reference_polygons: list[PolygonPoints],
) -> tuple[_DirectLocalCandidate | None, dict[str, Any]]:
    reference_bbox = bbox_from_polygon(item.reference_polygon)
    if reference_bbox is None:
        return None, _direct_local_reject("invalid_reference_bbox")

    reference_window = _direct_local_reference_window(reference_bbox, projection_data)
    selected_ref: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []
    for ref_point, frame_point in zip(reference_points, frame_points, strict=False):
        if not np.isfinite(ref_point).all() or not np.isfinite(frame_point).all():
            continue
        if not _point_in_bbox(ref_point, reference_window):
            continue
        # Use the slot context ring. Object interiors are the least trustworthy
        # matches in a missing-object/distractor task.
        if _point_inside_any_polygon(ref_point, all_reference_polygons):
            continue
        selected_ref.append(ref_point.astype(np.float32, copy=False))
        selected_frame.append(frame_point.astype(np.float32, copy=False))

    raw_count = len(selected_ref)
    if raw_count < _DIRECT_LOCAL_MIN_RAW_MATCHES:
        return None, _direct_local_reject(
            "too_few_reference_context_matches",
            raw_matches=raw_count,
            reference_window=reference_window,
        )

    ref = np.asarray(selected_ref, dtype=np.float32).reshape(-1, 2)
    frm = np.asarray(selected_frame, dtype=np.float32).reshape(-1, 2)
    ref_span_factor = _point_span_factor(ref, reference_bbox)
    frame_span_factor = _point_span_factor(frm, reference_bbox)
    if ref_span_factor < 0.18:
        return None, _direct_local_reject(
            "reference_context_too_clustered",
            raw_matches=raw_count,
            reference_span_factor=ref_span_factor,
            frame_span_factor=frame_span_factor,
            reference_window=reference_window,
        )

    try:
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            ref,
            frm,
            method=cv2.RANSAC,
            ransacReprojThreshold=6.5,
            maxIters=2200,
            confidence=0.995,
            refineIters=10,
        )
    except cv2.error:
        return None, _direct_local_reject("affine_exception", raw_matches=raw_count)

    if affine is None or inlier_mask is None or affine.shape != (2, 3):
        return None, _direct_local_reject("affine_failed", raw_matches=raw_count)
    if not np.isfinite(affine).all():
        return None, _direct_local_reject("affine_non_finite", raw_matches=raw_count)

    mask = inlier_mask.reshape(-1).astype(bool)
    inliers = int(mask.sum())
    inlier_ratio = inliers / max(float(raw_count), 1.0)
    if inliers < _DIRECT_LOCAL_MIN_INLIERS:
        return None, _direct_local_reject(
            "too_few_affine_inliers",
            raw_matches=raw_count,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reference_window=reference_window,
        )
    if inlier_ratio < _DIRECT_LOCAL_MIN_INLIER_RATIO:
        return None, _direct_local_reject(
            "low_affine_inlier_ratio",
            raw_matches=raw_count,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reference_window=reference_window,
        )

    median_error = _direct_local_affine_median_error(affine, ref[mask], frm[mask])
    if median_error > _DIRECT_LOCAL_MAX_MEDIAN_ERROR:
        return None, _direct_local_reject(
            "affine_median_error_bad",
            raw_matches=raw_count,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            reference_window=reference_window,
        )

    scale_x, scale_y = _affine_axis_scales(affine)
    if not (
        _DIRECT_LOCAL_MIN_SCALE <= scale_x <= _DIRECT_LOCAL_MAX_SCALE
        and _DIRECT_LOCAL_MIN_SCALE <= scale_y <= _DIRECT_LOCAL_MAX_SCALE
    ):
        return None, _direct_local_reject(
            "affine_scale_bad",
            raw_matches=raw_count,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            scale_x=scale_x,
            scale_y=scale_y,
            reference_window=reference_window,
        )

    source_points = np.asarray(item.reference_polygon, dtype=np.float32).reshape(-1, 1, 2)
    try:
        projected = cv2.transform(source_points, affine.astype(np.float32)).reshape(-1, 2)
    except cv2.error:
        return None, _direct_local_reject("polygon_projection_failed", raw_matches=raw_count)
    if len(projected) < 3 or not np.isfinite(projected).all():
        return None, _direct_local_reject("polygon_projection_non_finite", raw_matches=raw_count)

    polygon = [[float(x), float(y)] for x, y in projected]
    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return None, _direct_local_reject("invalid_projected_bbox", raw_matches=raw_count)
    if frame_size is not None and not is_visible_in_frame(bbox, frame_size, min_visible_fraction=0.04):
        return None, _direct_local_reject(
            "projected_bbox_outside_frame",
            raw_matches=raw_count,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            reference_window=reference_window,
        )

    area_score = bbox_area_similarity(bbox, reference_bbox)
    if area_score < _DIRECT_LOCAL_MIN_AREA_SCORE:
        return None, _direct_local_reject(
            "projected_area_bad",
            raw_matches=raw_count,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            area_score=area_score,
            reference_window=reference_window,
        )

    frame_distance_factor = _mean_pair_distance_factor(frm[mask], bbox)
    if frame_distance_factor > _DIRECT_LOCAL_MAX_PAIR_DISTANCE_FACTOR:
        return None, _direct_local_reject(
            "inlier_frame_points_far_from_projection",
            raw_matches=raw_count,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            area_score=area_score,
            frame_distance_factor=frame_distance_factor,
            reference_window=reference_window,
        )

    return (
        _DirectLocalCandidate(
            index=index,
            polygon=polygon,
            bbox=bbox,
            source="v3_slot_first_raw_matches",
            raw_match_count=raw_count,
            inlier_count=inliers,
            inlier_ratio=inlier_ratio,
            median_error=median_error,
            area_score=area_score,
            reference_context_pairs=raw_count,
            reference_window=reference_window,
            scale_x=scale_x,
            scale_y=scale_y,
            frame_span_factor=frame_span_factor,
            reference_span_factor=ref_span_factor,
        ),
        {},
    )


def _direct_local_non_overlapping_indices(
    candidates: dict[int, _DirectLocalCandidate],
) -> set[int]:
    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            item.inlier_count,
            item.inlier_ratio,
            item.area_score,
            -item.median_error,
        ),
        reverse=True,
    )
    accepted: list[_DirectLocalCandidate] = []
    accepted_indices: set[int] = set()
    for candidate in ordered:
        if any(bbox_iou(candidate.bbox, other.bbox) > 0.62 for other in accepted):
            continue
        accepted.append(candidate)
        accepted_indices.add(candidate.index)
    return accepted_indices


def _direct_local_reference_window(
    bbox: BBox,
    projection_data: LocalProjectionData,
) -> BBox:
    margin = float(
        np.clip(
            bbox_diag(bbox) * _DIRECT_LOCAL_CONTEXT_MARGIN_FACTOR,
            _DIRECT_LOCAL_MIN_CONTEXT_MARGIN,
            _DIRECT_LOCAL_MAX_CONTEXT_MARGIN,
        )
    )
    x1, y1, x2, y2 = bbox
    if projection_data.reference_frame is not None:
        height, width = projection_data.reference_frame.shape[:2]
    else:
        width = height = None
    left = float(x1 - margin)
    top = float(y1 - margin)
    right = float(x2 + margin)
    bottom = float(y2 + margin)
    if width is not None and height is not None:
        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(width), right)
        bottom = min(float(height), bottom)
    return left, top, right, bottom


def _point_in_bbox(point: np.ndarray, bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    return float(x1) <= float(point[0]) <= float(x2) and float(y1) <= float(point[1]) <= float(y2)


def _point_span_factor(points: np.ndarray, reference_bbox: BBox) -> float:
    if len(points) < 2:
        return 0.0
    span = np.ptp(points.astype(np.float32), axis=0)
    return float(max(span[0], span[1]) / max(bbox_diag(reference_bbox), 1.0))


def _direct_local_affine_median_error(
    affine: np.ndarray,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
) -> float:
    if len(reference_points) == 0:
        return float("inf")
    projected = cv2.transform(
        reference_points.reshape(-1, 1, 2).astype(np.float32),
        affine.astype(np.float32),
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - frame_points, axis=1)
    finite = errors[np.isfinite(errors)]
    if len(finite) == 0:
        return float("inf")
    return float(np.median(finite))


def _affine_axis_scales(affine: np.ndarray) -> tuple[float, float]:
    sx = float(np.linalg.norm(affine[:, 0]))
    sy = float(np.linalg.norm(affine[:, 1]))
    return sx, sy


def _mean_pair_distance_factor(points: np.ndarray, bbox: BBox) -> float:
    if len(points) == 0:
        return float("inf")
    cx, cy = _bbox_center_xy(bbox)
    center = np.asarray([cx, cy], dtype=np.float32)
    distances = np.linalg.norm(points.astype(np.float32) - center, axis=1)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return float("inf")
    return float(np.median(finite) / max(bbox_diag(bbox), 1.0))


def _direct_local_reject(
    reason: str,
    *,
    raw_matches: int = 0,
    inliers: int = 0,
    inlier_ratio: float | None = None,
    median_error: float | None = None,
    area_score: float | None = None,
    scale_x: float | None = None,
    scale_y: float | None = None,
    reference_span_factor: float | None = None,
    frame_span_factor: float | None = None,
    frame_distance_factor: float | None = None,
    reference_window: BBox | None = None,
) -> dict[str, Any]:
    return {
        "v3_slot_first_attempted": True,
        "v3_slot_first_accepted": False,
        "v3_slot_first_reject_reason": reason,
        "v3_slot_first_raw_matches": int(raw_matches),
        "v3_slot_first_inliers": int(inliers),
        "v3_slot_first_inlier_ratio": _round_debug(inlier_ratio),
        "v3_slot_first_median_error": _round_debug(median_error),
        "v3_slot_first_area_score": _round_debug(area_score),
        "v3_slot_first_scale_x": _round_debug(scale_x),
        "v3_slot_first_scale_y": _round_debug(scale_y),
        "v3_slot_first_reference_span_factor": _round_debug(reference_span_factor),
        "v3_slot_first_frame_span_factor": _round_debug(frame_span_factor),
        "v3_slot_first_frame_distance_factor": _round_debug(frame_distance_factor),
        "v3_slot_first_reference_window": _bbox_debug(reference_window),
    }


def _should_try_v2_slot_local(refinement: _LocalRefinement | None) -> bool:
    if refinement is None:
        return True
    if refinement.source in {"v2_context_seed", "v2_context_phase", "v2_context_ecc"}:
        return True
    if refinement.shift_factor > _V2_LARGE_SHIFT_FACTOR:
        return True
    if refinement.center_factor > 0.30:
        return True
    if refinement.max_other_overlap > 0.40:
        return True
    if refinement.source == "v2_context_points":
        if refinement.candidate_count <= 0:
            return True
    return False


def _try_v2_slot_local_refinement(
    seed: _ProjectedSeed,
    *,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected],
    projected_expected: ProjectedExpected | None,
    budget: Any,
) -> tuple[_LocalRefinement | None, dict[str, Any]]:
    debug: dict[str, Any] = {
        "v2_slot_local_rescue_attempted": False,
        "v2_slot_local_rescue_accepted": False,
    }
    if projected_expected is None:
        debug["v2_slot_local_rescue_reject_reason"] = "missing_projected_expected"
        return None, debug

    slot_projection, slot_debug = try_slot_local_lightglue_projection(
        projected_expected,
        slot=None,
        projection_data=projection_data,
        all_expected=all_expected,
        budget=budget,
    )
    debug.update(slot_debug)
    debug["v2_slot_local_rescue_attempted"] = bool(
        slot_debug.get("slot_local_lightglue_attempted")
    )
    if slot_projection is None:
        debug["v2_slot_local_rescue_reject_reason"] = slot_debug.get(
            "slot_local_lightglue_reject_reason", "slot_local_rejected"
        )
        return None, debug

    debug["v2_slot_local_rescue_accepted"] = True
    return _slot_local_projection_to_refinement(seed, slot_projection), debug


def _slot_local_projection_to_refinement(
    seed: _ProjectedSeed,
    slot_projection: SlotLocalProjection,
) -> _LocalRefinement:
    seed_cx, seed_cy = _bbox_center_xy(seed.bbox)
    slot_cx, slot_cy = _bbox_center_xy(slot_projection.bbox)
    dx = float(slot_cx - seed_cx)
    dy = float(slot_cy - seed_cy)
    shift_factor = float(np.hypot(dx, dy) / max(bbox_diag(seed.bbox), 1.0))
    return _LocalRefinement(
        polygon=slot_projection.polygon,
        bbox=slot_projection.bbox,
        shift_x=dx,
        shift_y=dy,
        shift_factor=shift_factor,
        ecc_score=None,
        phase_response=None,
        ring_fraction=1.0,
        max_other_overlap=slot_projection.max_other_overlap,
        center_factor=slot_projection.center_factor,
        other_center_factor=None,
        source="slot_local_lightglue",
        candidate_score=None,
        candidate_selection_score=None,
        crop_mode=slot_projection.mode,
        start_mode="slot_local_lightglue",
        candidate_count=1,
    )


def _select_confirmed_v2_refinement(
    *,
    context_refinement: _LocalRefinement | None,
    slot_refinement: _LocalRefinement | None,
) -> _LocalRefinement | None:
    if slot_refinement is not None and _v2_refinement_reject_reason(slot_refinement) is None:
        if context_refinement is None:
            return slot_refinement
        context_reject = _v2_refinement_reject_reason(context_refinement)
        if context_reject is not None:
            return slot_refinement
        if _slot_local_beats_context(slot_refinement, context_refinement):
            return slot_refinement

    if context_refinement is None:
        return None
    if _v2_refinement_reject_reason(context_refinement) is not None:
        return None
    return context_refinement


def _slot_local_beats_context(
    slot_refinement: _LocalRefinement,
    context_refinement: _LocalRefinement,
) -> bool:
    if context_refinement.source in {"v2_context_seed", "v2_context_phase", "v2_context_ecc"}:
        return True
    if context_refinement.shift_factor > _V2_LARGE_SHIFT_FACTOR:
        return True
    if context_refinement.max_other_overlap > 0.42 and slot_refinement.max_other_overlap <= 0.30:
        return True
    if context_refinement.center_factor > 0.34 and slot_refinement.center_factor <= 0.24:
        return True
    return False


def _v2_refinement_reject_reason(refinement: _LocalRefinement) -> str | None:
    if refinement.ring_fraction < _V2_MIN_RING_FRACTION:
        return "low_context_ring_fraction"
    if refinement.shift_factor > _V2_MAX_SHIFT_FACTOR:
        return "shift_too_large"
    if refinement.center_factor > _V2_MAX_CENTER_FACTOR:
        return "center_shift_too_large"
    if refinement.max_other_overlap > _V2_MAX_OTHER_OVERLAP:
        return "overlaps_other_expected_slot"

    if refinement.source == "v2_context_points":
        if refinement.candidate_count <= 0:
            return "point_candidate_missing"
        # point_pair_count is not stored on _LocalRefinement, but point candidates
        # that survived to this point were built from _LocalCandidate and encoded
        # into score/selection.  Keep the common geometry guards above as the hard
        # safety gate for this source.
        return None

    if refinement.source == "v2_context_ecc":
        if refinement.ecc_score is None or refinement.ecc_score < _V2_MIN_ECC_SCORE:
            return "weak_ecc_score"
        if refinement.shift_factor > _V2_LARGE_SHIFT_FACTOR and refinement.ecc_score < _V2_STRONG_ECC_SCORE:
            return "ecc_shift_without_strong_score"
        return None

    if refinement.source == "v2_context_phase":
        if refinement.phase_response is None or refinement.phase_response < _V2_MIN_PHASE_RESPONSE:
            return "weak_phase_response"
        if refinement.shift_factor > _V2_SMALL_SHIFT_FACTOR and refinement.phase_response < _V2_STRONG_PHASE_RESPONSE:
            return "phase_shift_without_strong_response"
        return None

    if refinement.source == "v2_context_seed":
        # Scene seed is allowed only when it stayed near the globally registered slot.
        if refinement.shift_factor > _V2_SMALL_SHIFT_FACTOR:
            return "seed_shifted_too_far"
        return None

    if refinement.source == "slot_local_lightglue":
        if refinement.center_factor > 0.30:
            return "slot_local_center_shift_too_large"
        if refinement.max_other_overlap > 0.42:
            return "slot_local_overlaps_other_expected_slot"
        return None

    return "unknown_refinement_source"

def _estimate_scene_registration(
    projection_data: LocalProjectionData | None,
    *,
    frame_size: tuple[int, int] | None = None,
) -> tuple[_SceneRegistration | None, dict[str, Any]]:
    debug: dict[str, Any] = {"v2_scene_registration_attempted": True}
    if projection_data is None:
        debug["v2_scene_registration_reject_reason"] = "no_projection_data"
        return None, debug

    reference_points = _as_points(projection_data.reference_points)
    frame_points = _as_points(projection_data.frame_points)
    homography = projection_data.global_homography

    if homography is not None and _valid_homography(homography):
        reference_points, frame_points, expansion_debug = _expand_matches_from_all_keypoints(
            projection_data,
            seed_homography=homography.astype(np.float32),
            reference_points=reference_points,
            frame_points=frame_points,
            frame_size=frame_size,
        )
        debug.update(expansion_debug)

    debug.update(
        _scene_match_distribution_debug(
            reference_points=reference_points,
            frame_points=frame_points,
            frame_size=frame_size,
        )
    )

    if homography is not None and _valid_homography(homography):
        selected_homography = homography.astype(np.float32)
        selected_source = "provided_homography"
        raw, inliers, ratio, median = _homography_quality(
            selected_homography,
            reference_points=reference_points,
            frame_points=frame_points,
        )
        refit_homography = _refit_homography_from_matches(reference_points, frame_points)
        if refit_homography is not None:
            refit_raw, refit_inliers, refit_ratio, refit_median = _homography_quality(
                refit_homography,
                reference_points=reference_points,
                frame_points=frame_points,
            )
            debug.update(
                {
                    "v2_guided_homography_available": True,
                    "v2_guided_homography_raw_matches": refit_raw,
                    "v2_guided_homography_inliers": refit_inliers,
                    "v2_guided_homography_inlier_ratio": _round_debug(refit_ratio),
                    "v2_guided_homography_median_error": _round_debug(refit_median),
                }
            )
            if _homography_candidate_key(refit_inliers, refit_median) > _homography_candidate_key(
                inliers, median
            ):
                selected_homography = refit_homography
                selected_source = "all_keypoints_guided_homography"
                raw, inliers, ratio, median = refit_raw, refit_inliers, refit_ratio, refit_median
        else:
            debug["v2_guided_homography_available"] = False
        debug.update(
            {
                "v2_homography_raw_matches": raw,
                "v2_homography_inliers": inliers,
                "v2_homography_inlier_ratio": _round_debug(ratio),
                "v2_homography_median_error": _round_debug(median),
            }
        )
        debug.update(
            _scene_model_distribution_debug(
                selected_homography.astype(np.float32),
                mode="homography",
                source=selected_source,
                reference_points=reference_points,
                frame_points=frame_points,
                frame_size=frame_size,
                prefix="v2_homography",
            )
        )
        if _scene_quality_is_acceptable(raw=raw, inliers=inliers, ratio=ratio, median_error=median):
            registration = _SceneRegistration(
                matrix=selected_homography.astype(np.float32),
                mode="homography",
                raw_match_count=raw,
                inlier_count=inliers,
                inlier_ratio=ratio,
                median_error=median,
                reference_points=reference_points,
                frame_points=frame_points,
            )
            return registration, {
                **debug,
                **_scene_model_distribution_debug(
                    registration.matrix,
                    mode=registration.mode,
                    source=selected_source,
                    reference_points=reference_points,
                    frame_points=frame_points,
                    frame_size=frame_size,
                    prefix="v2_scene_model",
                ),
                "v2_scene_registration_accepted": True,
                "v2_scene_registration_mode": registration.mode,
                "v2_scene_model_source": selected_source,
                "v2_scene_raw_matches": raw,
                "v2_scene_inliers": inliers,
                "v2_scene_inlier_ratio": _round_debug(ratio),
                "v2_scene_median_error": _round_debug(median),
            }
        debug["v2_homography_reject_reason"] = _scene_quality_reject_reason(
            raw=raw,
            inliers=inliers,
            ratio=ratio,
            median_error=median,
        )

    affine, affine_debug = _estimate_scene_affine(reference_points, frame_points)
    if affine is not None:
        registration = _SceneRegistration(
            matrix=affine,
            mode="affine",
            raw_match_count=int(affine_debug.get("raw_match_count") or 0),
            inlier_count=int(affine_debug.get("inlier_count") or 0),
            inlier_ratio=float(affine_debug.get("inlier_ratio") or 0.0),
            median_error=affine_debug.get("median_error"),
            reference_points=reference_points,
            frame_points=frame_points,
        )
        return registration, {
            **debug,
            **_scene_model_distribution_debug(
                registration.matrix,
                mode=registration.mode,
                source="affine_fallback",
                reference_points=reference_points,
                frame_points=frame_points,
                frame_size=frame_size,
                prefix="v2_scene_model",
            ),
            "v2_scene_registration_accepted": True,
            "v2_scene_registration_mode": registration.mode,
            "v2_scene_model_source": "affine_fallback",
            "v2_scene_raw_matches": registration.raw_match_count,
            "v2_scene_inliers": registration.inlier_count,
            "v2_scene_inlier_ratio": _round_debug(registration.inlier_ratio),
            "v2_scene_median_error": _round_debug(registration.median_error),
        }

    debug["v2_scene_registration_accepted"] = False
    debug["v2_scene_registration_reject_reason"] = affine_debug.get("reject_reason", "no_model")
    return None, debug


def _expand_matches_from_all_keypoints(
    projection_data: LocalProjectionData,
    *,
    seed_homography: np.ndarray,
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
    frame_size: tuple[int, int] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Use every available SuperPoint keypoint to densify the match set.

    LightGlue returns a sparse, high-confidence one-to-one subset.  That is good
    for a first homography, but bad for repeated hardware/fixture scenes: stable
    bolts, line ends and background marks may be detected as keypoints yet never
    become LightGlue matches.  Once we have a seed homography, project all
    reference keypoints and snap them to nearby frame keypoints.  These are still
    correspondences, not arbitrary points; they simply come from geometry-guided
    all-keypoint matching instead of descriptor-only LightGlue.
    """

    debug: dict[str, Any] = {
        "v2_all_keypoint_match_expansion_attempted": True,
        "v2_all_keypoint_match_radius_px": 7.5,
    }
    reference_keypoints = _as_points(
        projection_data.masked_reference_keypoints
        if projection_data.masked_reference_keypoints is not None
        else projection_data.original_reference_keypoints
    )
    frame_keypoints = _as_points(projection_data.frame_keypoints)
    base_reference = _empty_points() if reference_points is None else reference_points
    base_frame = _empty_points() if frame_points is None else frame_points
    debug["v2_all_keypoint_base_matches"] = int(len(base_reference))
    debug["v2_all_keypoint_reference_keypoints"] = (
        int(len(reference_keypoints)) if reference_keypoints is not None else 0
    )
    debug["v2_all_keypoint_frame_keypoints"] = int(len(frame_keypoints)) if frame_keypoints is not None else 0

    if reference_keypoints is None or frame_keypoints is None or frame_size is None:
        debug["v2_all_keypoint_match_expansion_reason"] = "missing_keypoints_or_frame_size"
        return reference_points, frame_points, debug

    guided_reference, guided_frame = _guided_keypoint_pairs(
        reference_keypoints=reference_keypoints,
        frame_keypoints=frame_keypoints,
        seed_homography=seed_homography,
        frame_size=frame_size,
        radius_px=7.5,
    )
    debug["v2_all_keypoint_guided_pairs"] = int(len(guided_reference))
    if len(guided_reference) == 0:
        debug["v2_all_keypoint_match_expansion_reason"] = "no_guided_pairs"
        return reference_points, frame_points, debug

    combined_reference = np.vstack([base_reference, guided_reference]).astype(np.float32)
    combined_frame = np.vstack([base_frame, guided_frame]).astype(np.float32)
    combined_reference, combined_frame = _dedupe_match_pairs(combined_reference, combined_frame)
    debug["v2_all_keypoint_total_matches"] = int(len(combined_reference))
    debug["v2_all_keypoint_added_matches"] = max(
        0, int(len(combined_reference)) - int(len(base_reference))
    )
    debug["v2_all_keypoint_match_expansion_reason"] = "ok"
    debug.update(
        _point_distribution_debug(
            guided_frame,
            frame_size=frame_size,
            prefix="v2_all_keypoint_guided_frame",
        )
    )
    return combined_reference, combined_frame, debug


def _guided_keypoint_pairs(
    *,
    reference_keypoints: np.ndarray,
    frame_keypoints: np.ndarray,
    seed_homography: np.ndarray,
    frame_size: tuple[int, int],
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = frame_size
    if width <= 0 or height <= 0 or len(reference_keypoints) == 0 or len(frame_keypoints) == 0:
        return _empty_points(), _empty_points()
    try:
        projected = cv2.perspectiveTransform(
            reference_keypoints.reshape(-1, 1, 2).astype(np.float32),
            seed_homography.astype(np.float32),
        ).reshape(-1, 2)
    except cv2.error:
        return _empty_points(), _empty_points()
    finite = np.isfinite(projected).all(axis=1)
    inside = (
        finite
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < float(width))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < float(height))
    )
    if not inside.any():
        return _empty_points(), _empty_points()

    cell_size = max(float(radius_px), 1.0)
    cells: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(frame_keypoints):
        if not np.isfinite(point).all():
            continue
        key = (int(point[0] // cell_size), int(point[1] // cell_size))
        cells.setdefault(key, []).append(index)

    candidates: list[tuple[float, int, int]] = []
    radius_sq = float(radius_px * radius_px)
    for ref_index, point in enumerate(projected):
        if not inside[ref_index]:
            continue
        cx = int(point[0] // cell_size)
        cy = int(point[1] // cell_size)
        best: tuple[float, int] | None = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for frame_index in cells.get((cx + dx, cy + dy), []):
                    delta = frame_keypoints[frame_index] - point
                    dist_sq = float(np.dot(delta, delta))
                    if dist_sq > radius_sq:
                        continue
                    if best is None or dist_sq < best[0]:
                        best = (dist_sq, frame_index)
        if best is not None:
            candidates.append((best[0], ref_index, best[1]))

    if not candidates:
        return _empty_points(), _empty_points()

    candidates.sort(key=lambda item: item[0])
    used_reference: set[int] = set()
    used_frame: set[int] = set()
    selected_reference: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []
    for _, ref_index, frame_index in candidates:
        if ref_index in used_reference or frame_index in used_frame:
            continue
        used_reference.add(ref_index)
        used_frame.add(frame_index)
        selected_reference.append(reference_keypoints[ref_index])
        selected_frame.append(frame_keypoints[frame_index])

    if not selected_reference:
        return _empty_points(), _empty_points()
    return (
        np.asarray(selected_reference, dtype=np.float32).reshape(-1, 2),
        np.asarray(selected_frame, dtype=np.float32).reshape(-1, 2),
    )


def _dedupe_match_pairs(
    reference_points: np.ndarray,
    frame_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference_points) == 0 or len(reference_points) != len(frame_points):
        return _empty_points(), _empty_points()
    selected_ref: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []
    seen: set[tuple[int, int, int, int]] = set()
    for ref_point, frame_point in zip(reference_points, frame_points, strict=False):
        if not np.isfinite(ref_point).all() or not np.isfinite(frame_point).all():
            continue
        key = (
            int(round(float(ref_point[0]) * 2.0)),
            int(round(float(ref_point[1]) * 2.0)),
            int(round(float(frame_point[0]) * 2.0)),
            int(round(float(frame_point[1]) * 2.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        selected_ref.append(ref_point.astype(np.float32, copy=False))
        selected_frame.append(frame_point.astype(np.float32, copy=False))
    if not selected_ref:
        return _empty_points(), _empty_points()
    return (
        np.asarray(selected_ref, dtype=np.float32).reshape(-1, 2),
        np.asarray(selected_frame, dtype=np.float32).reshape(-1, 2),
    )


def _refit_homography_from_matches(
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
) -> np.ndarray | None:
    if reference_points is None or frame_points is None:
        return None
    if len(reference_points) != len(frame_points) or len(reference_points) < 4:
        return None
    try:
        homography, _ = cv2.findHomography(
            reference_points,
            frame_points,
            cv2.RANSAC,
            5.5,
            maxIters=4000,
            confidence=0.995,
        )
    except cv2.error:
        return None
    if homography is None or homography.shape != (3, 3) or not np.isfinite(homography).all():
        return None
    return homography.astype(np.float32)


def _homography_candidate_key(inliers: int, median_error: float | None) -> tuple[int, float]:
    median_key = -float(median_error) if median_error is not None and np.isfinite(median_error) else -1e9
    return int(inliers), median_key


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _scene_quality_is_acceptable(
    *,
    raw: int,
    inliers: int,
    ratio: float,
    median_error: float | None,
) -> bool:
    if raw < _V2_MIN_SCENE_INLIERS:
        return False
    if inliers < _V2_MIN_SCENE_INLIERS:
        return False
    if median_error is None or not np.isfinite(float(median_error)):
        return False
    if float(median_error) > _V2_MAX_SCENE_MEDIAN_ERROR:
        return False
    if ratio < _V2_MIN_SCENE_INLIER_RATIO:
        return False
    return True


def _scene_quality_reject_reason(
    *,
    raw: int,
    inliers: int,
    ratio: float,
    median_error: float | None,
) -> str:
    if raw < _V2_MIN_SCENE_INLIERS:
        return "too_few_match_points"
    if inliers < _V2_MIN_SCENE_INLIERS:
        return "weak_scene_inliers"
    if median_error is None:
        return "scene_reprojection_error_missing"
    if median_error > _V2_MAX_SCENE_MEDIAN_ERROR:
        return "scene_reprojection_error_high"
    if ratio < _V2_MIN_SCENE_INLIER_RATIO:
        return "weak_scene_inlier_ratio"
    return "weak_scene_consensus"



def _scene_match_distribution_debug(
    *,
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
    frame_size: tuple[int, int] | None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"v2_scene_diag_version": "v89_report_only"}
    if reference_points is None or frame_points is None or len(reference_points) != len(frame_points):
        diagnostics.update(
            {
                "v2_scene_match_total": 0,
                "v2_scene_match_cells": 0,
                "v2_scene_match_bottom_count": 0,
                "v2_scene_match_top_count": 0,
                "v2_scene_match_left_count": 0,
                "v2_scene_match_right_count": 0,
            }
        )
        return diagnostics
    diagnostics.update(
        _point_distribution_debug(
            frame_points,
            frame_size=frame_size,
            prefix="v2_scene_match",
        )
    )
    diagnostics["v2_scene_reference_match_total"] = int(len(reference_points))
    return diagnostics


def _scene_model_distribution_debug(
    matrix: np.ndarray,
    *,
    mode: str,
    source: str,
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
    frame_size: tuple[int, int] | None,
    prefix: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {f"{prefix}_source": source}
    errors = _scene_projection_errors(
        matrix,
        mode=mode,
        reference_points=reference_points,
        frame_points=frame_points,
    )
    if errors is None or frame_points is None:
        diagnostics.update(
            {
                f"{prefix}_inlier_cells": 0,
                f"{prefix}_bottom_inliers": 0,
                f"{prefix}_top_inliers": 0,
                f"{prefix}_left_inliers": 0,
                f"{prefix}_right_inliers": 0,
            }
        )
        return diagnostics
    finite = np.isfinite(errors)
    inlier_mask = finite & (errors <= 6.5)
    inlier_points = frame_points[inlier_mask]
    diagnostics[f"{prefix}_finite_error_count"] = int(np.count_nonzero(finite))
    diagnostics[f"{prefix}_inlier_count"] = int(np.count_nonzero(inlier_mask))
    if finite.any():
        finite_errors = errors[finite]
        diagnostics[f"{prefix}_median_error"] = _round_debug(float(np.median(finite_errors)))
        diagnostics[f"{prefix}_p90_error"] = _round_debug(float(np.percentile(finite_errors, 90)))
    diagnostics.update(
        _point_distribution_debug(
            inlier_points,
            frame_size=frame_size,
            prefix=f"{prefix}_inlier",
        )
    )
    return diagnostics


def _scene_projection_errors(
    matrix: np.ndarray,
    *,
    mode: str,
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
) -> np.ndarray | None:
    if reference_points is None or frame_points is None or len(reference_points) != len(frame_points):
        return None
    if len(reference_points) == 0:
        return np.zeros((0,), dtype=np.float32)
    try:
        if mode == "homography":
            projected = cv2.perspectiveTransform(
                reference_points.reshape(-1, 1, 2),
                matrix.astype(np.float32),
            ).reshape(-1, 2)
        else:
            projected = cv2.transform(
                reference_points.reshape(-1, 1, 2),
                matrix.astype(np.float32),
            ).reshape(-1, 2)
    except cv2.error:
        return None
    return np.linalg.norm(projected - frame_points, axis=1)


def _point_distribution_debug(
    points: np.ndarray,
    *,
    frame_size: tuple[int, int] | None,
    prefix: str,
    grid_size: int = 4,
) -> dict[str, Any]:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    total = int(len(values))
    diagnostics: dict[str, Any] = {
        f"{prefix}_total": total,
        f"{prefix}_cells": 0,
        f"{prefix}_bottom_count": 0,
        f"{prefix}_top_count": 0,
        f"{prefix}_left_count": 0,
        f"{prefix}_right_count": 0,
        f"{prefix}_max_cell_fraction": None,
        f"{prefix}_span_x": None,
        f"{prefix}_span_y": None,
        f"{prefix}_hull_fraction": None,
    }
    if total == 0 or frame_size is None:
        return diagnostics
    width, height = frame_size
    if width <= 0 or height <= 0:
        return diagnostics

    x = np.clip(values[:, 0], 0.0, float(width - 1))
    y = np.clip(values[:, 1], 0.0, float(height - 1))
    gx = np.clip((x / max(float(width), 1.0) * grid_size).astype(int), 0, grid_size - 1)
    gy = np.clip((y / max(float(height), 1.0) * grid_size).astype(int), 0, grid_size - 1)
    cell_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    for cx, cy in zip(gx, gy, strict=False):
        cell_counts[int(cy), int(cx)] += 1

    nonzero_cells = int(np.count_nonzero(cell_counts))
    diagnostics[f"{prefix}_cells"] = nonzero_cells
    diagnostics[f"{prefix}_cell_counts"] = cell_counts.tolist()
    diagnostics[f"{prefix}_max_cell_fraction"] = _round_debug(
        float(cell_counts.max()) / max(float(total), 1.0)
    )
    diagnostics[f"{prefix}_span_x"] = _round_debug(
        float(x.max() - x.min()) / max(float(width), 1.0)
    )
    diagnostics[f"{prefix}_span_y"] = _round_debug(
        float(y.max() - y.min()) / max(float(height), 1.0)
    )
    diagnostics[f"{prefix}_top_count"] = int(np.count_nonzero(y <= height * 0.25))
    diagnostics[f"{prefix}_bottom_count"] = int(np.count_nonzero(y >= height * 0.75))
    diagnostics[f"{prefix}_left_count"] = int(np.count_nonzero(x <= width * 0.25))
    diagnostics[f"{prefix}_right_count"] = int(np.count_nonzero(x >= width * 0.75))
    diagnostics[f"{prefix}_hull_fraction"] = _round_debug(
        _point_hull_fraction(np.column_stack([x, y]), width=width, height=height)
    )
    return diagnostics


def _point_hull_fraction(points: np.ndarray, *, width: int, height: int) -> float | None:
    if len(points) < 3 or width <= 0 or height <= 0:
        return None
    try:
        hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
        area = float(cv2.contourArea(hull))
    except cv2.error:
        return None
    return area / max(float(width * height), 1.0)

def _estimate_scene_affine(
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if reference_points is None or frame_points is None:
        return None, {"reject_reason": "missing_match_points"}
    if len(reference_points) != len(frame_points) or len(reference_points) < _V2_MIN_SCENE_INLIERS:
        return None, {
            "reject_reason": "too_few_match_points",
            "raw_match_count": int(len(reference_points)),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "median_error": None,
        }

    try:
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            reference_points,
            frame_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.5,
            maxIters=1600,
            confidence=0.995,
            refineIters=10,
        )
    except cv2.error:
        return None, {"reject_reason": "estimate_affine_exception"}

    if affine is None or inlier_mask is None or affine.shape != (2, 3):
        return None, {"reject_reason": "estimate_affine_failed"}
    if not np.isfinite(affine).all():
        return None, {"reject_reason": "estimate_affine_non_finite"}

    mask = inlier_mask.reshape(-1).astype(bool)
    inliers = int(mask.sum())
    raw = int(len(reference_points))
    ratio = inliers / max(raw, 1)
    median = _affine_median_error(affine, reference_points[mask], frame_points[mask])
    if not _scene_quality_is_acceptable(
        raw=raw,
        inliers=inliers,
        ratio=ratio,
        median_error=median,
    ):
        return None, {
            "reject_reason": _scene_quality_reject_reason(
                raw=raw,
                inliers=inliers,
                ratio=ratio,
                median_error=median,
            ),
            "raw_match_count": raw,
            "inlier_count": inliers,
            "inlier_ratio": ratio,
            "median_error": median,
        }

    return affine.astype(np.float32), {
        "raw_match_count": raw,
        "inlier_count": inliers,
        "inlier_ratio": ratio,
        "median_error": median,
    }


def _project_expected_segments(
    expected: list[ExpectedSegment],
    *,
    registration: _SceneRegistration,
    frame_size: tuple[int, int] | None,
) -> list[_ProjectedSeed]:
    seeds: list[_ProjectedSeed] = []
    for index, item in enumerate(expected):
        polygon = _project_polygon(item.reference_polygon, registration=registration)
        bbox = bbox_from_polygon(polygon)
        if bbox is None:
            continue
        if frame_size is not None and not is_visible_in_frame(
            bbox,
            frame_size,
            min_visible_fraction=0.02,
        ):
            continue
        seeds.append(_ProjectedSeed(index=index, item=item, polygon=polygon, bbox=bbox))
    return seeds


def _refine_seed_with_context_registration(
    seed: _ProjectedSeed,
    *,
    projection_data: LocalProjectionData | None,
    registration: _SceneRegistration,
    warped_reference: np.ndarray | None,
    frame_size: tuple[int, int] | None,
    all_seed_polygons: list[PolygonPoints],
    budget: _TransferV2Budget,
) -> tuple[_LocalRefinement | None, dict[str, Any]]:
    debug: dict[str, Any] = {"v2_local_registration_attempted": False}
    if projection_data is None or projection_data.frame is None:
        debug["v2_reject_reason"] = "missing_frame"
        return None, debug
    if warped_reference is None:
        debug["v2_reject_reason"] = "missing_warped_reference"
        return None, debug
    if not budget.take():
        debug["v2_reject_reason"] = "budget_exhausted"
        return None, debug

    debug["v2_local_registration_attempted"] = True
    crop_rects = _candidate_crop_rects(seed.bbox, frame_size=frame_size)
    if not crop_rects:
        debug["v2_reject_reason"] = "empty_crop"
        return None, debug

    candidates: list[_LocalCandidate] = []
    crop_debug: list[dict[str, Any]] = []
    total_ecc_attempts = 0
    total_ecc_successes = 0

    for crop_mode, crop_rect in crop_rects:
        crop_candidates, crop_info = _build_local_candidates_for_crop(
            seed,
            projection_data=projection_data,
            registration=registration,
            warped_reference=warped_reference,
            frame_size=frame_size,
            all_seed_polygons=all_seed_polygons,
            crop_rect=crop_rect,
            crop_mode=crop_mode,
        )
        candidates.extend(crop_candidates)
        crop_debug.append(crop_info)
        total_ecc_attempts += int(crop_info.get("ecc_attempts") or 0)
        total_ecc_successes += int(crop_info.get("ecc_successes") or 0)

    debug["v2_candidate_crop_count"] = len(crop_rects)
    debug["v2_candidate_count"] = len(candidates)
    debug["v2_candidate_ecc_attempts"] = total_ecc_attempts
    debug["v2_candidate_ecc_successes"] = total_ecc_successes
    debug["v2_candidate_crop_debug"] = crop_debug[:4]

    if not candidates:
        debug["v2_reject_reason"] = "no_local_candidates"
        return None, debug

    candidate = _select_best_local_candidate(candidates)
    debug["v2_local_registration_source"] = candidate.source
    debug["v2_candidate_score"] = _round_debug(candidate.score)
    debug["v2_candidate_selection_score"] = _round_debug(candidate.selection_score)
    debug["v2_candidate_crop_mode"] = candidate.crop_mode
    debug["v2_candidate_start_mode"] = candidate.start_mode
    debug["v2_shift_factor"] = _round_debug(candidate.shift_factor)
    debug["v2_phase_response"] = _round_debug(candidate.phase_response)
    debug["v2_ecc_score"] = _round_debug(candidate.ecc_score)
    debug["v2_ring_fraction"] = _round_debug(candidate.ring_fraction)
    debug["v2_max_other_overlap"] = _round_debug(candidate.max_other_overlap)
    debug["v2_center_factor"] = _round_debug(candidate.center_factor)
    debug["v2_other_center_factor"] = _round_debug(candidate.other_center_factor)
    debug["v2_topology_margin"] = _round_debug(
        candidate.other_center_factor - candidate.center_factor
        if candidate.other_center_factor is not None
        else None
    )

    candidate_refinement = _LocalRefinement(
        polygon=candidate.polygon,
        bbox=candidate.bbox,
        shift_x=candidate.shift_x,
        shift_y=candidate.shift_y,
        shift_factor=candidate.shift_factor,
        ecc_score=candidate.ecc_score,
        phase_response=candidate.phase_response,
        ring_fraction=candidate.ring_fraction,
        max_other_overlap=candidate.max_other_overlap,
        center_factor=candidate.center_factor,
        other_center_factor=candidate.other_center_factor,
        source=candidate.source,
        candidate_score=candidate.score,
        candidate_selection_score=candidate.selection_score,
        crop_mode=candidate.crop_mode,
        start_mode=candidate.start_mode,
        candidate_count=len(candidates),
    )
    reject_reason = _v2_refinement_reject_reason(candidate_refinement)
    if reject_reason is not None:
        debug["v2_local_registration_accepted"] = False
        debug["v2_reject_reason"] = reject_reason
        return None, debug

    debug["v2_local_registration_accepted"] = True
    return candidate_refinement, debug


def _candidate_crop_rects(
    bbox: BBox,
    *,
    frame_size: tuple[int, int] | None,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    variants: list[tuple[str, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for mode, margin_factor, min_margin, max_margin in (
        ("normal", 0.95, 56.0, 220.0),
        ("wide", 1.55, 92.0, 340.0),
    ):
        rect = _expanded_crop_rect(
            bbox,
            frame_size=frame_size,
            margin_factor=margin_factor,
            min_margin=min_margin,
            max_margin=max_margin,
        )
        if rect is None or rect in seen:
            continue
        seen.add(rect)
        variants.append((mode, rect))
    return variants


def _build_local_candidates_for_crop(
    seed: _ProjectedSeed,
    *,
    projection_data: LocalProjectionData,
    registration: _SceneRegistration,
    warped_reference: np.ndarray,
    frame_size: tuple[int, int] | None,
    all_seed_polygons: list[PolygonPoints],
    crop_rect: tuple[int, int, int, int],
    crop_mode: str,
) -> tuple[list[_LocalCandidate], dict[str, Any]]:
    x1, y1, x2, y2 = crop_rect
    ref_crop = warped_reference[y1:y2, x1:x2]
    frame_crop = projection_data.frame[y1:y2, x1:x2]
    crop_info: dict[str, Any] = {
        "mode": crop_mode,
        "rect": [int(x1), int(y1), int(x2), int(y2)],
        "candidate_count": 0,
        "ecc_attempts": 0,
        "ecc_successes": 0,
    }
    if ref_crop.size == 0 or frame_crop.size == 0:
        crop_info["reject_reason"] = "empty_crop"
        return [], crop_info

    local_seed_polygon = _offset_polygon(seed.polygon, -float(x1), -float(y1))
    local_all_polygons = [
        _offset_polygon(polygon, -float(x1), -float(y1)) for polygon in all_seed_polygons
    ]
    ref_context = _erase_polygons(ref_crop, local_all_polygons)
    frame_context = _erase_polygons(frame_crop, local_all_polygons)
    ring_mask = _context_ring_mask(ref_crop.shape[:2], local_seed_polygon, local_all_polygons)
    ring_fraction = float(np.count_nonzero(ring_mask)) / max(float(ring_mask.size), 1.0)
    crop_info["ring_fraction"] = _round_debug(ring_fraction)

    ref_gray = _to_gray_float(ref_context)
    frame_gray = _to_gray_float(frame_context)
    ref_gray, frame_gray = _normalize_pair(ref_gray, frame_gray, ring_mask)

    phase_shift = _phase_translation(ref_gray, frame_gray, ring_mask)
    phase_dx, phase_dy, phase_response = phase_shift
    crop_info["phase_dx"] = _round_debug(phase_dx)
    crop_info["phase_dy"] = _round_debug(phase_dy)
    crop_info["phase_response"] = _round_debug(phase_response)

    candidates: list[_LocalCandidate] = []
    diagonal = max(bbox_diag(seed.bbox), 1.0)
    starts = _candidate_translation_starts(
        phase_dx=phase_dx,
        phase_dy=phase_dy,
        diagonal=diagonal,
    )
    crop_info["start_count"] = len(starts)

    # Always include the pure scene seed. This is not a safety fallback: it is a
    # normal candidate in the accuracy-ceiling search and competes by image score.
    seed_score = _masked_translation_score(ref_gray, frame_gray, ring_mask, 0.0, 0.0)
    seed_candidate = _make_local_candidate(
        seed,
        dx=0.0,
        dy=0.0,
        score=seed_score,
        ecc_score=None,
        phase_response=phase_response,
        ring_fraction=ring_fraction,
        crop_mode=crop_mode,
        start_mode="scene_seed",
        source="v2_context_seed",
        frame_size=frame_size,
        all_seed_polygons=all_seed_polygons,
    )
    if seed_candidate is not None:
        candidates.append(seed_candidate)

    if phase_dx is not None and phase_dy is not None:
        phase_score = _masked_translation_score(ref_gray, frame_gray, ring_mask, phase_dx, phase_dy)
        phase_candidate = _make_local_candidate(
            seed,
            dx=phase_dx,
            dy=phase_dy,
            score=phase_score,
            ecc_score=None,
            phase_response=phase_response,
            ring_fraction=ring_fraction,
            crop_mode=crop_mode,
            start_mode="phase_raw",
            source="v2_context_phase",
            frame_size=frame_size,
            all_seed_polygons=all_seed_polygons,
        )
        if phase_candidate is not None:
            candidates.append(phase_candidate)

    point_candidates, point_info = _build_local_point_residual_candidates(
        seed,
        registration=registration,
        frame_size=frame_size,
        all_seed_polygons=all_seed_polygons,
        crop_rect=crop_rect,
        crop_mode=crop_mode,
        ref_gray=ref_gray,
        frame_gray=frame_gray,
        ring_mask=ring_mask,
        ring_fraction=ring_fraction,
        phase_response=phase_response,
    )
    candidates.extend(point_candidates)
    crop_info.update(point_info)

    for start_mode, start_dx, start_dy in starts:
        crop_info["ecc_attempts"] += 1
        ecc = _run_translation_ecc(
            ref_gray,
            frame_gray,
            ring_mask,
            start_dx=start_dx,
            start_dy=start_dy,
        )
        if ecc is None:
            continue
        ecc_dx, ecc_dy, ecc_score = ecc
        crop_info["ecc_successes"] += 1
        candidate = _make_local_candidate(
            seed,
            dx=ecc_dx,
            dy=ecc_dy,
            score=ecc_score,
            ecc_score=ecc_score,
            phase_response=phase_response,
            ring_fraction=ring_fraction,
            crop_mode=crop_mode,
            start_mode=start_mode,
            source="v2_context_ecc",
            frame_size=frame_size,
            all_seed_polygons=all_seed_polygons,
        )
        if candidate is not None:
            candidates.append(candidate)

    crop_info["candidate_count"] = len(candidates)
    best = _select_best_local_candidate(candidates) if candidates else None
    if best is not None:
        crop_info["best_score"] = _round_debug(best.score)
        crop_info["best_source"] = best.source
        crop_info["best_start"] = best.start_mode
        crop_info["best_shift_factor"] = _round_debug(best.shift_factor)
    return candidates, crop_info




def _build_local_point_residual_candidates(
    seed: _ProjectedSeed,
    *,
    registration: _SceneRegistration,
    frame_size: tuple[int, int] | None,
    all_seed_polygons: list[PolygonPoints],
    crop_rect: tuple[int, int, int, int],
    crop_mode: str,
    ref_gray: np.ndarray,
    frame_gray: np.ndarray,
    ring_mask: np.ndarray,
    ring_fraction: float,
    phase_response: float | None,
) -> tuple[list[_LocalCandidate], dict[str, Any]]:
    """Build local translation candidates from scene match residuals around the slot.

    The global scene may already have hundreds of matches after v90 expansion, but
    local ECC can still choose a repeated-looking neighbour.  This candidate uses
    the matched keypoints themselves: project reference points through the scene
    model, keep pairs around this slot, take the robust residual frame - projected,
    and let that translation compete with ECC/phase candidates by image score.
    """

    info: dict[str, Any] = {
        "point_residual_attempted": True,
        "point_residual_pair_count": 0,
        "point_residual_inliers": 0,
    }
    reference_points = registration.reference_points
    frame_points = registration.frame_points
    if reference_points is None or frame_points is None or len(reference_points) != len(frame_points):
        info["point_residual_reject_reason"] = "missing_scene_match_points"
        return [], info
    if len(reference_points) < 2:
        info["point_residual_reject_reason"] = "too_few_scene_match_points"
        return [], info

    projected = _project_reference_points_array(reference_points, registration=registration)
    if projected is None:
        info["point_residual_reject_reason"] = "projection_failed"
        return [], info

    frame_search_rect = _inflate_rect(
        crop_rect,
        margin=max(24.0, min(96.0, bbox_diag(seed.bbox) * 0.30)),
        frame_size=frame_size,
    )
    projected_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    for projected_point, frame_point in zip(projected, frame_points, strict=False):
        if not np.isfinite(projected_point).all() or not np.isfinite(frame_point).all():
            continue
        if not _point_in_rect(projected_point, crop_rect):
            continue
        if not _point_in_rect(frame_point, frame_search_rect):
            continue
        # Use context points, not points that sit inside an expected object.  In a
        # missing-object scene these object-interior matches are exactly the ones
        # most likely to jump to a distractor copy.
        if _point_inside_any_polygon(projected_point, all_seed_polygons):
            continue
        if _point_inside_any_polygon(frame_point, all_seed_polygons):
            continue
        residual = frame_point.astype(np.float32) - projected_point.astype(np.float32)
        if not np.isfinite(residual).all():
            continue
        projected_values.append(projected_point.astype(np.float32, copy=False))
        residual_values.append(residual.astype(np.float32, copy=False))

    if not residual_values:
        info["point_residual_reject_reason"] = "no_context_pairs_in_crop"
        return [], info

    residuals = np.asarray(residual_values, dtype=np.float32).reshape(-1, 2)
    projected_points = np.asarray(projected_values, dtype=np.float32).reshape(-1, 2)
    info["point_residual_pair_count"] = int(len(residuals))
    info.update(
        _point_distribution_debug(
            projected_points,
            frame_size=frame_size,
            prefix="point_residual_projected",
        )
    )

    median = np.median(residuals, axis=0)
    distances = np.linalg.norm(residuals - median.reshape(1, 2), axis=1)
    inlier_threshold = max(6.0, min(18.0, bbox_diag(seed.bbox) * 0.08))
    inlier_mask = np.isfinite(distances) & (distances <= float(inlier_threshold))
    if int(np.count_nonzero(inlier_mask)) >= 2:
        robust = np.median(residuals[inlier_mask], axis=0)
        residual_error = float(np.median(distances[inlier_mask]))
    else:
        robust = median
        residual_error = float(np.median(distances)) if len(distances) else None
    inlier_count = int(np.count_nonzero(inlier_mask))
    info["point_residual_inliers"] = inlier_count
    info["point_residual_threshold_px"] = _round_debug(inlier_threshold)
    info["point_residual_error_px"] = _round_debug(residual_error)
    info["point_residual_dx"] = _round_debug(float(robust[0]))
    info["point_residual_dy"] = _round_debug(float(robust[1]))

    candidates: list[_LocalCandidate] = []
    # Raw point residual candidate: this is the direct "use the matched points"
    # translation.  It remains an accuracy candidate, not a guard.
    for mode, dx, dy in _dedupe_starts(
        [
            ("point_residual", float(robust[0]), float(robust[1])),
            ("point_half_residual", float(robust[0]) * 0.5, float(robust[1]) * 0.5),
        ]
    ):
        score = _masked_translation_score(ref_gray, frame_gray, ring_mask, dx, dy)
        candidate = _make_local_candidate(
            seed,
            dx=dx,
            dy=dy,
            score=score,
            ecc_score=None,
            phase_response=phase_response,
            ring_fraction=ring_fraction,
            crop_mode=crop_mode,
            start_mode=mode,
            source="v2_context_points",
            frame_size=frame_size,
            all_seed_polygons=all_seed_polygons,
            point_pair_count=int(len(residuals)),
            point_inlier_count=inlier_count,
            point_residual_error=residual_error,
        )
        if candidate is not None:
            candidates.append(candidate)

    info["point_residual_ecc_attempts"] = 0
    info["point_residual_ecc_successes"] = 0
    for mode, start_dx, start_dy in _dedupe_starts(
        [
            ("point_residual_ecc", float(robust[0]), float(robust[1])),
            ("point_half_residual_ecc", float(robust[0]) * 0.5, float(robust[1]) * 0.5),
        ]
    ):
        info["point_residual_ecc_attempts"] += 1
        ecc = _run_translation_ecc(
            ref_gray,
            frame_gray,
            ring_mask,
            start_dx=start_dx,
            start_dy=start_dy,
        )
        if ecc is None:
            continue
        ecc_dx, ecc_dy, ecc_score = ecc
        info["point_residual_ecc_successes"] += 1
        candidate = _make_local_candidate(
            seed,
            dx=ecc_dx,
            dy=ecc_dy,
            score=ecc_score,
            ecc_score=ecc_score,
            phase_response=phase_response,
            ring_fraction=ring_fraction,
            crop_mode=crop_mode,
            start_mode=mode,
            source="v2_context_points",
            frame_size=frame_size,
            all_seed_polygons=all_seed_polygons,
            point_pair_count=int(len(residuals)),
            point_inlier_count=inlier_count,
            point_residual_error=residual_error,
        )
        if candidate is not None:
            candidates.append(candidate)

    if candidates:
        best = _select_best_local_candidate(candidates)
        info["point_residual_candidate_count"] = len(candidates)
        info["point_residual_best_score"] = _round_debug(best.score)
        info["point_residual_best_shift_factor"] = _round_debug(best.shift_factor)
    else:
        info["point_residual_reject_reason"] = "candidate_build_failed"
    return candidates, info


def _project_reference_points_array(
    reference_points: np.ndarray,
    *,
    registration: _SceneRegistration,
) -> np.ndarray | None:
    if len(reference_points) == 0:
        return _empty_points()
    try:
        if registration.mode == "homography":
            projected = cv2.perspectiveTransform(
                reference_points.reshape(-1, 1, 2).astype(np.float32),
                registration.matrix.astype(np.float32),
            ).reshape(-1, 2)
        else:
            projected = cv2.transform(
                reference_points.reshape(-1, 1, 2).astype(np.float32),
                registration.matrix.astype(np.float32),
            ).reshape(-1, 2)
    except cv2.error:
        return None
    if not np.isfinite(projected).all():
        return None
    return projected.astype(np.float32)


def _point_in_rect(point: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return float(x1) <= float(point[0]) < float(x2) and float(y1) <= float(point[1]) < float(y2)


def _inflate_rect(
    rect: tuple[int, int, int, int],
    *,
    margin: float,
    frame_size: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    m = max(0, int(round(float(margin))))
    if frame_size is None:
        return int(x1 - m), int(y1 - m), int(x2 + m), int(y2 + m)
    width, height = frame_size
    return (
        max(0, int(x1 - m)),
        max(0, int(y1 - m)),
        min(int(width), int(x2 + m)),
        min(int(height), int(y2 + m)),
    )


def _point_inside_any_polygon(point: np.ndarray, polygons: list[PolygonPoints]) -> bool:
    x = float(point[0])
    y = float(point[1])
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        try:
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                return True
        except cv2.error:
            continue
    return False

def _phase_translation(
    reference: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    try:
        window = (mask.astype(np.float32) / 255.0)
        (shift_x, shift_y), response = cv2.phaseCorrelate(reference * window, frame * window)
    except cv2.error:
        return None, None, None
    if not np.isfinite(shift_x) or not np.isfinite(shift_y):
        return None, None, None
    return float(shift_x), float(shift_y), float(response) if np.isfinite(response) else None


def _candidate_translation_starts(
    *,
    phase_dx: float | None,
    phase_dy: float | None,
    diagonal: float,
) -> list[tuple[str, float, float]]:
    step = float(np.clip(diagonal * 0.075, 5.0, 24.0))
    starts: list[tuple[str, float, float]] = [
        ("zero", 0.0, 0.0),
        ("grid_pos_x", step, 0.0),
        ("grid_neg_x", -step, 0.0),
        ("grid_pos_y", 0.0, step),
        ("grid_neg_y", 0.0, -step),
    ]
    if phase_dx is not None and phase_dy is not None:
        starts.insert(1, ("phase", phase_dx, phase_dy))
    return _dedupe_starts(starts)


def _dedupe_starts(starts: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    result: list[tuple[str, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for mode, dx, dy in starts:
        key = (int(round(dx * 10.0)), int(round(dy * 10.0)))
        if key in seen:
            continue
        seen.add(key)
        result.append((mode, float(dx), float(dy)))
    return result


def _run_translation_ecc(
    reference: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    start_dx: float,
    start_dy: float,
) -> tuple[float, float, float] | None:
    try:
        warp = np.asarray([[1.0, 0.0, start_dx], [0.0, 1.0, start_dy]], dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            _V2_ECC_MAX_ITERATIONS,
            _V2_ECC_EPS,
        )
        score, warp = cv2.findTransformECC(
            reference,
            frame,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None
    if not np.isfinite(score) or not np.isfinite(warp).all():
        return None
    return float(warp[0, 2]), float(warp[1, 2]), float(score)


def _make_local_candidate(
    seed: _ProjectedSeed,
    *,
    dx: float,
    dy: float,
    score: float | None,
    ecc_score: float | None,
    phase_response: float | None,
    ring_fraction: float,
    crop_mode: str,
    start_mode: str,
    source: str,
    frame_size: tuple[int, int] | None,
    all_seed_polygons: list[PolygonPoints],
    point_pair_count: int = 0,
    point_inlier_count: int = 0,
    point_residual_error: float | None = None,
) -> _LocalCandidate | None:
    if not np.isfinite(dx) or not np.isfinite(dy):
        return None
    polygon = translate_polygon(seed.polygon, dx=dx, dy=dy)
    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return None
    if frame_size is not None and not is_visible_in_frame(
        bbox,
        frame_size,
        min_visible_fraction=_V2_MIN_VISIBLE_FRACTION,
    ):
        return None
    diagonal = max(bbox_diag(seed.bbox), 1.0)
    center_factor = bbox_center_distance_factor(bbox, seed.bbox)
    other_center_factor = _nearest_other_seed_center_factor(
        seed.index,
        bbox,
        all_seed_polygons,
        seed_bbox=seed.bbox,
    )
    return _LocalCandidate(
        polygon=polygon,
        bbox=bbox,
        shift_x=float(dx),
        shift_y=float(dy),
        shift_factor=float(np.hypot(dx, dy) / diagonal),
        ecc_score=ecc_score,
        phase_response=phase_response,
        ring_fraction=ring_fraction,
        max_other_overlap=_max_other_seed_overlap(seed.index, bbox, all_seed_polygons),
        center_factor=float(center_factor),
        other_center_factor=other_center_factor,
        source=source,
        score=score if score is None else float(score),
        selection_score=None,
        crop_mode=crop_mode,
        start_mode=start_mode,
        point_pair_count=int(point_pair_count),
        point_inlier_count=int(point_inlier_count),
        point_residual_error=(
            float(point_residual_error)
            if point_residual_error is not None and np.isfinite(point_residual_error)
            else None
        ),
    )


def _select_best_local_candidate(candidates: list[_LocalCandidate]) -> _LocalCandidate:
    for candidate in candidates:
        candidate.selection_score = _local_candidate_selection_score(candidate)

    def key(candidate: _LocalCandidate) -> tuple[float, float, int, float]:
        selection = candidate.selection_score
        selection_key = (
            float(selection)
            if selection is not None and np.isfinite(selection)
            else -1_000_000.0
        )
        raw_score = candidate.score
        raw_score_key = (
            float(raw_score)
            if raw_score is not None and np.isfinite(raw_score)
            else -1_000_000.0
        )
        source_bonus = 2 if candidate.source == "v2_context_points" else (1 if candidate.source == "v2_context_ecc" else 0)
        return selection_key, raw_score_key, source_bonus, -candidate.shift_factor

    return max(candidates, key=key)


def _local_candidate_selection_score(candidate: _LocalCandidate) -> float | None:
    """Rank local candidates without rejecting any of them.

    Raw ECC/NCC scores are not comparable enough on their own: a far-away
    wrong alignment can get an excellent image score on repeated context.  This
    is not a safety guard; it is only the ordering function for candidates.
    The scene-projected seed is still the prior, so a candidate must buy its
    displacement with a better image score.
    """

    score = candidate.score
    if score is None or not np.isfinite(score):
        return None
    shift = float(candidate.shift_factor) if np.isfinite(candidate.shift_factor) else 1_000_000.0

    # A repeated neighbour can produce a high NCC/ECC score. Keep this as ranking
    # only: large motion is still allowed, but it must beat the scene-projected
    # slot by a much larger image-score margin.
    shift_penalty = 0.16 * min(max(shift, 0.0), 1.0)
    shift_penalty += 0.34 * min(max(shift - 1.0, 0.0), 3.0)

    # Multi-object cases are where v92 still fails most often: the local candidate
    # may jump toward a different expected slot/distractor. Penalize that ordering
    # mistake without rejecting the candidate outright.
    topology_penalty = 0.0
    if np.isfinite(candidate.max_other_overlap):
        topology_penalty += 0.30 * min(max(candidate.max_other_overlap, 0.0), 1.0)
    if candidate.other_center_factor is not None and np.isfinite(candidate.other_center_factor):
        center_factor = (
            float(candidate.center_factor)
            if np.isfinite(candidate.center_factor)
            else 1_000_000.0
        )
        other_factor = max(float(candidate.other_center_factor), 0.0)
        if other_factor < center_factor:
            topology_penalty += 0.22 * min(center_factor - other_factor, 3.0)
        if other_factor < 0.35 and center_factor > 0.35:
            topology_penalty += 0.10 * min(0.35 - other_factor, 0.35) / 0.35

    source_bonus = 0.0
    if candidate.source == "v2_context_ecc":
        # After point-residual candidates exist, plain ECC is mostly a last-resort
        # image matcher and should not get an artificial boost.
        source_bonus = -0.020
    if candidate.source == "v2_context_points":
        pair_bonus = min(max(float(candidate.point_pair_count), 0.0), 32.0) / 32.0
        inlier_bonus = min(max(float(candidate.point_inlier_count), 0.0), 16.0) / 16.0
        point_bonus = 0.050 + 0.060 * inlier_bonus + 0.020 * pair_bonus
        if candidate.ecc_score is not None:
            point_bonus += 0.010
        if candidate.point_residual_error is not None:
            point_bonus -= 0.014 * min(max(candidate.point_residual_error / 10.0, 0.0), 2.0)
        if candidate.point_inlier_count < 2:
            point_bonus -= 0.050
        source_bonus = max(source_bonus, point_bonus)

    crop_penalty = 0.015 if candidate.crop_mode == "wide" else 0.0
    if candidate.source == "v2_context_points":
        # Good point residuals often involve a real local correction; keep the
        # topology prior, but do not over-penalize coherent point motion.
        shift_penalty *= 0.75
    return float(score) + source_bonus - shift_penalty - topology_penalty - crop_penalty


def _masked_translation_score(
    reference: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    dx: float,
    dy: float,
) -> float | None:
    if reference.size == 0 or frame.size == 0:
        return None
    try:
        warp = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted_reference = cv2.warpAffine(
            reference,
            warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        shifted_mask = cv2.warpAffine(
            mask,
            warp,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error:
        return None
    valid = (shifted_mask > 0) & (mask > 0)
    if int(np.count_nonzero(valid)) < 16:
        return None
    ref_values = shifted_reference[valid].astype(np.float32)
    frame_values = frame[valid].astype(np.float32)
    ref_values = ref_values - float(np.mean(ref_values))
    frame_values = frame_values - float(np.mean(frame_values))
    denom = float(np.linalg.norm(ref_values) * np.linalg.norm(frame_values))
    if denom <= 1e-6:
        return None
    score = float(np.dot(ref_values, frame_values) / denom)
    if not np.isfinite(score):
        return None
    return score


def _warp_reference_to_frame(
    projection_data: LocalProjectionData | None,
    *,
    registration: _SceneRegistration,
    frame_size: tuple[int, int] | None,
) -> np.ndarray | None:
    if projection_data is None or projection_data.reference_frame is None or frame_size is None:
        return None
    width, height = frame_size
    if registration.mode == "homography":
        return cv2.warpPerspective(
            projection_data.reference_frame,
            registration.matrix.astype(np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return cv2.warpAffine(
        projection_data.reference_frame,
        registration.matrix.astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _project_polygon(
    polygon: PolygonPoints,
    *,
    registration: _SceneRegistration,
) -> PolygonPoints:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 3:
        return []
    if registration.mode == "homography":
        projected = cv2.perspectiveTransform(points, registration.matrix.astype(np.float32))
    else:
        projected = cv2.transform(points, registration.matrix.astype(np.float32))
    values = projected.reshape(-1, 2)
    if not np.isfinite(values).all():
        return []
    return [[float(x), float(y)] for x, y in values]


def _as_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(values) == 0 or not np.isfinite(values).all():
        return None
    return values


def _valid_homography(matrix: np.ndarray) -> bool:
    values = np.asarray(matrix, dtype=np.float32)
    return values.shape == (3, 3) and np.isfinite(values).all()


def _homography_quality(
    homography: np.ndarray,
    *,
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
) -> tuple[int, int, float, float | None]:
    if reference_points is None or frame_points is None or len(reference_points) != len(frame_points):
        return 0, 0, 0.0, None
    if len(reference_points) == 0:
        return 0, 0, 0.0, None
    projected = cv2.perspectiveTransform(
        reference_points.reshape(-1, 1, 2),
        homography.astype(np.float32),
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - frame_points, axis=1)
    finite = np.isfinite(errors)
    if not finite.any():
        return int(len(reference_points)), 0, 0.0, None
    errors = errors[finite]
    inliers = int(np.count_nonzero(errors <= 6.5))
    raw = int(len(reference_points))
    ratio = inliers / max(raw, 1)
    median = float(np.median(errors)) if len(errors) else None
    return raw, inliers, ratio, median


def _affine_median_error(
    affine: np.ndarray,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
) -> float | None:
    if len(reference_points) == 0:
        return None
    projected = cv2.transform(reference_points.reshape(-1, 1, 2), affine).reshape(-1, 2)
    errors = np.linalg.norm(projected - frame_points, axis=1)
    if not np.isfinite(errors).any():
        return None
    return float(np.median(errors[np.isfinite(errors)]))


def _expanded_crop_rect(
    bbox: BBox,
    *,
    frame_size: tuple[int, int] | None,
    margin_factor: float = 0.95,
    min_margin: float = 56.0,
    max_margin: float = 220.0,
) -> tuple[int, int, int, int] | None:
    if frame_size is None:
        return None
    width, height = frame_size
    x1, y1, x2, y2 = bbox
    margin = float(np.clip(bbox_diag(bbox) * margin_factor, min_margin, max_margin))
    left = max(0, int(np.floor(x1 - margin)))
    top = max(0, int(np.floor(y1 - margin)))
    right = min(width, int(np.ceil(x2 + margin)))
    bottom = min(height, int(np.ceil(y2 + margin)))
    if right - left < 24 or bottom - top < 24:
        return None
    return left, top, right, bottom


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float32)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _normalize_pair(
    reference: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask > 0
    if np.count_nonzero(valid) < 16:
        return reference, frame
    ref_values = reference[valid]
    frame_values = frame[valid]
    ref_mean = float(np.mean(ref_values))
    frame_mean = float(np.mean(frame_values))
    ref_std = float(np.std(ref_values)) or 1.0
    frame_std = float(np.std(frame_values)) or 1.0
    reference = (reference - ref_mean) / ref_std
    frame = (frame - frame_mean) / frame_std
    return reference.astype(np.float32), frame.astype(np.float32)


def _erase_polygons(image: np.ndarray, polygons: list[PolygonPoints]) -> np.ndarray:
    result = image.copy()
    fill = _median_color(result)
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(result, [pts], fill)
    return result


def _context_ring_mask(
    shape_hw: tuple[int, int],
    seed_polygon: PolygonPoints,
    all_polygons: list[PolygonPoints],
) -> np.ndarray:
    height, width = shape_hw
    mask = np.full((height, width), 255, dtype=np.uint8)
    for polygon in all_polygons:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 0)
    seed_pts = np.asarray(seed_polygon, dtype=np.int32).reshape(-1, 1, 2)
    if len(seed_pts) >= 3:
        dilated = np.zeros_like(mask)
        cv2.fillPoly(dilated, [seed_pts], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated = cv2.dilate(dilated, kernel, iterations=1)
        mask[dilated > 0] = 0
    return mask


def _median_color(image: np.ndarray) -> tuple[int, ...]:
    if image.ndim == 2:
        return (int(np.median(image)),)
    values = np.median(image.reshape(-1, image.shape[-1]), axis=0)
    return tuple(int(np.clip(v, 0, 255)) for v in values)


def _offset_polygon(polygon: PolygonPoints, dx: float, dy: float) -> PolygonPoints:
    return [[float(x) + dx, float(y) + dy] for x, y in polygon]


def _max_other_seed_overlap(
    current_index: int,
    bbox: BBox,
    all_seed_polygons: list[PolygonPoints],
) -> float:
    best = 0.0
    for index, polygon in enumerate(all_seed_polygons):
        if index == current_index:
            continue
        other_bbox = bbox_from_polygon(polygon)
        if other_bbox is None:
            continue
        best = max(best, float(bbox_iou(bbox, other_bbox)))
    return best


def _nearest_other_seed_center_factor(
    current_index: int,
    bbox: BBox,
    all_seed_polygons: list[PolygonPoints],
    *,
    seed_bbox: BBox,
) -> float | None:
    if len(all_seed_polygons) <= 1:
        return None
    cx, cy = _bbox_center_xy(bbox)
    diagonal = max(bbox_diag(seed_bbox), 1.0)
    best: float | None = None
    for index, polygon in enumerate(all_seed_polygons):
        if index == current_index:
            continue
        other_bbox = bbox_from_polygon(polygon)
        if other_bbox is None:
            continue
        ox, oy = _bbox_center_xy(other_bbox)
        factor = float(np.hypot(cx - ox, cy - oy) / diagonal)
        if not np.isfinite(factor):
            continue
        best = factor if best is None else min(best, factor)
    return best


def _bbox_center_xy(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def _v2_match(
    item: ExpectedSegment,
    *,
    status: str,
    polygon: PolygonPoints | None,
    debug: dict[str, Any],
) -> SegmentMatch:
    return SegmentMatch(
        annotation_id=item.annotation_id,
        segment_class_id=item.segment_class_id,
        class_key=item.class_key,
        name=item.name,
        hue=item.hue,
        status=status,
        iou=None,
        confidence=None,
        expected_polygon=polygon,
        detected_polygon=None,
        detected_bbox=None,
        debug=debug,
    )


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


def _round_debug(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
