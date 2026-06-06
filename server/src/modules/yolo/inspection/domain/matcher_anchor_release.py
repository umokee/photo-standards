from __future__ import annotations

from typing import Any

import numpy as np
from modules.yolo.inspection.domain.alignment import LocalProjectionData
from modules.yolo.inspection.domain.matcher_anchor_diagnostics import (
    set_anchor_release_reject,
    trusted_anchor_source_counts,
    update_anchor_release_debug,
)
from modules.yolo.inspection.domain.matcher_anchor_utils import (
    _trusted_anchor_source_weight,
)
from modules.yolo.inspection.domain.matcher_debug import _round_debug
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_center,
    bbox_center_distance_factor,
    bbox_containment,
    bbox_diag,
    bbox_from_polygon,
    bbox_iou,
    is_visible_in_frame,
    translate_bbox,
    translate_polygon,
)
from modules.yolo.inspection.domain.matcher_structs import (
    AnchorOverlapDecision,
    AnchorReleaseConsensus,
    BBox,
    ExpectedSlot,
    MissingTranslationRescue,
    ProjectedExpected,
    TrustedAnchor,
)
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds


def _try_missing_anchor_release(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    fallback_bbox: BBox | None,
    all_expected: list[ProjectedExpected] | None = None,
    all_slots: dict[int, ExpectedSlot] | None = None,
    trusted_anchors: list[TrustedAnchor] | None = None,
    reject_debug: dict[str, Any] | None = None,
) -> MissingTranslationRescue | None:
    update_anchor_release_debug(
        reject_debug,
        attempted=True,
        total_anchors=len(trusted_anchors or []),
    )

    def reject(reason: str, **fields: Any) -> None:
        set_anchor_release_reject(reject_debug, reason, **fields)
        return None

    if (
        slot is None
        or projection_data is None
        or projection_data.global_homography is None
    ):
        return reject("anchor_release_rejected_unavailable_inputs")
    if not all_expected or len(all_expected) <= 1 or not all_slots:
        return reject("anchor_release_rejected_not_multi_object")
    if not trusted_anchors:
        return reject("anchor_release_rejected_no_trusted_anchors")

    min_anchors = max(1, int(_thresholds.missing_anchor_release_min_anchors))
    max_anchor_error = float(_thresholds.missing_anchor_release_max_anchor_error)
    current_center = np.asarray(bbox_center(global_bbox), dtype=np.float32)
    current_diag = max(1.0, bbox_diag(global_bbox))
    max_neighbor_distance = (
        _thresholds.missing_anchor_release_max_neighbor_distance_factor * current_diag
    )

    all_candidates: list[tuple[TrustedAnchor, np.ndarray, float, float]] = []
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
        if residual_shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
            rejected_shift += 1
            continue

        overlap_with_current = max(
            bbox_iou(anchor.local_bbox, global_bbox),
            bbox_containment(anchor.local_bbox, global_bbox),
            bbox_containment(global_bbox, anchor.local_bbox),
        )
        if overlap_with_current > _thresholds.missing_anchor_release_max_other_overlap:
            rejected_overlap += 1
            continue

        anchor_center = np.asarray(bbox_center(anchor.global_bbox), dtype=np.float32)
        distance = float(np.linalg.norm(anchor_center - current_center))
        distance_weight = 1.0 / max(1.0, distance)
        source_weight = _trusted_anchor_source_weight(anchor.source)
        weight = max(0.01, float(anchor.weight) * distance_weight * source_weight)
        all_candidates.append((anchor, residual, weight, distance))
        candidate_sources[anchor.source] = candidate_sources.get(anchor.source, 0) + 1

    update_anchor_release_debug(
        reject_debug,
        candidate_count=len(all_candidates),
        rejected_self=rejected_self,
        rejected_bad_residual=rejected_bad_residual,
        rejected_shift=rejected_shift,
        rejected_overlap=rejected_overlap,
        candidate_sources=candidate_sources,
    )

    if not all_candidates:
        return reject("anchor_release_rejected_no_candidates")

    strong_candidates = [
        item
        for item in all_candidates
        if item[0].source != "weak_local_global_slot_hint"
    ]
    candidate_pool = (
        strong_candidates if len(strong_candidates) >= min_anchors else all_candidates
    )
    update_anchor_release_debug(
        reject_debug,
        strong_candidate_count=len(strong_candidates),
        weak_candidate_count=len(all_candidates) - len(strong_candidates),
        weak_candidate_pool_used=(candidate_pool is all_candidates),
    )

    neighbor_candidates = [
        (anchor, residual, weight)
        for anchor, residual, weight, distance in candidate_pool
        if distance <= max_neighbor_distance
    ]
    update_anchor_release_debug(
        reject_debug,
        neighbor_candidate_count=len(neighbor_candidates),
        max_neighbor_distance=_round_debug(max_neighbor_distance),
    )
    if len(neighbor_candidates) >= min_anchors:
        candidates = neighbor_candidates
    else:
        candidates = [
            (anchor, residual, weight) for anchor, residual, weight, _ in candidate_pool
        ]

    update_anchor_release_debug(
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
            return reject(
                "anchor_release_rejected_single_anchor_guard",
                candidates=len(candidates),
                min_anchors=min_anchors,
            )
        min_anchors = 1

    consensus = _select_anchor_release_consensus(
        candidates,
        min_anchors=min_anchors,
        max_anchor_error=max_anchor_error,
        global_bbox=global_bbox,
    )
    if consensus is None:
        return reject(
            "anchor_release_rejected_high_dispersion",
            candidates=len(candidates),
            min_anchors=min_anchors,
        )

    update_anchor_release_debug(
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
    source_counts = trusted_anchor_source_counts(inlier_anchors)
    weak_slot_anchor_count = sum(
        1 for anchor in inlier_anchors if anchor.source == "weak_local_global_slot_hint"
    )
    weak_slot_only = weak_slot_anchor_count > 0 and weak_slot_anchor_count == len(
        inlier_anchors
    )
    if weak_slot_only:
        if (
            weak_slot_anchor_count
            < _thresholds.missing_anchor_release_weak_slot_min_inliers
        ):
            return reject(
                "anchor_release_rejected_weak_slot_guard",
                weak_slot_anchors=weak_slot_anchor_count,
            )
        if (
            consensus.inlier_ratio
            < _thresholds.missing_anchor_release_weak_slot_min_inlier_ratio
        ):
            return reject(
                "anchor_release_rejected_weak_slot_guard",
                weak_slot_inlier_ratio=_round_debug(consensus.inlier_ratio),
            )
        if (
            consensus.dispersion
            > _thresholds.missing_anchor_release_weak_slot_max_dispersion_factor
        ):
            return reject(
                "anchor_release_rejected_weak_slot_guard",
                weak_slot_dispersion=_round_debug(consensus.dispersion),
            )
    update_anchor_release_debug(
        reject_debug,
        weak_slot_anchor_count=weak_slot_anchor_count,
        weak_slot_only=weak_slot_only,
    )
    source_counts = trusted_anchor_source_counts(inlier_anchors)
    single_anchor = consensus.inlier_count == 1
    median_error = consensus.median_error
    if single_anchor:
        anchor_median_error = max(anchor.median_error for anchor in inlier_anchors)
        median_error = max(median_error, float(anchor_median_error))
    if median_error > max_anchor_error:
        return reject(
            "anchor_release_rejected_high_dispersion",
            median_error=_round_debug(median_error),
            max_anchor_error=_round_debug(max_anchor_error),
        )

    shift = consensus.shift
    shift_x = float(shift[0])
    shift_y = float(shift[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / current_diag
    update_anchor_release_debug(
        reject_debug,
        shift_x=_round_debug(shift_x),
        shift_y=_round_debug(shift_y),
        shift_factor=_round_debug(shift_factor),
    )
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return reject(
            "anchor_release_rejected_large_shift",
            shift_factor=_round_debug(shift_factor),
        )

    polygon = translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        return reject("anchor_release_rejected_bad_polygon")

    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return reject("anchor_release_rejected_bad_bbox")

    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=_thresholds.missing_anchor_release_min_visible_fraction,
    ):
        return reject("anchor_release_rejected_bad_overlap")

    if fallback_bbox is not None:
        local_area_score = bbox_area_similarity(bbox, fallback_bbox)
        local_center_factor = bbox_center_distance_factor(bbox, fallback_bbox)
        update_anchor_release_debug(
            reject_debug,
            local_area_score=_round_debug(local_area_score),
            local_center_factor=_round_debug(local_center_factor),
        )
        if local_area_score < _thresholds.missing_anchor_release_min_local_area_score:
            return reject(
                "anchor_release_rejected_bad_overlap",
                local_area_score=_round_debug(local_area_score),
            )
        if (
            local_center_factor
            > _thresholds.missing_anchor_release_max_local_center_factor
        ):
            return reject(
                "anchor_release_rejected_bad_overlap",
                local_center_factor=_round_debug(local_center_factor),
            )

        if single_anchor:
            single_min_area = max(
                _thresholds.missing_anchor_release_min_local_area_score,
                _thresholds.missing_anchor_release_single_min_local_area_score,
            )
            single_max_center = min(
                _thresholds.missing_anchor_release_max_local_center_factor,
                _thresholds.missing_anchor_release_single_max_local_center_factor,
            )
            if local_area_score < single_min_area:
                return reject(
                    "anchor_release_rejected_single_anchor_guard",
                    local_area_score=_round_debug(local_area_score),
                )
            if local_center_factor > single_max_center:
                return reject(
                    "anchor_release_rejected_single_anchor_guard",
                    local_center_factor=_round_debug(local_center_factor),
                )

    if (
        single_anchor
        and shift_factor > _thresholds.missing_anchor_release_single_max_shift_factor
    ):
        return reject(
            "anchor_release_rejected_single_anchor_guard",
            shift_factor=_round_debug(shift_factor),
        )

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        reference_area_score = bbox_area_similarity(bbox, reference_bbox)
        update_anchor_release_debug(
            reject_debug,
            reference_area_score=_round_debug(reference_area_score),
        )
        if reference_area_score < 0.12:
            return reject(
                "anchor_release_rejected_bad_overlap",
                reference_area_score=_round_debug(reference_area_score),
            )

    overlap_decision = _anchor_release_overlap_decision(
        expected_item,
        bbox=bbox,
        global_bbox=global_bbox,
        fallback_bbox=fallback_bbox,
        all_expected=all_expected,
        trusted_anchors=trusted_anchors,
        consensus=consensus,
        inlier_anchors=inlier_anchors,
        single_anchor=single_anchor,
        shift_factor=shift_factor,
    )
    update_anchor_release_debug(
        reject_debug,
        overlap_count=overlap_decision.overlap_count,
        overlap_unresolved_count=overlap_decision.unresolved_count,
        overlap_strong_anchor_count=overlap_decision.strong_anchor_count,
        overlap_max=_round_debug(overlap_decision.max_overlap),
        overlap_resolved_max=_round_debug(overlap_decision.max_resolved_overlap),
    )
    if not overlap_decision.allowed:
        return reject(
            overlap_decision.reason or "anchor_release_rejected_bad_overlap",
        )

    return MissingTranslationRescue(
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
    )


def _select_anchor_release_consensus(
    candidates: list[tuple[TrustedAnchor, np.ndarray, float]],
    *,
    min_anchors: int,
    max_anchor_error: float,
    global_bbox: BBox,
) -> AnchorReleaseConsensus | None:
    residual_array = np.asarray([item[1] for item in candidates], dtype=np.float32)
    weights_array = np.asarray([item[2] for item in candidates], dtype=np.float32)
    if residual_array.ndim != 2 or residual_array.shape[1] != 2:
        return None
    if not np.isfinite(residual_array).all():
        return None
    if not np.isfinite(weights_array).all() or float(np.sum(weights_array)) <= 0.0:
        return None

    min_local_ratio = float(_thresholds.missing_anchor_release_local_min_inlier_ratio)
    best: AnchorReleaseConsensus | None = None
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
) -> AnchorReleaseConsensus | None:
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
        / max(1.0, bbox_diag(global_bbox))
    )
    if dispersion > _thresholds.missing_anchor_release_max_dispersion_factor:
        return None

    return AnchorReleaseConsensus(
        shift=shift,
        inlier_mask=inlier_mask,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / max(1, candidate_count),
        median_error=median_error,
        dispersion=dispersion,
    )


def _anchor_release_allows_single_anchor(
    candidates: list[tuple[TrustedAnchor, np.ndarray, float]],
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
    fallback_bbox: BBox | None,
) -> bool:
    if len(candidates) != 1:
        return False

    anchor, residual, _ = candidates[0]
    if anchor.source in {"local_global_slot_hint", "weak_local_global_slot_hint"}:
        return False
    if anchor.inlier_count < _thresholds.missing_anchor_release_single_min_inliers:
        return False
    if anchor.inlier_ratio < _thresholds.missing_anchor_release_single_min_inlier_ratio:
        return False
    if anchor.median_error > _thresholds.missing_anchor_release_single_max_anchor_error:
        return False

    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    if shift_factor > _thresholds.missing_anchor_release_single_max_shift_factor:
        return False

    polygon_bbox = translate_bbox(
        global_bbox, dx=float(residual[0]), dy=float(residual[1])
    )
    local_bbox = fallback_bbox or slot.projected_bbox
    local_area_score = bbox_area_similarity(polygon_bbox, local_bbox)
    local_center_factor = bbox_center_distance_factor(polygon_bbox, local_bbox)

    if (
        local_area_score
        < _thresholds.missing_anchor_release_single_min_local_area_score
    ):
        return False
    if (
        local_center_factor
        > _thresholds.missing_anchor_release_single_max_local_center_factor
    ):
        return False

    return True


def _anchor_overlap_result(
    *,
    allowed: bool,
    reason: str | None,
    overlap_count: int = 0,
    unresolved_count: int = 0,
    strong_anchor_count: int = 0,
    max_overlap: float = 0.0,
    max_resolved_overlap: float = 0.0,
) -> AnchorOverlapDecision:
    return AnchorOverlapDecision(
        allowed=allowed,
        reason=reason,
        overlap_count=overlap_count,
        unresolved_count=unresolved_count,
        strong_anchor_count=strong_anchor_count,
        max_overlap=max_overlap,
        max_resolved_overlap=max_resolved_overlap,
    )


def _anchor_release_overlap_decision(
    expected_item: ProjectedExpected,
    *,
    bbox: BBox,
    global_bbox: BBox,
    fallback_bbox: BBox | None,
    all_expected: list[ProjectedExpected] | None,
    trusted_anchors: list[TrustedAnchor] | None,
    consensus: AnchorReleaseConsensus,
    inlier_anchors: list[TrustedAnchor],
    single_anchor: bool,
    shift_factor: float,
) -> AnchorOverlapDecision:
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
        return _anchor_overlap_result(
            allowed=True,
            reason=None,
        )

    overlap_limit = float(_thresholds.missing_anchor_release_max_other_overlap)
    strong_anchor_by_index = _best_strong_anchor_by_expected_index(
        trusted_anchors or []
    )
    target_owner_bbox = fallback_bbox or global_bbox
    target_center = np.asarray(bbox_center(target_owner_bbox), dtype=np.float32)
    released_center = np.asarray(bbox_center(bbox), dtype=np.float32)
    current_diag = max(1.0, bbox_diag(global_bbox))

    overlap_count = 0
    unresolved_count = 0
    strong_anchor_count = 0
    max_overlap = 0.0
    max_resolved_overlap = 0.0

    def decision(allowed: bool, reason: str | None) -> AnchorOverlapDecision:
        return _anchor_overlap_result(
            allowed=allowed,
            reason=reason,
            overlap_count=overlap_count,
            unresolved_count=unresolved_count,
            strong_anchor_count=strong_anchor_count,
            max_overlap=max_overlap,
            max_resolved_overlap=max_resolved_overlap,
        )

    for other in all_expected:
        if other.index == expected_item.index:
            continue

        overlap = max(
            bbox_iou(bbox, other.bbox),
            bbox_containment(bbox, other.bbox),
            bbox_containment(other.bbox, bbox),
        )
        max_overlap = max(max_overlap, float(overlap))
        if overlap <= overlap_limit:
            continue

        overlap_count += 1
        strong_anchor = strong_anchor_by_index.get(other.index)
        if strong_anchor is not None:
            strong_anchor_count += 1
            resolved_overlap = max(
                bbox_iou(bbox, strong_anchor.local_bbox),
                bbox_containment(bbox, strong_anchor.local_bbox),
                bbox_containment(strong_anchor.local_bbox, bbox),
            )
            max_resolved_overlap = max(max_resolved_overlap, float(resolved_overlap))
            if resolved_overlap > overlap_limit:
                return decision(
                    False,
                    "anchor_release_rejected_overlap_with_strong_anchor",
                )
            continue

        unresolved_count += 1
        other_center = np.asarray(bbox_center(other.bbox), dtype=np.float32)
        target_distance = float(np.linalg.norm(released_center - target_center))
        other_distance = float(np.linalg.norm(released_center - other_center))
        ownership_margin = float(
            _thresholds.missing_anchor_release_overlap_center_margin
        )
        if other_distance * ownership_margin < target_distance:
            return decision(
                False,
                "anchor_release_rejected_overlap_owned_by_other_slot",
            )

    if overlap_count == 0:
        return decision(True, None)

    if unresolved_count > 0:
        if single_anchor:
            target_drift = (
                float(np.linalg.norm(released_center - target_center)) / current_diag
            )
            if _anchor_release_allows_strict_single_overlap(
                inlier_anchors,
                consensus=consensus,
                shift_factor=shift_factor,
                unresolved_count=unresolved_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
                target_drift=target_drift,
            ):
                return decision(True, None)
            return decision(
                False,
                "anchor_release_rejected_overlap_single_anchor_guard",
            )
        if (
            consensus.inlier_count
            < _thresholds.missing_anchor_release_overlap_min_inliers
        ):
            return decision(
                False,
                "anchor_release_rejected_overlap_weak_consensus",
            )
        if (
            consensus.inlier_ratio
            < _thresholds.missing_anchor_release_overlap_min_inlier_ratio
        ):
            return decision(
                False,
                "anchor_release_rejected_overlap_weak_consensus",
            )
        if (
            consensus.dispersion
            > _thresholds.missing_anchor_release_overlap_max_dispersion_factor
        ):
            return decision(
                False,
                "anchor_release_rejected_overlap_high_dispersion",
            )
        if shift_factor > _thresholds.missing_anchor_release_overlap_max_shift_factor:
            return decision(
                False,
                "anchor_release_rejected_overlap_large_shift",
            )

        target_drift = (
            float(np.linalg.norm(released_center - target_center)) / current_diag
        )
        max_target_drift = _thresholds.missing_anchor_release_max_local_center_factor
        if target_drift > max_target_drift:
            return decision(
                False,
                "anchor_release_rejected_overlap_target_drift",
            )

    return decision(True, None)


def _anchor_release_allows_strict_single_overlap(
    inlier_anchors: list[TrustedAnchor],
    *,
    consensus: AnchorReleaseConsensus,
    shift_factor: float,
    unresolved_count: int,
    max_overlap: float,
    max_resolved_overlap: float,
    target_drift: float,
) -> bool:
    if len(inlier_anchors) != 1:
        return False
    if (
        unresolved_count
        > _thresholds.missing_anchor_release_strict_single_overlap_max_unresolved
    ):
        return False
    if max_resolved_overlap > _thresholds.missing_anchor_release_max_other_overlap:
        return False
    if not np.isfinite(max_overlap):
        return False
    if target_drift > _thresholds.missing_anchor_release_single_max_local_center_factor:
        return False

    anchor = inlier_anchors[0]
    if anchor.source not in {"translation_rescue", "context_feature_affine"}:
        return False
    if (
        anchor.inlier_count
        < _thresholds.missing_anchor_release_strict_single_overlap_min_inliers
    ):
        return False
    if (
        anchor.inlier_ratio
        < _thresholds.missing_anchor_release_strict_single_overlap_min_inlier_ratio
    ):
        return False
    if (
        anchor.median_error
        > _thresholds.missing_anchor_release_strict_single_overlap_max_anchor_error
    ):
        return False
    if (
        consensus.dispersion
        > _thresholds.missing_anchor_release_strict_single_overlap_max_dispersion_factor
    ):
        return False
    if (
        shift_factor
        > _thresholds.missing_anchor_release_strict_single_overlap_max_shift_factor
    ):
        return False

    return True


def _best_strong_anchor_by_expected_index(
    trusted_anchors: list[TrustedAnchor],
) -> dict[int, TrustedAnchor]:
    result: dict[int, TrustedAnchor] = {}

    for anchor in trusted_anchors:
        if anchor.source not in {"translation_rescue", "context_feature_affine"}:
            continue
        if anchor.inlier_count < 2:
            continue
        min_ratio = _thresholds.missing_anchor_release_min_inlier_ratio
        if anchor.inlier_ratio < min_ratio:
            continue
        current = result.get(anchor.expected_index)
        if current is None or anchor.weight > current.weight:
            result[anchor.expected_index] = anchor

    return result
