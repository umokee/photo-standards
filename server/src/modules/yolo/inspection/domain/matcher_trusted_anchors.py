from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from modules.core.standards.reference_constants import IOU_MATCH_THRESHOLD
from modules.yolo.inspection.domain.alignment import LocalProjectionData
from modules.yolo.inspection.domain.matcher_anchor_utils import (
    _bbox_area_center_scores,
    _bbox_center_residual,
    _residual_shift_factor,
    _slot_feature_ratio,
    _translation_residual,
    _trusted_anchor_source_weight,
)
from modules.yolo.inspection.domain.matcher_debug import _debug_float, _debug_int
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_diag,
)
from modules.yolo.inspection.domain.matcher_projection_validation import (
    _missing_global_projection,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    DetectionCandidate,
    ExpectedSlot,
    MissingPolygonRefinement,
    MissingTranslationRescue,
    ProjectedExpected,
    TrustedAnchor,
)
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds

TranslationRescueFn = Callable[..., MissingTranslationRescue | None]
PolygonRefinementFn = Callable[..., MissingPolygonRefinement | None]


def _build_trusted_missing_anchors(
    all_expected: list[ProjectedExpected],
    *,
    slot_by_index: dict[int, ExpectedSlot],
    projection_data: LocalProjectionData | None,
    translation_rescue_fn: TranslationRescueFn,
    polygon_refinement_fn: PolygonRefinementFn,
) -> list[TrustedAnchor]:
    if projection_data is None or projection_data.global_homography is None:
        return []
    if len(all_expected) <= 1:
        return []

    anchors: list[TrustedAnchor] = []

    def add_anchor(anchor: TrustedAnchor | None) -> bool:
        if anchor is None:
            return False
        anchors.append(anchor)
        return True

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

        rescue = translation_rescue_fn(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            global_polygon=global_polygon,
            global_bbox=global_bbox,
            all_expected=all_expected,
        )
        if (
            rescue is not None
            and _trusted_anchor_translation_reject_reason(rescue) is None
        ):
            if add_anchor(
                _trusted_anchor_from_translation_rescue(
                    expected_item,
                    rescue=rescue,
                    global_bbox=global_bbox,
                )
            ):
                continue

        refinement = polygon_refinement_fn(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            all_expected=all_expected,
        )
        if (
            refinement is not None
            and _trusted_anchor_affine_reject_reason(refinement, global_bbox) is None
        ):
            if add_anchor(
                _trusted_anchor_from_affine_refinement(
                    expected_item,
                    refinement=refinement,
                    global_bbox=global_bbox,
                )
            ):
                continue

        if (
            _trusted_anchor_local_global_reject_reason(
                expected_item,
                slot=slot,
                global_bbox=global_bbox,
            )
            is None
        ):
            add_anchor(
                _trusted_anchor_from_local_global_slot(
                    expected_item,
                    slot=slot,
                    global_bbox=global_bbox,
                )
            )
            continue

        if (
            _trusted_anchor_weak_local_global_reject_reason(
                expected_item,
                slot=slot,
                global_bbox=global_bbox,
            )
            is None
        ):
            add_anchor(
                _trusted_anchor_from_weak_local_global_slot(
                    expected_item,
                    slot=slot,
                    global_bbox=global_bbox,
                )
            )

    return anchors


def _trusted_anchor_translation_reject_reason(
    rescue: MissingTranslationRescue,
) -> str | None:
    max_anchor_error = float(_thresholds.missing_anchor_release_max_anchor_error)
    min_anchor_ratio = max(
        _thresholds.missing_rescue_translation_min_inlier_ratio,
        _thresholds.missing_anchor_release_min_inlier_ratio,
    )

    if rescue.inlier_count < _thresholds.missing_rescue_translation_min_support:
        return "low_support"
    if rescue.inlier_ratio < min_anchor_ratio:
        return "low_inlier_ratio"
    if rescue.median_error > max_anchor_error:
        return "high_median_error"
    if rescue.shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return "large_shift"

    residual = np.asarray([rescue.shift_x, rescue.shift_y], dtype=np.float32)
    if not np.isfinite(residual).all():
        return "bad_residual"
    return None


def _trusted_anchor_affine_reject_reason(
    refinement: MissingPolygonRefinement,
    global_bbox: BBox,
) -> str | None:
    max_anchor_error = float(_thresholds.missing_anchor_release_max_anchor_error)
    min_support = max(6, _thresholds.missing_polygon_min_feature_support)
    min_ratio = max(0.55, _thresholds.missing_anchor_release_min_inlier_ratio)

    if refinement.inlier_count < min_support:
        return "low_support"
    if refinement.inlier_ratio < min_ratio:
        return "low_inlier_ratio"
    if refinement.median_error > max_anchor_error:
        return "high_median_error"
    if refinement.area_score < 0.46:
        return "low_area_score"
    if refinement.center_drift_factor > 0.70:
        return "large_center_drift"

    residual = _bbox_center_residual(refinement.bbox, global_bbox)
    if not np.isfinite(residual).all():
        return "bad_residual"

    shift_factor = _residual_shift_factor(residual, global_bbox)
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return "large_shift"

    local_global_area_score = bbox_area_similarity(refinement.bbox, global_bbox)
    if local_global_area_score < 0.30:
        return "bad_local_global_area"
    return None


def _trusted_anchor_local_global_reject_reason(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> str | None:
    min_support = max(4, _thresholds.missing_polygon_min_feature_support)
    if slot.feature_support < min_support:
        return "low_support"

    feature_ratio = _slot_feature_ratio(slot)
    if feature_ratio < 0.10:
        return "low_feature_ratio"

    local_global_area_score, local_global_center_factor = _bbox_area_center_scores(
        expected_item.bbox,
        global_bbox,
    )
    if local_global_area_score < 0.55:
        return "bad_area"
    if local_global_center_factor > 0.65:
        return "large_center_drift"

    residual = _bbox_center_residual(expected_item.bbox, global_bbox)
    if not np.isfinite(residual).all():
        return "bad_residual"

    shift_factor = _residual_shift_factor(residual, global_bbox)
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return "large_shift"
    return None


def _trusted_anchor_weak_local_global_reject_reason(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> str | None:
    min_support = max(
        1,
        int(_thresholds.missing_anchor_release_weak_slot_min_feature_support),
    )
    if slot.feature_support < min_support:
        return "low_support"

    feature_ratio = _slot_feature_ratio(slot)
    if feature_ratio < _thresholds.missing_anchor_release_weak_slot_min_feature_ratio:
        return "low_feature_ratio"

    local_global_area_score, local_global_center_factor = _bbox_area_center_scores(
        expected_item.bbox,
        global_bbox,
    )
    if (
        local_global_area_score
        < _thresholds.missing_anchor_release_weak_slot_min_area_score
    ):
        return "bad_area"
    if (
        local_global_center_factor
        > _thresholds.missing_anchor_release_weak_slot_max_center_factor
    ):
        return "large_center_drift"

    residual = _bbox_center_residual(expected_item.bbox, global_bbox)
    if not np.isfinite(residual).all():
        return "bad_residual"

    shift_factor = _residual_shift_factor(residual, global_bbox)
    if shift_factor > _thresholds.missing_anchor_release_weak_slot_max_shift_factor:
        return "large_shift"
    return None


def _trusted_anchor_from_translation_rescue(
    expected_item: ProjectedExpected,
    *,
    rescue: MissingTranslationRescue,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if _trusted_anchor_translation_reject_reason(rescue) is not None:
        return None

    residual = _translation_residual(rescue)

    support_weight = float(rescue.inlier_count) / max(
        1.0, float(rescue.candidate_count)
    )
    error_weight = 1.0 / max(1.0, float(rescue.median_error))

    return TrustedAnchor(
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
    expected_item: ProjectedExpected,
    *,
    refinement: MissingPolygonRefinement,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if _trusted_anchor_affine_reject_reason(refinement, global_bbox) is not None:
        return None

    residual = _bbox_center_residual(refinement.bbox, global_bbox)
    shift_factor = _residual_shift_factor(residual, global_bbox)
    local_global_area_score = bbox_area_similarity(refinement.bbox, global_bbox)

    support_weight = refinement.inlier_count / max(
        1.0, float(refinement.candidate_count)
    )
    geometry_weight = refinement.area_score * max(0.20, local_global_area_score)
    error_weight = 1.0 / max(1.0, float(refinement.median_error))

    return TrustedAnchor(
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
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if (
        _trusted_anchor_local_global_reject_reason(
            expected_item,
            slot=slot,
            global_bbox=global_bbox,
        )
        is not None
    ):
        return None

    return _trusted_anchor_from_slot_hint(
        expected_item,
        slot=slot,
        global_bbox=global_bbox,
        source="local_global_slot_hint",
        min_weight=0.01,
        use_min_geometry_weight=False,
    )


def _trusted_anchor_from_weak_local_global_slot(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if (
        _trusted_anchor_weak_local_global_reject_reason(
            expected_item,
            slot=slot,
            global_bbox=global_bbox,
        )
        is not None
    ):
        return None

    return _trusted_anchor_from_slot_hint(
        expected_item,
        slot=slot,
        global_bbox=global_bbox,
        source="weak_local_global_slot_hint",
        min_weight=0.005,
        use_min_geometry_weight=True,
    )


def _trusted_anchor_from_slot_hint(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
    source: str,
    min_weight: float,
    use_min_geometry_weight: bool,
) -> TrustedAnchor:
    feature_ratio = _slot_feature_ratio(slot)
    local_global_area_score, local_global_center_factor = _bbox_area_center_scores(
        expected_item.bbox,
        global_bbox,
    )
    residual = _bbox_center_residual(expected_item.bbox, global_bbox)
    shift_factor = _residual_shift_factor(residual, global_bbox)
    median_error = float(local_global_center_factor * bbox_diag(global_bbox))
    support_weight = min(
        1.0, slot.feature_support / max(1.0, float(slot.feature_total))
    )
    geometry_weight = (
        max(0.02, local_global_area_score)
        if use_min_geometry_weight
        else local_global_area_score
    )
    error_weight = 1.0 / max(1.0, median_error)
    return TrustedAnchor(
        expected_index=expected_item.index,
        source=source,
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=expected_item.bbox,
        weight=max(min_weight, support_weight * geometry_weight * error_weight),
        candidate_count=max(1, slot.feature_total),
        inlier_count=slot.feature_support,
        inlier_ratio=feature_ratio,
        median_error=median_error,
        shift_factor=shift_factor,
    )


def _trusted_anchor_from_matched_detection(
    expected_item: ProjectedExpected,
    *,
    detection_item: DetectionCandidate,
    match_iou: float,
    projection_data: LocalProjectionData | None,
    match_debug: dict[str, Any] | None = None,
) -> TrustedAnchor | None:
    if projection_data is None:
        return None

    runtime_trusted = bool(
        match_debug is not None
        and match_debug.get("runtime_yolo_anchor_trusted") is True
    )
    if not runtime_trusted and match_iou < max(0.55, IOU_MATCH_THRESHOLD):
        return None

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return None
    _, global_bbox = global_projection

    local_bbox = detection_item.bbox
    residual = _bbox_center_residual(local_bbox, global_bbox)
    if residual.shape != (2,) or not np.isfinite(residual).all():
        return None

    shift_factor = _residual_shift_factor(residual, global_bbox)
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return None

    area_score = bbox_area_similarity(local_bbox, global_bbox)
    if area_score < 0.24:
        return None

    confidence = max(0.0, min(1.0, float(detection_item.detection.confidence or 0.0)))
    if runtime_trusted and match_debug is not None:
        anchor_iou = _debug_float(
            match_debug.get("runtime_anchor_slot_iou"), default=match_iou
        )
        coverage = _debug_float(
            match_debug.get("runtime_anchor_slot_coverage"), default=match_iou
        )
        containment = _debug_float(
            match_debug.get("runtime_anchor_detection_containment"),
            default=match_iou,
        )
        quality = max(0.01, min(1.0, (anchor_iou + coverage + containment) / 3.0))
    else:
        quality = max(0.01, min(1.0, float(match_iou)))
    return TrustedAnchor(
        expected_index=expected_item.index,
        source="matched_detection",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=local_bbox,
        weight=max(0.01, quality * max(0.10, confidence) * max(0.10, area_score)),
        candidate_count=1,
        inlier_count=1,
        inlier_ratio=quality,
        median_error=max(0.0, (1.0 - quality) * bbox_diag(global_bbox)),
        shift_factor=shift_factor,
    )


def _trusted_anchor_from_resolved_projection(
    expected_item: ProjectedExpected,
    *,
    polygon: list[list[float]] | None,
    bbox: BBox | None,
    projection_data: LocalProjectionData | None,
    debug: dict[str, Any] | None,
) -> TrustedAnchor | None:
    if polygon is None or bbox is None or debug is None or projection_data is None:
        return None

    projection_name = str(
        debug.get("missing_polygon_projection") or debug.get("projection") or ""
    )
    source_by_projection = {
        "context_feature_affine_translation_rescue": "resolved_context_translation",
        "context_feature_affine_scene_translation_rescue": "resolved_context_translation",
        "expected_slot_context_translation_rescue": "resolved_context_translation",
        "expected_slot_scene_translation_rescue": "resolved_scene_translation",
        "expected_slot_anchor_release": "resolved_anchor_release",
        "context_feature_affine": "resolved_context_affine",
    }
    source = source_by_projection.get(projection_name)
    if source is None:
        return None

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return None
    _, global_bbox = global_projection

    residual = _bbox_center_residual(bbox, global_bbox)
    if residual.shape != (2,) or not np.isfinite(residual).all():
        return None

    shift_factor = _residual_shift_factor(residual, global_bbox)
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return None

    area_score = bbox_area_similarity(bbox, global_bbox)
    if area_score < 0.24:
        return None

    candidate_count = _debug_int(
        debug.get("missing_polygon_candidate_count"), default=1
    )
    inlier_count = _debug_int(debug.get("missing_polygon_inliers"), default=1)
    inlier_ratio = _debug_float(debug.get("missing_polygon_inlier_ratio"), default=1.0)
    median_error = _debug_float(debug.get("missing_polygon_median_error"), default=0.0)

    if source == "resolved_context_affine":
        if inlier_count < max(6, _thresholds.missing_polygon_min_feature_support):
            return None
        if inlier_ratio < 0.55:
            return None
        if median_error > _thresholds.missing_anchor_release_max_anchor_error:
            return None
        if _debug_float(debug.get("missing_polygon_area_score"), default=1.0) < 0.46:
            return None
    elif source == "resolved_anchor_release":
        if inlier_count < 2:
            return None
        if inlier_ratio < 0.70:
            return None
    else:
        if inlier_count < 3:
            return None
        if inlier_ratio < 0.48:
            return None
        if median_error > _thresholds.missing_anchor_release_max_anchor_error:
            return None

    support_weight = inlier_count / max(1.0, float(candidate_count))
    error_weight = 1.0 / max(1.0, float(median_error))
    return TrustedAnchor(
        expected_index=expected_item.index,
        source=source,
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=bbox,
        weight=max(0.01, support_weight * max(0.10, area_score) * error_weight),
        candidate_count=max(1, candidate_count),
        inlier_count=max(1, inlier_count),
        inlier_ratio=max(0.0, min(1.0, float(inlier_ratio))),
        median_error=max(0.0, float(median_error)),
        shift_factor=shift_factor,
    )


def _append_or_replace_trusted_anchor(
    anchors: list[TrustedAnchor],
    anchor: TrustedAnchor | None,
) -> None:
    if anchor is None:
        return
    for index, existing in enumerate(list(anchors)):
        if existing.expected_index != anchor.expected_index:
            continue
        if _trusted_anchor_source_weight(
            existing.source
        ) >= _trusted_anchor_source_weight(anchor.source):
            return
        anchors[index] = anchor
        return
    anchors.append(anchor)
