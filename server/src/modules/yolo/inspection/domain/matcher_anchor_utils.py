from __future__ import annotations

import numpy as np
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_center,
    bbox_center_distance_factor,
    bbox_diag,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    ExpectedSlot,
    MissingTranslationRescue,
)

_TRUSTED_ANCHOR_SOURCE_WEIGHTS = {
    "matched_detection": 1.15,
    "translation_rescue": 1.0,
    "resolved_context_translation": 0.92,
    "context_feature_affine": 0.78,
    "resolved_anchor_release": 0.72,
    "resolved_context_affine": 0.66,
    "resolved_scene_translation": 0.58,
    "local_global_slot_hint": 0.42,
    "weak_local_global_slot_hint": 0.18,
}


def _trusted_anchor_source_weight(source: str) -> float:
    return _TRUSTED_ANCHOR_SOURCE_WEIGHTS.get(source, 0.42)


def _bbox_center_residual(local_bbox: BBox, global_bbox: BBox) -> np.ndarray:
    local_center = bbox_center(local_bbox)
    global_center = bbox_center(global_bbox)
    return np.asarray(
        [
            local_center[0] - global_center[0],
            local_center[1] - global_center[1],
        ],
        dtype=np.float32,
    )


def _translation_residual(rescue: MissingTranslationRescue) -> np.ndarray:
    return np.asarray([rescue.shift_x, rescue.shift_y], dtype=np.float32)


def _residual_shift_factor(residual: np.ndarray, bbox: BBox) -> float:
    return float(np.linalg.norm(residual) / max(1.0, bbox_diag(bbox)))


def _slot_feature_ratio(slot: ExpectedSlot) -> float:
    return slot.feature_support / max(1, slot.feature_total)


def _bbox_area_center_scores(
    source_bbox: BBox, target_bbox: BBox
) -> tuple[float, float]:
    return (
        bbox_area_similarity(source_bbox, target_bbox),
        bbox_center_distance_factor(source_bbox, target_bbox),
    )
