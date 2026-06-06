from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatcherThresholds:
    """Internal matcher thresholds that are not runtime environment knobs."""

    slot_search_expansion: float = 2.35
    slot_min_score: float = 0.43
    slot_min_detection_containment: float = 0.1
    slot_min_yolo_confidence: float = 0.08
    slot_min_feature_support: int = 4
    slot_feature_search_expansion: float = 2.75
    missing_polygon_refinement: bool = True
    missing_polygon_min_feature_support: int = 4
    missing_polygon_max_reprojection_error: float = 14.0
    missing_polygon_context_expansion: float = 3.0
    missing_polygon_context_exclusion_margin: float = 8.0
    missing_polygon_multi_context_exclusion_expansion: float = 1.35
    missing_polygon_context_min_inlier_ratio: float = 0.3
    missing_polygon_context_min_spread: float = 0.16
    missing_polygon_context_min_quadrants: int = 2
    missing_polygon_context_max_center_drift_factor: float = 0.85
    missing_polygon_context_min_area_score: float = 0.38
    missing_fallback_min_global_area_score: float = 0.08
    missing_fallback_min_local_global_area_score: float = 0.32
    missing_fallback_max_local_global_center_factor: float = 0.75
    missing_local_displacement_min_support: int = 6
    missing_local_displacement_nearest_points: int = 8
    missing_local_displacement_max_residual_error: float = 10.0
    missing_local_displacement_max_shift_factor: float = 0.58
    missing_local_displacement_min_spread: float = 0.1
    missing_local_displacement_min_quadrants: int = 2
    missing_local_displacement_min_search_containment: float = 0.1
    missing_local_displacement_min_local_area_score: float = 0.26
    missing_local_displacement_max_local_center_factor: float = 1.08
    missing_local_displacement_max_other_overlap: float = 0.3
    missing_rescue_translation_min_support: int = 6
    missing_rescue_translation_min_inlier_ratio: float = 0.46
    missing_rescue_translation_max_residual_error: float = 9.0
    missing_rescue_translation_max_shift_factor: float = 0.48
    missing_scene_rescue_context_expansion: float = 6.0
    missing_scene_rescue_min_support: int = 10
    missing_scene_rescue_min_inlier_ratio: float = 0.58
    missing_scene_rescue_max_residual_error: float = 8.0
    missing_scene_rescue_max_shift_factor: float = 0.42
    missing_scene_rescue_min_spread: float = 0.1
    missing_scene_rescue_max_other_overlap: float = 0.32
    missing_anchor_release_min_anchors: int = 2
    missing_anchor_release_min_inlier_ratio: float = 0.55
    missing_anchor_release_max_anchor_error: float = 9.0
    missing_anchor_release_max_shift_factor: float = 0.55
    missing_anchor_release_min_local_area_score: float = 0.24
    missing_anchor_release_max_local_center_factor: float = 0.95
    missing_anchor_release_max_other_overlap: float = 0.32
    missing_anchor_release_overlap_min_inliers: int = 2
    missing_anchor_release_overlap_min_inlier_ratio: float = 0.75
    missing_anchor_release_overlap_max_dispersion_factor: float = 0.035
    missing_anchor_release_overlap_max_shift_factor: float = 0.36
    missing_anchor_release_overlap_center_margin: float = 0.88
    missing_anchor_release_min_visible_fraction: float = 0.12
    missing_anchor_release_single_min_inliers: int = 5
    missing_anchor_release_single_min_inlier_ratio: float = 0.5
    missing_anchor_release_single_max_anchor_error: float = 7.0
    missing_anchor_release_single_max_shift_factor: float = 0.24
    missing_anchor_release_single_min_local_area_score: float = 0.42
    missing_anchor_release_single_max_local_center_factor: float = 0.52
    missing_anchor_release_strict_single_overlap_min_inliers: int = 8
    missing_anchor_release_strict_single_overlap_min_inlier_ratio: float = 0.62
    missing_anchor_release_strict_single_overlap_max_anchor_error: float = 6.5
    missing_anchor_release_strict_single_overlap_max_shift_factor: float = 0.18
    missing_anchor_release_strict_single_overlap_max_dispersion_factor: float = 0.018
    missing_anchor_release_strict_single_overlap_max_unresolved: int = 1
    missing_anchor_release_local_min_inlier_ratio: float = 0.25
    missing_anchor_release_max_dispersion_factor: float = 0.14
    missing_anchor_release_max_neighbor_distance_factor: float = 6.0
    missing_anchor_release_weak_slot_min_inliers: int = 3
    missing_anchor_release_weak_slot_min_inlier_ratio: float = 0.72
    missing_anchor_release_weak_slot_max_dispersion_factor: float = 0.045
    missing_anchor_release_weak_slot_min_feature_support: int = 2
    missing_anchor_release_weak_slot_min_feature_ratio: float = 0.03
    missing_anchor_release_weak_slot_min_area_score: float = 0.28
    missing_anchor_release_weak_slot_max_center_factor: float = 1.2
    missing_anchor_release_weak_slot_max_shift_factor: float = 0.36


thresholds = MatcherThresholds()


__all__ = ["MatcherThresholds", "thresholds"]
