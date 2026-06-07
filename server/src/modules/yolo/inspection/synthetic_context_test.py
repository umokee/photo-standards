from __future__ import annotations

import argparse
import html
import json
import math
import random
import shutil
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import cv2
import numpy as np
from app.config import settings
from modules.core.standards.reference_constants import (
    SUPERPOINT_PHOTO_GRID_COLS,
    SUPERPOINT_PHOTO_GRID_ROWS,
    SUPERPOINT_PHOTO_MAX_KEYPOINTS,
    SUPERPOINT_PHOTO_MAX_SIDE,
    SUPERPOINT_REFERENCE_GRID_COLS,
    SUPERPOINT_REFERENCE_GRID_ROWS,
    SUPERPOINT_REFERENCE_MAX_KEYPOINTS,
    SUPERPOINT_REFERENCE_MAX_SIDE,
)
from modules.core.standards.reference_features import compute_features
from modules.yolo.inspection.adapters.features import (
    AlignmentValidationConfig,
    align_with_features,
)
# Synthetic-only constants formerly stored in matcher_thresholds.py.
# Production matcher no longer uses that threshold bag.
_SYNTHETIC_MISSING_POLYGON_MIN_FEATURE_SUPPORT = 4
_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXPANSION = 3.0
_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXCLUSION_MARGIN = 8.0
from modules.yolo.inspection.domain.reference_masking import mask_reference_polygons
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.validation import make_valid


def _ensure_optional_storage_import() -> None:
    try:
        __import__("infra.storage.file_storage")
        return
    except ModuleNotFoundError as exc:
        if exc.name not in {"infra.storage", "infra.storage.file_storage"}:
            raise

    storage_module = types.ModuleType("infra.storage")
    file_storage_module = types.ModuleType("infra.storage.file_storage")

    def resolve_storage_path(relative_path: str | Path) -> Path:
        return settings.STORAGE_ROOT / Path(relative_path)

    file_storage_module.resolve_storage_path = resolve_storage_path  # type: ignore[attr-defined]
    sys.modules.setdefault("infra.storage", storage_module)
    sys.modules.setdefault("infra.storage.file_storage", file_storage_module)


_ensure_optional_storage_import()

from modules.yolo.inspection.domain.alignment import LocalProjectionData, project_polygon
from modules.yolo.inspection.domain.matcher import match_segments
from modules.yolo.inspection.domain.polygon_transfer_v2 import transfer_missing_segments_v2
from modules.yolo.inspection.domain.types import ExpectedSegment, SegmentMatch, YoloDetection

CanvasSize = tuple[int, int]
Point = tuple[float, float]
PolygonPoints = list[list[float]]

_CANVAS_SIZE: CanvasSize = (720, 480)
_OBJECT_SHAPES: dict[str, PolygonPoints] = {
    "long_asymmetric_lever": [
        [262.0, 184.0],
        [292.0, 172.0],
        [348.0, 184.0],
        [405.0, 198.0],
        [421.0, 216.0],
        [404.0, 236.0],
        [345.0, 231.0],
        [327.0, 248.0],
        [287.0, 240.0],
        [296.0, 219.0],
        [254.0, 209.0],
    ],
    "concave_l_bracket": [
        [286.0, 162.0],
        [396.0, 173.0],
        [392.0, 202.0],
        [354.0, 201.0],
        [348.0, 249.0],
        [306.0, 243.0],
        [312.0, 202.0],
        [282.0, 196.0],
    ],
    "thin_fork": [
        [260.0, 180.0],
        [418.0, 190.0],
        [414.0, 210.0],
        [346.0, 206.0],
        [344.0, 221.0],
        [408.0, 226.0],
        [404.0, 245.0],
        [254.0, 232.0],
        [260.0, 214.0],
        [319.0, 218.0],
        [320.0, 203.0],
        [258.0, 198.0],
    ],
    "stepped_plate": [
        [274.0, 170.0],
        [342.0, 170.0],
        [346.0, 188.0],
        [407.0, 196.0],
        [402.0, 226.0],
        [366.0, 225.0],
        [363.0, 248.0],
        [301.0, 239.0],
        [305.0, 215.0],
        [270.0, 210.0],
    ],
    "hook_like_part": [
        [296.0, 158.0],
        [386.0, 170.0],
        [423.0, 204.0],
        [411.0, 238.0],
        [376.0, 254.0],
        [354.0, 237.0],
        [379.0, 224.0],
        [388.0, 205.0],
        [364.0, 188.0],
        [303.0, 184.0],
        [282.0, 206.0],
        [260.0, 192.0],
    ],
}
_OBJECT_SHAPE_NAMES = tuple(_OBJECT_SHAPES)
_DEFAULT_OBJECT_SHAPE = "long_asymmetric_lever"
_CONTEXT_COLOR = (66, 180, 255)
_GT_COLOR = (0, 220, 0)
_PREDICTED_COLOR = (0, 0, 255)
_DISTRACTOR_COLOR = (0, 165, 255)
_OBJECT_BAD_COLOR = (180, 80, 255)
_TEXT_COLOR = (245, 245, 245)
_PANEL_BG = (35, 35, 35)
_PASS_COLOR = (70, 210, 70)
_FAIL_COLOR = (70, 70, 240)


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    name: str
    description: str
    seed: int
    rotation_deg: float
    scale: float
    shift_x: float
    shift_y: float
    perspective: float
    global_bias_x: float
    global_bias_y: float
    noise_px: float
    context_points: int
    object_bad_points: int
    outlier_points: int
    distractor: bool
    occluder: bool
    weak_context: bool = False
    object_shape: str = _DEFAULT_OBJECT_SHAPE
    distractor_count: int = 2
    context_cluster: float = 0.0


@dataclass(frozen=True, slots=True)
class SyntheticMultiScenario:
    name: str
    description: str
    seed: int
    object_shapes: tuple[str, ...]
    duplicate_class_groups: tuple[tuple[int, ...], ...]
    rotation_deg: float
    scale: float
    shift_x: float
    shift_y: float
    perspective: float
    global_bias_x: float
    global_bias_y: float
    noise_px: float
    context_points_per_object: int
    object_bad_points_per_object: int
    outlier_points: int
    distractor_count_per_object: int
    occluder: bool
    weak_context: bool = False
    context_cluster: float = 0.0


@dataclass(slots=True)
class SyntheticObjectMetric:
    name: str
    object_shape: str
    status: str
    projection: str
    passed: bool
    safety_passed: bool
    dangerous_projection: bool
    unsafe_hidden: bool
    hidden_reason: str | None
    iou: float
    center_drift_px: float
    area_ratio: float
    axis_angle_error_deg: float | None
    major_length_ratio: float | None
    closer_to_distractor: bool
    notes: list[str]
    v2_scene_diag_version: str | None = None
    v2_scene_match_total: int = 0
    v2_scene_match_cells: int = 0
    v2_scene_match_span_x: float | None = None
    v2_scene_match_span_y: float | None = None
    v2_scene_match_hull_fraction: float | None = None
    v2_scene_match_top_count: int = 0
    v2_scene_match_bottom_count: int = 0
    v2_scene_match_left_count: int = 0
    v2_scene_match_right_count: int = 0
    v2_scene_match_max_cell_fraction: float | None = None
    v2_scene_model_source: str | None = None
    v2_scene_model_inlier_total: int = 0
    v2_scene_model_inlier_cells: int = 0
    v2_scene_model_inlier_span_x: float | None = None
    v2_scene_model_inlier_span_y: float | None = None
    v2_scene_model_inlier_hull_fraction: float | None = None
    v2_scene_model_inlier_top_count: int = 0
    v2_scene_model_inlier_bottom_count: int = 0
    v2_scene_model_inlier_left_count: int = 0
    v2_scene_model_inlier_right_count: int = 0
    v2_scene_model_inlier_max_cell_fraction: float | None = None
    v2_scene_model_p90_error: float | None = None
    v2_candidate_count: int = 0
    v2_candidate_crop_count: int = 0
    v2_candidate_ecc_attempts: int = 0
    v2_candidate_ecc_successes: int = 0
    v2_candidate_score: float | None = None
    v2_candidate_selection_score: float | None = None
    v2_candidate_crop_mode: str | None = None
    v2_candidate_start_mode: str | None = None
    v2_shift_factor: float | None = None
    v2_center_factor: float | None = None
    v2_ecc_score: float | None = None
    v2_phase_response: float | None = None
    v2_ring_fraction: float | None = None
    v2_max_other_overlap: float | None = None
    reference_keypoints_total: int = 0
    frame_keypoints_total: int = 0
    frame_max_keypoints: int = 0
    frame_keypoint_grid_rows: int = 0
    frame_keypoint_grid_cols: int = 0
    lightglue_reference_matches_total: int = 0
    lightglue_frame_matches_total: int = 0
    masked_alignment_used: bool = False
    original_reference_keypoints_total: int = 0
    masked_reference_keypoints_total: int = 0
    masked_lightglue_matches_total: int = 0
    slot_local_lightglue_attempted: bool = False
    slot_local_lightglue_accepted: bool = False
    slot_local_lightglue_reject_reason: str | None = None
    slot_local_lightglue_mode: str | None = None
    slot_local_lightglue_reference_keypoints: int = 0
    slot_local_lightglue_frame_keypoints: int = 0
    slot_local_lightglue_max_keypoints: int = 0
    slot_local_lightglue_grid_rows: int = 0
    slot_local_lightglue_grid_cols: int = 0
    slot_local_lightglue_raw_matches: int = 0
    slot_local_lightglue_inliers: int = 0
    slot_local_lightglue_inlier_ratio: float | None = None
    slot_local_lightglue_median_error: float | None = None
    slot_local_lightglue_area_score: float | None = None
    slot_local_lightglue_center_factor: float | None = None
    slot_local_lightglue_max_other_overlap: float | None = None
    hidden_shadow_available: bool = False
    hidden_shadow_projection: str | None = None
    hidden_shadow_fallback_source: str | None = None
    hidden_shadow_reason: str | None = None
    hidden_shadow_iou: float | None = None
    hidden_shadow_center_drift_px: float | None = None
    hidden_shadow_area_ratio: float | None = None
    hidden_shadow_closer_to_distractor: bool = False
    hidden_shadow_dangerous: bool = False
    hidden_shadow_would_pass: bool = False
    hidden_shadow_notes: list[str] = field(default_factory=list)
    reason_code: str | None = None
    fallback_source: str | None = None
    fallback_reason: str | None = None
    fallback_global_available: bool = False
    fallback_local_global_disagrees: bool = False
    fallback_local_global_area_score: float | None = None
    fallback_local_global_center_factor: float | None = None
    fallback_slot_feature_support: int = 0
    fallback_slot_feature_total: int = 0
    selective_hidden_release: bool = False
    selective_hidden_release_source: str | None = None
    selective_hidden_release_hidden_reason: str | None = None
    selective_hidden_release_rejected: bool = False
    selective_hidden_release_reject_reason: str | None = None
    selective_hidden_release_slot_support: int = 0
    selective_hidden_release_slot_total: int = 0
    selective_hidden_release_slot_ratio: float | None = None
    selective_hidden_release_reference_area_score: float | None = None
    selective_hidden_release_center_factor: float | None = None
    selective_hidden_release_max_other_overlap: float | None = None
    projection_candidate_count: int = 0
    projection_candidate_sources: list[str] = field(default_factory=list)
    projection_candidate_selected: str | None = None
    candidate_agreement_available_count: int = 0
    candidate_agreement_comparison_count: int = 0
    candidate_agreement_selected: str | None = None
    candidate_agreement_count: int = 0
    candidate_agreement_level: str | None = None
    candidate_agreement_best_iou: float | None = None
    candidate_agreement_best_area_score: float | None = None
    candidate_agreement_min_center_factor: float | None = None
    candidate_agreement_closest_source: str | None = None
    candidate_confidence: str | None = None
    candidate_recommended_action: str | None = None
    candidate_oracle_available_count: int = 0
    candidate_oracle_pass_count: int = 0
    candidate_oracle_safe_pass_count: int = 0
    candidate_oracle_dangerous_count: int = 0
    candidate_oracle_selected_source: str | None = None
    candidate_oracle_selected_would_pass: bool = False
    candidate_oracle_best_source: str | None = None
    candidate_oracle_best_iou: float | None = None
    candidate_oracle_best_center_drift_px: float | None = None
    candidate_oracle_best_area_ratio: float | None = None
    candidate_oracle_best_would_pass: bool = False
    candidate_oracle_best_dangerous: bool = False
    candidate_oracle_has_safe_alternative: bool = False
    candidate_oracle_safe_gain_source: str | None = None
    candidate_oracle_failure_mode: str | None = None
    candidate_oracle_sources: list[str] = field(default_factory=list)
    crop_verification_attempted: bool = False
    crop_verification_candidate_count: int = 0
    crop_verification_source_count: int = 0
    crop_verification_best_source: str | None = None
    crop_verification_best_score: float | None = None
    crop_verification_best_object_matches: int = 0
    crop_verification_best_ref_containment: float | None = None
    crop_verification_best_frame_containment: float | None = None
    crop_verification_best_iou: float | None = None
    crop_verification_best_center_drift_px: float | None = None
    crop_verification_best_area_ratio: float | None = None
    crop_verification_best_would_pass: bool = False
    crop_verification_best_dangerous: bool = False
    crop_verification_selected_score: float | None = None
    crop_verification_selected_object_matches: int = 0
    crop_verification_object_crop_candidate_available: bool = False
    crop_verification_object_crop_candidate_source: str | None = None
    crop_verification_object_crop_candidate_iou: float | None = None
    crop_verification_object_crop_candidate_center_drift_px: float | None = None
    crop_verification_object_crop_candidate_would_pass: bool = False
    crop_verification_object_crop_candidate_dangerous: bool = False
    crop_verification_object_crop_candidate_inliers: int = 0
    crop_verification_object_crop_candidate_inlier_ratio: float | None = None
    crop_verification_assessment: str | None = None
    crop_verification_sources: list[str] = field(default_factory=list)
    global_translation_rescue_attempted: bool = False
    global_translation_rescue_accepted: bool = False
    global_translation_rescue_reject_reason: str | None = None
    global_translation_rescue_local_point_count: int = 0
    global_translation_rescue_min_support: int = 0
    global_translation_rescue_candidate_count: int = 0
    global_translation_rescue_inlier_count: int = 0
    global_translation_rescue_inlier_ratio: float | None = None
    global_translation_rescue_median_error: float | None = None
    global_translation_rescue_shift_factor: float | None = None
    global_translation_rescue_context_spread: float | None = None
    global_translation_rescue_search_containment: float | None = None
    global_translation_rescue_local_area_score: float | None = None
    global_translation_rescue_local_center_factor: float | None = None
    global_translation_rescue_slot_feature_support: int = 0
    global_translation_rescue_slot_feature_total: int = 0
    none_reason: str | None = None
    none_has_homography: bool = False
    none_projected_point_count: int = 0
    anchor_release_attempted: bool = False
    anchor_release_reject_reason: str | None = None
    anchor_release_candidate_count: int = 0
    anchor_release_neighbor_candidate_count: int = 0
    anchor_release_selected_candidate_count: int = 0
    anchor_release_total_anchors: int = 0
    anchor_release_inliers: int = 0
    anchor_release_inlier_ratio: float | None = None
    anchor_release_median_error: float | None = None
    anchor_release_dispersion: float | None = None
    anchor_release_shift_factor: float | None = None
    anchor_release_build_expected_count: int = 0
    anchor_release_build_attempt_count: int = 0
    anchor_release_build_built_count: int = 0
    anchor_release_build_source_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_build_reject_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_build_probe_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_runtime_trusted_anchor_count: int = 0
    anchor_release_runtime_trusted_anchor_source_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_runtime_anchor_before_current_count: int = 0
    anchor_release_runtime_anchor_after_current_count: int = 0
    anchor_release_runtime_processed_expected_count: int = 0
    anchor_release_runtime_future_expected_count: int = 0
    anchor_release_runtime_built_count: int = 0
    anchor_release_runtime_source_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_runtime_reject_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_runtime_probe_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_final_trusted_anchor_count: int = 0
    anchor_release_final_trusted_anchor_source_counts: dict[str, int] = field(default_factory=dict)
    anchor_release_final_anchor_before_current_count: int = 0
    anchor_release_final_anchor_after_current_count: int = 0
    anchor_release_final_resolved_anchor_count: int = 0
    anchor_release_final_resolved_anchor_source_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SyntheticResult:
    index: int
    name: str
    description: str
    object_shape: str
    passed: bool
    safety_passed: bool
    dangerous_projection: bool
    unsafe_hidden: bool
    status: str
    projection: str
    reason_code: str | None
    iou: float
    center_drift_px: float
    area_ratio: float
    axis_angle_error_deg: float | None
    major_length_ratio: float | None
    context_feature_support: int
    context_feature_total: int
    missing_candidate_count: int
    missing_inliers: int
    missing_median_error: float | None
    used_context_refinement: bool
    used_edge_refinement: bool
    closer_to_distractor: bool
    nearest_distractor_distance_px: float | None
    gt_distance_px: float
    image_path: str
    notes: list[str]
    keypoints_image_path: str | None = None
    scenario_kind: str = "single"
    object_count: int = 1
    object_shapes: list[str] | None = None
    object_failures: int = 0
    object_metrics: list[dict[str, Any]] | None = None
    reference_keypoints_total: int = 0
    frame_keypoints_total: int = 0
    frame_max_keypoints: int = 0
    frame_keypoint_grid_rows: int = 0
    frame_keypoint_grid_cols: int = 0
    lightglue_reference_matches_total: int = 0
    lightglue_frame_matches_total: int = 0
    masked_alignment_used: bool = False
    original_reference_keypoints_total: int = 0
    masked_reference_keypoints_total: int = 0
    masked_lightglue_matches_total: int = 0


@dataclass(slots=True)
class _SyntheticScene:
    scenario: SyntheticScenario
    reference_image: np.ndarray
    target_image: np.ndarray
    true_homography: np.ndarray
    approximate_homography: np.ndarray
    reference_polygon: PolygonPoints
    ground_truth_polygon: PolygonPoints
    distractor_polygons: list[PolygonPoints]
    reference_points: np.ndarray
    frame_points: np.ndarray
    context_reference_points: np.ndarray
    context_frame_points: np.ndarray
    object_bad_reference_points: np.ndarray
    object_bad_frame_points: np.ndarray


@dataclass(slots=True)
class _SyntheticObjectInstance:
    index: int
    name: str
    class_key: str
    object_shape: str
    reference_polygon: PolygonPoints
    ground_truth_polygon: PolygonPoints
    distractor_polygons: list[PolygonPoints]


@dataclass(slots=True)
class _SyntheticMultiScene:
    scenario: SyntheticMultiScenario
    reference_image: np.ndarray
    target_image: np.ndarray
    true_homography: np.ndarray
    approximate_homography: np.ndarray
    objects: list[_SyntheticObjectInstance]
    reference_points: np.ndarray
    frame_points: np.ndarray
    context_reference_points: np.ndarray
    context_frame_points: np.ndarray
    object_bad_reference_points: np.ndarray
    object_bad_frame_points: np.ndarray


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.out).resolve()

    if args.real_yolo_synthetic:
        return _run_real_yolo_synthetic_pipeline(args=args, output_dir=output_dir)

    summary = _run_lightglue_synthetic_suite(args=args, output_dir=output_dir)
    if args.fail_on_fail and summary["failed"] > 0:
        return 1
    return 0


def _run_lightglue_synthetic_suite(*, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _build_scenarios(total=args.cases, seed=args.seed, profile=args.profile)
    results: list[SyntheticResult] = []

    for index, scenario in enumerate(scenarios, start=1):
        scene = _build_scene(scenario)
        result = _run_case(
            index=index,
            scene=scene,
            images_dir=images_dir,
            feature_source=args.feature_source,
            projection_pipeline=args.projection_pipeline,
            synthetic_yolo_anchor_pose=args.synthetic_yolo_anchor_pose,
            synthetic_yolo_anchor_noise_px=args.synthetic_yolo_anchor_noise_px,
            synthetic_yolo_anchor_dropout=args.synthetic_yolo_anchor_dropout,
            synthetic_yolo_anchor_min_anchors=args.synthetic_yolo_anchor_min_anchors,
            synthetic_yolo_anchor_ransac_px=args.synthetic_yolo_anchor_ransac_px,
        )
        results.append(result)

    multi_cases = _resolve_multi_object_cases(
        requested=args.multi_object_cases,
        profile=args.profile,
        base_count=len(scenarios),
    )
    if multi_cases > 0:
        multi_scenarios = _build_multi_scenarios(
            total=multi_cases,
            seed=args.seed,
            profile=args.profile,
        )
        start_index = len(results) + 1
        for offset, scenario in enumerate(multi_scenarios):
            scene = _build_multi_scene(scenario)
            result = _run_multi_case(
                index=start_index + offset,
                scene=scene,
                images_dir=images_dir,
                feature_source=args.feature_source,
                projection_pipeline=args.projection_pipeline,
                synthetic_yolo_anchor_pose=args.synthetic_yolo_anchor_pose,
                synthetic_yolo_anchor_noise_px=args.synthetic_yolo_anchor_noise_px,
                synthetic_yolo_anchor_dropout=args.synthetic_yolo_anchor_dropout,
                synthetic_yolo_anchor_min_anchors=args.synthetic_yolo_anchor_min_anchors,
                synthetic_yolo_anchor_ransac_px=args.synthetic_yolo_anchor_ransac_px,
            )
            results.append(result)

    summary = _build_summary(results)
    summary["feature_source"] = args.feature_source
    summary["projection_pipeline"] = args.projection_pipeline
    summary["synthetic_yolo_anchor_pose"] = bool(args.synthetic_yolo_anchor_pose)
    summary["synthetic_yolo_anchor_noise_px"] = float(args.synthetic_yolo_anchor_noise_px)
    summary["synthetic_yolo_anchor_dropout"] = float(args.synthetic_yolo_anchor_dropout)
    summary["synthetic_yolo_anchor_min_anchors"] = int(args.synthetic_yolo_anchor_min_anchors)
    summary["synthetic_yolo_anchor_ransac_px"] = float(args.synthetic_yolo_anchor_ransac_px)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "results.json", _serialize_results(results))
    _write_html_report(
        output_dir / "report.html",
        results=results,
        summary=summary,
    )

    print(
        "synthetic_context_test "
        f"cases={summary['total']} single={summary['single_cases']} multi={summary['multi_cases']} "
        f"objects={summary['total_expected_objects']} passed={summary['passed']} failed={summary['failed']} "
        f"pass_rate={summary['pass_rate']:.1f}% safety={summary['safety_rate']:.1f}% "
        f"feature_source={summary.get('feature_source', 'real')} "
        f"projection_pipeline={summary.get('projection_pipeline', 'legacy')} "
        f"synthetic_yolo_anchor_pose={summary.get('synthetic_yolo_anchor_pose', False)}"
    )
    print(f"report={output_dir / 'report.html'}")
    print(f"summary={output_dir / 'summary.json'}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic visual stress test for context-ring missing polygon transfer. "
            "By default the generated images are processed by the real "
            "SuperPoint/LightGlue feature extractor and matcher before match_segments. "
            "Runtime YOLO is not run unless --real-yolo-synthetic is used."
        )
    )
    parser.add_argument(
        "--out",
        default=str(settings.STORAGE_ROOT / "debug" / "synthetic_context_refinement"),
        help="Directory for report.html, json summaries and rendered images.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=160,
        help="Total number of synthetic scenarios, including curated stress cases.",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "brutal", "nightmare"),
        default="brutal",
        help="quick = fewer cases, brutal = default hard bake, nightmare = more extreme randomization.",
    )
    parser.add_argument(
        "--multi-object-cases",
        type=int,
        default=None,
        help=(
            "Additional multi-object transfer scenarios. "
            "Default: for nightmare profile add the same number of multi-object "
            "cases as single-object cases; for other profiles add none. "
            "Pass 0 to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Return exit code 1 when at least one synthetic case fails.",
    )
    parser.add_argument(
        "--feature-source",
        choices=("real",),
        default="real",
        help=(
            "Only real mode is supported here: generated images are passed through "
            "the real SuperPoint/LightGlue feature extraction and matching pipeline "
            "before calling match_segments. The old generated-pair geometry fixture "
            "was intentionally removed from this end-to-end script."
        ),
    )
    parser.add_argument(
        "--projection-pipeline",
        choices=("legacy", "v2"),
        default="legacy",
        help=(
            "legacy = current matcher/fallback stack; v2 = experimental "
            "registration-first context-only polygon transfer without YOLO."
        ),
    )
    parser.add_argument(
        "--synthetic-yolo-anchor-pose",
        action="store_true",
        help=(
            "Synthetic V4 oracle for the next real pipeline: for every missing "
            "object in a multi-object scene, generate noisy YOLO-like detections "
            "from the other objects, estimate product pose with RANSAC, and "
            "project the current missing slot from that pose. This tests the "
            "object-anchor pose idea without training YOLO in the loop."
        ),
    )
    parser.add_argument(
        "--synthetic-yolo-anchor-noise-px",
        type=float,
        default=1.5,
        help="Gaussian corner jitter applied to synthetic YOLO anchor polygons.",
    )
    parser.add_argument(
        "--synthetic-yolo-anchor-dropout",
        type=float,
        default=0.06,
        help="Probability that a visible synthetic YOLO anchor detection is missing.",
    )
    parser.add_argument(
        "--synthetic-yolo-anchor-min-anchors",
        type=int,
        default=2,
        help="Minimum number of other visible object anchors required to trust V4 pose.",
    )
    parser.add_argument(
        "--synthetic-yolo-anchor-ransac-px",
        type=float,
        default=6.0,
        help="RANSAC reprojection threshold for synthetic YOLO-anchor pose.",
    )
    parser.add_argument(
        "--real-yolo-synthetic",
        action="store_true",
        help=(
            "Run the real synthetic YOLO-seg loop: first LightGlue baseline, then "
            "export a synthetic YOLO dataset, train/load a real Ultralytics model, "
            "run inference and evaluate YOLO as an anchor source for geometry rescue."
        ),
    )
    parser.add_argument(
        "--real-yolo-model-path",
        default=None,
        help="Optional trained YOLO-seg .pt file. If omitted, the test trains a model first.",
    )
    parser.add_argument(
        "--real-yolo-base-model",
        default="yolov8n-seg.pt",
        help="Base Ultralytics segmentation checkpoint used when training is needed.",
    )
    parser.add_argument("--real-yolo-train-cases", type=int, default=160)
    parser.add_argument("--real-yolo-val-cases", type=int, default=40)
    parser.add_argument("--real-yolo-test-cases", type=int, default=None)
    parser.add_argument("--real-yolo-epochs", type=int, default=20)
    parser.add_argument("--real-yolo-imgsz", type=int, default=640)
    parser.add_argument("--real-yolo-batch", type=int, default=8)
    parser.add_argument("--real-yolo-device", default="auto", help="auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--real-yolo-conf", type=float, default=0.25)
    parser.add_argument("--real-yolo-iou", type=float, default=0.70)
    parser.add_argument(
        "--real-yolo-anchor-min-conf",
        type=float,
        default=0.90,
        help=(
            "Minimum YOLO confidence for a detection to become a trusted anchor. "
            "No anchor never means missing; it only means YOLO did not provide geometry help."
        ),
    )
    parser.add_argument(
        "--real-yolo-anchor-max-center-factor",
        type=float,
        default=0.28,
        help="Maximum normalized distance between expected slot center and YOLO anchor center.",
    )
    parser.add_argument(
        "--real-yolo-anchor-min-slot-iou",
        type=float,
        default=0.32,
        help="Minimum overlap between expected slot and YOLO anchor mask for trusted anchor assignment.",
    )
    parser.add_argument(
        "--real-yolo-anchor-min-containment",
        type=float,
        default=0.45,
        help="Minimum fraction of YOLO anchor mask lying inside the expected slot.",
    )
    parser.add_argument(
        "--real-yolo-anchor-min-coverage",
        type=float,
        default=0.50,
        help=(
            "Minimum fraction of the expected slot covered by the YOLO anchor mask. "
            "This is the expected→factual agreement gate: the factual object must "
            "really occupy the expected area before it can become an anchor."
        ),
    )
    parser.add_argument(
        "--real-yolo-anchor-min-area-ratio",
        type=float,
        default=0.72,
        help=(
            "Minimum YOLO anchor mask area divided by expected slot area. "
            "Keeps undersized factual detections from becoming trusted anchors."
        ),
    )
    parser.add_argument(
        "--real-yolo-anchor-max-area-ratio",
        type=float,
        default=1.60,
        help="Maximum YOLO anchor mask area divided by expected slot area.",
    )
    parser.add_argument(
        "--real-yolo-keep-training-dir",
        action="store_true",
        help="Keep previous YOLO training directory instead of deleting it before a new run.",
    )
    return parser.parse_args()


def _build_scenarios(*, total: int, seed: int, profile: str) -> list[SyntheticScenario]:
    total = max(1, int(total))
    if profile == "quick":
        total = min(total, 48)
    elif profile == "nightmare":
        total = max(total, 240)

    curated = [
        SyntheticScenario(
            name="complex_lever_missing_good_context",
            description="Сложный рычаг из 11 точек удалён; вокруг достаточно стабильных признаков.",
            seed=seed + 1,
            rotation_deg=6.0,
            scale=1.02,
            shift_x=18.0,
            shift_y=-12.0,
            perspective=0.00018,
            global_bias_x=13.0,
            global_bias_y=-9.0,
            noise_px=1.0,
            context_points=72,
            object_bad_points=30,
            outlier_points=12,
            distractor=False,
            occluder=False,
            object_shape="long_asymmetric_lever",
            distractor_count=0,
        ),
        SyntheticScenario(
            name="concave_l_bracket_near_copy",
            description="Вогнутый L-кронштейн удалён; рядом похожая копия-приманка.",
            seed=seed + 2,
            rotation_deg=-8.0,
            scale=0.98,
            shift_x=-22.0,
            shift_y=16.0,
            perspective=-0.00022,
            global_bias_x=-16.0,
            global_bias_y=11.0,
            noise_px=1.2,
            context_points=76,
            object_bad_points=70,
            outlier_points=18,
            distractor=True,
            occluder=False,
            object_shape="concave_l_bracket",
            distractor_count=3,
        ),
        SyntheticScenario(
            name="thin_fork_under_occluder",
            description="Тонкая вилкообразная деталь отсутствует и частично закрыта чужим объектом.",
            seed=seed + 3,
            rotation_deg=11.0,
            scale=1.04,
            shift_x=12.0,
            shift_y=20.0,
            perspective=0.00026,
            global_bias_x=15.0,
            global_bias_y=12.0,
            noise_px=1.4,
            context_points=84,
            object_bad_points=58,
            outlier_points=18,
            distractor=True,
            occluder=True,
            object_shape="thin_fork",
            distractor_count=3,
        ),
        SyntheticScenario(
            name="stepped_plate_strong_perspective",
            description="Ступенчатая пластина, сильная перспектива и несколько похожих соседей.",
            seed=seed + 4,
            rotation_deg=-13.0,
            scale=1.05,
            shift_x=-8.0,
            shift_y=-24.0,
            perspective=0.00055,
            global_bias_x=12.0,
            global_bias_y=-15.0,
            noise_px=1.6,
            context_points=88,
            object_bad_points=48,
            outlier_points=20,
            distractor=True,
            occluder=False,
            object_shape="stepped_plate",
            distractor_count=4,
        ),
        SyntheticScenario(
            name="hook_shape_many_bad_object_matches",
            description="Крючкообразная деталь; много ложных совпадений внутри отсутствующего объекта ведут к приманкам.",
            seed=seed + 5,
            rotation_deg=7.0,
            scale=0.97,
            shift_x=-14.0,
            shift_y=-15.0,
            perspective=-0.00034,
            global_bias_x=-12.0,
            global_bias_y=14.0,
            noise_px=1.1,
            context_points=82,
            object_bad_points=110,
            outlier_points=22,
            distractor=True,
            occluder=False,
            object_shape="hook_like_part",
            distractor_count=4,
        ),
        SyntheticScenario(
            name="weak_context_should_fallback",
            description="Слабое окружение и много плохих внутренних точек: безопаснее fallback, а не ложное уточнение.",
            seed=seed + 6,
            rotation_deg=5.0,
            scale=1.0,
            shift_x=16.0,
            shift_y=10.0,
            perspective=0.00012,
            global_bias_x=6.0,
            global_bias_y=-5.0,
            noise_px=0.8,
            context_points=2,
            object_bad_points=90,
            outlier_points=34,
            distractor=True,
            occluder=False,
            weak_context=True,
            object_shape="concave_l_bracket",
            distractor_count=4,
        ),
        SyntheticScenario(
            name="clustered_context_degenerate_geometry",
            description="Контекстные точки собраны в одном углу — проверка на плохую локальную геометрию.",
            seed=seed + 7,
            rotation_deg=-10.0,
            scale=1.03,
            shift_x=22.0,
            shift_y=9.0,
            perspective=0.00028,
            global_bias_x=17.0,
            global_bias_y=8.0,
            noise_px=2.2,
            context_points=62,
            object_bad_points=78,
            outlier_points=30,
            distractor=True,
            occluder=True,
            object_shape="thin_fork",
            distractor_count=4,
            context_cluster=0.88,
        ),
        SyntheticScenario(
            name="repetitive_same_shape_distractors",
            description="Несколько почти одинаковых соседних деталей; predicted-зона не должна выбрать ближайшую копию.",
            seed=seed + 8,
            rotation_deg=12.0,
            scale=1.01,
            shift_x=-26.0,
            shift_y=18.0,
            perspective=0.00042,
            global_bias_x=-19.0,
            global_bias_y=13.0,
            noise_px=1.6,
            context_points=92,
            object_bad_points=100,
            outlier_points=28,
            distractor=True,
            occluder=False,
            object_shape="long_asymmetric_lever",
            distractor_count=5,
        ),
        SyntheticScenario(
            name="thin_part_low_iou_sensitive",
            description="Очень тонкая форма: даже маленький сдвиг быстро режет IoU, поэтому проверяется drift и distractor distance.",
            seed=seed + 9,
            rotation_deg=-4.0,
            scale=0.95,
            shift_x=24.0,
            shift_y=-20.0,
            perspective=-0.00040,
            global_bias_x=18.0,
            global_bias_y=-14.0,
            noise_px=1.3,
            context_points=70,
            object_bad_points=76,
            outlier_points=24,
            distractor=True,
            occluder=True,
            object_shape="thin_fork",
            distractor_count=4,
        ),
        SyntheticScenario(
            name="noisy_context_outlier_storm",
            description="Много outlier-точек и шумный контекст; RANSAC должен не принять мусор.",
            seed=seed + 10,
            rotation_deg=14.0,
            scale=1.07,
            shift_x=28.0,
            shift_y=-18.0,
            perspective=0.00062,
            global_bias_x=20.0,
            global_bias_y=-18.0,
            noise_px=2.8,
            context_points=86,
            object_bad_points=72,
            outlier_points=80,
            distractor=True,
            occluder=True,
            object_shape="stepped_plate",
            distractor_count=5,
        ),
    ]

    if total <= len(curated):
        return curated[:total]

    rng = random.Random(seed)
    scenarios = list(curated)
    nightmare = profile == "nightmare"
    shape_names = list(_OBJECT_SHAPE_NAMES)
    for index in range(total - len(curated)):
        has_distractor = rng.random() < (0.88 if nightmare else 0.78)
        weak_context = rng.random() < (0.18 if nightmare else 0.10)
        context_points = rng.randint(1, 5) if weak_context else rng.randint(42, 112)
        scenarios.append(
            SyntheticScenario(
                name=f"random_{profile}_stress_{index + 1:03d}",
                description="Случайная прожарка: сложная форма, приманки, шум, перспектива и ложные feature-точки.",
                seed=seed + 1000 + index,
                rotation_deg=rng.uniform(-18.0, 18.0) if nightmare else rng.uniform(-14.0, 14.0),
                scale=rng.uniform(0.90, 1.11) if nightmare else rng.uniform(0.94, 1.07),
                shift_x=rng.uniform(-38.0, 38.0) if nightmare else rng.uniform(-28.0, 28.0),
                shift_y=rng.uniform(-32.0, 32.0) if nightmare else rng.uniform(-24.0, 24.0),
                perspective=rng.uniform(-0.00080, 0.00085) if nightmare else rng.uniform(-0.00050, 0.00060),
                global_bias_x=rng.uniform(-26.0, 26.0) if nightmare else rng.uniform(-18.0, 18.0),
                global_bias_y=rng.uniform(-24.0, 24.0) if nightmare else rng.uniform(-16.0, 16.0),
                noise_px=rng.uniform(0.7, 3.2) if nightmare else rng.uniform(0.6, 2.2),
                context_points=context_points,
                object_bad_points=(rng.randint(40, 130) if has_distractor else rng.randint(8, 34)),
                outlier_points=rng.randint(18, 96) if nightmare else rng.randint(8, 42),
                distractor=has_distractor,
                occluder=rng.random() < (0.50 if nightmare else 0.38),
                weak_context=weak_context,
                object_shape=rng.choice(shape_names),
                distractor_count=rng.randint(2, 6) if has_distractor else 0,
                context_cluster=rng.uniform(0.0, 0.90) if rng.random() < 0.32 else 0.0,
            )
        )
    return scenarios


def _build_scene(scenario: SyntheticScenario) -> _SyntheticScene:
    width, height = _CANVAS_SIZE
    rng = np.random.default_rng(scenario.seed)
    object_polygon = _scenario_object_polygon(scenario)

    reference = _draw_reference_scene(
        width=width,
        height=height,
        object_polygon=object_polygon,
        object_shape=scenario.object_shape,
    )
    true_h = _build_homography(scenario, width=width, height=height)
    approximate_h = true_h.copy().astype(np.float32)
    approximate_h[0, 2] += float(scenario.global_bias_x)
    approximate_h[1, 2] += float(scenario.global_bias_y)

    distractor_count = scenario.distractor_count if scenario.distractor else 0
    target = _draw_target_scene(
        width=width,
        height=height,
        homography=true_h,
        object_polygon=object_polygon,
        distractor_count=distractor_count,
        occluder=scenario.occluder,
    )
    gt_polygon = project_polygon(object_polygon, true_h)
    distractors = _target_distractor_polygons(
        true_h,
        object_polygon=object_polygon,
        count=distractor_count,
    )

    context_ref = _sample_context_points(
        rng,
        count=scenario.context_points,
        reference_polygon=object_polygon,
        width=width,
        height=height,
        cluster_bias=scenario.context_cluster,
    )
    context_frame = _transform_points(context_ref, true_h)
    context_frame += rng.normal(0.0, scenario.noise_px, size=context_frame.shape).astype(
        np.float32
    )

    object_bad_ref = _sample_inside_polygon(
        rng,
        polygon=object_polygon,
        count=scenario.object_bad_points,
    )
    object_bad_frame = _bad_object_frame_points(
        rng,
        reference_points=object_bad_ref,
        homography=true_h,
        distractors=distractors,
        noise_px=scenario.noise_px,
    )

    outlier_ref = rng.uniform(
        low=(0.0, 0.0),
        high=(float(width - 1), float(height - 1)),
        size=(scenario.outlier_points, 2),
    ).astype(np.float32)
    outlier_frame = rng.uniform(
        low=(0.0, 0.0),
        high=(float(width - 1), float(height - 1)),
        size=(scenario.outlier_points, 2),
    ).astype(np.float32)

    reference_points = np.concatenate(
        [context_ref, object_bad_ref, outlier_ref],
        axis=0,
    ).astype(np.float32)
    frame_points = np.concatenate(
        [context_frame, object_bad_frame, outlier_frame],
        axis=0,
    ).astype(np.float32)

    return _SyntheticScene(
        scenario=scenario,
        reference_image=reference,
        target_image=target,
        true_homography=true_h,
        approximate_homography=approximate_h,
        reference_polygon=object_polygon,
        ground_truth_polygon=gt_polygon,
        distractor_polygons=distractors,
        reference_points=reference_points,
        frame_points=frame_points,
        context_reference_points=context_ref,
        context_frame_points=context_frame,
        object_bad_reference_points=object_bad_ref,
        object_bad_frame_points=object_bad_frame,
    )


def _resolve_multi_object_cases(
    *,
    requested: int | None,
    profile: str,
    base_count: int,
) -> int:
    if requested is not None:
        return max(0, int(requested))
    if profile == "nightmare":
        return max(0, int(base_count))
    return 0


def _build_multi_scenarios(
    *,
    total: int,
    seed: int,
    profile: str,
) -> list[SyntheticMultiScenario]:
    total = max(0, int(total))
    if total <= 0:
        return []

    curated = [
        SyntheticMultiScenario(
            name="multi_mixed_product_good_context",
            description="Несколько разных деталей на одном изделии; все отсутствуют, окружение стабильное.",
            seed=seed + 9001,
            object_shapes=("long_asymmetric_lever", "concave_l_bracket", "thin_fork", "stepped_plate"),
            duplicate_class_groups=(),
            rotation_deg=7.0,
            scale=1.02,
            shift_x=18.0,
            shift_y=-10.0,
            perspective=0.00022,
            global_bias_x=11.0,
            global_bias_y=-8.0,
            noise_px=1.1,
            context_points_per_object=44,
            object_bad_points_per_object=36,
            outlier_points=34,
            distractor_count_per_object=2,
            occluder=False,
        ),
        SyntheticMultiScenario(
            name="multi_duplicate_same_class_nearby",
            description="Несколько одинаковых деталей одного класса рядом: перенос не должен перескочить между слотами.",
            seed=seed + 9002,
            object_shapes=("thin_fork", "thin_fork", "thin_fork", "thin_fork"),
            duplicate_class_groups=((0, 1, 2, 3),),
            rotation_deg=-9.0,
            scale=0.98,
            shift_x=-20.0,
            shift_y=18.0,
            perspective=-0.00028,
            global_bias_x=-16.0,
            global_bias_y=12.0,
            noise_px=1.4,
            context_points_per_object=40,
            object_bad_points_per_object=62,
            outlier_points=42,
            distractor_count_per_object=3,
            occluder=True,
        ),
        SyntheticMultiScenario(
            name="multi_mixed_with_hook_and_weak_context",
            description="Смешанные детали, включая крючок; часть окружения слабая и шумная.",
            seed=seed + 9003,
            object_shapes=("hook_like_part", "stepped_plate", "hook_like_part", "concave_l_bracket", "long_asymmetric_lever"),
            duplicate_class_groups=((0, 2),),
            rotation_deg=13.0,
            scale=1.06,
            shift_x=24.0,
            shift_y=-18.0,
            perspective=0.00058,
            global_bias_x=20.0,
            global_bias_y=-16.0,
            noise_px=2.3,
            context_points_per_object=18,
            object_bad_points_per_object=84,
            outlier_points=90,
            distractor_count_per_object=3,
            occluder=True,
            weak_context=True,
            context_cluster=0.42,
        ),
        SyntheticMultiScenario(
            name="multi_many_slots_repetitive_background",
            description="Много слотов на одном кадре, повторяющийся фон и похожие приманки вокруг каждого объекта.",
            seed=seed + 9004,
            object_shapes=("long_asymmetric_lever", "long_asymmetric_lever", "stepped_plate", "stepped_plate", "thin_fork", "hook_like_part"),
            duplicate_class_groups=((0, 1), (2, 3)),
            rotation_deg=-15.0,
            scale=1.04,
            shift_x=-28.0,
            shift_y=24.0,
            perspective=0.00072,
            global_bias_x=-22.0,
            global_bias_y=18.0,
            noise_px=2.6,
            context_points_per_object=34,
            object_bad_points_per_object=70,
            outlier_points=110,
            distractor_count_per_object=2,
            occluder=True,
            context_cluster=0.55,
        ),
    ]

    if total <= len(curated):
        return curated[:total]

    rng = random.Random(seed + 91000)
    scenarios = list(curated)
    nightmare = profile == "nightmare"
    shape_names = list(_OBJECT_SHAPE_NAMES)
    for index in range(total - len(curated)):
        object_count = rng.randint(3, 7 if nightmare else 5)
        use_duplicates = rng.random() < (0.68 if nightmare else 0.46)
        if use_duplicates:
            duplicate_shape = rng.choice(shape_names)
            object_shapes = [duplicate_shape if i < max(2, object_count // 2) else rng.choice(shape_names) for i in range(object_count)]
            duplicate_group = tuple(i for i, shape in enumerate(object_shapes) if shape == duplicate_shape)
            duplicate_groups = (duplicate_group,) if len(duplicate_group) >= 2 else ()
        else:
            object_shapes = [rng.choice(shape_names) for _ in range(object_count)]
            duplicate_groups = ()

        weak_context = rng.random() < (0.20 if nightmare else 0.12)
        scenarios.append(
            SyntheticMultiScenario(
                name=f"multi_{profile}_stress_{index + 1:03d}",
                description=(
                    "Мультиобъектная прожарка: несколько разных/одинаковых деталей, "
                    "общая сцена, приманки, шум и ложные feature-точки."
                ),
                seed=seed + 10000 + index,
                object_shapes=tuple(object_shapes),
                duplicate_class_groups=duplicate_groups,
                rotation_deg=rng.uniform(-19.0, 19.0) if nightmare else rng.uniform(-14.0, 14.0),
                scale=rng.uniform(0.90, 1.12) if nightmare else rng.uniform(0.94, 1.07),
                shift_x=rng.uniform(-42.0, 42.0) if nightmare else rng.uniform(-30.0, 30.0),
                shift_y=rng.uniform(-36.0, 36.0) if nightmare else rng.uniform(-24.0, 24.0),
                perspective=rng.uniform(-0.00085, 0.00090) if nightmare else rng.uniform(-0.00055, 0.00060),
                global_bias_x=rng.uniform(-28.0, 28.0) if nightmare else rng.uniform(-18.0, 18.0),
                global_bias_y=rng.uniform(-26.0, 26.0) if nightmare else rng.uniform(-16.0, 16.0),
                noise_px=rng.uniform(0.8, 3.4) if nightmare else rng.uniform(0.6, 2.2),
                context_points_per_object=(rng.randint(4, 14) if weak_context else rng.randint(24, 58)),
                object_bad_points_per_object=rng.randint(28, 92),
                outlier_points=rng.randint(36, 128) if nightmare else rng.randint(18, 64),
                distractor_count_per_object=rng.randint(1, 3),
                occluder=rng.random() < (0.55 if nightmare else 0.35),
                weak_context=weak_context,
                context_cluster=rng.uniform(0.0, 0.72) if rng.random() < 0.38 else 0.0,
            )
        )
    return scenarios


def _build_multi_scene(scenario: SyntheticMultiScenario) -> _SyntheticMultiScene:
    width, height = _CANVAS_SIZE
    rng = np.random.default_rng(scenario.seed)
    reference_polygons = _multi_reference_polygons(scenario)
    true_h = _build_multi_homography(scenario, width=width, height=height)
    approximate_h = true_h.copy().astype(np.float32)
    approximate_h[0, 2] += float(scenario.global_bias_x)
    approximate_h[1, 2] += float(scenario.global_bias_y)

    objects: list[_SyntheticObjectInstance] = []
    all_distractors: list[PolygonPoints] = []
    class_keys = _multi_class_keys(scenario)
    for object_index, (shape, polygon) in enumerate(zip(scenario.object_shapes, reference_polygons, strict=True)):
        gt_polygon = project_polygon(polygon, true_h)
        distractors = _target_distractor_polygons(
            true_h,
            object_polygon=polygon,
            count=scenario.distractor_count_per_object,
        )
        all_distractors.extend(distractors)
        objects.append(
            _SyntheticObjectInstance(
                index=object_index,
                name=f"object_{object_index + 1}_{shape}",
                class_key=class_keys[object_index],
                object_shape=shape,
                reference_polygon=polygon,
                ground_truth_polygon=gt_polygon,
                distractor_polygons=distractors,
            )
        )

    reference = _draw_reference_scene_multi(
        width=width,
        height=height,
        objects=objects,
    )
    target = _draw_target_scene_multi(
        width=width,
        height=height,
        homography=true_h,
        objects=objects,
        occluder=scenario.occluder,
    )

    context_reference_sets: list[np.ndarray] = []
    object_bad_reference_sets: list[np.ndarray] = []
    all_reference_polygons = [obj.reference_polygon for obj in objects]
    for obj in objects:
        context_reference_sets.append(
            _sample_context_points_around_object(
                rng,
                count=scenario.context_points_per_object,
                reference_polygon=obj.reference_polygon,
                all_object_polygons=all_reference_polygons,
                width=width,
                height=height,
                cluster_bias=scenario.context_cluster,
            )
        )
        object_bad_reference_sets.append(
            _sample_inside_polygon(
                rng,
                polygon=obj.reference_polygon,
                count=scenario.object_bad_points_per_object,
            )
        )

    context_ref = _concat_points(context_reference_sets)
    context_frame = _transform_points(context_ref, true_h)
    context_frame += rng.normal(0.0, scenario.noise_px, size=context_frame.shape).astype(np.float32)

    object_bad_ref = _concat_points(object_bad_reference_sets)
    object_bad_frame = _bad_object_frame_points(
        rng,
        reference_points=object_bad_ref,
        homography=true_h,
        distractors=all_distractors,
        noise_px=scenario.noise_px,
    )

    outlier_ref = rng.uniform(
        low=(0.0, 0.0),
        high=(float(width - 1), float(height - 1)),
        size=(scenario.outlier_points, 2),
    ).astype(np.float32)
    outlier_frame = rng.uniform(
        low=(0.0, 0.0),
        high=(float(width - 1), float(height - 1)),
        size=(scenario.outlier_points, 2),
    ).astype(np.float32)

    reference_points = np.concatenate([context_ref, object_bad_ref, outlier_ref], axis=0).astype(np.float32)
    frame_points = np.concatenate([context_frame, object_bad_frame, outlier_frame], axis=0).astype(np.float32)

    return _SyntheticMultiScene(
        scenario=scenario,
        reference_image=reference,
        target_image=target,
        true_homography=true_h,
        approximate_homography=approximate_h,
        objects=objects,
        reference_points=reference_points,
        frame_points=frame_points,
        context_reference_points=context_ref,
        context_frame_points=context_frame,
        object_bad_reference_points=object_bad_ref,
        object_bad_frame_points=object_bad_frame,
    )


def _build_case_projection_data(
    scene: _SyntheticScene | _SyntheticMultiScene,
    *,
    feature_source: str,
) -> LocalProjectionData:
    if feature_source != "real":
        raise ValueError(
            "synthetic_context_test now supports only feature_source='real'. "
            "Generated fake match-pairs were removed from this end-to-end test."
        )

    return _real_feature_projection_data(scene)


def _match_synthetic_expected_segments(
    expected: list[ExpectedSegment],
    *,
    projection_data: LocalProjectionData,
    frame_size: tuple[int, int],
    projection_pipeline: str,
) -> list[SegmentMatch]:
    if projection_pipeline == "v2":
        return transfer_missing_segments_v2(
            expected,
            projection_data=projection_data,
            frame_size=frame_size,
        )

    return match_segments(
        expected,
        [],
        projection_data.global_homography,
        frame_size=frame_size,
        projection_data=projection_data,
    )


def _scene_reference_polygons(
    scene: _SyntheticScene | _SyntheticMultiScene,
) -> list[PolygonPoints]:
    if isinstance(scene, _SyntheticMultiScene):
        return [obj.reference_polygon for obj in scene.objects]
    return [scene.reference_polygon]


def _real_feature_projection_data(
    scene: _SyntheticScene | _SyntheticMultiScene,
) -> LocalProjectionData:
    return _real_feature_projection_data_for_target(scene, scene.target_image)


def _real_feature_projection_data_for_target(
    scene: _SyntheticScene | _SyntheticMultiScene,
    target_image: np.ndarray,
) -> LocalProjectionData:
    reference_selection_grid = (
        SUPERPOINT_REFERENCE_GRID_ROWS,
        SUPERPOINT_REFERENCE_GRID_COLS,
    )
    photo_selection_grid = (SUPERPOINT_PHOTO_GRID_ROWS, SUPERPOINT_PHOTO_GRID_COLS)
    reference_polygons = _scene_reference_polygons(scene)
    masked_reference_image = mask_reference_polygons(
        scene.reference_image,
        reference_polygons,
    )
    original_reference_features = compute_features(
        scene.reference_image,
        max_side=SUPERPOINT_REFERENCE_MAX_SIDE,
        max_keypoints=SUPERPOINT_REFERENCE_MAX_KEYPOINTS,
        selection_grid=reference_selection_grid,
    )
    masked_reference_features = compute_features(
        masked_reference_image,
        max_side=SUPERPOINT_REFERENCE_MAX_SIDE,
        max_keypoints=SUPERPOINT_REFERENCE_MAX_KEYPOINTS,
        selection_grid=reference_selection_grid,
    )
    frame_features = compute_features(
        target_image,
        max_side=SUPERPOINT_PHOTO_MAX_SIDE,
        max_keypoints=SUPERPOINT_PHOTO_MAX_KEYPOINTS,
        selection_grid=photo_selection_grid,
    )
    context = types.SimpleNamespace(
        standard=types.SimpleNamespace(id="synthetic-standard"),
        reference_image=types.SimpleNamespace(id="synthetic-reference"),
        reference_features=original_reference_features,
        selected_classes=[],
    )
    alignment = align_with_features(
        context=context,
        frame_features=frame_features,
        frame_shape=target_image.shape[:2],
        config=AlignmentValidationConfig.industrial(),
        max_keypoints=SUPERPOINT_PHOTO_MAX_KEYPOINTS,
        selection_grid=photo_selection_grid,
        reference_features=masked_reference_features,
        masked_alignment_used=True,
    )
    return LocalProjectionData(
        global_homography=alignment.homography if alignment.is_success else None,
        reference_points=alignment.reference_matches,
        frame_points=alignment.frame_matches,
        frame_size=(target_image.shape[1], target_image.shape[0]),
        frame=target_image,
        reference_frame=scene.reference_image,
        reference_feature_count=masked_reference_features.count,
        frame_feature_count=frame_features.count,
        frame_max_keypoints=SUPERPOINT_PHOTO_MAX_KEYPOINTS,
        frame_keypoint_grid=photo_selection_grid,
        masked_alignment_used=True,
        original_reference_feature_count=original_reference_features.count,
        masked_reference_feature_count=masked_reference_features.count,
        original_reference_keypoints=original_reference_features.keypoints,
        masked_reference_keypoints=masked_reference_features.keypoints,
        frame_keypoints=frame_features.keypoints,
    )



def _match_synthetic_expected_segments_with_yolo_anchor_pose(
    expected: list[ExpectedSegment],
    *,
    annotation_to_object: dict[Any, _SyntheticObjectInstance],
    scene: _SyntheticMultiScene,
    noise_px: float,
    dropout: float,
    min_anchor_objects: int,
    ransac_px: float,
) -> list[SegmentMatch]:
    """Project every missing slot from other synthetic YOLO object anchors.

    This deliberately does not match the missing object's local crop.  For each
    expected object we pretend that YOLO detected the other visible objects in
    the product, estimate a planar pose with RANSAC, and then project the current
    reference slot.  It is an oracle-style synthetic bridge for the next real
    pipeline where YOLO/segmentation detections become pose anchors.
    """

    matches: list[SegmentMatch] = []
    dropout = min(max(float(dropout), 0.0), 0.95)
    min_anchor_objects = max(1, int(min_anchor_objects))
    ransac_px = max(1.0, float(ransac_px))
    noise_px = max(0.0, float(noise_px))

    for item in expected:
        target_obj = annotation_to_object[item.annotation_id]
        rng = np.random.default_rng(
            int(scene.scenario.seed) * 104729 + int(target_obj.index) * 1009 + 17
        )
        anchor_objects: list[_SyntheticObjectInstance] = []
        for anchor in scene.objects:
            if anchor.index == target_obj.index:
                continue
            if dropout > 0.0 and float(rng.random()) < dropout:
                continue
            anchor_objects.append(anchor)

        if len(anchor_objects) < min_anchor_objects:
            matches.append(
                _synthetic_yolo_anchor_pose_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug=_synthetic_yolo_anchor_pose_debug(
                        accepted=False,
                        reason="too_few_anchor_objects",
                        anchor_count=len(anchor_objects),
                        min_anchor_count=min_anchor_objects,
                    ),
                )
            )
            continue

        reference_points, frame_points = _synthetic_yolo_anchor_pose_points(
            anchor_objects,
            noise_px=noise_px,
            rng=rng,
        )
        if len(reference_points) < 4 or len(frame_points) < 4:
            matches.append(
                _synthetic_yolo_anchor_pose_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug=_synthetic_yolo_anchor_pose_debug(
                        accepted=False,
                        reason="too_few_anchor_points",
                        anchor_count=len(anchor_objects),
                        min_anchor_count=min_anchor_objects,
                        point_count=len(reference_points),
                    ),
                )
            )
            continue

        pose = _estimate_synthetic_yolo_anchor_pose(
            reference_points,
            frame_points,
            ransac_px=ransac_px,
        )
        if pose is None:
            matches.append(
                _synthetic_yolo_anchor_pose_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug=_synthetic_yolo_anchor_pose_debug(
                        accepted=False,
                        reason="pose_estimation_failed",
                        anchor_count=len(anchor_objects),
                        min_anchor_count=min_anchor_objects,
                        point_count=len(reference_points),
                    ),
                )
            )
            continue

        matrix, model_source, inliers, median_error = pose
        inlier_ratio = float(inliers / max(1, len(reference_points)))
        if inliers < max(8, min_anchor_objects * 4):
            matches.append(
                _synthetic_yolo_anchor_pose_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug=_synthetic_yolo_anchor_pose_debug(
                        accepted=False,
                        reason="too_few_pose_inliers",
                        anchor_count=len(anchor_objects),
                        min_anchor_count=min_anchor_objects,
                        point_count=len(reference_points),
                        inliers=inliers,
                        inlier_ratio=inlier_ratio,
                        median_error=median_error,
                        model_source=model_source,
                    ),
                )
            )
            continue
        if inlier_ratio < 0.50:
            matches.append(
                _synthetic_yolo_anchor_pose_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug=_synthetic_yolo_anchor_pose_debug(
                        accepted=False,
                        reason="low_pose_inlier_ratio",
                        anchor_count=len(anchor_objects),
                        min_anchor_count=min_anchor_objects,
                        point_count=len(reference_points),
                        inliers=inliers,
                        inlier_ratio=inlier_ratio,
                        median_error=median_error,
                        model_source=model_source,
                    ),
                )
            )
            continue
        if not math.isfinite(median_error) or median_error > max(5.0, ransac_px * 0.95):
            matches.append(
                _synthetic_yolo_anchor_pose_match(
                    item,
                    status="missing",
                    polygon=None,
                    debug=_synthetic_yolo_anchor_pose_debug(
                        accepted=False,
                        reason="pose_error_too_large",
                        anchor_count=len(anchor_objects),
                        min_anchor_count=min_anchor_objects,
                        point_count=len(reference_points),
                        inliers=inliers,
                        inlier_ratio=inlier_ratio,
                        median_error=median_error,
                        model_source=model_source,
                    ),
                )
            )
            continue

        projected_polygon = project_polygon(item.reference_polygon, matrix)
        matches.append(
            _synthetic_yolo_anchor_pose_match(
                item,
                status="missing",
                polygon=projected_polygon,
                debug=_synthetic_yolo_anchor_pose_debug(
                    accepted=True,
                    reason="v4_yolo_anchor_pose_confirmed",
                    anchor_count=len(anchor_objects),
                    min_anchor_count=min_anchor_objects,
                    point_count=len(reference_points),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    median_error=median_error,
                    model_source=model_source,
                ),
            )
        )

    return matches


def _synthetic_yolo_anchor_pose_points(
    anchor_objects: Sequence[_SyntheticObjectInstance],
    *,
    noise_px: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    reference_points: list[list[float]] = []
    frame_points: list[list[float]] = []
    for anchor in anchor_objects:
        ref_poly = np.asarray(anchor.reference_polygon, dtype=np.float32).reshape(-1, 2)
        frame_poly = np.asarray(anchor.ground_truth_polygon, dtype=np.float32).reshape(-1, 2)
        if len(ref_poly) != len(frame_poly) or len(ref_poly) < 3:
            continue
        if noise_px > 0:
            jitter = rng.normal(0.0, noise_px, size=frame_poly.shape).astype(np.float32)
            frame_poly = frame_poly + jitter
        ref_center = np.mean(ref_poly, axis=0, keepdims=True)
        frame_center = np.mean(frame_poly, axis=0, keepdims=True)
        ref_augmented = np.concatenate([ref_poly, ref_center], axis=0)
        frame_augmented = np.concatenate([frame_poly, frame_center], axis=0)
        reference_points.extend(ref_augmented.astype(float).tolist())
        frame_points.extend(frame_augmented.astype(float).tolist())
    if not reference_points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return (
        np.asarray(reference_points, dtype=np.float32).reshape(-1, 2),
        np.asarray(frame_points, dtype=np.float32).reshape(-1, 2),
    )


def _estimate_synthetic_yolo_anchor_pose(
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    *,
    ransac_px: float,
) -> tuple[np.ndarray, str, int, float] | None:
    if len(reference_points) < 4 or len(frame_points) < 4:
        return None

    homography, homography_mask = cv2.findHomography(
        reference_points,
        frame_points,
        cv2.RANSAC,
        ransac_px,
        maxIters=2000,
        confidence=0.995,
    )
    if homography is not None and homography_mask is not None:
        homography = np.asarray(homography, dtype=np.float32).reshape(3, 3)
        if np.all(np.isfinite(homography)) and abs(float(homography[2, 2])) > 1e-6:
            homography = homography / float(homography[2, 2])
            inlier_mask = np.asarray(homography_mask, dtype=bool).reshape(-1)
            inliers = int(np.count_nonzero(inlier_mask))
            median_error = _pose_reprojection_median_error(
                homography,
                reference_points,
                frame_points,
                inlier_mask,
            )
            return homography, "homography", inliers, median_error

    affine, affine_mask = cv2.estimateAffinePartial2D(
        reference_points,
        frame_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px,
        maxIters=2000,
        confidence=0.995,
        refineIters=20,
    )
    if affine is None or affine_mask is None:
        return None
    affine = np.asarray(affine, dtype=np.float32).reshape(2, 3)
    if not np.all(np.isfinite(affine)):
        return None
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, :] = affine
    inlier_mask = np.asarray(affine_mask, dtype=bool).reshape(-1)
    inliers = int(np.count_nonzero(inlier_mask))
    median_error = _pose_reprojection_median_error(
        matrix,
        reference_points,
        frame_points,
        inlier_mask,
    )
    return matrix, "similarity_affine", inliers, median_error


def _pose_reprojection_median_error(
    matrix: np.ndarray,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    inlier_mask: np.ndarray,
) -> float:
    if len(reference_points) == 0:
        return float("inf")
    points_h = np.concatenate(
        [
            reference_points.astype(np.float32),
            np.ones((len(reference_points), 1), dtype=np.float32),
        ],
        axis=1,
    )
    projected = (np.asarray(matrix, dtype=np.float32).reshape(3, 3) @ points_h.T).T
    denom = projected[:, 2:3]
    finite = np.abs(denom[:, 0]) > 1e-6
    projected_xy = np.empty_like(reference_points, dtype=np.float32)
    projected_xy[:] = np.nan
    projected_xy[finite] = projected[finite, :2] / denom[finite]
    errors = np.linalg.norm(projected_xy - frame_points.astype(np.float32), axis=1)
    if len(inlier_mask) == len(errors) and np.any(inlier_mask):
        errors = errors[inlier_mask]
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return float("inf")
    return float(np.median(errors))


def _synthetic_yolo_anchor_pose_match(
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


def _synthetic_yolo_anchor_pose_debug(
    *,
    accepted: bool,
    reason: str,
    anchor_count: int,
    min_anchor_count: int,
    point_count: int = 0,
    inliers: int = 0,
    inlier_ratio: float | None = None,
    median_error: float | None = None,
    model_source: str | None = None,
) -> dict[str, Any]:
    projection = "v4_yolo_anchor_pose" if accepted else "v2_unconfirmed"
    debug: dict[str, Any] = {
        "projection": projection,
        "reason_code": reason,
        "missing_polygon_projection": projection,
        "missing_polygon_projection_safety": "confirmed" if accepted else "unsafe_hidden",
        "missing_polygon_hidden_reason": None if accepted else reason,
        "missing_polygon_feature_support": int(anchor_count),
        "missing_polygon_feature_total": int(point_count),
        "missing_polygon_candidate_count": int(anchor_count),
        "missing_polygon_inliers": int(inliers),
        "missing_polygon_median_error": median_error,
        "v4_yolo_anchor_pose_attempted": True,
        "v4_yolo_anchor_pose_accepted": bool(accepted),
        "v4_yolo_anchor_pose_reason": reason,
        "v4_yolo_anchor_pose_anchor_count": int(anchor_count),
        "v4_yolo_anchor_pose_min_anchor_count": int(min_anchor_count),
        "v4_yolo_anchor_pose_point_count": int(point_count),
        "v4_yolo_anchor_pose_inliers": int(inliers),
        "v4_yolo_anchor_pose_inlier_ratio": inlier_ratio,
        "v4_yolo_anchor_pose_median_error": median_error,
        "v4_yolo_anchor_pose_model_source": model_source,
    }
    return debug


def _run_case(
    *,
    index: int,
    scene: _SyntheticScene,
    images_dir: Path,
    feature_source: str,
    projection_pipeline: str,
    synthetic_yolo_anchor_pose: bool = False,
    synthetic_yolo_anchor_noise_px: float = 1.5,
    synthetic_yolo_anchor_dropout: float = 0.06,
    synthetic_yolo_anchor_min_anchors: int = 2,
    synthetic_yolo_anchor_ransac_px: float = 6.0,
) -> SyntheticResult:
    # The single-object synthetic cases cannot build pose from other objects.
    # Keep these arguments accepted so the same CLI can run single + multi suites.
    del (
        synthetic_yolo_anchor_pose,
        synthetic_yolo_anchor_noise_px,
        synthetic_yolo_anchor_dropout,
        synthetic_yolo_anchor_min_anchors,
        synthetic_yolo_anchor_ransac_px,
    )
    expected = [
        ExpectedSegment(
            annotation_id=uuid4(),
            segment_class_id=uuid4(),
            class_key="synthetic-lever",
            name="Synthetic lever",
            hue=0,
            reference_polygon=scene.reference_polygon,
        )
    ]
    projection_data = _build_case_projection_data(
        scene,
        feature_source=feature_source,
    )
    matches = _match_synthetic_expected_segments(
        expected,
        projection_data=projection_data,
        frame_size=_CANVAS_SIZE,
        projection_pipeline=projection_pipeline,
    )
    match = matches[0] if matches else None
    predicted_polygon = match.expected_polygon if match is not None else None
    debug = match.debug if match is not None and match.debug is not None else {}

    iou = _polygon_iou(predicted_polygon, scene.ground_truth_polygon)
    center_drift = _polygon_center_distance(predicted_polygon, scene.ground_truth_polygon)
    area_ratio = _polygon_area_ratio(predicted_polygon, scene.ground_truth_polygon)
    axis_angle_error_deg, major_length_ratio = _polygon_axis_delta(
        predicted_polygon,
        scene.ground_truth_polygon,
    )
    gt_distance = center_drift
    nearest_distractor_distance = _nearest_distractor_distance(
        predicted_polygon,
        scene.distractor_polygons,
    )
    closer_to_distractor = (
        nearest_distractor_distance is not None
        and nearest_distractor_distance + 1.0 < gt_distance
    )

    projection = str(debug.get("missing_polygon_projection") or debug.get("projection") or "none")
    reason_code = debug.get("reason_code")
    used_context_refinement = _projection_uses_context_refinement(projection)
    used_edge_refinement = bool(debug.get("missing_polygon_edge_snapped"))
    unsafe_hidden = debug.get("missing_polygon_projection_safety") == "unsafe_hidden"
    hidden_reason = (
        str(debug.get("missing_polygon_hidden_reason"))
        if debug.get("missing_polygon_hidden_reason") is not None
        else None
    )
    support = _int_debug(
        debug.get("missing_polygon_feature_support"),
        debug.get("slot_feature_support"),
        debug.get("slot", {}).get("feature_support") if isinstance(debug.get("slot"), dict) else None,
    )
    total = _int_debug(
        debug.get("missing_polygon_feature_total"),
        debug.get("slot_feature_total"),
        debug.get("slot", {}).get("feature_total") if isinstance(debug.get("slot"), dict) else None,
    )
    candidate_count = _int_debug(debug.get("missing_polygon_candidate_count"))
    inliers = _int_debug(debug.get("missing_polygon_inliers"))
    median_error = _float_or_none(debug.get("missing_polygon_median_error"))
    anchor_release_fields = _anchor_release_metric_fields(debug)
    fallback_fields = _fallback_metric_fields(debug, reason_code=reason_code)
    hidden_shadow_fields = _hidden_shadow_metric_fields(
        debug,
        scene=scene,
        ground_truth_polygon=scene.ground_truth_polygon,
        distractor_polygons=scene.distractor_polygons,
        support=support,
    )
    feature_telemetry_fields = _feature_telemetry_metric_fields(
        debug,
        projection_data=projection_data,
    )

    notes = _case_notes(
        scene=scene,
        status=match.status if match is not None else "none",
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
        closer_to_distractor=closer_to_distractor,
        support=support,
        projection=projection,
        used_context_refinement=used_context_refinement,
        unsafe_hidden=unsafe_hidden,
    )
    passed = not notes
    dangerous_projection = _is_dangerous_projection(
        predicted_polygon=predicted_polygon,
        center_drift=center_drift,
        area_ratio=area_ratio,
        closer_to_distractor=closer_to_distractor,
        unsafe_hidden=unsafe_hidden,
    )
    safety_passed = not dangerous_projection
    candidate_oracle_fields = _candidate_oracle_metric_fields(
        debug,
        scene=scene,
        current_projection=projection,
        current_polygon=predicted_polygon,
        ground_truth_polygon=scene.ground_truth_polygon,
        distractor_polygons=scene.distractor_polygons,
        support=support,
        current_passed=passed,
        current_dangerous=dangerous_projection,
    )
    crop_verification_fields = _object_crop_verification_metric_fields(
        debug,
        scene=scene,
        current_projection=projection,
        current_polygon=predicted_polygon,
        ground_truth_polygon=scene.ground_truth_polygon,
        distractor_polygons=scene.distractor_polygons,
        support=support,
    )

    image_path = images_dir / f"case_{index:03d}_{_safe_name(scene.scenario.name)}.png"
    panel = _render_case_panel(
        index=index,
        scene=scene,
        projection_data=projection_data,
        predicted_polygon=predicted_polygon,
        result_status="PASS" if passed else "FAIL",
        metrics={
            "IoU": f"{iou:.3f}",
            "drift": f"{center_drift:.1f}px",
            "area": f"{area_ratio:.2f}x",
            "support": f"{support}/{total}",
            "matches": str(_projection_match_count(projection_data)),
            "tilt": _scenario_tilt_label(scene.scenario),
            "projection": projection,
        },
        notes=notes,
    )
    cv2.imwrite(str(image_path), panel)

    keypoints_image_path = images_dir / f"case_{index:03d}_{_safe_name(scene.scenario.name)}_keypoints.png"
    keypoints_panel = _render_keypoint_diagnostic_panel(
        index=index,
        name=scene.scenario.name,
        scenario_kind="single",
        reference_image=scene.reference_image,
        target_image=scene.target_image,
        projection_data=projection_data,
        reference_polygons=[scene.reference_polygon],
        target_polygons=[scene.ground_truth_polygon],
        predicted_polygons=[predicted_polygon] if predicted_polygon else [],
        result_status="PASS" if passed else "FAIL",
    )
    cv2.imwrite(str(keypoints_image_path), keypoints_panel)

    return SyntheticResult(
        index=index,
        name=scene.scenario.name,
        description=scene.scenario.description,
        object_shape=scene.scenario.object_shape,
        passed=passed,
        safety_passed=safety_passed,
        dangerous_projection=dangerous_projection,
        unsafe_hidden=unsafe_hidden,
        status=match.status if match is not None else "none",
        projection=projection,
        reason_code=str(reason_code) if reason_code is not None else None,
        iou=float(iou),
        center_drift_px=float(center_drift),
        area_ratio=float(area_ratio),
        axis_angle_error_deg=(
            float(axis_angle_error_deg)
            if axis_angle_error_deg is not None
            else None
        ),
        major_length_ratio=(
            float(major_length_ratio)
            if major_length_ratio is not None
            else None
        ),
        context_feature_support=support,
        context_feature_total=total,
        missing_candidate_count=candidate_count,
        missing_inliers=inliers,
        missing_median_error=median_error,
        used_context_refinement=used_context_refinement,
        used_edge_refinement=used_edge_refinement,
        closer_to_distractor=closer_to_distractor,
        nearest_distractor_distance_px=(
            float(nearest_distractor_distance)
            if nearest_distractor_distance is not None
            else None
        ),
        gt_distance_px=float(gt_distance),
        image_path=str(image_path.relative_to(image_path.parent.parent)),
        notes=notes,
        keypoints_image_path=str(keypoints_image_path.relative_to(keypoints_image_path.parent.parent)),
        object_failures=0 if passed else 1,
        reference_keypoints_total=feature_telemetry_fields["reference_keypoints_total"],
        frame_keypoints_total=feature_telemetry_fields["frame_keypoints_total"],
        frame_max_keypoints=feature_telemetry_fields["frame_max_keypoints"],
        frame_keypoint_grid_rows=feature_telemetry_fields["frame_keypoint_grid_rows"],
        frame_keypoint_grid_cols=feature_telemetry_fields["frame_keypoint_grid_cols"],
        lightglue_reference_matches_total=feature_telemetry_fields[
            "lightglue_reference_matches_total"
        ],
        lightglue_frame_matches_total=feature_telemetry_fields[
            "lightglue_frame_matches_total"
        ],
        masked_alignment_used=feature_telemetry_fields["masked_alignment_used"],
        original_reference_keypoints_total=feature_telemetry_fields[
            "original_reference_keypoints_total"
        ],
        masked_reference_keypoints_total=feature_telemetry_fields[
            "masked_reference_keypoints_total"
        ],
        masked_lightglue_matches_total=feature_telemetry_fields[
            "masked_lightglue_matches_total"
        ],
        object_metrics=[
            asdict(
                SyntheticObjectMetric(
                    name=scene.scenario.name,
                    object_shape=scene.scenario.object_shape,
                    status=match.status if match is not None else "none",
                    projection=projection,
                    passed=passed,
                    safety_passed=safety_passed,
                    dangerous_projection=dangerous_projection,
                    unsafe_hidden=unsafe_hidden,
                    hidden_reason=hidden_reason,
                    iou=float(iou),
                    center_drift_px=float(center_drift),
                    area_ratio=float(area_ratio),
                    axis_angle_error_deg=(
                        float(axis_angle_error_deg)
                        if axis_angle_error_deg is not None
                        else None
                    ),
                    major_length_ratio=(
                        float(major_length_ratio)
                        if major_length_ratio is not None
                        else None
                    ),
                    closer_to_distractor=closer_to_distractor,
                    notes=notes,
                    **feature_telemetry_fields,
                    **hidden_shadow_fields,
                    **fallback_fields,
                    **candidate_oracle_fields,
                    **crop_verification_fields,
                    **anchor_release_fields,
                )
            )
        ],
    )


def _run_multi_case(
    *,
    index: int,
    scene: _SyntheticMultiScene,
    images_dir: Path,
    feature_source: str,
    projection_pipeline: str,
    synthetic_yolo_anchor_pose: bool = False,
    synthetic_yolo_anchor_noise_px: float = 1.5,
    synthetic_yolo_anchor_dropout: float = 0.06,
    synthetic_yolo_anchor_min_anchors: int = 2,
    synthetic_yolo_anchor_ransac_px: float = 6.0,
) -> SyntheticResult:
    expected: list[ExpectedSegment] = []
    annotation_to_object: dict[Any, _SyntheticObjectInstance] = {}
    for obj in scene.objects:
        annotation_id = uuid4()
        annotation_to_object[annotation_id] = obj
        expected.append(
            ExpectedSegment(
                annotation_id=annotation_id,
                segment_class_id=uuid4(),
                class_key=obj.class_key,
                name=obj.name,
                hue=(obj.index * 47) % 360,
                reference_polygon=obj.reference_polygon,
            )
        )

    projection_data = _build_case_projection_data(
        scene,
        feature_source=feature_source,
    )
    if synthetic_yolo_anchor_pose:
        matches = _match_synthetic_expected_segments_with_yolo_anchor_pose(
            expected,
            annotation_to_object=annotation_to_object,
            scene=scene,
            noise_px=synthetic_yolo_anchor_noise_px,
            dropout=synthetic_yolo_anchor_dropout,
            min_anchor_objects=synthetic_yolo_anchor_min_anchors,
            ransac_px=synthetic_yolo_anchor_ransac_px,
        )
    else:
        matches = _match_synthetic_expected_segments(
            expected,
            projection_data=projection_data,
            frame_size=_CANVAS_SIZE,
            projection_pipeline=projection_pipeline,
        )
    matches_by_annotation = {match.annotation_id: match for match in matches}

    object_metrics: list[SyntheticObjectMetric] = []
    predicted_by_object: dict[int, PolygonPoints | None] = {}
    support_total = 0
    feature_total = 0
    candidate_total = 0
    inlier_total = 0
    median_errors: list[float] = []
    all_notes: list[str] = []

    for expected_item in expected:
        obj = annotation_to_object[expected_item.annotation_id]
        match = matches_by_annotation.get(expected_item.annotation_id)
        predicted_polygon = match.expected_polygon if match is not None else None
        predicted_by_object[obj.index] = predicted_polygon
        debug = match.debug if match is not None and match.debug is not None else {}

        iou = _polygon_iou(predicted_polygon, obj.ground_truth_polygon)
        center_drift = _polygon_center_distance(predicted_polygon, obj.ground_truth_polygon)
        area_ratio = _polygon_area_ratio(predicted_polygon, obj.ground_truth_polygon)
        axis_angle_error_deg, major_length_ratio = _polygon_axis_delta(
            predicted_polygon,
            obj.ground_truth_polygon,
        )
        nearest_distractor_distance = _nearest_distractor_distance(
            predicted_polygon,
            obj.distractor_polygons,
        )
        closer_to_distractor = (
            nearest_distractor_distance is not None
            and nearest_distractor_distance + 1.0 < center_drift
        )
        projection = str(debug.get("missing_polygon_projection") or debug.get("projection") or "none")
        reason_code = debug.get("reason_code")
        unsafe_hidden = debug.get("missing_polygon_projection_safety") == "unsafe_hidden"
        hidden_reason = (
            str(debug.get("missing_polygon_hidden_reason"))
            if debug.get("missing_polygon_hidden_reason") is not None
            else None
        )
        support = _int_debug(
            debug.get("missing_polygon_feature_support"),
            debug.get("slot_feature_support"),
            debug.get("slot", {}).get("feature_support") if isinstance(debug.get("slot"), dict) else None,
        )
        total = _int_debug(
            debug.get("missing_polygon_feature_total"),
            debug.get("slot_feature_total"),
            debug.get("slot", {}).get("feature_total") if isinstance(debug.get("slot"), dict) else None,
        )
        candidate_count = _int_debug(debug.get("missing_polygon_candidate_count"))
        inliers = _int_debug(debug.get("missing_polygon_inliers"))
        median_error = _float_or_none(debug.get("missing_polygon_median_error"))
        if median_error is not None:
            median_errors.append(median_error)
        anchor_release_fields = _anchor_release_metric_fields(debug)
        fallback_fields = _fallback_metric_fields(debug, reason_code=reason_code)
        hidden_shadow_fields = _hidden_shadow_metric_fields(
            debug,
            scene=_scene_proxy_for_object(scene, obj),
            ground_truth_polygon=obj.ground_truth_polygon,
            distractor_polygons=obj.distractor_polygons,
            support=support,
        )
        feature_telemetry_fields = _feature_telemetry_metric_fields(
            debug,
            projection_data=projection_data,
        )
        v2_metric_fields = _v2_metric_fields(debug)

        used_context_refinement = _projection_uses_context_refinement(projection)
        notes = _case_notes(
            scene=_scene_proxy_for_object(scene, obj),
            status=match.status if match is not None else "none",
            iou=iou,
            center_drift=center_drift,
            area_ratio=area_ratio,
            axis_angle_error_deg=axis_angle_error_deg,
            major_length_ratio=major_length_ratio,
            closer_to_distractor=closer_to_distractor,
            support=support,
            projection=projection,
            used_context_refinement=used_context_refinement,
            unsafe_hidden=unsafe_hidden,
        )
        dangerous = _is_dangerous_projection(
            predicted_polygon=predicted_polygon,
            center_drift=center_drift,
            area_ratio=area_ratio,
            closer_to_distractor=closer_to_distractor,
            unsafe_hidden=unsafe_hidden,
        )
        object_scene = _scene_proxy_for_object(scene, obj)
        candidate_oracle_fields = _candidate_oracle_metric_fields(
            debug,
            scene=object_scene,
            current_projection=projection,
            current_polygon=predicted_polygon,
            ground_truth_polygon=obj.ground_truth_polygon,
            distractor_polygons=obj.distractor_polygons,
            support=support,
            current_passed=not notes,
            current_dangerous=dangerous,
        )
        crop_verification_fields = _object_crop_verification_metric_fields(
            debug,
            scene=object_scene,
            current_projection=projection,
            current_polygon=predicted_polygon,
            ground_truth_polygon=obj.ground_truth_polygon,
            distractor_polygons=obj.distractor_polygons,
            support=support,
        )
        metric = SyntheticObjectMetric(
            name=obj.name,
            object_shape=obj.object_shape,
            status=match.status if match is not None else "none",
            projection=projection,
            passed=not notes,
            safety_passed=not dangerous,
            dangerous_projection=dangerous,
            unsafe_hidden=unsafe_hidden,
            hidden_reason=hidden_reason,
            iou=float(iou),
            center_drift_px=float(center_drift),
            area_ratio=float(area_ratio),
            axis_angle_error_deg=(
                float(axis_angle_error_deg)
                if axis_angle_error_deg is not None
                else None
            ),
            major_length_ratio=(
                float(major_length_ratio)
                if major_length_ratio is not None
                else None
            ),
            closer_to_distractor=closer_to_distractor,
            notes=notes,
            **v2_metric_fields,
            **feature_telemetry_fields,
            **hidden_shadow_fields,
            **fallback_fields,
            **candidate_oracle_fields,
            **crop_verification_fields,
            **anchor_release_fields,
        )
        object_metrics.append(metric)
        support_total += support
        feature_total += total
        candidate_total += candidate_count
        inlier_total += inliers
        for note in notes:
            all_notes.append(f"{obj.name}: {note}")

    finite_ious = [metric.iou for metric in object_metrics if math.isfinite(metric.iou)]
    finite_drifts = [metric.center_drift_px for metric in object_metrics if math.isfinite(metric.center_drift_px)]
    finite_area = [metric.area_ratio for metric in object_metrics if math.isfinite(metric.area_ratio)]
    passed = all(metric.passed for metric in object_metrics)
    dangerous_projection = any(metric.dangerous_projection for metric in object_metrics)
    safety_passed = not dangerous_projection
    unsafe_hidden = any(metric.unsafe_hidden for metric in object_metrics)
    projection = _projection_summary([metric.projection for metric in object_metrics])
    status = "missing" if all(metric.status == "missing" for metric in object_metrics) else "mixed"
    case_feature_telemetry_fields = _feature_telemetry_metric_fields(
        {},
        projection_data=projection_data,
    )

    image_path = images_dir / f"case_{index:03d}_{_safe_name(scene.scenario.name)}.png"
    panel = _render_multi_case_panel(
        index=index,
        scene=scene,
        projection_data=projection_data,
        predicted_by_object=predicted_by_object,
        result_status="PASS" if passed else "FAIL",
        metrics={
            "objects": str(len(scene.objects)),
            "mean IoU": f"{(float(np.mean(finite_ious)) if finite_ious else 0.0):.3f}",
            "max drift": f"{(float(np.max(finite_drifts)) if finite_drifts else float('inf')):.1f}px",
            "hidden": str(sum(1 for metric in object_metrics if metric.unsafe_hidden)),
            "matches": str(_projection_match_count(projection_data)),
            "tilt": _scenario_tilt_label(scene.scenario),
            "projection": projection,
        },
        notes=all_notes,
    )
    cv2.imwrite(str(image_path), panel)

    keypoints_image_path = images_dir / f"case_{index:03d}_{_safe_name(scene.scenario.name)}_keypoints.png"
    keypoints_panel = _render_keypoint_diagnostic_panel(
        index=index,
        name=scene.scenario.name,
        scenario_kind="multi",
        reference_image=scene.reference_image,
        target_image=scene.target_image,
        projection_data=projection_data,
        reference_polygons=[obj.reference_polygon for obj in scene.objects],
        target_polygons=[obj.ground_truth_polygon for obj in scene.objects],
        predicted_polygons=[poly for poly in predicted_by_object.values() if poly],
        result_status="PASS" if passed else "FAIL",
    )
    cv2.imwrite(str(keypoints_image_path), keypoints_panel)

    return SyntheticResult(
        index=index,
        name=scene.scenario.name,
        description=scene.scenario.description,
        object_shape="multi",
        passed=passed,
        safety_passed=safety_passed,
        dangerous_projection=dangerous_projection,
        unsafe_hidden=unsafe_hidden,
        status=status,
        projection=projection,
        reason_code=None,
        iou=float(np.mean(finite_ious)) if finite_ious else 0.0,
        center_drift_px=float(np.max(finite_drifts)) if finite_drifts else float("inf"),
        area_ratio=float(np.mean(finite_area)) if finite_area else 0.0,
        axis_angle_error_deg=None,
        major_length_ratio=None,
        context_feature_support=support_total,
        context_feature_total=feature_total,
        missing_candidate_count=candidate_total,
        missing_inliers=inlier_total,
        missing_median_error=float(np.mean(median_errors)) if median_errors else None,
        used_context_refinement=any(_projection_uses_context_refinement(metric.projection) for metric in object_metrics),
        used_edge_refinement=False,
        closer_to_distractor=any(metric.closer_to_distractor for metric in object_metrics),
        nearest_distractor_distance_px=None,
        gt_distance_px=float(np.max(finite_drifts)) if finite_drifts else float("inf"),
        image_path=str(image_path.relative_to(image_path.parent.parent)),
        notes=all_notes[:12],
        keypoints_image_path=str(keypoints_image_path.relative_to(keypoints_image_path.parent.parent)),
        scenario_kind="multi",
        object_count=len(scene.objects),
        object_shapes=[obj.object_shape for obj in scene.objects],
        object_failures=sum(1 for metric in object_metrics if not metric.passed),
        object_metrics=[asdict(metric) for metric in object_metrics],
        reference_keypoints_total=case_feature_telemetry_fields["reference_keypoints_total"],
        frame_keypoints_total=case_feature_telemetry_fields["frame_keypoints_total"],
        frame_max_keypoints=case_feature_telemetry_fields["frame_max_keypoints"],
        frame_keypoint_grid_rows=case_feature_telemetry_fields["frame_keypoint_grid_rows"],
        frame_keypoint_grid_cols=case_feature_telemetry_fields["frame_keypoint_grid_cols"],
        lightglue_reference_matches_total=case_feature_telemetry_fields[
            "lightglue_reference_matches_total"
        ],
        lightglue_frame_matches_total=case_feature_telemetry_fields[
            "lightglue_frame_matches_total"
        ],
        masked_alignment_used=case_feature_telemetry_fields["masked_alignment_used"],
        original_reference_keypoints_total=case_feature_telemetry_fields[
            "original_reference_keypoints_total"
        ],
        masked_reference_keypoints_total=case_feature_telemetry_fields[
            "masked_reference_keypoints_total"
        ],
        masked_lightglue_matches_total=case_feature_telemetry_fields[
            "masked_lightglue_matches_total"
        ],
    )


def _is_dangerous_projection(
    *,
    predicted_polygon: PolygonPoints | None,
    center_drift: float,
    area_ratio: float,
    closer_to_distractor: bool,
    unsafe_hidden: bool,
) -> bool:
    if unsafe_hidden or predicted_polygon is None:
        return False
    if closer_to_distractor:
        return True
    if center_drift > 60.0:
        return True
    return area_ratio < 0.35 or area_ratio > 2.75


def _debug_polygon_points(value: Any) -> PolygonPoints | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    polygon: PolygonPoints = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            polygon.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return polygon if len(polygon) >= 3 else None


def _candidate_oracle_projection_name(source: str) -> str:
    normalized = source
    if normalized.startswith("current:"):
        normalized = normalized.removeprefix("current:")
    if normalized.startswith("selected:"):
        normalized = normalized.removeprefix("selected:")
    if normalized.startswith("hidden_shadow:"):
        normalized = normalized.removeprefix("hidden_shadow:")
    if normalized.startswith("selective_hidden_release:"):
        release = normalized.removeprefix("selective_hidden_release:")
        if release == "global_fallback":
            return "expected_slot_global_fallback_hidden_release"
        if release == "expected_slot_agreement":
            return "expected_slot_agreement_hidden_release"
        return release
    mapping = {
        "global_fallback": "expected_slot_global_fallback",
        "context_translation_rescue": "expected_slot_context_translation_rescue",
        "scene_translation_rescue": "expected_slot_scene_translation_rescue",
        "anchor_release": "expected_slot_anchor_release",
        "local_displacement": "expected_slot_local_displacement",
    }
    return mapping.get(normalized, normalized)


def _polygon_from_debug_bbox(value: Any) -> PolygonPoints | None:
    if not isinstance(value, dict):
        return None
    try:
        x = float(value.get("x"))
        y = float(value.get("y"))
        width = float(value.get("w"))
        height = float(value.get("h"))
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def _candidate_oracle_empty_fields(mode: str) -> dict[str, Any]:
    return {
        "candidate_oracle_available_count": 0,
        "candidate_oracle_pass_count": 0,
        "candidate_oracle_safe_pass_count": 0,
        "candidate_oracle_dangerous_count": 0,
        "candidate_oracle_selected_source": None,
        "candidate_oracle_selected_would_pass": False,
        "candidate_oracle_best_source": None,
        "candidate_oracle_best_iou": None,
        "candidate_oracle_best_center_drift_px": None,
        "candidate_oracle_best_area_ratio": None,
        "candidate_oracle_best_would_pass": False,
        "candidate_oracle_best_dangerous": False,
        "candidate_oracle_has_safe_alternative": False,
        "candidate_oracle_safe_gain_source": None,
        "candidate_oracle_failure_mode": mode,
        "candidate_oracle_sources": [],
    }


def _candidate_oracle_metric_fields(
    debug: dict[str, Any],
    *,
    scene: _SyntheticScene,
    current_projection: str,
    current_polygon: PolygonPoints | None,
    ground_truth_polygon: PolygonPoints,
    distractor_polygons: Sequence[PolygonPoints],
    support: int,
    current_passed: bool,
    current_dangerous: bool,
) -> dict[str, Any]:
    """Evaluate already available candidates against synthetic ground truth.

    This is report-only oracle logic.  It does not change matcher behavior and it
    does not release hidden candidates.  The goal is to answer whether a failed
    result had a safe candidate already available, or whether the current
    LightGlue-only evidence simply has no good option.
    """
    candidate_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, tuple[int, ...]]] = set()

    def add_candidate(source: str, polygon: PolygonPoints | None, *, selected: bool = False) -> None:
        if polygon is None or len(polygon) < 3:
            return
        key = (
            source,
            tuple(
                int(round(coord * 10.0))
                for point in polygon
                for coord in (float(point[0]), float(point[1]))
            ),
        )
        if key in seen_keys:
            return
        seen_keys.add(key)
        projection = _candidate_oracle_projection_name(source)
        iou = _polygon_iou(polygon, ground_truth_polygon)
        center_drift = _polygon_center_distance(polygon, ground_truth_polygon)
        area_ratio = _polygon_area_ratio(polygon, ground_truth_polygon)
        axis_angle_error_deg, major_length_ratio = _polygon_axis_delta(
            polygon,
            ground_truth_polygon,
        )
        nearest_distractor_distance = _nearest_distractor_distance(
            polygon,
            distractor_polygons,
        )
        closer_to_distractor = (
            nearest_distractor_distance is not None
            and nearest_distractor_distance + 1.0 < center_drift
        )
        notes = _case_notes(
            scene=scene,
            status="missing",
            iou=iou,
            center_drift=center_drift,
            area_ratio=area_ratio,
            axis_angle_error_deg=axis_angle_error_deg,
            major_length_ratio=major_length_ratio,
            closer_to_distractor=closer_to_distractor,
            support=support,
            projection=projection,
            used_context_refinement=_projection_uses_context_refinement(projection),
            unsafe_hidden=False,
        )
        dangerous = _is_dangerous_projection(
            predicted_polygon=polygon,
            center_drift=center_drift,
            area_ratio=area_ratio,
            closer_to_distractor=closer_to_distractor,
            unsafe_hidden=False,
        )
        candidate_rows.append(
            {
                "source": source,
                "projection": projection,
                "selected": selected,
                "iou": float(iou),
                "center_drift_px": float(center_drift),
                "area_ratio": float(area_ratio),
                "would_pass": not notes,
                "dangerous": bool(dangerous),
                "safe_pass": bool(not notes and not dangerous),
            }
        )

    if current_polygon is not None and current_projection not in {"none", "unsafe_hidden"}:
        add_candidate(f"current:{current_projection}", current_polygon, selected=True)

    raw_registry = debug.get("missing_polygon_candidate_registry")
    if isinstance(raw_registry, list):
        for raw_item in raw_registry:
            if not isinstance(raw_item, dict) or not bool(raw_item.get("available", True)):
                continue
            source = str(raw_item.get("name") or "candidate")
            polygon = _debug_polygon_points(raw_item.get("polygon"))
            if polygon is None:
                polygon = _polygon_from_debug_bbox(raw_item.get("bbox"))
            add_candidate(source, polygon, selected=bool(raw_item.get("selected")))

    if bool(debug.get("missing_polygon_hidden_shadow_available")):
        hidden_shadow_polygon = _debug_polygon_points(
            debug.get("missing_polygon_hidden_shadow_polygon")
        )
        hidden_shadow_projection = str(
            debug.get("missing_polygon_hidden_shadow_projection")
            or "unsafe_hidden_shadow"
        )
        add_candidate(
            f"hidden_shadow:{hidden_shadow_projection}",
            hidden_shadow_polygon,
        )

    if not candidate_rows:
        return _candidate_oracle_empty_fields(
            "current_passed" if current_passed else "no_available_candidate"
        )

    selected_row = next(
        (row for row in candidate_rows if bool(row.get("selected"))),
        None,
    )
    pass_rows = [row for row in candidate_rows if bool(row["would_pass"])]
    safe_rows = [row for row in candidate_rows if bool(row["safe_pass"])]
    dangerous_rows = [row for row in candidate_rows if bool(row["dangerous"])]
    best_pool = safe_rows or pass_rows or candidate_rows
    best = max(
        best_pool,
        key=lambda row: (
            1 if bool(row["safe_pass"]) else 0,
            1 if bool(row["would_pass"]) else 0,
            0 if bool(row["dangerous"]) else 1,
            float(row["iou"]),
            -float(row["center_drift_px"])
            if math.isfinite(float(row["center_drift_px"]))
            else -1e9,
        ),
    )
    has_safe_alternative = bool(safe_rows)
    best_is_safe = bool(best["safe_pass"])
    selected_would_pass = bool(selected_row and selected_row["would_pass"])
    safe_gain_source = str(best["source"]) if (not current_passed and best_is_safe) else None

    if current_passed:
        failure_mode = "current_passed"
    elif has_safe_alternative:
        failure_mode = "wrong_selection_has_safe_candidate"
    elif current_dangerous or dangerous_rows:
        failure_mode = "no_safe_candidate_dangerous_candidates"
    else:
        failure_mode = "no_safe_candidate"

    return {
        "candidate_oracle_available_count": len(candidate_rows),
        "candidate_oracle_pass_count": len(pass_rows),
        "candidate_oracle_safe_pass_count": len(safe_rows),
        "candidate_oracle_dangerous_count": len(dangerous_rows),
        "candidate_oracle_selected_source": (
            str(selected_row["source"]) if selected_row is not None else None
        ),
        "candidate_oracle_selected_would_pass": selected_would_pass,
        "candidate_oracle_best_source": str(best["source"]),
        "candidate_oracle_best_iou": float(best["iou"]),
        "candidate_oracle_best_center_drift_px": float(best["center_drift_px"]),
        "candidate_oracle_best_area_ratio": float(best["area_ratio"]),
        "candidate_oracle_best_would_pass": bool(best["would_pass"]),
        "candidate_oracle_best_dangerous": bool(best["dangerous"]),
        "candidate_oracle_has_safe_alternative": has_safe_alternative,
        "candidate_oracle_safe_gain_source": safe_gain_source,
        "candidate_oracle_failure_mode": failure_mode,
        "candidate_oracle_sources": [str(row["source"]) for row in candidate_rows],
    }



def _candidate_polygon_rows_from_debug(
    debug: dict[str, Any],
    *,
    current_projection: str,
    current_polygon: PolygonPoints | None,
) -> list[tuple[str, PolygonPoints, bool]]:
    """Collect candidate polygons that already exist in matcher debug output."""
    rows: list[tuple[str, PolygonPoints, bool]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    def add(source: str, polygon: PolygonPoints | None, *, selected: bool = False) -> None:
        if polygon is None or len(polygon) < 3:
            return
        key = (
            source,
            tuple(
                int(round(float(coord) * 10.0))
                for point in polygon
                for coord in (float(point[0]), float(point[1]))
            ),
        )
        if key in seen:
            return
        seen.add(key)
        rows.append((source, polygon, selected))

    if current_polygon is not None and current_projection not in {"none", "unsafe_hidden"}:
        add(f"current:{current_projection}", current_polygon, selected=True)

    raw_registry = debug.get("missing_polygon_candidate_registry")
    if isinstance(raw_registry, list):
        for raw_item in raw_registry:
            if not isinstance(raw_item, dict) or not bool(raw_item.get("available", True)):
                continue
            source = str(raw_item.get("name") or "candidate")
            polygon = _debug_polygon_points(raw_item.get("polygon"))
            if polygon is None:
                polygon = _polygon_from_debug_bbox(raw_item.get("bbox"))
            add(source, polygon, selected=bool(raw_item.get("selected")))

    if bool(debug.get("missing_polygon_hidden_shadow_available")):
        shadow_polygon = _debug_polygon_points(debug.get("missing_polygon_hidden_shadow_polygon"))
        shadow_projection = str(
            debug.get("missing_polygon_hidden_shadow_projection")
            or "unsafe_hidden_shadow"
        )
        add(f"hidden_shadow:{shadow_projection}", shadow_polygon)

    return rows


def _points_inside_polygon_mask(points: np.ndarray, polygon: PolygonPoints | None) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    if polygon is None or len(polygon) < 3:
        return np.zeros((len(points),), dtype=bool)
    polygon_np = np.asarray(polygon, dtype=np.float32)
    return np.asarray(
        [_point_inside_polygon((float(point[0]), float(point[1])), polygon_np) for point in points],
        dtype=bool,
    )


def _transform_polygon_affine(
    polygon: PolygonPoints,
    matrix: np.ndarray,
) -> PolygonPoints | None:
    try:
        affine = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
        points = np.asarray(polygon, dtype=np.float32)
        ones = np.ones((len(points), 1), dtype=np.float32)
        transformed = np.hstack([points, ones]) @ affine.T
        return [[float(x), float(y)] for x, y in transformed]
    except (TypeError, ValueError):
        return None


def _translate_polygon(
    polygon: PolygonPoints,
    shift: np.ndarray,
) -> PolygonPoints | None:
    try:
        dx = float(shift[0])
        dy = float(shift[1])
    except (TypeError, ValueError, IndexError):
        return None
    return [[float(x) + dx, float(y) + dy] for x, y in polygon]


def _object_crop_feature_candidate_rows(scene: _SyntheticScene) -> list[dict[str, Any]]:
    """Build object-crop candidates from reference-object feature matches.

    These candidates are shadow-only.  In this synthetic missing-object test a
    strong object-crop match often means "the feature evidence points to a
    distractor", not that the missing slot should be released.  The stats tell us
    whether a future real-image object verifier can add evidence, or whether it
    mainly adds risk.
    """
    ref_points = scene.reference_points
    frame_points = scene.frame_points
    if len(ref_points) == 0 or len(ref_points) != len(frame_points):
        return []

    ref_mask = _points_inside_polygon_mask(ref_points, scene.reference_polygon)
    ref_obj = ref_points[ref_mask].astype(np.float32)
    frame_obj = frame_points[ref_mask].astype(np.float32)
    if len(ref_obj) == 0:
        return []

    rows: list[dict[str, Any]] = []

    def add_row(source: str, polygon: PolygonPoints | None, *, inliers: int, ratio: float | None) -> None:
        if polygon is None or len(polygon) < 3:
            return
        rows.append(
            {
                "source": source,
                "polygon": polygon,
                "inliers": int(max(0, inliers)),
                "inlier_ratio": float(ratio) if ratio is not None and math.isfinite(float(ratio)) else None,
            }
        )

    if len(ref_obj) >= 3:
        try:
            matrix, inlier_mask = cv2.estimateAffinePartial2D(
                ref_obj,
                frame_obj,
                method=cv2.RANSAC,
                ransacReprojThreshold=8.0,
                maxIters=500,
                confidence=0.98,
            )
        except cv2.error:
            matrix, inlier_mask = None, None
        if matrix is not None:
            if inlier_mask is not None:
                inliers = int(np.asarray(inlier_mask).reshape(-1).astype(bool).sum())
            else:
                inliers = len(ref_obj)
            ratio = inliers / max(1, len(ref_obj))
            add_row(
                "object_crop_affine",
                _transform_polygon_affine(scene.reference_polygon, matrix),
                inliers=inliers,
                ratio=ratio,
            )

    if len(ref_obj) >= 1:
        shifts = frame_obj - ref_obj
        median_shift = np.median(shifts, axis=0)
        distances = np.linalg.norm(shifts - median_shift, axis=1)
        inliers = int(np.sum(distances <= 10.0))
        ratio = inliers / max(1, len(ref_obj))
        add_row(
            "object_crop_translation",
            _translate_polygon(scene.reference_polygon, median_shift),
            inliers=inliers,
            ratio=ratio,
        )

    return rows


def _object_crop_candidate_would_pass(
    *,
    iou: float,
    center_drift: float,
    area_ratio: float,
    closer_to_distractor: bool,
    weak_context: bool,
) -> bool:
    min_iou = 0.42 if weak_context else 0.55
    max_drift = 38.0 if weak_context else 28.0
    if not math.isfinite(iou) or not math.isfinite(center_drift):
        return False
    if iou < min_iou or center_drift > max_drift:
        return False
    if area_ratio < 0.55 or area_ratio > 1.75:
        return False
    if closer_to_distractor:
        return False
    return True


def _object_crop_verification_empty_fields(assessment: str) -> dict[str, Any]:
    return {
        "crop_verification_attempted": False,
        "crop_verification_candidate_count": 0,
        "crop_verification_source_count": 0,
        "crop_verification_best_source": None,
        "crop_verification_best_score": None,
        "crop_verification_best_object_matches": 0,
        "crop_verification_best_ref_containment": None,
        "crop_verification_best_frame_containment": None,
        "crop_verification_best_iou": None,
        "crop_verification_best_center_drift_px": None,
        "crop_verification_best_area_ratio": None,
        "crop_verification_best_would_pass": False,
        "crop_verification_best_dangerous": False,
        "crop_verification_selected_score": None,
        "crop_verification_selected_object_matches": 0,
        "crop_verification_object_crop_candidate_available": False,
        "crop_verification_object_crop_candidate_source": None,
        "crop_verification_object_crop_candidate_iou": None,
        "crop_verification_object_crop_candidate_center_drift_px": None,
        "crop_verification_object_crop_candidate_would_pass": False,
        "crop_verification_object_crop_candidate_dangerous": False,
        "crop_verification_object_crop_candidate_inliers": 0,
        "crop_verification_object_crop_candidate_inlier_ratio": None,
        "crop_verification_assessment": assessment,
        "crop_verification_sources": [],
    }


def _object_crop_verification_metric_fields(
    debug: dict[str, Any],
    *,
    scene: _SyntheticScene,
    current_projection: str,
    current_polygon: PolygonPoints | None,
    ground_truth_polygon: PolygonPoints,
    distractor_polygons: Sequence[PolygonPoints],
    support: int,
) -> dict[str, Any]:
    ref_points = scene.reference_points
    frame_points = scene.frame_points
    if len(ref_points) == 0 or len(ref_points) != len(frame_points):
        return _object_crop_verification_empty_fields("no_feature_pairs")

    ref_object_mask = _points_inside_polygon_mask(ref_points, scene.reference_polygon)
    ref_object_count = int(ref_object_mask.sum())
    if ref_object_count <= 0:
        return _object_crop_verification_empty_fields("no_reference_object_features")

    candidate_rows: list[dict[str, Any]] = []
    for source, polygon, selected in _candidate_polygon_rows_from_debug(
        debug,
        current_projection=current_projection,
        current_polygon=current_polygon,
    ):
        candidate_rows.append(
            {
                "source": source,
                "polygon": polygon,
                "selected": selected,
                "derived": False,
                "inliers": 0,
                "inlier_ratio": None,
            }
        )

    for row in _object_crop_feature_candidate_rows(scene):
        candidate_rows.append(
            {
                "source": str(row["source"]),
                "polygon": row["polygon"],
                "selected": False,
                "derived": True,
                "inliers": int(row.get("inliers") or 0),
                "inlier_ratio": row.get("inlier_ratio"),
            }
        )

    if not candidate_rows:
        return _object_crop_verification_empty_fields("no_candidate")

    weak_context = scene.scenario.weak_context or support < _SYNTHETIC_MISSING_POLYGON_MIN_FEATURE_SUPPORT
    scored_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        polygon = row.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        frame_mask = _points_inside_polygon_mask(frame_points, polygon)
        frame_count = int(frame_mask.sum())
        pair_matches = int(np.logical_and(ref_object_mask, frame_mask).sum())
        ref_containment = pair_matches / max(1, ref_object_count)
        frame_containment = pair_matches / max(1, frame_count)
        volume = math.log1p(pair_matches) / math.log1p(max(2, ref_object_count))
        object_like_score = math.sqrt(max(0.0, ref_containment) * max(0.0, frame_containment)) * volume
        iou = _polygon_iou(polygon, ground_truth_polygon)
        center_drift = _polygon_center_distance(polygon, ground_truth_polygon)
        area_ratio = _polygon_area_ratio(polygon, ground_truth_polygon)
        nearest_distractor_distance = _nearest_distractor_distance(polygon, distractor_polygons)
        closer_to_distractor = (
            nearest_distractor_distance is not None
            and nearest_distractor_distance + 1.0 < center_drift
        )
        dangerous = _is_dangerous_projection(
            predicted_polygon=polygon,
            center_drift=center_drift,
            area_ratio=area_ratio,
            closer_to_distractor=closer_to_distractor,
            unsafe_hidden=False,
        )
        would_pass = _object_crop_candidate_would_pass(
            iou=iou,
            center_drift=center_drift,
            area_ratio=area_ratio,
            closer_to_distractor=closer_to_distractor,
            weak_context=weak_context,
        )
        scored_rows.append(
            {
                "source": str(row.get("source") or "candidate"),
                "selected": bool(row.get("selected")),
                "derived": bool(row.get("derived")),
                "score": float(object_like_score),
                "matches": pair_matches,
                "ref_containment": float(ref_containment),
                "frame_containment": float(frame_containment),
                "iou": float(iou),
                "center_drift_px": float(center_drift),
                "area_ratio": float(area_ratio),
                "would_pass": bool(would_pass),
                "dangerous": bool(dangerous),
                "inliers": int(row.get("inliers") or 0),
                "inlier_ratio": row.get("inlier_ratio"),
            }
        )

    if not scored_rows:
        return _object_crop_verification_empty_fields("no_scored_candidate")

    best = max(
        scored_rows,
        key=lambda row: (
            float(row["score"]),
            int(row["matches"]),
            float(row["iou"]),
            -float(row["center_drift_px"]) if math.isfinite(float(row["center_drift_px"])) else -1e9,
        ),
    )
    selected = next((row for row in scored_rows if bool(row.get("selected"))), None)
    derived_candidates = [row for row in scored_rows if bool(row.get("derived"))]
    best_derived = max(
        derived_candidates,
        key=lambda row: (
            1 if bool(row["would_pass"]) and not bool(row["dangerous"]) else 0,
            float(row["iou"]),
            float(row["score"]),
        ),
    ) if derived_candidates else None

    if float(best["score"]) >= 0.38 and int(best["matches"]) >= 8:
        assessment = "strong_object_like_candidate"
    elif float(best["score"]) >= 0.16 and int(best["matches"]) >= 4:
        assessment = "weak_object_like_candidate"
    else:
        assessment = "no_object_like_candidate"
    if best_derived is not None:
        if bool(best_derived["would_pass"]) and not bool(best_derived["dangerous"]):
            assessment = f"{assessment}+object_crop_safe_candidate"
        elif bool(best_derived["dangerous"]):
            assessment = f"{assessment}+object_crop_dangerous_candidate"
        else:
            assessment = f"{assessment}+object_crop_unconfirmed_candidate"

    return {
        "crop_verification_attempted": True,
        "crop_verification_candidate_count": len(scored_rows),
        "crop_verification_source_count": len({str(row["source"]) for row in scored_rows}),
        "crop_verification_best_source": str(best["source"]),
        "crop_verification_best_score": float(best["score"]),
        "crop_verification_best_object_matches": int(best["matches"]),
        "crop_verification_best_ref_containment": float(best["ref_containment"]),
        "crop_verification_best_frame_containment": float(best["frame_containment"]),
        "crop_verification_best_iou": float(best["iou"]),
        "crop_verification_best_center_drift_px": float(best["center_drift_px"]),
        "crop_verification_best_area_ratio": float(best["area_ratio"]),
        "crop_verification_best_would_pass": bool(best["would_pass"]),
        "crop_verification_best_dangerous": bool(best["dangerous"]),
        "crop_verification_selected_score": (
            float(selected["score"]) if selected is not None else None
        ),
        "crop_verification_selected_object_matches": (
            int(selected["matches"]) if selected is not None else 0
        ),
        "crop_verification_object_crop_candidate_available": best_derived is not None,
        "crop_verification_object_crop_candidate_source": (
            str(best_derived["source"]) if best_derived is not None else None
        ),
        "crop_verification_object_crop_candidate_iou": (
            float(best_derived["iou"]) if best_derived is not None else None
        ),
        "crop_verification_object_crop_candidate_center_drift_px": (
            float(best_derived["center_drift_px"]) if best_derived is not None else None
        ),
        "crop_verification_object_crop_candidate_would_pass": (
            bool(best_derived["would_pass"]) if best_derived is not None else False
        ),
        "crop_verification_object_crop_candidate_dangerous": (
            bool(best_derived["dangerous"]) if best_derived is not None else False
        ),
        "crop_verification_object_crop_candidate_inliers": (
            int(best_derived["inliers"]) if best_derived is not None else 0
        ),
        "crop_verification_object_crop_candidate_inlier_ratio": (
            float(best_derived["inlier_ratio"])
            if best_derived is not None and best_derived.get("inlier_ratio") is not None
            else None
        ),
        "crop_verification_assessment": assessment,
        "crop_verification_sources": [str(row["source"]) for row in scored_rows],
    }

def _feature_telemetry_metric_fields(
    debug: dict[str, Any],
    *,
    projection_data: Any | None = None,
) -> dict[str, Any]:
    reference_points = getattr(projection_data, "reference_points", None)
    frame_points = getattr(projection_data, "frame_points", None)

    return {
        "reference_keypoints_total": _int_debug(
            debug.get("reference_keypoints_total"),
            getattr(projection_data, "reference_feature_count", None),
        ),
        "frame_keypoints_total": _int_debug(
            debug.get("frame_keypoints_total"),
            getattr(projection_data, "frame_feature_count", None),
        ),
        "frame_max_keypoints": _int_debug(
            debug.get("frame_max_keypoints"),
            getattr(projection_data, "frame_max_keypoints", None),
        ),
        "frame_keypoint_grid_rows": _int_debug(
            debug.get("frame_keypoint_grid_rows"),
            (getattr(projection_data, "frame_keypoint_grid", None) or (None, None))[0],
        ),
        "frame_keypoint_grid_cols": _int_debug(
            debug.get("frame_keypoint_grid_cols"),
            (getattr(projection_data, "frame_keypoint_grid", None) or (None, None))[1],
        ),
        "lightglue_reference_matches_total": _int_debug(
            debug.get("lightglue_reference_matches_total"),
            len(reference_points) if reference_points is not None else None,
        ),
        "lightglue_frame_matches_total": _int_debug(
            debug.get("lightglue_frame_matches_total"),
            len(frame_points) if frame_points is not None else None,
        ),
        "masked_alignment_used": bool(debug.get("masked_alignment_used")),
        "original_reference_keypoints_total": _int_debug(
            debug.get("original_reference_keypoints_total"),
            getattr(projection_data, "original_reference_feature_count", None),
        ),
        "masked_reference_keypoints_total": _int_debug(
            debug.get("masked_reference_keypoints_total"),
            getattr(projection_data, "masked_reference_feature_count", None),
        ),
        "masked_lightglue_matches_total": _int_debug(
            debug.get("masked_lightglue_matches_total"),
            len(reference_points)
            if bool(getattr(projection_data, "masked_alignment_used", False))
            and reference_points is not None
            else None,
        ),
        "slot_local_lightglue_attempted": bool(
            debug.get("slot_local_lightglue_attempted")
        ),
        "slot_local_lightglue_accepted": bool(
            debug.get("slot_local_lightglue_accepted")
        ),
        "slot_local_lightglue_reject_reason": (
            str(debug.get("slot_local_lightglue_reject_reason"))
            if debug.get("slot_local_lightglue_reject_reason") is not None
            else None
        ),
        "slot_local_lightglue_mode": (
            str(debug.get("slot_local_lightglue_mode"))
            if debug.get("slot_local_lightglue_mode") is not None
            else None
        ),
        "slot_local_lightglue_reference_keypoints": _int_debug(
            debug.get("slot_local_lightglue_reference_keypoints")
        ),
        "slot_local_lightglue_frame_keypoints": _int_debug(
            debug.get("slot_local_lightglue_frame_keypoints")
        ),
        "slot_local_lightglue_max_keypoints": _int_debug(
            debug.get("slot_local_lightglue_max_keypoints")
        ),
        "slot_local_lightglue_grid_rows": _int_debug(
            debug.get("slot_local_lightglue_grid_rows")
        ),
        "slot_local_lightglue_grid_cols": _int_debug(
            debug.get("slot_local_lightglue_grid_cols")
        ),
        "slot_local_lightglue_raw_matches": _int_debug(
            debug.get("slot_local_lightglue_raw_matches")
        ),
        "slot_local_lightglue_inliers": _int_debug(
            debug.get("slot_local_lightglue_inliers")
        ),
        "slot_local_lightglue_inlier_ratio": _float_or_none(
            debug.get("slot_local_lightglue_inlier_ratio")
        ),
        "slot_local_lightglue_median_error": _float_or_none(
            debug.get("slot_local_lightglue_median_error")
        ),
        "slot_local_lightglue_area_score": _float_or_none(
            debug.get("slot_local_lightglue_area_score")
        ),
        "slot_local_lightglue_center_factor": _float_or_none(
            debug.get("slot_local_lightglue_center_factor")
        ),
        "slot_local_lightglue_max_other_overlap": _float_or_none(
            debug.get("slot_local_lightglue_max_other_overlap")
        ),
    }


def _v2_metric_fields(debug: dict[str, Any]) -> dict[str, Any]:
    def text_or_none(value: Any) -> str | None:
        return str(value) if value is not None else None

    return {
        "v2_scene_diag_version": text_or_none(debug.get("v2_scene_diag_version")),
        "v2_scene_match_total": _int_debug(debug.get("v2_scene_match_total")),
        "v2_scene_match_cells": _int_debug(debug.get("v2_scene_match_cells")),
        "v2_scene_match_span_x": _float_or_none(debug.get("v2_scene_match_span_x")),
        "v2_scene_match_span_y": _float_or_none(debug.get("v2_scene_match_span_y")),
        "v2_scene_match_hull_fraction": _float_or_none(
            debug.get("v2_scene_match_hull_fraction")
        ),
        "v2_scene_match_top_count": _int_debug(debug.get("v2_scene_match_top_count")),
        "v2_scene_match_bottom_count": _int_debug(
            debug.get("v2_scene_match_bottom_count")
        ),
        "v2_scene_match_left_count": _int_debug(debug.get("v2_scene_match_left_count")),
        "v2_scene_match_right_count": _int_debug(debug.get("v2_scene_match_right_count")),
        "v2_scene_match_max_cell_fraction": _float_or_none(
            debug.get("v2_scene_match_max_cell_fraction")
        ),
        "v2_scene_model_source": text_or_none(debug.get("v2_scene_model_source")),
        "v2_scene_model_inlier_total": _int_debug(
            debug.get("v2_scene_model_inlier_total")
        ),
        "v2_scene_model_inlier_cells": _int_debug(
            debug.get("v2_scene_model_inlier_cells")
        ),
        "v2_scene_model_inlier_span_x": _float_or_none(
            debug.get("v2_scene_model_inlier_span_x")
        ),
        "v2_scene_model_inlier_span_y": _float_or_none(
            debug.get("v2_scene_model_inlier_span_y")
        ),
        "v2_scene_model_inlier_hull_fraction": _float_or_none(
            debug.get("v2_scene_model_inlier_hull_fraction")
        ),
        "v2_scene_model_inlier_top_count": _int_debug(
            debug.get("v2_scene_model_inlier_top_count")
        ),
        "v2_scene_model_inlier_bottom_count": _int_debug(
            debug.get("v2_scene_model_inlier_bottom_count")
        ),
        "v2_scene_model_inlier_left_count": _int_debug(
            debug.get("v2_scene_model_inlier_left_count")
        ),
        "v2_scene_model_inlier_right_count": _int_debug(
            debug.get("v2_scene_model_inlier_right_count")
        ),
        "v2_scene_model_inlier_max_cell_fraction": _float_or_none(
            debug.get("v2_scene_model_inlier_max_cell_fraction")
        ),
        "v2_scene_model_p90_error": _float_or_none(debug.get("v2_scene_model_p90_error")),
        "v2_candidate_count": _int_debug(debug.get("v2_candidate_count")),
        "v2_candidate_crop_count": _int_debug(debug.get("v2_candidate_crop_count")),
        "v2_candidate_ecc_attempts": _int_debug(debug.get("v2_candidate_ecc_attempts")),
        "v2_candidate_ecc_successes": _int_debug(debug.get("v2_candidate_ecc_successes")),
        "v2_candidate_score": _float_or_none(debug.get("v2_candidate_score")),
        "v2_candidate_selection_score": _float_or_none(
            debug.get("v2_candidate_selection_score")
        ),
        "v2_candidate_crop_mode": text_or_none(debug.get("v2_candidate_crop_mode")),
        "v2_candidate_start_mode": text_or_none(debug.get("v2_candidate_start_mode")),
        "v2_shift_factor": _float_or_none(debug.get("v2_shift_factor")),
        "v2_center_factor": _float_or_none(debug.get("v2_center_factor")),
        "v2_ecc_score": _float_or_none(debug.get("v2_ecc_score")),
        "v2_phase_response": _float_or_none(debug.get("v2_phase_response")),
        "v2_ring_fraction": _float_or_none(debug.get("v2_ring_fraction")),
        "v2_max_other_overlap": _float_or_none(debug.get("v2_max_other_overlap")),
    }


def _hidden_shadow_metric_fields(
    debug: dict[str, Any],
    *,
    scene: _SyntheticScene,
    ground_truth_polygon: PolygonPoints,
    distractor_polygons: Sequence[PolygonPoints],
    support: int,
) -> dict[str, Any]:
    available = bool(debug.get("missing_polygon_hidden_shadow_available"))
    polygon = _debug_polygon_points(debug.get("missing_polygon_hidden_shadow_polygon"))
    if not available or polygon is None:
        return {
            "hidden_shadow_available": False,
            "hidden_shadow_projection": (
                str(debug.get("missing_polygon_hidden_shadow_projection"))
                if debug.get("missing_polygon_hidden_shadow_projection") is not None
                else None
            ),
            "hidden_shadow_fallback_source": (
                str(debug.get("missing_polygon_hidden_shadow_fallback_source"))
                if debug.get("missing_polygon_hidden_shadow_fallback_source") is not None
                else None
            ),
            "hidden_shadow_reason": (
                str(debug.get("missing_polygon_hidden_shadow_reason"))
                if debug.get("missing_polygon_hidden_shadow_reason") is not None
                else None
            ),
        }

    projection = str(debug.get("missing_polygon_hidden_shadow_projection") or "unsafe_hidden_shadow")
    iou = _polygon_iou(polygon, ground_truth_polygon)
    center_drift = _polygon_center_distance(polygon, ground_truth_polygon)
    area_ratio = _polygon_area_ratio(polygon, ground_truth_polygon)
    axis_angle_error_deg, major_length_ratio = _polygon_axis_delta(
        polygon,
        ground_truth_polygon,
    )
    nearest_distractor_distance = _nearest_distractor_distance(
        polygon,
        distractor_polygons,
    )
    closer_to_distractor = (
        nearest_distractor_distance is not None
        and nearest_distractor_distance + 1.0 < center_drift
    )
    dangerous = _is_dangerous_projection(
        predicted_polygon=polygon,
        center_drift=center_drift,
        area_ratio=area_ratio,
        closer_to_distractor=closer_to_distractor,
        unsafe_hidden=False,
    )
    notes = _case_notes(
        scene=scene,
        status="missing",
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
        closer_to_distractor=closer_to_distractor,
        support=support,
        projection=projection,
        used_context_refinement=_projection_uses_context_refinement(projection),
        unsafe_hidden=False,
    )
    return {
        "hidden_shadow_available": True,
        "hidden_shadow_projection": projection,
        "hidden_shadow_fallback_source": (
            str(debug.get("missing_polygon_hidden_shadow_fallback_source"))
            if debug.get("missing_polygon_hidden_shadow_fallback_source") is not None
            else None
        ),
        "hidden_shadow_reason": (
            str(debug.get("missing_polygon_hidden_shadow_reason"))
            if debug.get("missing_polygon_hidden_shadow_reason") is not None
            else None
        ),
        "hidden_shadow_iou": float(iou),
        "hidden_shadow_center_drift_px": float(center_drift),
        "hidden_shadow_area_ratio": float(area_ratio),
        "hidden_shadow_closer_to_distractor": bool(closer_to_distractor),
        "hidden_shadow_dangerous": bool(dangerous),
        "hidden_shadow_would_pass": not notes,
        "hidden_shadow_notes": notes,
    }


def _case_notes(
    *,
    scene: _SyntheticScene,
    status: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
    axis_angle_error_deg: float | None,
    major_length_ratio: float | None,
    closer_to_distractor: bool,
    support: int,
    projection: str,
    used_context_refinement: bool,
    unsafe_hidden: bool,
) -> list[str]:
    notes: list[str] = []
    if status != "missing":
        notes.append("status должен остаться missing, потому что YOLO-детекций в тесте нет")

    if unsafe_hidden:
        notes.append(
            "точная expected-зона скрыта как небезопасная: это accuracy FAIL, "
            "но safety PASS, если полигон не нарисован на distractor"
        )
        return notes

    weak_context = scene.scenario.weak_context or support < _SYNTHETIC_MISSING_POLYGON_MIN_FEATURE_SUPPORT
    min_iou = 0.42 if weak_context else 0.55
    max_drift = 38.0 if weak_context else 28.0

    shape_aware_ok = _shape_aware_projection_ok(
        object_shape=scene.scenario.object_shape,
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
        weak_context=weak_context,
    )
    slot_local_quality_ok = _slot_local_projection_quality_ok(
        projection=projection,
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
        weak_context=weak_context,
    )
    anchor_pose_quality_ok = _anchor_pose_projection_quality_ok(
        projection=projection,
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
        weak_context=weak_context,
    )

    if (
        iou < min_iou
        and not shape_aware_ok
        and not slot_local_quality_ok
        and not anchor_pose_quality_ok
    ):
        notes.append(f"IoU с ground truth ниже порога: {iou:.3f} < {min_iou:.2f}")
    if (
        center_drift > max_drift
        and not shape_aware_ok
        and not slot_local_quality_ok
        and not anchor_pose_quality_ok
    ):
        notes.append(f"центр expected-зоны уехал слишком далеко: {center_drift:.1f}px > {max_drift:.1f}px")
    if area_ratio < 0.55 or area_ratio > 1.75:
        notes.append(f"площадь полигона изменилась слишком сильно: {area_ratio:.2f}x")
    if closer_to_distractor:
        notes.append("predicted-зона ближе к distractor-объекту, чем к правильному месту")

    conservative_rescue_ok = _conservative_translation_rescue_quality_ok(
        projection=projection,
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
    )
    fallback_quality_ok = _fallback_without_context_quality_ok(
        projection=projection,
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
    )

    if (
        not weak_context
        and not used_context_refinement
        and not fallback_quality_ok
        and not anchor_pose_quality_ok
    ):
        notes.append("достаточно контекста, но context_feature_affine не сработал")
    context_affine_quality_ok = _context_affine_refinement_quality_ok(
        projection=projection,
        iou=iou,
        center_drift=center_drift,
        area_ratio=area_ratio,
        axis_angle_error_deg=axis_angle_error_deg,
        major_length_ratio=major_length_ratio,
    )
    if (
        weak_context
        and used_context_refinement
        and not conservative_rescue_ok
        and not context_affine_quality_ok
        and not slot_local_quality_ok
    ):
        notes.append("при слабом контексте refinement сработал, хотя безопаснее fallback")

    return notes


def _slot_local_projection_quality_ok(
    *,
    projection: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
    axis_angle_error_deg: float | None,
    major_length_ratio: float | None,
    weak_context: bool,
) -> bool:
    if projection != "slot_local_lightglue":
        return False
    if not math.isfinite(iou) or not math.isfinite(center_drift):
        return False

    min_iou = 0.42 if weak_context else 0.55
    max_drift = 38.0 if weak_context else 28.0
    if iou < min_iou or center_drift > max_drift:
        return False
    if area_ratio < 0.58 or area_ratio > 1.65:
        return False
    if axis_angle_error_deg is not None and axis_angle_error_deg > 18.0:
        return False
    if major_length_ratio is not None and not (0.62 <= major_length_ratio <= 1.45):
        return False
    return True


def _anchor_pose_projection_quality_ok(
    *,
    projection: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
    axis_angle_error_deg: float | None,
    major_length_ratio: float | None,
    weak_context: bool,
) -> bool:
    if projection != "v4_yolo_anchor_pose":
        return False
    if not math.isfinite(iou) or not math.isfinite(center_drift):
        return False

    # Anchor-pose projection is not a context_feature_affine refinement.
    # It is a separate pose source built from synthetic YOLO anchors, so the
    # synthetic scorer must judge it by resulting geometry, not by whether the
    # context-affine branch ran.  This mirrors the runtime intent: if a robust
    # pose source projects the slot accurately, the missing zone is valid.
    min_iou = 0.42 if weak_context else 0.55
    max_drift = 38.0 if weak_context else 28.0
    if iou < min_iou or center_drift > max_drift:
        return False
    if area_ratio < 0.58 or area_ratio > 1.65:
        return False
    if axis_angle_error_deg is not None and axis_angle_error_deg > 18.0:
        return False
    if major_length_ratio is not None and not (0.62 <= major_length_ratio <= 1.45):
        return False
    return True


def _context_affine_refinement_quality_ok(
    *,
    projection: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
    axis_angle_error_deg: float | None,
    major_length_ratio: float | None,
) -> bool:
    if projection not in {"context_feature_affine", "context_feature_affine_edge_guarded"}:
        return False
    if not math.isfinite(iou) or not math.isfinite(center_drift):
        return False

    # Weak-context synthetic cases used to fail every affine refinement solely
    # because affine was used. That was too strict: if the polygon is actually
    # close to the synthetic ground truth and keeps sane shape geometry, it should
    # count as a correct transfer. This changes only the test/report scoring;
    # runtime safety is still enforced by the matcher guards.
    if iou < 0.70 or center_drift > 8.0:
        return False
    if area_ratio < 0.62 or area_ratio > 1.50:
        return False
    if axis_angle_error_deg is not None and axis_angle_error_deg > 12.0:
        return False
    if major_length_ratio is not None and (major_length_ratio < 0.70 or major_length_ratio > 1.35):
        return False
    return True


def _conservative_translation_rescue_quality_ok(
    *,
    projection: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
    axis_angle_error_deg: float | None,
    major_length_ratio: float | None,
) -> bool:
    if not _projection_is_conservative_translation_rescue(projection):
        return False
    if not math.isfinite(iou) or not math.isfinite(center_drift):
        return False
    if iou < 0.70 or center_drift > 8.0:
        return False
    if area_ratio < 0.60 or area_ratio > 1.55:
        return False
    if axis_angle_error_deg is not None and axis_angle_error_deg > 14.0:
        return False
    if major_length_ratio is not None and (major_length_ratio < 0.65 or major_length_ratio > 1.45):
        return False
    return True


def _fallback_without_context_quality_ok(
    *,
    projection: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
    axis_angle_error_deg: float | None,
    major_length_ratio: float | None,
) -> bool:
    if projection not in {
        "expected_slot",
        "expected_slot_global_fallback",
        "expected_slot_global_fallback_hidden_release",
        "expected_slot_agreement_hidden_release",
    }:
        return False
    if not math.isfinite(iou) or not math.isfinite(center_drift):
        return False
    if iou < 0.70 or center_drift > 10.0:
        return False
    if area_ratio < 0.65 or area_ratio > 1.45:
        return False
    if axis_angle_error_deg is not None and axis_angle_error_deg > 12.0:
        return False
    if major_length_ratio is not None and (major_length_ratio < 0.70 or major_length_ratio > 1.35):
        return False
    return True


def _projection_is_conservative_translation_rescue(projection: str) -> bool:
    return projection in {
        "context_feature_affine_translation_rescue",
        "context_feature_affine_scene_translation_rescue",
        "expected_slot_context_translation_rescue",
        "expected_slot_scene_translation_rescue",
        "expected_slot_global_translation_rescue",
    }


def _projection_uses_context_refinement(projection: str) -> bool:
    return projection.startswith("context_feature_affine") or projection in {
        "expected_slot_context_translation_rescue",
        "expected_slot_scene_translation_rescue",
        "expected_slot_global_translation_rescue",
        "slot_local_lightglue",
    }


def _projection_summary(projections: list[str]) -> str:
    counts: dict[str, int] = {}
    for projection in projections:
        counts[projection] = counts.get(projection, 0) + 1
    return ", ".join(f"{name}:{count}" for name, count in sorted(counts.items())) or "none"


def _scene_proxy_for_object(
    scene: _SyntheticMultiScene,
    obj: _SyntheticObjectInstance,
) -> _SyntheticScene:
    return _SyntheticScene(
        scenario=SyntheticScenario(
            name=scene.scenario.name,
            description=scene.scenario.description,
            seed=scene.scenario.seed,
            rotation_deg=scene.scenario.rotation_deg,
            scale=scene.scenario.scale,
            shift_x=scene.scenario.shift_x,
            shift_y=scene.scenario.shift_y,
            perspective=scene.scenario.perspective,
            global_bias_x=scene.scenario.global_bias_x,
            global_bias_y=scene.scenario.global_bias_y,
            noise_px=scene.scenario.noise_px,
            context_points=scene.scenario.context_points_per_object,
            object_bad_points=scene.scenario.object_bad_points_per_object,
            outlier_points=scene.scenario.outlier_points,
            distractor=bool(obj.distractor_polygons),
            occluder=scene.scenario.occluder,
            weak_context=scene.scenario.weak_context,
            object_shape=obj.object_shape,
            distractor_count=len(obj.distractor_polygons),
            context_cluster=scene.scenario.context_cluster,
        ),
        reference_image=scene.reference_image,
        target_image=scene.target_image,
        true_homography=scene.true_homography,
        approximate_homography=scene.approximate_homography,
        reference_polygon=obj.reference_polygon,
        ground_truth_polygon=obj.ground_truth_polygon,
        distractor_polygons=obj.distractor_polygons,
        reference_points=scene.reference_points,
        frame_points=scene.frame_points,
        context_reference_points=scene.context_reference_points,
        context_frame_points=scene.context_frame_points,
        object_bad_reference_points=scene.object_bad_reference_points,
        object_bad_frame_points=scene.object_bad_frame_points,
    )


def _multi_reference_polygons(scenario: SyntheticMultiScenario) -> list[PolygonPoints]:
    layouts = [
        (-166.0, -78.0, -7.0, 0.82),
        (56.0, -84.0, 8.0, 0.78),
        (-104.0, 82.0, 12.0, 0.72),
        (116.0, 72.0, -10.0, 0.76),
        (0.0, 0.0, 3.0, 0.70),
        (194.0, -8.0, 16.0, 0.66),
        (-214.0, 22.0, -18.0, 0.64),
    ]
    polygons: list[PolygonPoints] = []
    for index, shape in enumerate(scenario.object_shapes):
        dx, dy, angle, scale = layouts[index % len(layouts)]
        base = _OBJECT_SHAPES.get(shape, _OBJECT_SHAPES[_DEFAULT_OBJECT_SHAPE])
        polygons.append(
            _local_transform_polygon(
                [[float(x), float(y)] for x, y in base],
                dx=dx,
                dy=dy,
                angle_deg=angle,
                scale=scale,
            )
        )
    return polygons


def _multi_class_keys(scenario: SyntheticMultiScenario) -> list[str]:
    keys = [f"synthetic-{index}-{shape}" for index, shape in enumerate(scenario.object_shapes)]
    for group_index, group in enumerate(scenario.duplicate_class_groups):
        key = f"synthetic-duplicate-{group_index}"
        for item_index in group:
            if 0 <= item_index < len(keys):
                keys[item_index] = key
    return keys


def _build_multi_homography(
    scenario: SyntheticMultiScenario,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    proxy = SyntheticScenario(
        name=scenario.name,
        description=scenario.description,
        seed=scenario.seed,
        rotation_deg=scenario.rotation_deg,
        scale=scenario.scale,
        shift_x=scenario.shift_x,
        shift_y=scenario.shift_y,
        perspective=scenario.perspective,
        global_bias_x=scenario.global_bias_x,
        global_bias_y=scenario.global_bias_y,
        noise_px=scenario.noise_px,
        context_points=scenario.context_points_per_object,
        object_bad_points=scenario.object_bad_points_per_object,
        outlier_points=scenario.outlier_points,
        distractor=scenario.distractor_count_per_object > 0,
        occluder=scenario.occluder,
        weak_context=scenario.weak_context,
        object_shape=scenario.object_shapes[0] if scenario.object_shapes else _DEFAULT_OBJECT_SHAPE,
        distractor_count=scenario.distractor_count_per_object,
        context_cluster=scenario.context_cluster,
    )
    return _build_homography(proxy, width=width, height=height)


def _draw_reference_scene_multi(
    *,
    width: int,
    height: int,
    objects: list[_SyntheticObjectInstance],
) -> np.ndarray:
    image = _base_scene(width=width, height=height)
    _draw_stable_context(image)
    for obj in objects:
        color = _object_color(obj.index)
        _draw_polygon(image, obj.reference_polygon, color, fill=True, alpha=0.72)
        _draw_polygon(image, obj.reference_polygon, (20, 40, 160), thickness=2)
        # Keep context search windows out of the synthetic source image.
        # They are diagnostic overlays only: drawing them into the reference
        # creates many SuperPoint keypoints that cannot exist in the target
        # frame, so global LightGlue collapses in multi-object scenes.
        # Do not draw object labels into the synthetic source image.
        # They are report annotations, not real product features; if they stay in
        # the reference only, SuperPoint/LightGlue sees many unmatched text
        # keypoints around every missing polygon and multi-scene alignment
        # collapses to a handful of pairs.
    return image


def _draw_target_scene_multi(
    *,
    width: int,
    height: int,
    homography: np.ndarray,
    objects: list[_SyntheticObjectInstance],
    occluder: bool,
) -> np.ndarray:
    image = _base_scene(width=width, height=height)
    _draw_stable_context(image, homography=homography)
    for obj in objects:
        for distractor in obj.distractor_polygons:
            _draw_polygon(image, distractor, (42, 130, 230), fill=True, alpha=0.60)
            _draw_polygon(image, distractor, _DISTRACTOR_COLOR, thickness=2)
    if occluder:
        occluder_poly = project_polygon(
            [[160.0, 140.0], [560.0, 170.0], [532.0, 344.0], [135.0, 312.0]],
            homography,
        )
        _draw_polygon(image, occluder_poly, (88, 90, 98), fill=True, alpha=0.72)
        _draw_polygon(image, occluder_poly, (180, 180, 190), thickness=2)
    return image


def _sample_context_points_around_object(
    rng: np.random.Generator,
    *,
    count: int,
    reference_polygon: PolygonPoints,
    all_object_polygons: list[PolygonPoints],
    width: int,
    height: int,
    cluster_bias: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)

    bbox = _bbox_from_polygon(reference_polygon)
    expanded = _expand_bbox(bbox, factor=_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXPANSION)
    x1, y1, x2, y2 = expanded
    object_arrays = [np.asarray(poly, dtype=np.float32) for poly in all_object_polygons]
    margin = _context_exclusion_margin(bbox)
    anchors = [
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
        ((x1 + x2) * 0.5, y1),
        ((x1 + x2) * 0.5, y2),
    ]
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < count and attempts < max(600, count * 140):
        attempts += 1
        if cluster_bias > 0 and float(rng.random()) < cluster_bias:
            ax, ay = anchors[int(rng.integers(0, len(anchors)))]
            x = float(rng.normal(ax, max(5.0, (x2 - x1) * 0.08)))
            y = float(rng.normal(ay, max(5.0, (y2 - y1) * 0.08)))
        else:
            x = float(rng.uniform(max(0.0, x1), min(float(width - 1), x2)))
            y = float(rng.uniform(max(0.0, y1), min(float(height - 1), y2)))
        x = min(max(x, 0.0), float(width - 1))
        y = min(max(y, 0.0), float(height - 1))
        if any(
            not _point_outside_polygon_margin((x, y), poly, margin=margin)
            for poly in object_arrays
        ):
            continue
        points.append((x, y))
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _concat_points(items: list[np.ndarray]) -> np.ndarray:
    non_empty = [np.asarray(item, dtype=np.float32).reshape(-1, 2) for item in items if len(item) > 0]
    if not non_empty:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(non_empty, axis=0).astype(np.float32)


def _object_color(index: int) -> tuple[int, int, int]:
    palette = [
        (70, 120, 235),
        (90, 180, 140),
        (210, 140, 80),
        (180, 105, 210),
        (215, 190, 80),
        (95, 170, 220),
        (190, 120, 120),
    ]
    return palette[index % len(palette)]


def _as_points_array(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return array.reshape(-1, 2)


def _count_points_inside_polygons(
    points: np.ndarray | None,
    polygons: Sequence[PolygonPoints],
) -> int:
    array = _as_points_array(points)
    if len(array) == 0 or not polygons:
        return 0
    polygon_arrays = [
        np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
        for poly in polygons
        if len(poly) >= 3
    ]
    if not polygon_arrays:
        return 0
    count = 0
    for x, y in array:
        if any(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for poly in polygon_arrays):
            count += 1
    return count


def _count_points_in_context_windows(
    points: np.ndarray | None,
    polygons: Sequence[PolygonPoints],
) -> int:
    array = _as_points_array(points)
    if len(array) == 0 or not polygons:
        return 0
    windows: list[tuple[float, float, float, float]] = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        windows.append(
            _expand_bbox(
                _bbox_from_polygon(poly),
                factor=_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXPANSION,
            )
        )
    if not windows:
        return 0
    count = 0
    for x, y in array:
        xf = float(x)
        yf = float(y)
        if any(x1 <= xf <= x2 and y1 <= yf <= y2 for x1, y1, x2, y2 in windows):
            count += 1
    return count


def _draw_keypoint_diagnostic_base(
    image: np.ndarray,
    *,
    polygons: Sequence[PolygonPoints],
    polygon_color: tuple[int, int, int],
    show_context_windows: bool,
) -> None:
    for polygon in polygons:
        if show_context_windows:
            _draw_context_ring(image, polygon)
        _draw_polygon(image, polygon, polygon_color, thickness=2)


def _draw_keypoints_with_count(
    image: np.ndarray,
    points: np.ndarray | None,
    *,
    color: tuple[int, int, int],
    label: str,
    polygons: Sequence[PolygonPoints],
    y: int,
) -> None:
    array = _as_points_array(points)
    _draw_points(image, array, color, radius=1)
    inside = _count_points_inside_polygons(array, polygons)
    in_context = _count_points_in_context_windows(array, polygons)
    cv2.putText(
        image,
        f"{label}: all={len(array)} | inside expected={inside} | inside context windows={in_context}",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        _TEXT_COLOR,
        1,
    )


def _render_keypoint_diagnostic_panel(
    *,
    index: int,
    name: str,
    scenario_kind: str,
    reference_image: np.ndarray,
    target_image: np.ndarray,
    projection_data: LocalProjectionData,
    reference_polygons: Sequence[PolygonPoints],
    target_polygons: Sequence[PolygonPoints],
    predicted_polygons: Sequence[PolygonPoints],
    result_status: str,
) -> np.ndarray:
    width, height = _CANVAS_SIZE
    header_h = 74
    footer_h = 80
    gutter = 12
    panel_w = width
    panel_count = 4
    canvas = np.full(
        (height + header_h + footer_h, panel_w * panel_count + gutter * (panel_count - 1), 3),
        _PANEL_BG,
        dtype=np.uint8,
    )

    masked_reference = mask_reference_polygons(reference_image, list(reference_polygons))
    panels = [
        reference_image.copy(),
        masked_reference.copy(),
        target_image.copy(),
        target_image.copy(),
    ]

    original_reference_keypoints = _as_points_array(projection_data.original_reference_keypoints)
    masked_reference_keypoints = _as_points_array(projection_data.masked_reference_keypoints)
    frame_keypoints = _as_points_array(projection_data.frame_keypoints)
    matched_reference_points = _as_points_array(projection_data.reference_points)
    matched_frame_points = _as_points_array(projection_data.frame_points)

    _draw_keypoint_diagnostic_base(
        panels[0],
        polygons=reference_polygons,
        polygon_color=_PREDICTED_COLOR,
        show_context_windows=True,
    )
    _draw_keypoints_with_count(
        panels[0],
        original_reference_keypoints,
        color=(0, 255, 255),
        label="original reference SuperPoint",
        polygons=reference_polygons,
        y=28,
    )

    _draw_keypoint_diagnostic_base(
        panels[1],
        polygons=reference_polygons,
        polygon_color=_PREDICTED_COLOR,
        show_context_windows=True,
    )
    _draw_keypoints_with_count(
        panels[1],
        masked_reference_keypoints,
        color=(90, 255, 90),
        label="masked reference SuperPoint",
        polygons=reference_polygons,
        y=28,
    )

    for polygon in target_polygons:
        _draw_polygon(panels[2], polygon, _GT_COLOR, thickness=2)
        _draw_polygon(panels[3], polygon, _GT_COLOR, thickness=1)
    for polygon in predicted_polygons:
        _draw_polygon(panels[2], polygon, _PREDICTED_COLOR, thickness=2)
        _draw_polygon(panels[3], polygon, _PREDICTED_COLOR, thickness=2)
    _draw_keypoints_with_count(
        panels[2],
        frame_keypoints,
        color=(255, 220, 90),
        label="target frame SuperPoint",
        polygons=target_polygons,
        y=28,
    )

    _draw_points(panels[0], matched_reference_points, _CONTEXT_COLOR, radius=2)
    _draw_points(panels[3], matched_frame_points, _CONTEXT_COLOR, radius=2)
    _draw_feature_vectors_for_points(
        panels[3],
        reference_points=matched_reference_points,
        frame_points=matched_frame_points,
        homography=projection_data.global_homography,
        color=_CONTEXT_COLOR,
        max_lines=360,
    )
    cv2.putText(
        panels[3],
        f"post-LightGlue matched pairs: {_projection_match_count(projection_data)}",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        _TEXT_COLOR,
        1,
    )

    labels = [
        "1 GLOBAL REF BEFORE MASK: all SuperPoint points",
        "2 GLOBAL REF AFTER MASK: points sent to global LightGlue",
        "3 GLOBAL TARGET BEFORE MATCH: all target points",
        "4 GLOBAL MATCHES ONLY: not per-object slot-local",
    ]
    for panel_index, panel in enumerate(panels):
        x = panel_index * (panel_w + gutter)
        canvas[header_h : header_h + height, x : x + panel_w] = panel
        cv2.putText(
            canvas,
            labels[panel_index],
            (x + 14, header_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            _TEXT_COLOR,
            2,
        )

    status_color = _PASS_COLOR if result_status == "PASS" else _FAIL_COLOR
    cv2.putText(
        canvas,
        f"#{index:03d} {name} - keypoint diagnostic - {result_status}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        status_color,
        2,
    )
    cv2.putText(
        canvas,
        f"{scenario_kind} | panels 1-4 are GLOBAL diagnostics; slot-local object matches are in HTML table below",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        _TEXT_COLOR,
        1,
    )
    footer = (
        f"orig_ref={len(original_reference_keypoints)} | masked_ref={len(masked_reference_keypoints)} | "
        f"frame={len(frame_keypoints)} | global_matched_pairs={_projection_match_count(projection_data)} | "
        f"masked_alignment={projection_data.masked_alignment_used}"
    )
    cv2.putText(canvas, footer, (18, header_h + height + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, _TEXT_COLOR, 1)
    cv2.putText(
        canvas,
        "Panel 4 is global LightGlue evidence only. For object-level failures use the slot-local debug table below the case.",
        (18, header_h + height + 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        _TEXT_COLOR,
        1,
    )
    return canvas


def _render_multi_case_panel(
    *,
    index: int,
    scene: _SyntheticMultiScene,
    projection_data: LocalProjectionData,
    predicted_by_object: dict[int, PolygonPoints | None],
    result_status: str,
    metrics: dict[str, str],
    notes: list[str],
) -> np.ndarray:
    width, height = _CANVAS_SIZE
    header_h = 76
    footer_h = 92
    gutter = 12
    panel_w = width
    panel_count = 4
    canvas = np.full(
        (height + header_h + footer_h, panel_w * panel_count + gutter * (panel_count - 1), 3),
        _PANEL_BG,
        dtype=np.uint8,
    )
    panels = [
        scene.reference_image.copy(),
        scene.target_image.copy(),
        scene.target_image.copy(),
        scene.target_image.copy(),
    ]

    for obj in scene.objects:
        color = _object_color(obj.index)
        _draw_polygon(panels[0], obj.reference_polygon, color, thickness=3)
        _draw_polygon(panels[1], obj.ground_truth_polygon, _GT_COLOR, thickness=2)
        for distractor in obj.distractor_polygons:
            _draw_polygon(panels[1], distractor, _DISTRACTOR_COLOR, thickness=2)
        _draw_polygon(panels[2], obj.ground_truth_polygon, _GT_COLOR, thickness=2)
        predicted = predicted_by_object.get(obj.index)
        if predicted:
            _draw_polygon(panels[2], predicted, _PREDICTED_COLOR, thickness=3)
        for distractor in obj.distractor_polygons:
            _draw_polygon(panels[2], distractor, _DISTRACTOR_COLOR, thickness=2)
        _draw_prediction_error_overlay(
            panels[2],
            ground_truth=obj.ground_truth_polygon,
            predicted=predicted,
            label=f"obj{obj.index}",
        )
        _draw_polygon(panels[3], obj.ground_truth_polygon, _GT_COLOR, thickness=1)
        if predicted:
            _draw_polygon(panels[3], predicted, _PREDICTED_COLOR, thickness=2)
        _draw_prediction_error_overlay(
            panels[3],
            ground_truth=obj.ground_truth_polygon,
            predicted=predicted,
            label=f"obj{obj.index}",
        )

    _draw_real_projection_matches(
        panels=panels,
        projection_data=projection_data,
        max_points=260,
        max_vectors=180,
    )

    labels = [
        "REFERENCE: multiple annotated expected objects",
        "TARGET: all objects removed + ground truth",
        "PREDICTION: all missing expected zones",
        "REAL MATCHES: LightGlue global->actual",
    ]
    for panel_index, panel in enumerate(panels):
        x = panel_index * (panel_w + gutter)
        canvas[header_h : header_h + height, x : x + panel_w] = panel
        cv2.putText(canvas, labels[panel_index], (x + 14, header_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64, _TEXT_COLOR, 2)

    status_color = _PASS_COLOR if result_status == "PASS" else _FAIL_COLOR
    title = f"#{index:03d} {scene.scenario.name} - {result_status}"
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, status_color, 2)
    cv2.putText(
        canvas,
        f"MULTI | objects={len(scene.objects)} | {scene.scenario.description[:132]}",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        _TEXT_COLOR,
        1,
    )
    legend = "red=predicted zones | green=GT zones | orange=distractors | cyan=real LightGlue matches"
    cv2.putText(canvas, legend, (18, header_h + height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _TEXT_COLOR, 1)
    metric_text = " | ".join(f"{key}: {value}" for key, value in metrics.items())
    cv2.putText(canvas, metric_text[:260], (18, header_h + height + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT_COLOR, 1)
    if notes:
        cv2.putText(canvas, f"FAIL reason: {notes[0][:230]}", (18, header_h + height + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _FAIL_COLOR, 1)
    else:
        cv2.putText(canvas, "PASS: all expected zones stayed with their own ground-truth slots", (18, header_h + height + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _PASS_COLOR, 1)
    return canvas


def _projection_match_count(projection_data: LocalProjectionData) -> int:
    reference_points = projection_data.reference_points
    frame_points = projection_data.frame_points
    if reference_points is None or frame_points is None:
        return 0
    return int(min(len(reference_points), len(frame_points)))


def _limited_point_pairs(
    reference_points: np.ndarray | None,
    frame_points: np.ndarray | None,
    *,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    if reference_points is None or frame_points is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    count = int(min(len(reference_points), len(frame_points)))
    if count <= 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    ref = np.asarray(reference_points[:count], dtype=np.float32).reshape(-1, 2)
    frm = np.asarray(frame_points[:count], dtype=np.float32).reshape(-1, 2)
    if count <= limit:
        return ref, frm
    step = max(1, int(math.ceil(count / max(1, limit))))
    return ref[::step][:limit], frm[::step][:limit]


def _draw_real_projection_matches(
    *,
    panels: list[np.ndarray],
    projection_data: LocalProjectionData,
    max_points: int,
    max_vectors: int,
) -> None:
    reference_points, frame_points = _limited_point_pairs(
        projection_data.reference_points,
        projection_data.frame_points,
        limit=max(max_points, max_vectors),
    )
    if len(reference_points) == 0 or len(frame_points) == 0:
        cv2.putText(
            panels[3],
            "NO REAL LIGHTGLUE MATCHES",
            (18, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            _FAIL_COLOR,
            2,
        )
        return

    _draw_points(panels[0], reference_points[:max_points], _CONTEXT_COLOR, radius=2)
    _draw_points(panels[2], frame_points[:max_points], _CONTEXT_COLOR, radius=2)
    _draw_feature_vectors_for_points(
        panels[3],
        reference_points=reference_points[:max_vectors],
        frame_points=frame_points[:max_vectors],
        homography=projection_data.global_homography,
        color=_CONTEXT_COLOR,
        max_lines=max_vectors,
    )
    cv2.putText(
        panels[3],
        f"real LightGlue matches: {_projection_match_count(projection_data)}",
        (18, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        _TEXT_COLOR,
        2,
    )


def _draw_feature_vectors_for_points(
    image: np.ndarray,
    *,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    homography: np.ndarray | None,
    color: tuple[int, int, int],
    max_lines: int,
) -> None:
    if homography is None:
        _draw_points(image, frame_points, color, radius=2)
        cv2.putText(
            image,
            "homography failed: showing frame-side real matches only",
            (18, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            _FAIL_COLOR,
            1,
        )
        return
    approx = _transform_points(reference_points, homography)
    _draw_vector_set(image, approx, frame_points, color, max_lines=max_lines)


def _draw_reference_scene(
    *,
    width: int,
    height: int,
    object_polygon: PolygonPoints,
    object_shape: str,
) -> np.ndarray:
    image = _base_scene(width=width, height=height)
    _draw_stable_context(image)
    _draw_polygon(image, object_polygon, (70, 120, 235), fill=True, alpha=0.82)
    _draw_polygon(image, object_polygon, (20, 40, 160), thickness=2)
    # Keep synthetic source images label-free: text is a diagnostic overlay only,
    # otherwise global feature matching learns reference-only glyphs.
    return image


def _draw_target_scene(
    *,
    width: int,
    height: int,
    homography: np.ndarray,
    object_polygon: PolygonPoints,
    distractor_count: int,
    occluder: bool,
) -> np.ndarray:
    image = _base_scene(width=width, height=height)
    _draw_stable_context(image, homography=homography)
    for polygon in _target_distractor_polygons(
        homography,
        object_polygon=object_polygon,
        count=distractor_count,
    ):
        _draw_polygon(image, polygon, (42, 130, 230), fill=True, alpha=0.75)
        _draw_polygon(image, polygon, _DISTRACTOR_COLOR, thickness=2)
    if occluder:
        occluder_poly = project_polygon(
            [[266.0, 156.0], [430.0, 180.0], [408.0, 264.0], [252.0, 238.0]],
            homography,
        )
        _draw_polygon(image, occluder_poly, (90, 92, 100), fill=True, alpha=0.90)
        _draw_polygon(image, occluder_poly, (180, 180, 190), thickness=2)
        cv2.putText(image, "occluder", _poly_label_point(occluder_poly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return image


def _base_scene(*, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), (54, 56, 58), dtype=np.uint8)
    for y in range(0, height, 32):
        cv2.line(image, (0, y), (width, y), (62, 64, 66), 1)
    for x in range(0, width, 32):
        cv2.line(image, (x, 0), (x, height), (62, 64, 66), 1)
    return image


def _draw_stable_context(image: np.ndarray, homography: np.ndarray | None = None) -> None:
    shapes = [
        ([[206, 130], [470, 142], [456, 306], [190, 288]], (82, 88, 92), "panel"),
        ([[246, 248], [426, 258], [420, 286], [240, 275]], (72, 78, 82), "rib"),
        ([[228, 154], [256, 158], [252, 186], [224, 181]], (110, 115, 120), "bolt"),
        ([[438, 164], [468, 166], [464, 196], [434, 192]], (110, 115, 120), "bolt"),
        ([[206, 268], [236, 270], [232, 300], [202, 296]], (110, 115, 120), "bolt"),
        ([[462, 264], [492, 268], [488, 298], [458, 294]], (110, 115, 120), "bolt"),
        ([[150, 96], [200, 104], [190, 150], [142, 140]], (95, 100, 106), "bracket"),
        ([[510, 92], [580, 108], [568, 158], [500, 145]], (96, 101, 107), "housing"),
    ]
    for polygon, color, label in shapes:
        points = [[float(x), float(y)] for x, y in polygon]
        if homography is not None:
            points = project_polygon(points, homography)
        _draw_polygon(image, points, color, fill=True, alpha=0.86)
        _draw_polygon(image, points, (145, 150, 155), thickness=1)
        cv2.putText(image, label, _poly_label_point(points), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

    circles = [(250, 150), (445, 176), (215, 282), (470, 280), (350, 156), (355, 282)]
    for point in circles:
        center = np.asarray([[point]], dtype=np.float32)
        if homography is not None:
            center = cv2.perspectiveTransform(center, homography.astype(np.float32))
        x, y = center.reshape(2)
        cv2.circle(image, (int(round(x)), int(round(y))), 6, (158, 164, 168), -1)
        cv2.circle(image, (int(round(x)), int(round(y))), 8, (44, 46, 48), 1)


def _target_distractor_polygons(
    homography: np.ndarray,
    *,
    object_polygon: PolygonPoints,
    count: int,
) -> list[PolygonPoints]:
    offsets = [
        (118.0, 20.0, 4.0, 0.98),
        (-104.0, 118.0, -8.0, 0.92),
        (70.0, -92.0, 12.0, 0.88),
        (-150.0, -34.0, -15.0, 1.05),
        (162.0, 112.0, 18.0, 0.95),
        (-190.0, 44.0, -20.0, 0.90),
    ]
    polygons: list[PolygonPoints] = []
    for index in range(max(0, count)):
        dx, dy, angle, scale = offsets[index % len(offsets)]
        local = _local_transform_polygon(
            object_polygon,
            dx=dx,
            dy=dy,
            angle_deg=angle,
            scale=scale,
        )
        polygons.append(project_polygon(local, homography))
    return polygons


def _scenario_projective_terms(scenario: Any) -> tuple[float, float]:
    perspective = float(getattr(scenario, "perspective", 0.0) or 0.0)
    if abs(perspective) <= 1e-12:
        return 0.0, 0.0

    # The old synthetic transform always leaned in the same direction.  That made
    # the test visually repetitive and allowed accidental overfitting to one
    # projective skew.  Use the seed to cover eight stable tilt directions while
    # preserving the existing perspective magnitude in every scenario.
    bucket = int(abs(int(getattr(scenario, "seed", 0) or 0))) % 8
    direction = math.radians(bucket * 45.0)
    return perspective * math.cos(direction), perspective * math.sin(direction)


def _scenario_tilt_label(scenario: Any) -> str:
    tx, ty = _scenario_projective_terms(scenario)
    magnitude = float(math.hypot(tx, ty))
    if magnitude <= 1e-12:
        return "0°/0"
    angle = (math.degrees(math.atan2(ty, tx)) + 360.0) % 360.0
    return f"{angle:.0f}°/{magnitude:.5f}"


def _build_homography(
    scenario: SyntheticScenario,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    angle = math.radians(scenario.rotation_deg)
    cos_a = math.cos(angle) * scenario.scale
    sin_a = math.sin(angle) * scenario.scale
    cx, cy = width * 0.5, height * 0.5
    perspective_x, perspective_y = _scenario_projective_terms(scenario)
    affine = np.asarray(
        [
            [cos_a, -sin_a, cx + scenario.shift_x - cos_a * cx + sin_a * cy],
            [sin_a, cos_a, cy + scenario.shift_y - sin_a * cx - cos_a * cy],
            [perspective_x, perspective_y, 1.0],
        ],
        dtype=np.float32,
    )
    return affine


def _sample_context_points(
    rng: np.random.Generator,
    *,
    count: int,
    reference_polygon: PolygonPoints,
    width: int,
    height: int,
    cluster_bias: float = 0.0,
) -> np.ndarray:
    bbox = _bbox_from_polygon(reference_polygon)
    expanded = _expand_bbox(bbox, factor=_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXPANSION)
    x1, y1, x2, y2 = expanded
    polygon_np = np.asarray(reference_polygon, dtype=np.float32)
    margin = _context_exclusion_margin(bbox)
    anchors = [
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
        ((x1 + x2) * 0.5, y1),
        ((x1 + x2) * 0.5, y2),
    ]

    points: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(400, count * 100)
    while len(points) < count and attempts < max_attempts:
        attempts += 1
        if cluster_bias > 0 and float(rng.random()) < cluster_bias:
            ax, ay = anchors[int(rng.integers(0, len(anchors)))]
            x = float(rng.normal(ax, max(5.0, (x2 - x1) * 0.08)))
            y = float(rng.normal(ay, max(5.0, (y2 - y1) * 0.08)))
        else:
            x = float(rng.uniform(max(0.0, x1), min(float(width - 1), x2)))
            y = float(rng.uniform(max(0.0, y1), min(float(height - 1), y2)))
        x = min(max(x, 0.0), float(width - 1))
        y = min(max(y, 0.0), float(height - 1))
        if not _point_outside_polygon_margin((x, y), polygon_np, margin=margin):
            continue
        points.append((x, y))

    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _sample_inside_polygon(
    rng: np.random.Generator,
    *,
    polygon: PolygonPoints,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    bbox = _bbox_from_polygon(polygon)
    x1, y1, x2, y2 = bbox
    polygon_np = np.asarray(polygon, dtype=np.float32)
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < count and attempts < count * 80:
        attempts += 1
        x = float(rng.uniform(x1, x2))
        y = float(rng.uniform(y1, y2))
        if _point_inside_polygon((x, y), polygon_np):
            points.append((x, y))
    return np.asarray(points, dtype=np.float32)


def _bad_object_frame_points(
    rng: np.random.Generator,
    *,
    reference_points: np.ndarray,
    homography: np.ndarray,
    distractors: list[PolygonPoints],
    noise_px: float,
) -> np.ndarray:
    if len(reference_points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if not distractors:
        transformed = _transform_points(reference_points, homography)
        return transformed + rng.normal(0.0, noise_px * 3.0, transformed.shape).astype(np.float32)

    target_points: list[np.ndarray] = []
    for point in reference_points:
        polygon = distractors[int(rng.integers(0, len(distractors)))]
        bbox = _bbox_from_polygon(polygon)
        x1, y1, x2, y2 = bbox
        target_points.append(
            np.asarray(
                [
                    float(rng.uniform(x1, x2)),
                    float(rng.uniform(y1, y2)),
                ],
                dtype=np.float32,
            )
        )
    frame = np.asarray(target_points, dtype=np.float32)
    frame += rng.normal(0.0, max(1.0, noise_px), frame.shape).astype(np.float32)
    return frame


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    transformed = cv2.perspectiveTransform(
        points.astype(np.float32).reshape(-1, 1, 2),
        homography.astype(np.float32),
    ).reshape(-1, 2)
    return transformed.astype(np.float32)



def _polygon_center_xy(polygon: PolygonPoints | None) -> tuple[float, float] | None:
    if not polygon or len(polygon) < 3:
        return None
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all():
        return None
    return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))


def _draw_prediction_error_overlay(
    image: np.ndarray,
    *,
    ground_truth: PolygonPoints | None,
    predicted: PolygonPoints | None,
    label: str = "pred-vs-gt",
) -> None:
    gt_center = _polygon_center_xy(ground_truth)
    predicted_center = _polygon_center_xy(predicted)
    if gt_center is None:
        return

    gx, gy = int(round(gt_center[0])), int(round(gt_center[1]))
    cv2.circle(image, (gx, gy), 7, _GT_COLOR, -1)
    cv2.circle(image, (gx, gy), 10, (20, 20, 20), 2)
    cv2.putText(
        image,
        "GT",
        (gx + 10, gy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        _GT_COLOR,
        2,
    )

    if predicted_center is None:
        cv2.putText(
            image,
            f"{label}: projection hidden / not confirmed",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            _FAIL_COLOR,
            2,
        )
        return

    px, py = int(round(predicted_center[0])), int(round(predicted_center[1]))
    cv2.circle(image, (px, py), 7, _PREDICTED_COLOR, -1)
    cv2.circle(image, (px, py), 10, (20, 20, 20), 2)
    cv2.putText(
        image,
        "PRED",
        (px + 10, py + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        _PREDICTED_COLOR,
        2,
    )
    cv2.arrowedLine(image, (gx, gy), (px, py), _TEXT_COLOR, 2, tipLength=0.08)
    drift = float(np.hypot(predicted_center[0] - gt_center[0], predicted_center[1] - gt_center[1]))
    mx = int(round((gx + px) * 0.5))
    my = int(round((gy + py) * 0.5))
    cv2.putText(
        image,
        f"drift={drift:.1f}px",
        (max(10, mx - 42), max(24, my - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        _TEXT_COLOR,
        2,
    )

def _render_case_panel(
    *,
    index: int,
    scene: _SyntheticScene,
    projection_data: LocalProjectionData,
    predicted_polygon: PolygonPoints | None,
    result_status: str,
    metrics: dict[str, str],
    notes: list[str],
) -> np.ndarray:
    width, height = _CANVAS_SIZE
    header_h = 76
    footer_h = 84
    gutter = 12
    panel_w = width
    panel_count = 4
    canvas = np.full(
        (height + header_h + footer_h, panel_w * panel_count + gutter * (panel_count - 1), 3),
        _PANEL_BG,
        dtype=np.uint8,
    )

    panels = [
        scene.reference_image.copy(),
        scene.target_image.copy(),
        scene.target_image.copy(),
        scene.target_image.copy(),
    ]

    _draw_context_ring(panels[0], scene.reference_polygon)
    _draw_polygon(panels[0], scene.reference_polygon, _PREDICTED_COLOR, thickness=3)

    _draw_polygon(panels[1], scene.ground_truth_polygon, _GT_COLOR, thickness=3)
    for distractor in scene.distractor_polygons:
        _draw_polygon(panels[1], distractor, _DISTRACTOR_COLOR, thickness=3)

    _draw_polygon(panels[2], scene.ground_truth_polygon, _GT_COLOR, thickness=2)
    if predicted_polygon:
        _draw_polygon(panels[2], predicted_polygon, _PREDICTED_COLOR, thickness=3)
    for distractor in scene.distractor_polygons:
        _draw_polygon(panels[2], distractor, _DISTRACTOR_COLOR, thickness=2)
    _draw_prediction_error_overlay(
        panels[2],
        ground_truth=scene.ground_truth_polygon,
        predicted=predicted_polygon,
        label="case",
    )

    _draw_polygon(panels[3], scene.ground_truth_polygon, _GT_COLOR, thickness=2)
    if predicted_polygon:
        _draw_polygon(panels[3], predicted_polygon, _PREDICTED_COLOR, thickness=2)
    for distractor in scene.distractor_polygons:
        _draw_polygon(panels[3], distractor, _DISTRACTOR_COLOR, thickness=2)
    _draw_prediction_error_overlay(
        panels[3],
        ground_truth=scene.ground_truth_polygon,
        predicted=predicted_polygon,
        label="case",
    )
    _draw_real_projection_matches(
        panels=panels,
        projection_data=projection_data,
        max_points=220,
        max_vectors=160,
    )

    labels = [
        "REFERENCE: complex object + context ring",
        "TARGET: object removed + ground truth",
        "PREDICTION: missing expected zone",
        "REAL MATCHES: LightGlue global->actual",
    ]
    for panel_index, panel in enumerate(panels):
        x = panel_index * (panel_w + gutter)
        canvas[header_h : header_h + height, x : x + panel_w] = panel
        cv2.putText(canvas, labels[panel_index], (x + 14, header_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64, _TEXT_COLOR, 2)

    status_color = _PASS_COLOR if result_status == "PASS" else _FAIL_COLOR
    title = f"#{index:03d} {scene.scenario.name} - {result_status}"
    cv2.putText(canvas, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, status_color, 2)
    cv2.putText(canvas, f"shape={scene.scenario.object_shape} | tilt={_scenario_tilt_label(scene.scenario)} | {scene.scenario.description[:120]}", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _TEXT_COLOR, 1)

    legend = (
        "red=predicted missing zone | green=ground truth | orange=distractor | "
        "cyan=real LightGlue matches from compute_features/LightGlue"
    )
    cv2.putText(canvas, legend, (18, header_h + height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _TEXT_COLOR, 1)

    metric_text = " | ".join(f"{key}: {value}" for key, value in metrics.items())
    cv2.putText(canvas, metric_text[:260], (18, header_h + height + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT_COLOR, 1)
    if notes:
        cv2.putText(canvas, f"FAIL reason: {notes[0][:230]}", (18, header_h + height + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _FAIL_COLOR, 1)
    else:
        cv2.putText(canvas, "PASS: expected zone stayed with the ground-truth missing location", (18, header_h + height + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _PASS_COLOR, 1)

    return canvas


def _scenario_object_polygon(scenario: SyntheticScenario) -> PolygonPoints:
    polygon = _OBJECT_SHAPES.get(scenario.object_shape)
    if polygon is None:
        polygon = _OBJECT_SHAPES[_DEFAULT_OBJECT_SHAPE]
    return [[float(x), float(y)] for x, y in polygon]


def _local_transform_polygon(
    polygon: PolygonPoints,
    *,
    dx: float,
    dy: float,
    angle_deg: float,
    scale: float,
) -> PolygonPoints:
    points = np.asarray(polygon, dtype=np.float32)
    center = np.mean(points, axis=0)
    angle = math.radians(angle_deg)
    matrix = np.asarray(
        [
            [math.cos(angle) * scale, -math.sin(angle) * scale],
            [math.sin(angle) * scale, math.cos(angle) * scale],
        ],
        dtype=np.float32,
    )
    transformed = (points - center) @ matrix.T + center + np.asarray([dx, dy], dtype=np.float32)
    return [[float(x), float(y)] for x, y in transformed]


def _draw_feature_vectors(image: np.ndarray, scene: _SyntheticScene) -> None:
    approx_context = _transform_points(scene.context_reference_points, scene.approximate_homography)
    _draw_vector_set(
        image,
        approx_context,
        scene.context_frame_points,
        _CONTEXT_COLOR,
        max_lines=120,
    )
    approx_bad = _transform_points(scene.object_bad_reference_points, scene.approximate_homography)
    _draw_vector_set(
        image,
        approx_bad,
        scene.object_bad_frame_points,
        _OBJECT_BAD_COLOR,
        max_lines=80,
    )


def _draw_vector_set(
    image: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    color: tuple[int, int, int],
    *,
    max_lines: int,
) -> None:
    if len(starts) == 0 or len(ends) == 0:
        return
    step = max(1, int(math.ceil(len(starts) / max_lines)))
    for start, end in zip(starts[::step], ends[::step], strict=False):
        sx, sy = int(round(float(start[0]))), int(round(float(start[1])))
        ex, ey = int(round(float(end[0]))), int(round(float(end[1])))
        if not (0 <= ex < image.shape[1] and 0 <= ey < image.shape[0]):
            continue
        cv2.line(image, (sx, sy), (ex, ey), color, 1)
        cv2.circle(image, (ex, ey), 2, color, -1)


def _draw_context_ring(image: np.ndarray, polygon: PolygonPoints) -> None:
    bbox = _bbox_from_polygon(polygon)
    expanded = _expand_bbox(bbox, factor=_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXPANSION)
    x1, y1, x2, y2 = [int(round(v)) for v in expanded]
    cv2.rectangle(image, (x1, y1), (x2, y2), _CONTEXT_COLOR, 2)
    cv2.putText(image, "context search window", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _CONTEXT_COLOR, 1)


def _draw_polygon(
    image: np.ndarray,
    polygon: PolygonPoints,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
    fill: bool = False,
    alpha: float = 1.0,
) -> None:
    if not polygon or len(polygon) < 3:
        return
    points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    if fill:
        overlay = image.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)
    else:
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)


def _draw_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int,
) -> None:
    for point in points:
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), radius, color, -1)


def _poly_label_point(polygon: PolygonPoints) -> tuple[int, int]:
    points = np.asarray(polygon, dtype=np.float32)
    if points.size == 0:
        return (10, 20)
    center = np.mean(points, axis=0)
    return int(round(float(center[0]))), int(round(float(center[1])))



def _serialize_results(results: list[SyntheticResult]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for result in results:
        item = asdict(result)
        raw_metrics = item.get("object_metrics")
        if isinstance(raw_metrics, list):
            item["object_metrics"] = [
                _enrich_object_metric(dict(metric))
                for metric in raw_metrics
                if isinstance(metric, dict)
            ]
        serialized.append(item)
    return serialized


_REPORT_ONLY_OBJECT_METRIC_PREFIXES = (
    "candidate_oracle_",
    "crop_verification_",
    "yolo_fixture_",
    "yolo_gt_",
    "yolo_synthetic_",
)


def _strip_report_only_object_metric_fields(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metric.items()
        if not key.startswith(_REPORT_ONLY_OBJECT_METRIC_PREFIXES)
    }


def _enrich_object_metric(metric: dict[str, Any]) -> dict[str, Any]:
    enriched = _strip_report_only_object_metric_fields(dict(metric))
    enriched.update(_result_policy_metric_fields(enriched))
    return enriched


def _result_policy_metric_fields(metric: dict[str, Any]) -> dict[str, Any]:
    diagnostic = _metric_diagnostic_class(metric)
    projection = _metric_text(metric, "projection", "unknown")
    hidden_reason = _metric_text(metric, "hidden_reason", "not_hidden")

    if _metric_bool(metric, "passed"):
        status = "accepted"
        confidence = "high" if _metric_float(metric, "iou") >= 0.75 else "medium"
        action = "render_ok_overlay"
        user_label = "Контроль пройден"
        reason = "geometry_checks_passed"
    elif _metric_bool(metric, "dangerous_projection"):
        status = "rejected_dangerous"
        confidence = "high"
        action = "hide_projection_mark_missing"
        user_label = "Деталь не подтверждена"
        reason = diagnostic
    elif _metric_bool(metric, "unsafe_hidden"):
        status = "hidden_for_safety"
        confidence = "high"
        if _metric_bool(metric, "hidden_shadow_available"):
            action = "show_unconfirmed_shadow_with_safety_note"
            user_label = "Зона найдена, контур не подтверждён"
            reason = _metric_text(metric, "hidden_shadow_reason", hidden_reason)
        else:
            action = "show_missing_with_safety_note"
            user_label = "Деталь скрыта как небезопасная"
            reason = hidden_reason
    elif _metric_bool(metric, "hidden_shadow_would_pass") and not _metric_bool(
        metric,
        "hidden_shadow_dangerous",
    ):
        status = "reviewable_hidden_candidate"
        confidence = "medium"
        action = "show_manual_review_candidate"
        user_label = "Нужна проверка кандидата"
        reason = "hidden_shadow_safe_candidate"
    elif _metric_bool(metric, "candidate_oracle_has_safe_alternative"):
        status = "reviewable_alternative_candidate"
        confidence = "medium"
        action = "show_manual_review_candidate"
        user_label = "Есть безопасная альтернатива"
        reason = "candidate_oracle_safe_alternative"
    elif projection == "none":
        status = "missing_no_projection"
        confidence = "low"
        if _metric_bool(metric, "hidden_shadow_available"):
            action = "show_unconfirmed_shadow_with_safety_note"
            user_label = "Зона найдена, контур не подтверждён"
            reason = _metric_text(metric, "hidden_shadow_reason", diagnostic)
        else:
            action = "show_missing_without_overlay"
            user_label = "Деталь не найдена"
            reason = _metric_text(metric, "none_reason", diagnostic)
    else:
        status = "rejected_geometry"
        confidence = _result_policy_geometry_confidence(metric)
        action = "show_missing_or_review_overlay"
        user_label = "Деталь не подтверждена"
        reason = diagnostic

    return {
        "result_policy_status": status,
        "result_policy_confidence": confidence,
        "result_policy_action": action,
        "result_policy_user_label": user_label,
        "result_policy_reason": reason,
    }


def _yolo_synthetic_metric_fields(metric: dict[str, Any]) -> dict[str, Any]:
    projection = _metric_text(metric, "projection", "unknown")
    diagnostic = _metric_diagnostic_class(metric)

    if _metric_bool(metric, "passed"):
        bucket = "geometry_baseline_ok"
        expected_role = "not_required_for_this_case"
    elif _metric_bool(metric, "dangerous_projection"):
        bucket = "safety_guard_case"
        expected_role = "confirm_before_release"
    elif _metric_bool(metric, "unsafe_hidden"):
        bucket = "detector_oracle_required"
        expected_role = "confirm_object_presence_in_hidden_slot"
    elif projection == "none":
        bucket = "no_projection_detector_needed"
        expected_role = "detect_object_when_geometry_has_no_polygon"
    elif projection in {
        "expected_slot",
        "expected_slot_global_fallback",
        "expected_slot_global_fallback_hidden_release",
        "expected_slot_agreement_hidden_release",
    }:
        bucket = "slot_confirmation_needed"
        expected_role = "confirm_class_inside_expected_slot"
    elif _metric_bool(metric, "candidate_oracle_has_safe_alternative"):
        bucket = "candidate_selection_needed"
        expected_role = "choose_safe_candidate_among_geometry_options"
    elif _metric_bool(metric, "crop_verification_object_crop_candidate_available"):
        bucket = "object_crop_confirmation_needed"
        expected_role = "verify_object_crop_candidate"
    elif projection == "expected_slot_anchor_release":
        bucket = "anchor_policy_validation"
        expected_role = "audit_anchor_release_quality"
    elif diagnostic.startswith("context_affine") or "translation_rescue" in projection:
        bucket = "geometry_refinement_validation"
        expected_role = "validate_refined_projection"
    else:
        bucket = "manual_review_bucket"
        expected_role = "manual_review_or_real_yolo_test"

    return {
        "yolo_synthetic_test_bucket": bucket,
        "yolo_synthetic_expected_role": expected_role,
        "yolo_synthetic_fixture": "synthetic_geometry_only_empty_yolo_detections",
        "yolo_synthetic_limitation": (
            "This synthetic test does not train or run YOLO; it only maps where "
            "a detector would be useful as an oracle/confirmation layer."
        ),
    }



def _yolo_synthetic_fixture_metric_fields(metric: dict[str, Any]) -> dict[str, Any]:
    oracle = _yolo_synthetic_fixture_outcome(metric, profile="oracle")
    noisy = _yolo_synthetic_fixture_outcome(metric, profile="noisy")
    false_positive = _yolo_synthetic_fixture_outcome(metric, profile="false_positive")
    gt_perfect = _yolo_gt_detector_fixture_outcome(metric, profile="perfect")
    gt_jitter = _yolo_gt_detector_fixture_outcome(metric, profile="jitter")
    gt_false_positive = _yolo_gt_detector_fixture_outcome(metric, profile="false_positive")

    return {
        "yolo_fixture_oracle_status": oracle["status"],
        "yolo_fixture_oracle_action": oracle["action"],
        "yolo_fixture_oracle_candidate_source": oracle["candidate_source"],
        "yolo_fixture_oracle_would_gain": oracle["would_gain"],
        "yolo_fixture_oracle_would_pass": oracle["would_pass"],
        "yolo_fixture_oracle_would_be_dangerous": oracle["would_be_dangerous"],
        "yolo_fixture_oracle_reason": oracle["reason"],
        "yolo_fixture_noisy_status": noisy["status"],
        "yolo_fixture_noisy_action": noisy["action"],
        "yolo_fixture_noisy_candidate_source": noisy["candidate_source"],
        "yolo_fixture_noisy_would_gain": noisy["would_gain"],
        "yolo_fixture_noisy_would_pass": noisy["would_pass"],
        "yolo_fixture_noisy_would_be_dangerous": noisy["would_be_dangerous"],
        "yolo_fixture_noisy_reason": noisy["reason"],
        "yolo_fixture_false_positive_status": false_positive["status"],
        "yolo_fixture_false_positive_action": false_positive["action"],
        "yolo_fixture_false_positive_candidate_source": false_positive["candidate_source"],
        "yolo_fixture_false_positive_would_gain": false_positive["would_gain"],
        "yolo_fixture_false_positive_would_pass": false_positive["would_pass"],
        "yolo_fixture_false_positive_would_be_dangerous": false_positive["would_be_dangerous"],
        "yolo_fixture_false_positive_reason": false_positive["reason"],
        "yolo_gt_perfect_status": gt_perfect["status"],
        "yolo_gt_perfect_action": gt_perfect["action"],
        "yolo_gt_perfect_candidate_source": gt_perfect["candidate_source"],
        "yolo_gt_perfect_would_gain": gt_perfect["would_gain"],
        "yolo_gt_perfect_would_pass": gt_perfect["would_pass"],
        "yolo_gt_perfect_would_be_dangerous": gt_perfect["would_be_dangerous"],
        "yolo_gt_perfect_reason": gt_perfect["reason"],
        "yolo_gt_jitter_status": gt_jitter["status"],
        "yolo_gt_jitter_action": gt_jitter["action"],
        "yolo_gt_jitter_candidate_source": gt_jitter["candidate_source"],
        "yolo_gt_jitter_would_gain": gt_jitter["would_gain"],
        "yolo_gt_jitter_would_pass": gt_jitter["would_pass"],
        "yolo_gt_jitter_would_be_dangerous": gt_jitter["would_be_dangerous"],
        "yolo_gt_jitter_reason": gt_jitter["reason"],
        "yolo_gt_false_positive_status": gt_false_positive["status"],
        "yolo_gt_false_positive_action": gt_false_positive["action"],
        "yolo_gt_false_positive_candidate_source": gt_false_positive["candidate_source"],
        "yolo_gt_false_positive_would_gain": gt_false_positive["would_gain"],
        "yolo_gt_false_positive_would_pass": gt_false_positive["would_pass"],
        "yolo_gt_false_positive_would_be_dangerous": gt_false_positive["would_be_dangerous"],
        "yolo_gt_false_positive_reason": gt_false_positive["reason"],
    }


def _yolo_synthetic_fixture_outcome(
    metric: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    passed = _metric_bool(metric, "passed")
    bucket = _metric_text(metric, "yolo_synthetic_test_bucket", "manual_review_bucket")
    source = _yolo_synthetic_safe_candidate_source(metric)
    has_safe_candidate = source is not None
    has_risky_candidate = _yolo_synthetic_has_risky_candidate(metric)

    if passed:
        return _yolo_synthetic_fixture_result(
            status="baseline_pass",
            action="keep_current_result",
            candidate_source="current_geometry",
            would_gain=False,
            would_pass=True,
            would_be_dangerous=False,
            reason="geometry_already_passed",
        )

    if profile == "oracle":
        if has_safe_candidate:
            return _yolo_synthetic_fixture_result(
                status="accepted_by_oracle",
                action="would_rescue_failure",
                candidate_source=source,
                would_gain=True,
                would_pass=True,
                would_be_dangerous=False,
                reason="perfect_detector_confirms_safe_candidate",
            )
        if has_risky_candidate:
            return _yolo_synthetic_fixture_result(
                status="rejected_risky_candidate",
                action="keep_current_policy",
                candidate_source=_yolo_synthetic_risky_candidate_source(metric),
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason="oracle_has_only_risky_candidate_guarded",
            )
        return _yolo_synthetic_fixture_result(
            status="no_detector_rescue",
            action="keep_current_policy",
            candidate_source="none",
            would_gain=False,
            would_pass=False,
            would_be_dangerous=False,
            reason=f"{bucket}_has_no_safe_candidate",
        )

    if profile == "noisy":
        if not _yolo_synthetic_noisy_fixture_detects(metric):
            return _yolo_synthetic_fixture_result(
                status="detector_miss",
                action="keep_current_policy",
                candidate_source="none",
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason="deterministic_noisy_fixture_missed_candidate",
            )
        if has_safe_candidate and not has_risky_candidate:
            return _yolo_synthetic_fixture_result(
                status="accepted_by_noisy_detector",
                action="would_rescue_failure",
                candidate_source=source,
                would_gain=True,
                would_pass=True,
                would_be_dangerous=False,
                reason="noisy_detector_confirms_clean_safe_candidate",
            )
        if has_safe_candidate:
            return _yolo_synthetic_fixture_result(
                status="review_required_mixed_candidates",
                action="show_manual_review_candidate",
                candidate_source=source,
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason="safe_candidate_exists_but_false_positive_risk_is_present",
            )
        return _yolo_synthetic_fixture_result(
            status="no_detector_rescue",
            action="keep_current_policy",
            candidate_source="none",
            would_gain=False,
            would_pass=False,
            would_be_dangerous=False,
            reason=f"{bucket}_has_no_confirmed_safe_candidate",
        )

    if profile == "false_positive":
        if has_risky_candidate:
            return _yolo_synthetic_fixture_result(
                status="dangerous_if_trusted",
                action="must_reject_false_positive",
                candidate_source=_yolo_synthetic_risky_candidate_source(metric),
                would_gain=False,
                would_pass=False,
                would_be_dangerous=True,
                reason="false_positive_fixture_targets_distractor_or_bad_crop",
            )
        if has_safe_candidate:
            return _yolo_synthetic_fixture_result(
                status="guarded_safe_candidate",
                action="would_need_normal_detector_thresholds",
                candidate_source=source,
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason="false_positive_fixture_did_not_find_dangerous_candidate",
            )
        return _yolo_synthetic_fixture_result(
            status="no_false_positive_candidate",
            action="keep_current_policy",
            candidate_source="none",
            would_gain=False,
            would_pass=False,
            would_be_dangerous=False,
            reason="no_risky_candidate_seen_in_existing_diagnostics",
        )

    return _yolo_synthetic_fixture_result(
        status="unknown_profile",
        action="keep_current_policy",
        candidate_source="none",
        would_gain=False,
        would_pass=False,
        would_be_dangerous=False,
        reason=profile,
    )


def _yolo_synthetic_fixture_result(
    *,
    status: str,
    action: str,
    candidate_source: str | None,
    would_gain: bool,
    would_pass: bool,
    would_be_dangerous: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "action": action,
        "candidate_source": candidate_source or "none",
        "would_gain": bool(would_gain),
        "would_pass": bool(would_pass),
        "would_be_dangerous": bool(would_be_dangerous),
        "reason": reason,
    }


def _yolo_synthetic_safe_candidate_source(metric: dict[str, Any]) -> str | None:
    if _metric_bool(metric, "hidden_shadow_would_pass") and not _metric_bool(
        metric,
        "hidden_shadow_dangerous",
    ):
        return _metric_text(metric, "hidden_shadow_projection", "hidden_shadow")

    if _metric_bool(metric, "candidate_oracle_has_safe_alternative"):
        return _metric_text(
            metric,
            "candidate_oracle_safe_gain_source",
            _metric_text(metric, "candidate_oracle_best_source", "candidate_oracle"),
        )

    if _metric_bool(metric, "candidate_oracle_best_would_pass") and not _metric_bool(
        metric,
        "candidate_oracle_best_dangerous",
    ):
        return _metric_text(metric, "candidate_oracle_best_source", "candidate_oracle")

    if _metric_bool(metric, "crop_verification_object_crop_candidate_would_pass") and not _metric_bool(
        metric,
        "crop_verification_object_crop_candidate_dangerous",
    ):
        return _metric_text(
            metric,
            "crop_verification_object_crop_candidate_source",
            "object_crop_candidate",
        )

    if _metric_bool(metric, "crop_verification_best_would_pass") and not _metric_bool(
        metric,
        "crop_verification_best_dangerous",
    ):
        return _metric_text(metric, "crop_verification_best_source", "crop_candidate")

    return None


def _yolo_synthetic_has_risky_candidate(metric: dict[str, Any]) -> bool:
    return any(
        (
            _metric_bool(metric, "hidden_shadow_dangerous"),
            _metric_bool(metric, "candidate_oracle_best_dangerous"),
            _metric_bool(metric, "crop_verification_object_crop_candidate_dangerous"),
            _metric_bool(metric, "crop_verification_best_dangerous"),
        )
    )


def _yolo_synthetic_risky_candidate_source(metric: dict[str, Any]) -> str:
    if _metric_bool(metric, "hidden_shadow_dangerous"):
        return _metric_text(metric, "hidden_shadow_projection", "hidden_shadow")
    if _metric_bool(metric, "candidate_oracle_best_dangerous"):
        return _metric_text(metric, "candidate_oracle_best_source", "candidate_oracle")
    if _metric_bool(metric, "crop_verification_object_crop_candidate_dangerous"):
        return _metric_text(
            metric,
            "crop_verification_object_crop_candidate_source",
            "object_crop_candidate",
        )
    if _metric_bool(metric, "crop_verification_best_dangerous"):
        return _metric_text(metric, "crop_verification_best_source", "crop_candidate")
    return "none"


def _yolo_synthetic_noisy_fixture_detects(metric: dict[str, Any]) -> bool:
    stable_key = "|".join(
        [
            _metric_text(metric, "case_name", _metric_text(metric, "name")),
            _metric_text(metric, "name"),
            _metric_text(metric, "projection"),
            _metric_text(metric, "yolo_synthetic_test_bucket"),
        ]
    )
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(stable_key))
    return checksum % 10 not in {0, 1}


def _yolo_gt_detector_fixture_outcome(
    metric: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    if _metric_bool(metric, "passed"):
        return _yolo_synthetic_fixture_result(
            status="baseline_pass",
            action="keep_current_result",
            candidate_source="current_geometry",
            would_gain=False,
            would_pass=True,
            would_be_dangerous=False,
            reason="geometry_already_passed",
        )

    bucket = _metric_text(metric, "yolo_synthetic_test_bucket", "manual_review_bucket")
    has_reference_slot = _yolo_gt_detector_has_reference_slot(metric)
    can_auto_release = _yolo_gt_detector_can_auto_release(metric)

    if profile == "perfect":
        if not has_reference_slot:
            return _yolo_synthetic_fixture_result(
                status="detector_only_manual_review",
                action="show_detector_only_candidate",
                candidate_source="gt_detector_mask",
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason=f"{bucket}_has_no_reference_slot_for_automatic_release",
            )
        if can_auto_release:
            return _yolo_synthetic_fixture_result(
                status="accepted_by_gt_detector",
                action="would_rescue_failure",
                candidate_source="gt_detector_mask",
                would_gain=True,
                would_pass=True,
                would_be_dangerous=False,
                reason="perfect_gt_detector_supplies_independent_mask_inside_reference_slot",
            )
        return _yolo_synthetic_fixture_result(
            status="manual_review_gt_detector",
            action="show_manual_review_candidate",
            candidate_source="gt_detector_mask",
            would_gain=False,
            would_pass=False,
            would_be_dangerous=False,
            reason=f"{bucket}_requires_manual_review_even_with_detector_mask",
        )

    if profile == "jitter":
        if not _yolo_gt_detector_jitter_detects(metric):
            return _yolo_synthetic_fixture_result(
                status="detector_miss",
                action="keep_current_policy",
                candidate_source="none",
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason="deterministic_jitter_fixture_missed_detection",
            )
        if not has_reference_slot:
            return _yolo_synthetic_fixture_result(
                status="detector_only_manual_review",
                action="show_detector_only_candidate",
                candidate_source="jittered_gt_detector_mask",
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason=f"{bucket}_has_no_reference_slot_for_jittered_detector",
            )
        if can_auto_release and _yolo_gt_detector_jitter_passes(metric):
            return _yolo_synthetic_fixture_result(
                status="accepted_by_jittered_detector",
                action="would_rescue_failure",
                candidate_source="jittered_gt_detector_mask",
                would_gain=True,
                would_pass=True,
                would_be_dangerous=False,
                reason="jittered_gt_detector_stays_inside_reference_slot",
            )
        return _yolo_synthetic_fixture_result(
            status="rejected_jittered_detector",
            action="keep_current_policy",
            candidate_source="jittered_gt_detector_mask",
            would_gain=False,
            would_pass=False,
            would_be_dangerous=False,
            reason="jittered_gt_detector_not_confident_enough_for_auto_release",
        )

    if profile == "false_positive":
        if not _yolo_gt_detector_false_positive_present(metric):
            return _yolo_synthetic_fixture_result(
                status="no_false_positive_candidate",
                action="keep_current_policy",
                candidate_source="none",
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason="no_distractor_candidate_in_synthetic_scene",
            )
        if _yolo_gt_detector_false_positive_rejected_by_guards(metric):
            return _yolo_synthetic_fixture_result(
                status="rejected_by_slot_guards",
                action="keep_current_policy",
                candidate_source="synthetic_distractor_detection",
                would_gain=False,
                would_pass=False,
                would_be_dangerous=False,
                reason="slot_or_manual_review_guards_reject_false_positive",
            )
        return _yolo_synthetic_fixture_result(
            status="dangerous_if_auto_trusted",
            action="must_require_slot_and_shape_guards",
            candidate_source="synthetic_distractor_detection",
            would_gain=False,
            would_pass=False,
            would_be_dangerous=True,
            reason="false_positive_can_match_the_wrong_object_without_strict_guards",
        )

    return _yolo_synthetic_fixture_result(
        status="unknown_profile",
        action="keep_current_policy",
        candidate_source="none",
        would_gain=False,
        would_pass=False,
        would_be_dangerous=False,
        reason=profile,
    )


def _yolo_gt_detector_has_reference_slot(metric: dict[str, Any]) -> bool:
    projection = _metric_text(metric, "projection")
    if projection == "none":
        return False
    if _metric_bool(metric, "unsafe_hidden"):
        return True
    return bool(projection)


def _yolo_gt_detector_can_auto_release(metric: dict[str, Any]) -> bool:
    bucket = _metric_text(metric, "yolo_synthetic_test_bucket", "manual_review_bucket")
    if bucket == "no_projection_detector_needed":
        return False
    if bucket in {
        "slot_confirmation_needed",
        "object_crop_confirmation_needed",
        "detector_oracle_required",
        "candidate_selection_needed",
        "anchor_policy_validation",
        "geometry_refinement_validation",
    }:
        return True
    if bucket == "manual_review_bucket":
        return _result_policy_geometry_confidence(metric) != "very_low"
    return False


def _yolo_gt_detector_jitter_detects(metric: dict[str, Any]) -> bool:
    return _stable_metric_checksum(metric, "gt-jitter-detect") % 10 not in {0, 1}


def _yolo_gt_detector_jitter_passes(metric: dict[str, Any]) -> bool:
    shape = _metric_text(metric, "object_shape")
    checksum = _stable_metric_checksum(metric, "gt-jitter-quality")
    quality_bucket = checksum % 100
    if shape == "thin_fork":
        return quality_bucket >= 28
    if shape == "hook_like_part":
        return quality_bucket >= 24
    if shape == "concave_l_bracket":
        return quality_bucket >= 20
    return quality_bucket >= 16


def _yolo_gt_detector_false_positive_present(metric: dict[str, Any]) -> bool:
    if _yolo_synthetic_has_risky_candidate(metric):
        return True
    distance = _metric_float(metric, "nearest_distractor_distance_px", default=float("inf"))
    if math.isfinite(distance) and distance <= 180.0:
        return True
    return _stable_metric_checksum(metric, "gt-false-positive") % 10 in {0, 1, 2}


def _yolo_gt_detector_false_positive_rejected_by_guards(metric: dict[str, Any]) -> bool:
    if _metric_bool(metric, "unsafe_hidden"):
        return True
    if _metric_text(metric, "projection") == "none":
        return True
    if _metric_bool(metric, "candidate_oracle_best_dangerous"):
        return False
    if _metric_bool(metric, "crop_verification_object_crop_candidate_dangerous"):
        return False
    iou = _metric_float(metric, "iou")
    drift = _metric_float(metric, "center_drift_px", default=float("inf"))
    support = max(
        _metric_int(metric, "context_feature_support"),
        _metric_int(metric, "fallback_slot_feature_support"),
    )
    if iou >= 0.42 or drift <= 12.0 or support >= 10:
        return True
    return False


def _stable_metric_checksum(metric: dict[str, Any], salt: str) -> int:
    stable_key = "|".join(
        [
            salt,
            _metric_text(metric, "case_name", _metric_text(metric, "name")),
            _metric_text(metric, "name"),
            _metric_text(metric, "projection"),
            _metric_text(metric, "object_shape"),
            _metric_text(metric, "yolo_synthetic_test_bucket"),
        ]
    )
    return sum((index + 1) * ord(char) for index, char in enumerate(stable_key))


def _result_policy_geometry_confidence(metric: dict[str, Any]) -> str:
    iou = _metric_float(metric, "iou")
    drift = _metric_float(metric, "center_drift_px", default=float("inf"))
    support = max(
        _metric_int(metric, "context_feature_support"),
        _metric_int(metric, "fallback_slot_feature_support"),
    )

    if iou >= 0.55 and drift <= 16.0 and support >= 4:
        return "medium"
    if iou >= 0.35 or support >= 8:
        return "low"
    return "very_low"


def _object_metric_dicts(results: list[SyntheticResult]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for result in results:
        if result.object_metrics:
            for raw_metric in result.object_metrics:
                metric = dict(raw_metric)
                metric.setdefault("case_index", result.index)
                metric.setdefault("case_name", result.name)
                metric.setdefault("case_kind", result.scenario_kind)
                metric.setdefault("case_projection", result.projection)
                metric.setdefault("case_passed", result.passed)
                metrics.append(_enrich_object_metric(metric))
            continue

        fallback_metric = {
            "name": result.name,
            "object_shape": result.object_shape,
            "status": result.status,
            "projection": result.projection,
            "passed": result.passed,
            "safety_passed": result.safety_passed,
            "dangerous_projection": result.dangerous_projection,
            "unsafe_hidden": result.unsafe_hidden,
            "hidden_reason": None,
            "iou": result.iou,
            "center_drift_px": result.center_drift_px,
            "area_ratio": result.area_ratio,
            "axis_angle_error_deg": result.axis_angle_error_deg,
            "major_length_ratio": result.major_length_ratio,
            "closer_to_distractor": result.closer_to_distractor,
            "notes": result.notes,
            "global_translation_rescue_attempted": False,
            "global_translation_rescue_accepted": False,
            "global_translation_rescue_reject_reason": None,
            "global_translation_rescue_local_point_count": 0,
            "global_translation_rescue_min_support": 0,
            "global_translation_rescue_candidate_count": 0,
            "global_translation_rescue_inlier_count": 0,
            "global_translation_rescue_inlier_ratio": None,
            "global_translation_rescue_median_error": None,
            "global_translation_rescue_shift_factor": None,
            "global_translation_rescue_context_spread": None,
            "global_translation_rescue_search_containment": None,
            "global_translation_rescue_local_area_score": None,
            "global_translation_rescue_local_center_factor": None,
            "global_translation_rescue_slot_feature_support": 0,
            "global_translation_rescue_slot_feature_total": 0,
            "anchor_release_runtime_trusted_anchor_count": 0,
            "anchor_release_runtime_trusted_anchor_source_counts": {},
            "anchor_release_runtime_anchor_before_current_count": 0,
            "anchor_release_runtime_anchor_after_current_count": 0,
            "anchor_release_runtime_processed_expected_count": 0,
            "anchor_release_runtime_future_expected_count": 0,
            "anchor_release_runtime_built_count": 0,
            "anchor_release_runtime_source_counts": {},
            "anchor_release_runtime_reject_counts": {},
            "anchor_release_runtime_probe_counts": {},
            "anchor_release_final_trusted_anchor_count": 0,
            "anchor_release_final_trusted_anchor_source_counts": {},
            "anchor_release_final_anchor_before_current_count": 0,
            "anchor_release_final_anchor_after_current_count": 0,
            "anchor_release_final_resolved_anchor_count": 0,
            "anchor_release_final_resolved_anchor_source_counts": {},
            "crop_verification_attempted": False,
            "crop_verification_candidate_count": 0,
            "crop_verification_source_count": 0,
            "crop_verification_best_source": None,
            "crop_verification_best_score": None,
            "crop_verification_best_object_matches": 0,
            "crop_verification_best_ref_containment": None,
            "crop_verification_best_frame_containment": None,
            "crop_verification_best_iou": None,
            "crop_verification_best_center_drift_px": None,
            "crop_verification_best_area_ratio": None,
            "crop_verification_best_would_pass": False,
            "crop_verification_best_dangerous": False,
            "crop_verification_selected_score": None,
            "crop_verification_selected_object_matches": 0,
            "crop_verification_object_crop_candidate_available": False,
            "crop_verification_object_crop_candidate_source": None,
            "crop_verification_object_crop_candidate_iou": None,
            "crop_verification_object_crop_candidate_center_drift_px": None,
            "crop_verification_object_crop_candidate_would_pass": False,
            "crop_verification_object_crop_candidate_dangerous": False,
            "crop_verification_object_crop_candidate_inliers": 0,
            "crop_verification_object_crop_candidate_inlier_ratio": None,
            "crop_verification_assessment": None,
            "crop_verification_sources": [],
        }
        metrics.append(_enrich_object_metric(fallback_metric))
    return metrics


def _finite_metric_values(
    metrics: list[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for metric in metrics:
        value = metric.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _bucket_object_metric_stats(
    metrics: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        bucket = str(metric.get(key) or "unknown")
        buckets.setdefault(bucket, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for bucket, bucket_metrics in sorted(buckets.items()):
        total = len(bucket_metrics)
        passed = sum(1 for metric in bucket_metrics if bool(metric.get("passed")))
        unsafe_hidden = sum(1 for metric in bucket_metrics if bool(metric.get("unsafe_hidden")))
        dangerous = sum(
            1 for metric in bucket_metrics if bool(metric.get("dangerous_projection"))
        )
        ious = _finite_metric_values(bucket_metrics, "iou")
        drifts = _finite_metric_values(bucket_metrics, "center_drift_px")
        stats[bucket] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total * 100.0) if total else 0.0,
            "unsafe_hidden": unsafe_hidden,
            "dangerous": dangerous,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "mean_center_drift_px": float(np.mean(drifts)) if drifts else 0.0,
        }
    return stats


def _object_failure_reason_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if bool(metric.get("passed")):
            continue
        notes = metric.get("notes")
        if not isinstance(notes, list) or not notes:
            reason = "failed_without_note"
        else:
            reason = str(notes[0])
        buckets.setdefault(reason, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for reason, reason_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        projection_counts: dict[str, int] = {}
        shape_counts: dict[str, int] = {}
        for metric in reason_metrics:
            projection = str(metric.get("projection") or "unknown")
            shape = str(metric.get("object_shape") or "unknown")
            projection_counts[projection] = projection_counts.get(projection, 0) + 1
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        stats[reason] = {
            "total": len(reason_metrics),
            "top_projection": _top_bucket_name(projection_counts),
            "top_shape": _top_bucket_name(shape_counts),
        }
    return stats


def _object_hidden_reason_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if not bool(metric.get("unsafe_hidden")):
            continue
        reason = str(metric.get("hidden_reason") or "unknown_hidden_reason")
        buckets.setdefault(reason, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for reason, reason_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        shape_counts: dict[str, int] = {}
        for metric in reason_metrics:
            shape = str(metric.get("object_shape") or "unknown")
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        stats[reason] = {
            "total": len(reason_metrics),
            "top_shape": _top_bucket_name(shape_counts),
        }
    return stats


def _object_anchor_release_reject_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if not bool(metric.get("anchor_release_attempted")):
            continue
        reason = metric.get("anchor_release_reject_reason")
        if reason is None:
            reason = "anchor_release_accepted_or_no_reject"
        buckets.setdefault(str(reason), []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for reason, reason_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        projection_counts: dict[str, int] = {}
        shape_counts: dict[str, int] = {}
        hidden_counts: dict[str, int] = {}
        for metric in reason_metrics:
            projection = str(metric.get("projection") or "unknown")
            shape = str(metric.get("object_shape") or "unknown")
            hidden_reason = str(metric.get("hidden_reason") or "not_hidden")
            projection_counts[projection] = projection_counts.get(projection, 0) + 1
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            hidden_counts[hidden_reason] = hidden_counts.get(hidden_reason, 0) + 1
        stats[reason] = {
            "total": len(reason_metrics),
            "top_projection": _top_bucket_name(projection_counts),
            "top_shape": _top_bucket_name(shape_counts),
            "top_hidden_reason": _top_bucket_name(hidden_counts),
            "mean_candidates": _mean_metric_value(
                reason_metrics,
                "anchor_release_candidate_count",
            ),
            "mean_inliers": _mean_metric_value(
                reason_metrics,
                "anchor_release_inliers",
            ),
        }
    return stats



def _object_anchor_build_reject_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    reject_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    probe_counts: dict[str, int] = {}
    affected_objects = 0

    for metric in metrics:
        if not bool(metric.get("anchor_release_attempted")):
            continue
        if metric.get("anchor_release_reject_reason") != "anchor_release_rejected_no_trusted_anchors":
            continue
        affected_objects += 1
        for key, value in (metric.get("anchor_release_build_reject_counts") or {}).items():
            reject_counts[str(key)] = reject_counts.get(str(key), 0) + int(value)
        for key, value in (metric.get("anchor_release_build_source_counts") or {}).items():
            source_counts[str(key)] = source_counts.get(str(key), 0) + int(value)
        for key, value in (metric.get("anchor_release_build_probe_counts") or {}).items():
            probe_counts[str(key)] = probe_counts.get(str(key), 0) + int(value)

    stats: dict[str, dict[str, Any]] = {}
    for reason, total in sorted(reject_counts.items(), key=lambda item: item[1], reverse=True):
        stats[reason] = {
            "total": total,
            "affected_objects": affected_objects,
            "top_built_source": _top_bucket_name(source_counts),
            "top_probe": _top_bucket_name(probe_counts),
        }
    return stats



def _object_anchor_order_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    no_trusted_metrics = [
        metric
        for metric in metrics
        if bool(metric.get("anchor_release_attempted"))
        and metric.get("anchor_release_reject_reason") == "anchor_release_rejected_no_trusted_anchors"
    ]
    attempted_metrics = [
        metric for metric in metrics if bool(metric.get("anchor_release_attempted"))
    ]

    def sum_counts(metric_list: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metric in metric_list:
            for source, value in (metric.get(key) or {}).items():
                counts[str(source)] = counts.get(str(source), 0) + int(value)
        return counts

    def count_positive(metric_list: list[dict[str, Any]], key: str) -> int:
        return sum(1 for metric in metric_list if int(metric.get(key) or 0) > 0)

    return {
        "no_trusted_anchor_order": {
            "total": len(no_trusted_metrics),
            "runtime_has_any_anchor": count_positive(
                no_trusted_metrics,
                "anchor_release_runtime_trusted_anchor_count",
            ),
            "final_has_any_anchor": count_positive(
                no_trusted_metrics,
                "anchor_release_final_trusted_anchor_count",
            ),
            "final_has_previous_anchor": count_positive(
                no_trusted_metrics,
                "anchor_release_final_anchor_before_current_count",
            ),
            "final_has_future_anchor": count_positive(
                no_trusted_metrics,
                "anchor_release_final_anchor_after_current_count",
            ),
            "final_has_resolved_anchor": count_positive(
                no_trusted_metrics,
                "anchor_release_final_resolved_anchor_count",
            ),
            "top_runtime_source": _top_bucket_name(
                sum_counts(no_trusted_metrics, "anchor_release_runtime_trusted_anchor_source_counts")
            ),
            "top_final_source": _top_bucket_name(
                sum_counts(no_trusted_metrics, "anchor_release_final_trusted_anchor_source_counts")
            ),
            "top_final_resolved_source": _top_bucket_name(
                sum_counts(no_trusted_metrics, "anchor_release_final_resolved_anchor_source_counts")
            ),
        },
        "all_anchor_release_attempts": {
            "total": len(attempted_metrics),
            "runtime_has_any_anchor": count_positive(
                attempted_metrics,
                "anchor_release_runtime_trusted_anchor_count",
            ),
            "final_has_any_anchor": count_positive(
                attempted_metrics,
                "anchor_release_final_trusted_anchor_count",
            ),
            "runtime_built_count": sum(
                int(metric.get("anchor_release_runtime_built_count") or 0)
                for metric in attempted_metrics
            ),
            "top_runtime_source": _top_bucket_name(
                sum_counts(attempted_metrics, "anchor_release_runtime_source_counts")
            ),
            "top_runtime_reject": _top_bucket_name(
                sum_counts(attempted_metrics, "anchor_release_runtime_reject_counts")
            ),
            "top_runtime_probe": _top_bucket_name(
                sum_counts(attempted_metrics, "anchor_release_runtime_probe_counts")
            ),
        },
    }


def _object_none_projection_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if str(metric.get("projection") or "") != "none":
            continue
        reason = str(
            metric.get("none_reason")
            or metric.get("reason_code")
            or "unknown_none_reason"
        )
        buckets.setdefault(reason, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for reason, reason_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        shape_counts: dict[str, int] = {}
        code_counts: dict[str, int] = {}
        for metric in reason_metrics:
            shape = str(metric.get("object_shape") or "unknown")
            code = str(metric.get("reason_code") or "unknown")
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            code_counts[code] = code_counts.get(code, 0) + 1
        passed = sum(1 for metric in reason_metrics if bool(metric.get("passed")))
        has_homography = sum(
            1 for metric in reason_metrics if bool(metric.get("none_has_homography"))
        )
        stats[reason] = {
            "total": len(reason_metrics),
            "passed": passed,
            "failed": len(reason_metrics) - passed,
            "pass_rate": (passed / len(reason_metrics) * 100.0) if reason_metrics else 0.0,
            "top_shape": _top_bucket_name(shape_counts),
            "top_reason_code": _top_bucket_name(code_counts),
            "has_homography": has_homography,
            "mean_projected_point_count": _mean_metric_value(
                reason_metrics,
                "none_projected_point_count",
            ),
        }
    return stats


def _object_global_fallback_diagnostics(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if str(metric.get("projection") or "") != "expected_slot_global_fallback":
            continue
        reason = str(
            metric.get("fallback_reason")
            or metric.get("reason_code")
            or "unknown_global_fallback_reason"
        )
        buckets.setdefault(reason, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for reason, reason_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        shape_counts: dict[str, int] = {}
        code_counts: dict[str, int] = {}
        disagrees = 0
        global_available = 0
        for metric in reason_metrics:
            shape = str(metric.get("object_shape") or "unknown")
            code = str(metric.get("reason_code") or "unknown")
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            code_counts[code] = code_counts.get(code, 0) + 1
            if bool(metric.get("fallback_local_global_disagrees")):
                disagrees += 1
            if bool(metric.get("fallback_global_available")):
                global_available += 1
        passed = sum(1 for metric in reason_metrics if bool(metric.get("passed")))
        stats[reason] = {
            "total": len(reason_metrics),
            "passed": passed,
            "failed": len(reason_metrics) - passed,
            "pass_rate": (passed / len(reason_metrics) * 100.0) if reason_metrics else 0.0,
            "top_shape": _top_bucket_name(shape_counts),
            "top_reason_code": _top_bucket_name(code_counts),
            "global_available": global_available,
            "local_global_disagrees": disagrees,
            "mean_area_score": _mean_metric_value(
                reason_metrics,
                "fallback_local_global_area_score",
            ),
            "mean_center_factor": _mean_metric_value(
                reason_metrics,
                "fallback_local_global_center_factor",
            ),
            "mean_slot_support": _mean_metric_value(
                reason_metrics,
                "fallback_slot_feature_support",
            ),
            "mean_slot_total": _mean_metric_value(
                reason_metrics,
                "fallback_slot_feature_total",
            ),
        }
    return stats

def _object_global_translation_rescue_diagnostics(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if not bool(metric.get("global_translation_rescue_attempted")):
            continue
        reason = (
            "accepted"
            if bool(metric.get("global_translation_rescue_accepted"))
            else str(
                metric.get("global_translation_rescue_reject_reason")
                or "unknown_global_translation_reject"
            )
        )
        buckets.setdefault(reason, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for reason, reason_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        projection_counts: dict[str, int] = {}
        shape_counts: dict[str, int] = {}
        fallback_counts: dict[str, int] = {}
        passed = sum(1 for metric in reason_metrics if bool(metric.get("passed")))
        accepted = sum(
            1
            for metric in reason_metrics
            if bool(metric.get("global_translation_rescue_accepted"))
        )
        for metric in reason_metrics:
            projection = str(metric.get("projection") or "unknown")
            shape = str(metric.get("object_shape") or "unknown")
            fallback_reason = str(metric.get("fallback_reason") or "unknown")
            projection_counts[projection] = projection_counts.get(projection, 0) + 1
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            fallback_counts[fallback_reason] = fallback_counts.get(fallback_reason, 0) + 1
        stats[reason] = {
            "total": len(reason_metrics),
            "accepted": accepted,
            "passed": passed,
            "failed": len(reason_metrics) - passed,
            "pass_rate": (passed / len(reason_metrics) * 100.0) if reason_metrics else 0.0,
            "top_projection": _top_bucket_name(projection_counts),
            "top_shape": _top_bucket_name(shape_counts),
            "top_fallback_reason": _top_bucket_name(fallback_counts),
            "mean_local_points": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_local_point_count",
            ),
            "mean_min_support": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_min_support",
            ),
            "mean_candidates": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_candidate_count",
            ),
            "mean_inliers": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_inlier_count",
            ),
            "mean_inlier_ratio": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_inlier_ratio",
            ),
            "mean_median_error": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_median_error",
            ),
            "mean_shift_factor": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_shift_factor",
            ),
            "mean_context_spread": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_context_spread",
            ),
            "mean_search_containment": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_search_containment",
            ),
            "mean_local_area_score": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_local_area_score",
            ),
            "mean_local_center_factor": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_local_center_factor",
            ),
            "mean_slot_support": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_slot_feature_support",
            ),
            "mean_slot_total": _mean_metric_value(
                reason_metrics,
                "global_translation_rescue_slot_feature_total",
            ),
        }
    return stats


def _object_unsafe_hidden_shadow_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    hidden_metrics = [metric for metric in metrics if bool(metric.get("unsafe_hidden"))]
    shadow_metrics = [
        metric for metric in hidden_metrics if bool(metric.get("hidden_shadow_available"))
    ]
    if not hidden_metrics:
        return {
            "total_hidden": 0,
            "shadow_available": 0,
            "would_pass": 0,
            "would_fail": 0,
            "would_be_dangerous": 0,
            "safe_would_pass": 0,
            "shadow_pass_rate": 0.0,
            "shadow_dangerous_rate": 0.0,
            "mean_iou": 0.0,
            "mean_center_drift_px": 0.0,
            "top_shadow_projection": "—",
            "top_shadow_source": "—",
            "top_hidden_reason": "—",
            "top_shape": "—",
            "by_projection": {},
        }

    def count_values(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metric in shadow_metrics:
            value = str(metric.get(key) or "—")
            counts[value] = counts.get(value, 0) + 1
        return counts

    would_pass = sum(1 for metric in shadow_metrics if bool(metric.get("hidden_shadow_would_pass")))
    would_be_dangerous = sum(
        1 for metric in shadow_metrics if bool(metric.get("hidden_shadow_dangerous"))
    )
    safe_would_pass = sum(
        1
        for metric in shadow_metrics
        if bool(metric.get("hidden_shadow_would_pass"))
        and not bool(metric.get("hidden_shadow_dangerous"))
    )

    by_projection: dict[str, dict[str, Any]] = {}
    projection_buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in shadow_metrics:
        projection = str(metric.get("hidden_shadow_projection") or "—")
        projection_buckets.setdefault(projection, []).append(metric)
    for projection, bucket in sorted(
        projection_buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        bucket_pass = sum(
            1 for metric in bucket if bool(metric.get("hidden_shadow_would_pass"))
        )
        bucket_dangerous = sum(
            1 for metric in bucket if bool(metric.get("hidden_shadow_dangerous"))
        )
        by_projection[projection] = {
            "total": len(bucket),
            "would_pass": bucket_pass,
            "would_fail": len(bucket) - bucket_pass,
            "would_be_dangerous": bucket_dangerous,
            "safe_would_pass": sum(
                1
                for metric in bucket
                if bool(metric.get("hidden_shadow_would_pass"))
                and not bool(metric.get("hidden_shadow_dangerous"))
            ),
            "pass_rate": (bucket_pass / len(bucket) * 100.0) if bucket else 0.0,
            "dangerous_rate": (bucket_dangerous / len(bucket) * 100.0) if bucket else 0.0,
            "mean_iou": _mean_metric_value(bucket, "hidden_shadow_iou"),
            "mean_center_drift_px": _mean_metric_value(
                bucket,
                "hidden_shadow_center_drift_px",
            ),
        }

    return {
        "total_hidden": len(hidden_metrics),
        "shadow_available": len(shadow_metrics),
        "would_pass": would_pass,
        "would_fail": len(shadow_metrics) - would_pass,
        "would_be_dangerous": would_be_dangerous,
        "safe_would_pass": safe_would_pass,
        "shadow_pass_rate": (would_pass / len(shadow_metrics) * 100.0) if shadow_metrics else 0.0,
        "shadow_dangerous_rate": (
            would_be_dangerous / len(shadow_metrics) * 100.0
        ) if shadow_metrics else 0.0,
        "mean_iou": _mean_metric_value(shadow_metrics, "hidden_shadow_iou"),
        "mean_center_drift_px": _mean_metric_value(
            shadow_metrics,
            "hidden_shadow_center_drift_px",
        ),
        "top_shadow_projection": _top_bucket_name(count_values("hidden_shadow_projection")),
        "top_shadow_source": _top_bucket_name(count_values("hidden_shadow_fallback_source")),
        "top_hidden_reason": _top_bucket_name(count_values("hidden_shadow_reason")),
        "top_shape": _top_bucket_name(count_values("object_shape")),
        "by_projection": by_projection,
    }


def _metric_bool(metric: dict[str, Any], key: str) -> bool:
    return bool(metric.get(key))


def _metric_int(metric: dict[str, Any], key: str) -> int:
    value = metric.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _metric_float(metric: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metric.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _metric_text(metric: dict[str, Any], key: str, default: str = "—") -> str:
    value = metric.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _metric_notes_text(metric: dict[str, Any]) -> str:
    notes = metric.get("notes")
    if isinstance(notes, list):
        return " | ".join(str(note) for note in notes if note)
    if notes is None:
        return ""
    return str(notes)


def _metric_count_values(metrics: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in metrics:
        value = _metric_text(metric, key)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _metric_iou_bucket(metric: dict[str, Any], *, key: str = "iou") -> str:
    value = metric.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "not_projected"
    iou = float(value)
    if iou <= 0.0:
        return "0.000"
    if iou < 0.20:
        return "0.000-0.199"
    if iou < 0.42:
        return "0.200-0.419"
    if iou < 0.55:
        return "0.420-0.549"
    if iou < 0.75:
        return "0.550-0.749"
    return "0.750+"


def _metric_drift_bucket(metric: dict[str, Any], *, key: str = "center_drift_px") -> str:
    value = metric.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "not_projected"
    drift = float(value)
    if drift <= 3.0:
        return "<=3px"
    if drift <= 8.0:
        return "3-8px"
    if drift <= 16.0:
        return "8-16px"
    if drift <= 28.0:
        return "16-28px"
    if drift <= 40.0:
        return "28-40px"
    return ">40px"


def _metric_support_bucket(metric: dict[str, Any]) -> str:
    support = _metric_int(metric, "context_feature_support")
    if support <= 0:
        support = _metric_int(metric, "fallback_slot_feature_support")
    if support <= 0:
        return "0"
    if support <= 3:
        return "1-3"
    if support <= 9:
        return "4-9"
    if support <= 24:
        return "10-24"
    return "25+"


def _metric_area_bucket(metric: dict[str, Any], *, key: str = "area_ratio") -> str:
    value = metric.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "not_projected"
    ratio = float(value)
    if ratio <= 0.0:
        return "0.000"
    if ratio < 0.50:
        return "collapsed"
    if ratio < 0.75:
        return "shrink"
    if ratio <= 1.35:
        return "normal"
    if ratio <= 2.00:
        return "expanded"
    return "exploded"


def _metric_anchor_state(metric: dict[str, Any]) -> str:
    if _metric_text(metric, "projection") == "expected_slot_anchor_release":
        return "accepted"
    if _metric_bool(metric, "anchor_release_attempted"):
        return _metric_text(metric, "anchor_release_reject_reason", "attempted_unknown")
    return "not_attempted"


def _metric_global_translation_state(metric: dict[str, Any]) -> str:
    if _metric_bool(metric, "global_translation_rescue_accepted"):
        return "accepted"
    if _metric_bool(metric, "global_translation_rescue_attempted"):
        return _metric_text(
            metric,
            "global_translation_rescue_reject_reason",
            "attempted_unknown",
        )
    return "not_attempted"


def _metric_diagnostic_class(metric: dict[str, Any]) -> str:
    if _metric_bool(metric, "passed"):
        return "passed"
    if _metric_bool(metric, "dangerous_projection"):
        return "dangerous_current_projection"

    projection = _metric_text(metric, "projection")
    notes = _metric_notes_text(metric)
    if _metric_bool(metric, "unsafe_hidden"):
        if not _metric_bool(metric, "hidden_shadow_available"):
            return "hidden_no_shadow_candidate"
        if _metric_bool(metric, "hidden_shadow_dangerous"):
            return "hidden_shadow_dangerous_candidate"
        if _metric_bool(metric, "hidden_shadow_would_pass"):
            return "hidden_shadow_safe_release_candidate"
        return "hidden_shadow_bad_candidate"

    if projection == "none":
        reason = _metric_text(metric, "none_reason", "projection_failed")
        return f"none_{reason}"
    if projection == "expected_slot_global_fallback_hidden_release":
        return "selective_hidden_global_fallback_release_geometry_fail"
    if projection == "expected_slot_agreement_hidden_release":
        return "selective_hidden_expected_slot_agreement_release_geometry_fail"
    if projection == "expected_slot_global_fallback":
        return "weak_global_fallback_geometry_fail"
    if projection == "expected_slot":
        return "raw_expected_slot_geometry_fail"
    if projection == "expected_slot_anchor_release":
        return "anchor_release_geometry_or_consensus_fail"
    if projection == "context_feature_affine":
        if "при слабом контексте refinement" in notes:
            return "context_affine_weak_context_policy_fail"
        return "context_affine_geometry_fail"
    if projection in {
        "context_feature_affine_translation_rescue",
        "expected_slot_context_translation_rescue",
        "expected_slot_scene_translation_rescue",
    }:
        return "translation_rescue_geometry_fail"
    return "other_geometry_fail"


def _metric_recommended_action(metric: dict[str, Any]) -> str:
    diagnostic_class = _metric_diagnostic_class(metric)
    if diagnostic_class == "hidden_shadow_safe_release_candidate":
        projection = _metric_text(metric, "hidden_shadow_projection")
        if projection == "expected_slot_global_fallback":
            return "selective_release_hidden_global_fallback"
        return "selective_release_hidden_with_extra_guards"
    if diagnostic_class == "hidden_shadow_dangerous_candidate":
        return "keep_hidden_needs_extra_evidence"
    if diagnostic_class == "hidden_shadow_bad_candidate":
        return "keep_hidden_shadow_is_wrong"
    if diagnostic_class == "hidden_no_shadow_candidate":
        return "collect_shadow_candidate_or_leave_hidden"
    if diagnostic_class.startswith("none_"):
        return "add_none_clipped_or_slot_shadow_diagnostics"
    if diagnostic_class == "selective_hidden_global_fallback_release_geometry_fail":
        return "tighten_selective_hidden_release_guards"
    if diagnostic_class == "selective_hidden_expected_slot_agreement_release_geometry_fail":
        return "tighten_candidate_agreement_release_guards"
    if diagnostic_class == "weak_global_fallback_geometry_fail":
        return "do_not_trust_raw_global_fallback_need_confirmation"
    if diagnostic_class == "raw_expected_slot_geometry_fail":
        return "raw_slot_too_crude_need_local_or_anchor_evidence"
    if diagnostic_class == "context_affine_weak_context_policy_fail":
        return "separate_policy_fail_from_geometry_or_demote_weak_affine"
    if diagnostic_class == "context_affine_geometry_fail":
        return "tighten_or_refine_context_affine_selection"
    if diagnostic_class == "anchor_release_geometry_or_consensus_fail":
        return "improve_anchor_consensus_or_overlap_arbitration"
    if diagnostic_class == "dangerous_current_projection":
        return "must_hide_or_require_external_confirmation"
    return "inspect_case_manually"


def _metric_recoverability(metric: dict[str, Any]) -> str:
    if _metric_bool(metric, "passed"):
        return "already_passed"
    diagnostic_class = _metric_diagnostic_class(metric)
    if diagnostic_class == "hidden_shadow_safe_release_candidate":
        return "high_safe_gain"
    if diagnostic_class == "hidden_shadow_dangerous_candidate":
        return "high_risk"
    if diagnostic_class in {
        "selective_hidden_global_fallback_release_geometry_fail",
        "selective_hidden_expected_slot_agreement_release_geometry_fail",
    }:
        return "medium_needs_better_selector"
    if diagnostic_class in {"hidden_shadow_bad_candidate", "weak_global_fallback_geometry_fail"}:
        return "low_without_new_evidence"
    if diagnostic_class.startswith("none_"):
        return "unknown_needs_shadow"
    if diagnostic_class in {
        "raw_expected_slot_geometry_fail",
        "context_affine_geometry_fail",
        "anchor_release_geometry_or_consensus_fail",
    }:
        return "medium_needs_better_selector"
    return "manual_review"


def _metric_diagnostic_row(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_index": _metric_int(metric, "case_index"),
        "case_kind": _metric_text(metric, "case_kind"),
        "case_name": _metric_text(metric, "case_name", _metric_text(metric, "name")),
        "object_name": _metric_text(metric, "name"),
        "shape": _metric_text(metric, "object_shape"),
        "projection": _metric_text(metric, "projection"),
        "result_policy_status": _metric_text(metric, "result_policy_status"),
        "result_policy_confidence": _metric_text(metric, "result_policy_confidence"),
        "result_policy_action": _metric_text(metric, "result_policy_action"),
        "result_policy_user_label": _metric_text(metric, "result_policy_user_label"),
        "result_policy_reason": _metric_text(metric, "result_policy_reason"),
        "yolo_synthetic_test_bucket": _metric_text(metric, "yolo_synthetic_test_bucket"),
        "yolo_synthetic_expected_role": _metric_text(metric, "yolo_synthetic_expected_role"),
        "yolo_synthetic_fixture": _metric_text(metric, "yolo_synthetic_fixture"),
        "yolo_synthetic_limitation": _metric_text(metric, "yolo_synthetic_limitation"),
        "yolo_fixture_oracle_status": _metric_text(metric, "yolo_fixture_oracle_status"),
        "yolo_fixture_oracle_action": _metric_text(metric, "yolo_fixture_oracle_action"),
        "yolo_fixture_oracle_would_gain": _metric_bool(metric, "yolo_fixture_oracle_would_gain"),
        "yolo_fixture_oracle_reason": _metric_text(metric, "yolo_fixture_oracle_reason"),
        "yolo_fixture_noisy_status": _metric_text(metric, "yolo_fixture_noisy_status"),
        "yolo_fixture_noisy_action": _metric_text(metric, "yolo_fixture_noisy_action"),
        "yolo_fixture_noisy_would_gain": _metric_bool(metric, "yolo_fixture_noisy_would_gain"),
        "yolo_fixture_noisy_reason": _metric_text(metric, "yolo_fixture_noisy_reason"),
        "yolo_fixture_false_positive_status": _metric_text(metric, "yolo_fixture_false_positive_status"),
        "yolo_fixture_false_positive_action": _metric_text(metric, "yolo_fixture_false_positive_action"),
        "yolo_fixture_false_positive_would_be_dangerous": _metric_bool(
            metric,
            "yolo_fixture_false_positive_would_be_dangerous",
        ),
        "yolo_fixture_false_positive_reason": _metric_text(
            metric,
            "yolo_fixture_false_positive_reason",
        ),
        "yolo_gt_perfect_status": _metric_text(metric, "yolo_gt_perfect_status"),
        "yolo_gt_perfect_action": _metric_text(metric, "yolo_gt_perfect_action"),
        "yolo_gt_perfect_would_gain": _metric_bool(metric, "yolo_gt_perfect_would_gain"),
        "yolo_gt_perfect_reason": _metric_text(metric, "yolo_gt_perfect_reason"),
        "yolo_gt_jitter_status": _metric_text(metric, "yolo_gt_jitter_status"),
        "yolo_gt_jitter_action": _metric_text(metric, "yolo_gt_jitter_action"),
        "yolo_gt_jitter_would_gain": _metric_bool(metric, "yolo_gt_jitter_would_gain"),
        "yolo_gt_jitter_reason": _metric_text(metric, "yolo_gt_jitter_reason"),
        "yolo_gt_false_positive_status": _metric_text(metric, "yolo_gt_false_positive_status"),
        "yolo_gt_false_positive_action": _metric_text(metric, "yolo_gt_false_positive_action"),
        "yolo_gt_false_positive_would_be_dangerous": _metric_bool(
            metric,
            "yolo_gt_false_positive_would_be_dangerous",
        ),
        "yolo_gt_false_positive_reason": _metric_text(
            metric,
            "yolo_gt_false_positive_reason",
        ),
        "diagnostic_class": _metric_diagnostic_class(metric),
        "recoverability": _metric_recoverability(metric),
        "recommended_action": _metric_recommended_action(metric),
        "passed": _metric_bool(metric, "passed"),
        "unsafe_hidden": _metric_bool(metric, "unsafe_hidden"),
        "dangerous": _metric_bool(metric, "dangerous_projection"),
        "iou": _metric_float(metric, "iou"),
        "iou_bucket": _metric_iou_bucket(metric),
        "center_drift_px": _metric_float(metric, "center_drift_px"),
        "drift_bucket": _metric_drift_bucket(metric),
        "area_ratio": _metric_float(metric, "area_ratio"),
        "area_bucket": _metric_area_bucket(metric),
        "support": max(
            _metric_int(metric, "context_feature_support"),
            _metric_int(metric, "fallback_slot_feature_support"),
        ),
        "support_total": max(
            _metric_int(metric, "context_feature_total"),
            _metric_int(metric, "fallback_slot_feature_total"),
        ),
        "support_bucket": _metric_support_bucket(metric),
        "reference_keypoints_total": _metric_int(metric, "reference_keypoints_total"),
        "frame_keypoints_total": _metric_int(metric, "frame_keypoints_total"),
        "frame_max_keypoints": _metric_int(metric, "frame_max_keypoints"),
        "frame_keypoint_grid": (
            f"{_metric_int(metric, 'frame_keypoint_grid_rows')}x"
            f"{_metric_int(metric, 'frame_keypoint_grid_cols')}"
        ),
        "lightglue_reference_matches_total": _metric_int(
            metric,
            "lightglue_reference_matches_total",
        ),
        "lightglue_frame_matches_total": _metric_int(
            metric,
            "lightglue_frame_matches_total",
        ),
        "masked_alignment_used": _metric_bool(metric, "masked_alignment_used"),
        "original_reference_keypoints_total": _metric_int(
            metric,
            "original_reference_keypoints_total",
        ),
        "masked_reference_keypoints_total": _metric_int(
            metric,
            "masked_reference_keypoints_total",
        ),
        "masked_lightglue_matches_total": _metric_int(
            metric,
            "masked_lightglue_matches_total",
        ),
        "slot_local_lightglue_attempted": _metric_bool(
            metric,
            "slot_local_lightglue_attempted",
        ),
        "slot_local_lightglue_accepted": _metric_bool(
            metric,
            "slot_local_lightglue_accepted",
        ),
        "slot_local_lightglue_reject_reason": _metric_text(
            metric,
            "slot_local_lightglue_reject_reason",
        ),
        "slot_local_lightglue_mode": _metric_text(metric, "slot_local_lightglue_mode"),
        "slot_local_lightglue_keypoints": (
            f"{_metric_int(metric, 'slot_local_lightglue_reference_keypoints')}/"
            f"{_metric_int(metric, 'slot_local_lightglue_frame_keypoints')}"
        ),
        "slot_local_lightglue_matches": _metric_int(
            metric,
            "slot_local_lightglue_raw_matches",
        ),
        "slot_local_lightglue_inliers": _metric_int(
            metric,
            "slot_local_lightglue_inliers",
        ),
        "candidate_count": _metric_int(metric, "missing_candidate_count"),
        "inliers": _metric_int(metric, "missing_inliers"),
        "fallback_source": _metric_text(metric, "fallback_source"),
        "fallback_reason": _metric_text(metric, "fallback_reason"),
        "selective_hidden_release": _metric_bool(metric, "selective_hidden_release"),
        "selective_hidden_release_source": _metric_text(metric, "selective_hidden_release_source"),
        "selective_hidden_release_hidden_reason": _metric_text(metric, "selective_hidden_release_hidden_reason"),
        "projection_candidate_count": _metric_int(metric, "projection_candidate_count"),
        "projection_candidate_selected": _metric_text(metric, "projection_candidate_selected"),
        "candidate_agreement_level": _metric_text(metric, "candidate_agreement_level"),
        "candidate_agreement_count": _metric_int(metric, "candidate_agreement_count"),
        "candidate_confidence": _metric_text(metric, "candidate_confidence"),
        "candidate_recommended_action": _metric_text(metric, "candidate_recommended_action"),
        "candidate_oracle_failure_mode": _metric_text(metric, "candidate_oracle_failure_mode"),
        "candidate_oracle_best_source": _metric_text(metric, "candidate_oracle_best_source"),
        "candidate_oracle_best_iou": _metric_float(metric, "candidate_oracle_best_iou"),
        "candidate_oracle_best_drift_px": _metric_float(
            metric,
            "candidate_oracle_best_center_drift_px",
        ),
        "candidate_oracle_safe_gain_source": _metric_text(metric, "candidate_oracle_safe_gain_source"),
        "candidate_oracle_has_safe_alternative": _metric_bool(
            metric,
            "candidate_oracle_has_safe_alternative",
        ),
        "crop_verification_assessment": _metric_text(metric, "crop_verification_assessment"),
        "crop_verification_best_source": _metric_text(metric, "crop_verification_best_source"),
        "crop_verification_best_score": _metric_float(metric, "crop_verification_best_score"),
        "crop_verification_best_matches": _metric_int(metric, "crop_verification_best_object_matches"),
        "crop_verification_object_crop_source": _metric_text(
            metric,
            "crop_verification_object_crop_candidate_source",
        ),
        "crop_verification_object_crop_iou": _metric_float(
            metric,
            "crop_verification_object_crop_candidate_iou",
        ),
        "crop_verification_object_crop_would_pass": _metric_bool(
            metric,
            "crop_verification_object_crop_candidate_would_pass",
        ),
        "crop_verification_object_crop_dangerous": _metric_bool(
            metric,
            "crop_verification_object_crop_candidate_dangerous",
        ),
        "hidden_reason": _metric_text(metric, "hidden_reason"),
        "hidden_shadow_available": _metric_bool(metric, "hidden_shadow_available"),
        "hidden_shadow_projection": _metric_text(metric, "hidden_shadow_projection"),
        "hidden_shadow_iou": _metric_float(metric, "hidden_shadow_iou"),
        "hidden_shadow_drift_px": _metric_float(
            metric,
            "hidden_shadow_center_drift_px",
        ),
        "hidden_shadow_would_pass": _metric_bool(metric, "hidden_shadow_would_pass"),
        "hidden_shadow_dangerous": _metric_bool(metric, "hidden_shadow_dangerous"),
        "global_translation_state": _metric_global_translation_state(metric),
        "anchor_state": _metric_anchor_state(metric),
        "none_reason": _metric_text(metric, "none_reason"),
        "notes": _metric_notes_text(metric),
    }



def _object_feature_telemetry_stats(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_int(name: str) -> float:
        values = [
            _metric_int(metric, name)
            for metric in metrics
            if _metric_int(metric, name) > 0
        ]
        return float(np.mean(values)) if values else 0.0

    slot_attempted = [
        metric
        for metric in metrics
        if _metric_bool(metric, "slot_local_lightglue_attempted")
    ]
    slot_accepted = [
        metric
        for metric in metrics
        if _metric_bool(metric, "slot_local_lightglue_accepted")
    ]

    return {
        "objects": len(metrics),
        "mean_reference_keypoints_total": mean_int("reference_keypoints_total"),
        "mean_frame_keypoints_total": mean_int("frame_keypoints_total"),
        "mean_frame_max_keypoints": mean_int("frame_max_keypoints"),
        "mean_lightglue_reference_matches_total": mean_int(
            "lightglue_reference_matches_total"
        ),
        "mean_lightglue_frame_matches_total": mean_int(
            "lightglue_frame_matches_total"
        ),
        "masked_alignment_used": sum(
            1 for metric in metrics if _metric_bool(metric, "masked_alignment_used")
        ),
        "mean_original_reference_keypoints_total": mean_int(
            "original_reference_keypoints_total"
        ),
        "mean_masked_reference_keypoints_total": mean_int(
            "masked_reference_keypoints_total"
        ),
        "mean_masked_lightglue_matches_total": mean_int(
            "masked_lightglue_matches_total"
        ),
        "slot_local_attempted": len(slot_attempted),
        "slot_local_accepted": len(slot_accepted),
        "slot_local_accept_rate": (
            len(slot_accepted) / len(slot_attempted) * 100.0
            if slot_attempted
            else 0.0
        ),
        "mean_slot_local_reference_keypoints": mean_int(
            "slot_local_lightglue_reference_keypoints"
        ),
        "mean_slot_local_frame_keypoints": mean_int(
            "slot_local_lightglue_frame_keypoints"
        ),
        "mean_slot_local_raw_matches": mean_int("slot_local_lightglue_raw_matches"),
        "mean_slot_local_inliers": mean_int("slot_local_lightglue_inliers"),
    }


def _object_result_policy_stats(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        buckets.setdefault(_metric_text(metric, "result_policy_status"), []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for bucket, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        action_counts = _metric_count_values(bucket_metrics, "result_policy_action")
        label_counts = _metric_count_values(bucket_metrics, "result_policy_user_label")
        reason_counts = _metric_count_values(bucket_metrics, "result_policy_reason")
        projection_counts = _metric_count_values(bucket_metrics, "projection")
        shape_counts = _metric_count_values(bucket_metrics, "object_shape")
        confidence_counts = _metric_count_values(bucket_metrics, "result_policy_confidence")
        passed = sum(1 for metric in bucket_metrics if _metric_bool(metric, "passed"))
        stats[bucket] = {
            "total": len(bucket_metrics),
            "passed": passed,
            "failed": len(bucket_metrics) - passed,
            "unsafe_hidden": sum(1 for metric in bucket_metrics if _metric_bool(metric, "unsafe_hidden")),
            "dangerous": sum(1 for metric in bucket_metrics if _metric_bool(metric, "dangerous_projection")),
            "top_confidence": _top_bucket_name(confidence_counts),
            "top_action": _top_bucket_name(action_counts),
            "top_user_label": _top_bucket_name(label_counts),
            "top_reason": _top_bucket_name(reason_counts),
            "top_projection": _top_bucket_name(projection_counts),
            "top_shape": _top_bucket_name(shape_counts),
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(bucket_metrics, "center_drift_px"),
        }
    return stats


def _object_yolo_synthetic_feasibility_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        buckets.setdefault(_metric_text(metric, "yolo_synthetic_test_bucket"), []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for bucket, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        role_counts = _metric_count_values(bucket_metrics, "yolo_synthetic_expected_role")
        policy_counts = _metric_count_values(bucket_metrics, "result_policy_status")
        projection_counts = _metric_count_values(bucket_metrics, "projection")
        shape_counts = _metric_count_values(bucket_metrics, "object_shape")
        passed = sum(1 for metric in bucket_metrics if _metric_bool(metric, "passed"))
        stats[bucket] = {
            "total": len(bucket_metrics),
            "passed": passed,
            "failed": len(bucket_metrics) - passed,
            "unsafe_hidden": sum(1 for metric in bucket_metrics if _metric_bool(metric, "unsafe_hidden")),
            "dangerous": sum(1 for metric in bucket_metrics if _metric_bool(metric, "dangerous_projection")),
            "top_expected_role": _top_bucket_name(role_counts),
            "top_policy": _top_bucket_name(policy_counts),
            "top_projection": _top_bucket_name(projection_counts),
            "top_shape": _top_bucket_name(shape_counts),
            "fixture": _metric_text(bucket_metrics[0], "yolo_synthetic_fixture"),
            "limitation": _metric_text(bucket_metrics[0], "yolo_synthetic_limitation"),
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(bucket_metrics, "center_drift_px"),
        }
    return stats


def _group_metrics_by_key(
    metrics: list[dict[str, Any]],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        buckets.setdefault(_metric_text(metric, key, "unknown"), []).append(metric)
    return dict(
        sorted(
            buckets.items(),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
    )


def _object_yolo_synthetic_fixture_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return _object_detector_fixture_stats(
        metrics,
        profiles={
            "oracle": "Perfect detector oracle",
            "noisy": "Noisy detector fixture",
            "false_positive": "False-positive stress fixture",
        },
        prefix_root="yolo_fixture",
    )


def _object_yolo_gt_detector_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return _object_detector_fixture_stats(
        metrics,
        profiles={
            "perfect": "Perfect GT detector mask",
            "jitter": "Jittered GT detector mask",
            "false_positive": "Synthetic distractor detector",
        },
        prefix_root="yolo_gt",
    )


def _object_detector_fixture_stats(
    metrics: list[dict[str, Any]],
    *,
    profiles: dict[str, str],
    prefix_root: str,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    failed_metrics = [metric for metric in metrics if not _metric_bool(metric, "passed")]
    grouped_by_bucket = _group_metrics_by_key(metrics, "yolo_synthetic_test_bucket")
    for profile, title in profiles.items():
        prefix = f"{prefix_root}_{profile}"
        status_counts = _metric_count_values(metrics, f"{prefix}_status")
        action_counts = _metric_count_values(metrics, f"{prefix}_action")
        reason_counts = _metric_count_values(metrics, f"{prefix}_reason")
        source_counts = _metric_count_values(metrics, f"{prefix}_candidate_source")
        gains = [metric for metric in failed_metrics if _metric_bool(metric, f"{prefix}_would_gain")]
        dangerous = [
            metric
            for metric in metrics
            if _metric_bool(metric, f"{prefix}_would_be_dangerous")
        ]
        by_bucket: dict[str, dict[str, Any]] = {}
        for bucket, bucket_metrics in grouped_by_bucket.items():
            bucket_failed = [metric for metric in bucket_metrics if not _metric_bool(metric, "passed")]
            bucket_gains = [
                metric
                for metric in bucket_failed
                if _metric_bool(metric, f"{prefix}_would_gain")
            ]
            bucket_dangerous = [
                metric
                for metric in bucket_metrics
                if _metric_bool(metric, f"{prefix}_would_be_dangerous")
            ]
            bucket_status_counts = _metric_count_values(bucket_metrics, f"{prefix}_status")
            bucket_reason_counts = _metric_count_values(bucket_metrics, f"{prefix}_reason")
            bucket_source_counts = _metric_count_values(bucket_metrics, f"{prefix}_candidate_source")
            by_bucket[bucket] = {
                "total": len(bucket_metrics),
                "baseline_failed": len(bucket_failed),
                "would_rescue": len(bucket_gains),
                "remaining_failed": len(bucket_failed) - len(bucket_gains),
                "dangerous_if_trusted": len(bucket_dangerous),
                "top_status": _top_bucket_name(bucket_status_counts),
                "top_reason": _top_bucket_name(bucket_reason_counts),
                "top_candidate_source": _top_bucket_name(bucket_source_counts),
            }
        stats[profile] = {
            "title": title,
            "total": len(metrics),
            "baseline_failed": len(failed_metrics),
            "would_rescue": len(gains),
            "remaining_failed": len(failed_metrics) - len(gains),
            "dangerous_if_trusted": len(dangerous),
            "top_status": _top_bucket_name(status_counts),
            "top_action": _top_bucket_name(action_counts),
            "top_reason": _top_bucket_name(reason_counts),
            "top_candidate_source": _top_bucket_name(source_counts),
            "by_bucket": by_bucket,
        }
    return stats

def _object_failure_microscope(
    metrics: list[dict[str, Any]],
    *,
    max_rows: int = 160,
) -> list[dict[str, Any]]:
    rows = [_metric_diagnostic_row(metric) for metric in metrics if not _metric_bool(metric, "passed")]
    rows.sort(
        key=lambda row: (
            row["recoverability"] != "high_safe_gain",
            not row["dangerous"],
            row["recoverability"],
            -float(row["iou"]),
            float(row["center_drift_px"]),
            row["case_index"],
        )
    )
    return rows[:max_rows]


def _bucket_counter_as_top(counts: dict[str, int], *, limit: int = 5) -> list[str]:
    return [
        f"{name} ({count})"
        for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def _object_deep_failure_stats(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if _metric_bool(metric, "passed"):
            continue
        buckets.setdefault(_metric_diagnostic_class(metric), []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for bucket, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        projection_counts = _metric_count_values(bucket_metrics, "projection")
        shape_counts = _metric_count_values(bucket_metrics, "object_shape")
        hidden_reason_counts = _metric_count_values(bucket_metrics, "hidden_reason")
        action_counts: dict[str, int] = {}
        iou_bucket_counts: dict[str, int] = {}
        drift_bucket_counts: dict[str, int] = {}
        support_bucket_counts: dict[str, int] = {}
        for metric in bucket_metrics:
            action = _metric_recommended_action(metric)
            action_counts[action] = action_counts.get(action, 0) + 1
            iou_bucket = _metric_iou_bucket(metric)
            drift_bucket = _metric_drift_bucket(metric)
            support_bucket = _metric_support_bucket(metric)
            iou_bucket_counts[iou_bucket] = iou_bucket_counts.get(iou_bucket, 0) + 1
            drift_bucket_counts[drift_bucket] = drift_bucket_counts.get(drift_bucket, 0) + 1
            support_bucket_counts[support_bucket] = support_bucket_counts.get(support_bucket, 0) + 1
        safe_gain = sum(
            1
            for metric in bucket_metrics
            if _metric_bool(metric, "hidden_shadow_would_pass")
            and not _metric_bool(metric, "hidden_shadow_dangerous")
        )
        danger_risk = sum(
            1
            for metric in bucket_metrics
            if _metric_bool(metric, "dangerous_projection")
            or _metric_bool(metric, "hidden_shadow_dangerous")
        )
        stats[bucket] = {
            "total": len(bucket_metrics),
            "safe_gain_candidates": safe_gain,
            "danger_risk_candidates": danger_risk,
            "top_projection": _top_bucket_name(projection_counts),
            "top_shape": _top_bucket_name(shape_counts),
            "top_hidden_reason": _top_bucket_name(hidden_reason_counts),
            "top_action": _top_bucket_name(action_counts),
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(bucket_metrics, "center_drift_px"),
            "mean_support": float(
                np.mean(
                    [
                        max(
                            _metric_int(metric, "context_feature_support"),
                            _metric_int(metric, "fallback_slot_feature_support"),
                        )
                        for metric in bucket_metrics
                    ]
                )
            ) if bucket_metrics else 0.0,
            "iou_buckets": dict(
                sorted(iou_bucket_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "drift_buckets": dict(
                sorted(drift_bucket_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "support_buckets": dict(
                sorted(support_bucket_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "top_iou_buckets": _bucket_counter_as_top(iou_bucket_counts),
            "top_drift_buckets": _bucket_counter_as_top(drift_bucket_counts),
            "top_support_buckets": _bucket_counter_as_top(support_bucket_counts),
        }
    return stats


def _object_recovery_opportunity_stats(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if _metric_bool(metric, "passed"):
            continue
        buckets.setdefault(_metric_recoverability(metric), []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for bucket, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        diagnostic_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        for metric in bucket_metrics:
            diagnostic = _metric_diagnostic_class(metric)
            diagnostic_counts[diagnostic] = diagnostic_counts.get(diagnostic, 0) + 1
            action = _metric_recommended_action(metric)
            action_counts[action] = action_counts.get(action, 0) + 1
        safe_gain = sum(
            1
            for metric in bucket_metrics
            if _metric_bool(metric, "hidden_shadow_would_pass")
            and not _metric_bool(metric, "hidden_shadow_dangerous")
        )
        danger_risk = sum(
            1
            for metric in bucket_metrics
            if _metric_bool(metric, "dangerous_projection")
            or _metric_bool(metric, "hidden_shadow_dangerous")
        )
        stats[bucket] = {
            "total": len(bucket_metrics),
            "safe_gain_candidates": safe_gain,
            "danger_risk_candidates": danger_risk,
            "top_diagnostic": _top_bucket_name(diagnostic_counts),
            "top_action": _top_bucket_name(action_counts),
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(bucket_metrics, "center_drift_px"),
        }
    return stats


def _object_method_coverage_stats(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        buckets.setdefault(_metric_text(metric, "projection"), []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for projection, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        passed = sum(1 for metric in bucket_metrics if _metric_bool(metric, "passed"))
        failed = len(bucket_metrics) - passed
        hidden = sum(1 for metric in bucket_metrics if _metric_bool(metric, "unsafe_hidden"))
        dangerous = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "dangerous_projection")
        )
        shadow_available = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "hidden_shadow_available")
        )
        shadow_safe_pass = sum(
            1
            for metric in bucket_metrics
            if _metric_bool(metric, "hidden_shadow_would_pass")
            and not _metric_bool(metric, "hidden_shadow_dangerous")
        )
        shadow_danger = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "hidden_shadow_dangerous")
        )
        global_attempted = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "global_translation_rescue_attempted")
        )
        anchor_attempted = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "anchor_release_attempted")
        )
        failure_class_counts: dict[str, int] = {}
        global_reject_counts: dict[str, int] = {}
        anchor_reject_counts: dict[str, int] = {}
        fallback_source_counts: dict[str, int] = {}
        for metric in bucket_metrics:
            if not _metric_bool(metric, "passed"):
                failure_class = _metric_diagnostic_class(metric)
                failure_class_counts[failure_class] = failure_class_counts.get(failure_class, 0) + 1
            global_state = _metric_global_translation_state(metric)
            if global_state != "not_attempted":
                global_reject_counts[global_state] = global_reject_counts.get(global_state, 0) + 1
            anchor_state = _metric_anchor_state(metric)
            if anchor_state != "not_attempted":
                anchor_reject_counts[anchor_state] = anchor_reject_counts.get(anchor_state, 0) + 1
            fallback_source = _metric_text(metric, "fallback_source")
            if fallback_source != "—":
                fallback_source_counts[fallback_source] = fallback_source_counts.get(fallback_source, 0) + 1
        stats[projection] = {
            "total": len(bucket_metrics),
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / len(bucket_metrics) * 100.0) if bucket_metrics else 0.0,
            "hidden": hidden,
            "dangerous": dangerous,
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(bucket_metrics, "center_drift_px"),
            "shadow_available": shadow_available,
            "shadow_safe_pass": shadow_safe_pass,
            "shadow_dangerous": shadow_danger,
            "global_translation_attempted": global_attempted,
            "anchor_release_attempted": anchor_attempted,
            "top_failure_class": _top_bucket_name(failure_class_counts),
            "top_global_translation_state": _top_bucket_name(global_reject_counts),
            "top_anchor_state": _top_bucket_name(anchor_reject_counts),
            "top_fallback_source": _top_bucket_name(fallback_source_counts),
        }
    return stats


def _object_selective_hidden_release_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if _metric_bool(metric, "selective_hidden_release"):
            reason = _metric_text(
                metric,
                "selective_hidden_release_hidden_reason",
                "released_unknown_reason",
            )
            buckets.setdefault(reason, []).append(metric)
        elif _metric_bool(metric, "selective_hidden_release_rejected"):
            reason = _metric_text(
                metric,
                "selective_hidden_release_reject_reason",
                "release_rejected_unknown_reason",
            )
            buckets.setdefault(f"rejected:{reason}", []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for reason, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        passed = sum(1 for metric in bucket_metrics if _metric_bool(metric, "passed"))
        dangerous = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "dangerous_projection")
        )
        stats[reason] = {
            "total": len(bucket_metrics),
            "passed": passed,
            "failed": len(bucket_metrics) - passed,
            "pass_rate": (passed / len(bucket_metrics) * 100.0) if bucket_metrics else 0.0,
            "unsafe_hidden": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "unsafe_hidden")
            ),
            "dangerous": dangerous,
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(bucket_metrics, "center_drift_px"),
            "top_shape": _top_bucket_name(_metric_count_values(bucket_metrics, "object_shape")),
            "top_projection": _top_bucket_name(_metric_count_values(bucket_metrics, "projection")),
            "mean_slot_support": _mean_metric_value(
                bucket_metrics,
                "selective_hidden_release_slot_support",
            ),
            "mean_max_other_overlap": _mean_metric_value(
                bucket_metrics,
                "selective_hidden_release_max_other_overlap",
            ),
        }
    return stats


def _object_candidate_agreement_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        confidence = _metric_text(metric, "candidate_confidence")
        level = _metric_text(metric, "candidate_agreement_level")
        if confidence == "—" and level == "—":
            continue
        key = f"{confidence} / {level}"
        buckets.setdefault(key, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for key, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        passed = sum(1 for metric in bucket_metrics if _metric_bool(metric, "passed"))
        dangerous = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "dangerous_projection")
        )
        stats[key] = {
            "total": len(bucket_metrics),
            "passed": passed,
            "failed": len(bucket_metrics) - passed,
            "pass_rate": (passed / len(bucket_metrics) * 100.0) if bucket_metrics else 0.0,
            "unsafe_hidden": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "unsafe_hidden")
            ),
            "dangerous": dangerous,
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(
                bucket_metrics,
                "center_drift_px",
            ),
            "mean_agreement_count": _mean_metric_value(
                bucket_metrics,
                "candidate_agreement_count",
            ),
            "mean_best_iou": _mean_metric_value(
                bucket_metrics,
                "candidate_agreement_best_iou",
            ),
            "mean_min_center_factor": _mean_metric_value(
                bucket_metrics,
                "candidate_agreement_min_center_factor",
            ),
            "top_projection": _top_bucket_name(
                _metric_count_values(bucket_metrics, "projection")
            ),
            "top_action": _top_bucket_name(
                _metric_count_values(bucket_metrics, "candidate_recommended_action")
            ),
        }
    return stats


def _count_metric_diagnostic_classes(metrics: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in metrics:
        diagnostic = _metric_diagnostic_class(metric)
        counts[diagnostic] = counts.get(diagnostic, 0) + 1
    return counts


def _object_candidate_oracle_stats(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        mode = _metric_text(metric, "candidate_oracle_failure_mode")
        if mode in {"—", "current_passed"}:
            continue
        buckets.setdefault(mode, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for mode, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        stats[mode] = {
            "total": len(bucket_metrics),
            "safe_alternative_objects": sum(
                1
                for metric in bucket_metrics
                if _metric_bool(metric, "candidate_oracle_has_safe_alternative")
            ),
            "dangerous_candidate_count": sum(
                _metric_int(metric, "candidate_oracle_dangerous_count")
                for metric in bucket_metrics
            ),
            "top_projection": _top_bucket_name(
                _metric_count_values(bucket_metrics, "projection")
            ),
            "top_failure_class": _top_bucket_name(
                _count_metric_diagnostic_classes(bucket_metrics)
            ),
            "top_best_source": _top_bucket_name(
                _metric_count_values(bucket_metrics, "candidate_oracle_best_source")
            ),
            "top_safe_gain_source": _top_bucket_name(
                _metric_count_values(bucket_metrics, "candidate_oracle_safe_gain_source")
            ),
            "mean_available_candidates": _mean_metric_value(
                bucket_metrics,
                "candidate_oracle_available_count",
            ),
            "mean_safe_candidates": _mean_metric_value(
                bucket_metrics,
                "candidate_oracle_safe_pass_count",
            ),
            "mean_best_iou": _mean_metric_value(
                bucket_metrics,
                "candidate_oracle_best_iou",
            ),
            "mean_current_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_best_drift_px": _mean_metric_value(
                bucket_metrics,
                "candidate_oracle_best_center_drift_px",
            ),
            "mean_current_drift_px": _mean_metric_value(
                bucket_metrics,
                "center_drift_px",
            ),
        }
    return stats


def _object_candidate_oracle_matrix(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if _metric_bool(metric, "passed"):
            continue
        best_source = _metric_text(metric, "candidate_oracle_best_source")
        if best_source == "—":
            best_source = "no_candidate"
        key = f"{_metric_text(metric, 'projection')} -> {best_source}"
        buckets.setdefault(key, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for key, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        stats[key] = {
            "total": len(bucket_metrics),
            "safe_alternative_objects": sum(
                1
                for metric in bucket_metrics
                if _metric_bool(metric, "candidate_oracle_has_safe_alternative")
            ),
            "selected_would_pass": sum(
                1
                for metric in bucket_metrics
                if _metric_bool(metric, "candidate_oracle_selected_would_pass")
            ),
            "dangerous_candidate_count": sum(
                _metric_int(metric, "candidate_oracle_dangerous_count")
                for metric in bucket_metrics
            ),
            "top_failure_class": _top_bucket_name(
                _count_metric_diagnostic_classes(bucket_metrics)
            ),
            "top_shape": _top_bucket_name(
                _metric_count_values(bucket_metrics, "object_shape")
            ),
            "top_safe_gain_source": _top_bucket_name(
                _metric_count_values(bucket_metrics, "candidate_oracle_safe_gain_source")
            ),
            "mean_current_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_best_iou": _mean_metric_value(
                bucket_metrics,
                "candidate_oracle_best_iou",
            ),
            "mean_current_drift_px": _mean_metric_value(
                bucket_metrics,
                "center_drift_px",
            ),
            "mean_best_drift_px": _mean_metric_value(
                bucket_metrics,
                "candidate_oracle_best_center_drift_px",
            ),
        }
    return stats



def _object_crop_verification_stats(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        key = _metric_text(metric, "crop_verification_assessment")
        if key == "—":
            key = "not_available"
        buckets.setdefault(key, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for key, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        failed = [metric for metric in bucket_metrics if not _metric_bool(metric, "passed")]
        stats[key] = {
            "total": len(bucket_metrics),
            "failed": len(failed),
            "current_passed": len(bucket_metrics) - len(failed),
            "hidden": sum(1 for metric in bucket_metrics if _metric_bool(metric, "unsafe_hidden")),
            "dangerous_current": sum(1 for metric in bucket_metrics if _metric_bool(metric, "dangerous_projection")),
            "object_crop_candidate_available": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_object_crop_candidate_available")
            ),
            "object_crop_candidate_would_pass": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_object_crop_candidate_would_pass")
            ),
            "object_crop_candidate_dangerous": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_object_crop_candidate_dangerous")
            ),
            "best_would_pass": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_best_would_pass")
            ),
            "best_dangerous": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_best_dangerous")
            ),
            "top_projection": _top_bucket_name(_metric_count_values(bucket_metrics, "projection")),
            "top_failure_class": _top_bucket_name(_count_metric_diagnostic_classes(failed)),
            "top_best_source": _top_bucket_name(_metric_count_values(bucket_metrics, "crop_verification_best_source")),
            "top_object_crop_source": _top_bucket_name(
                _metric_count_values(bucket_metrics, "crop_verification_object_crop_candidate_source")
            ),
            "mean_best_score": _mean_metric_value(bucket_metrics, "crop_verification_best_score"),
            "mean_best_matches": _mean_metric_value(bucket_metrics, "crop_verification_best_object_matches"),
            "mean_best_iou": _mean_metric_value(bucket_metrics, "crop_verification_best_iou"),
            "mean_object_crop_iou": _mean_metric_value(
                bucket_metrics,
                "crop_verification_object_crop_candidate_iou",
            ),
        }
    return stats


def _object_crop_verification_matrix(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        if _metric_bool(metric, "passed"):
            continue
        assessment = _metric_text(metric, "crop_verification_assessment")
        source = _metric_text(metric, "crop_verification_object_crop_candidate_source")
        key = f"{_metric_text(metric, 'projection')} -> {assessment} / {source}"
        buckets.setdefault(key, []).append(metric)

    stats: dict[str, dict[str, Any]] = {}
    for key, bucket_metrics in sorted(
        buckets.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        stats[key] = {
            "total": len(bucket_metrics),
            "object_crop_candidate_available": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_object_crop_candidate_available")
            ),
            "object_crop_candidate_would_pass": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_object_crop_candidate_would_pass")
            ),
            "object_crop_candidate_dangerous": sum(
                1 for metric in bucket_metrics if _metric_bool(metric, "crop_verification_object_crop_candidate_dangerous")
            ),
            "top_failure_class": _top_bucket_name(_count_metric_diagnostic_classes(bucket_metrics)),
            "top_shape": _top_bucket_name(_metric_count_values(bucket_metrics, "object_shape")),
            "top_oracle_mode": _top_bucket_name(
                _metric_count_values(bucket_metrics, "candidate_oracle_failure_mode")
            ),
            "mean_current_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_best_score": _mean_metric_value(bucket_metrics, "crop_verification_best_score"),
            "mean_best_matches": _mean_metric_value(bucket_metrics, "crop_verification_best_object_matches"),
            "mean_object_crop_iou": _mean_metric_value(
                bucket_metrics,
                "crop_verification_object_crop_candidate_iou",
            ),
        }
    return stats

def _object_candidate_overlap_stats(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, list[dict[str, Any]]] = {}

    def add_row(kind: str, source: str, target: str, metric: dict[str, Any]) -> None:
        key = f"{kind}: {source} -> {target}"
        rows.setdefault(key, []).append(metric)

    for metric in metrics:
        projection = _metric_text(metric, "projection")
        if _metric_bool(metric, "hidden_shadow_available"):
            add_row(
                "hidden_shadow",
                projection,
                _metric_text(metric, "hidden_shadow_projection"),
                metric,
            )
        fallback_source = _metric_text(metric, "fallback_source")
        if fallback_source != "—":
            add_row("fallback_source", fallback_source, projection, metric)
        global_state = _metric_global_translation_state(metric)
        if global_state != "not_attempted":
            add_row("global_translation", projection, global_state, metric)
        anchor_state = _metric_anchor_state(metric)
        if anchor_state != "not_attempted":
            add_row("anchor_release", projection, anchor_state, metric)
        if _metric_bool(metric, "selective_hidden_release"):
            add_row(
                "selective_hidden_release",
                _metric_text(metric, "selective_hidden_release_source"),
                projection,
                metric,
            )
        selected_candidate = _metric_text(metric, "projection_candidate_selected")
        if selected_candidate != "—":
            add_row("candidate_registry_selected", selected_candidate, projection, metric)

    stats: dict[str, dict[str, Any]] = {}
    for key, bucket_metrics in sorted(
        rows.items(),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ):
        passed = sum(1 for metric in bucket_metrics if _metric_bool(metric, "passed"))
        hidden_shadow_safe = sum(
            1
            for metric in bucket_metrics
            if _metric_bool(metric, "hidden_shadow_would_pass")
            and not _metric_bool(metric, "hidden_shadow_dangerous")
        )
        hidden_shadow_danger = sum(
            1 for metric in bucket_metrics if _metric_bool(metric, "hidden_shadow_dangerous")
        )
        stats[key] = {
            "total": len(bucket_metrics),
            "passed": passed,
            "failed": len(bucket_metrics) - passed,
            "current_pass_rate": (passed / len(bucket_metrics) * 100.0) if bucket_metrics else 0.0,
            "shadow_safe_gain": hidden_shadow_safe,
            "shadow_danger_risk": hidden_shadow_danger,
            "top_shape": _top_bucket_name(_metric_count_values(bucket_metrics, "object_shape")),
            "top_hidden_reason": _top_bucket_name(
                _metric_count_values(bucket_metrics, "hidden_reason")
            ),
            "mean_iou": _mean_metric_value(bucket_metrics, "iou"),
            "mean_center_drift_px": _mean_metric_value(bucket_metrics, "center_drift_px"),
        }
    return stats

def _mean_metric_value(metrics: list[dict[str, Any]], key: str) -> float:
    values = _finite_metric_values(metrics, key)
    return float(np.mean(values)) if values else 0.0


def _top_bucket_name(counts: dict[str, int]) -> str:
    if not counts:
        return "—"
    name, value = max(counts.items(), key=lambda item: item[1])
    return f"{name} ({value})"


def _build_summary(results: list[SyntheticResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed
    safety_passed = sum(1 for result in results if result.safety_passed)
    dangerous = sum(1 for result in results if result.dangerous_projection)
    unsafe_hidden = sum(1 for result in results if result.unsafe_hidden)
    single_cases = sum(1 for result in results if result.scenario_kind == "single")
    multi_cases = sum(1 for result in results if result.scenario_kind == "multi")

    object_metrics = [
        _strip_report_only_object_metric_fields(metric)
        for metric in _object_metric_dicts(results)
    ]
    total_expected_objects = len(object_metrics)
    object_failures = sum(1 for metric in object_metrics if not bool(metric.get("passed")))
    object_safety_passed = sum(
        1 for metric in object_metrics if bool(metric.get("safety_passed"))
    )
    object_dangerous = sum(
        1 for metric in object_metrics if bool(metric.get("dangerous_projection"))
    )
    object_unsafe_hidden = sum(
        1 for metric in object_metrics if bool(metric.get("unsafe_hidden"))
    )

    ious = [result.iou for result in results if math.isfinite(result.iou)]
    drifts = [
        result.center_drift_px
        for result in results
        if math.isfinite(result.center_drift_px)
    ]
    object_ious = _finite_metric_values(object_metrics, "iou")
    object_drifts = _finite_metric_values(object_metrics, "center_drift_px")

    shape_names: set[str] = set()
    for result in results:
        if result.object_shapes:
            shape_names.update(result.object_shapes)
        elif result.object_shape != "multi":
            shape_names.add(result.object_shape)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total * 100.0) if total else 0.0,
        "safety_passed": safety_passed,
        "dangerous_projection_count": dangerous,
        "unsafe_hidden_count": unsafe_hidden,
        "safety_rate": (safety_passed / total * 100.0) if total else 0.0,
        "single_cases": single_cases,
        "multi_cases": multi_cases,
        "total_expected_objects": total_expected_objects,
        "object_failures": object_failures,
        "object_accuracy_rate": (
            (total_expected_objects - object_failures) / total_expected_objects * 100.0
            if total_expected_objects
            else 0.0
        ),
        "object_safety_passed": object_safety_passed,
        "object_safety_rate": (
            object_safety_passed / total_expected_objects * 100.0
            if total_expected_objects
            else 0.0
        ),
        "object_dangerous_projection_count": object_dangerous,
        "object_unsafe_hidden_count": object_unsafe_hidden,
        "object_mean_iou": float(np.mean(object_ious)) if object_ious else 0.0,
        "object_mean_center_drift_px": (
            float(np.mean(object_drifts)) if object_drifts else 0.0
        ),
        "object_projection_stats": _bucket_object_metric_stats(
            object_metrics,
            "projection",
        ),
        "object_shape_stats": _bucket_object_metric_stats(
            object_metrics,
            "object_shape",
        ),
        "object_failure_reason_stats": _object_failure_reason_stats(object_metrics),
        "object_hidden_reason_stats": _object_hidden_reason_stats(object_metrics),
        "object_anchor_release_reject_stats": _object_anchor_release_reject_stats(object_metrics),
        "object_anchor_build_reject_stats": _object_anchor_build_reject_stats(object_metrics),
        "object_anchor_order_stats": _object_anchor_order_stats(object_metrics),
        "object_none_projection_stats": _object_none_projection_stats(object_metrics),
        "object_global_fallback_diagnostics": _object_global_fallback_diagnostics(
            object_metrics
        ),
        "object_global_translation_rescue_diagnostics": _object_global_translation_rescue_diagnostics(
            object_metrics
        ),
        "object_unsafe_hidden_shadow_stats": _object_unsafe_hidden_shadow_stats(
            object_metrics
        ),
        "object_deep_failure_stats": _object_deep_failure_stats(object_metrics),
        "object_recovery_opportunity_stats": _object_recovery_opportunity_stats(
            object_metrics
        ),
        "object_method_coverage_stats": _object_method_coverage_stats(object_metrics),
        "object_selective_hidden_release_stats": _object_selective_hidden_release_stats(
            object_metrics
        ),
        "object_candidate_agreement_stats": _object_candidate_agreement_stats(
            object_metrics
        ),
        "object_candidate_overlap_stats": _object_candidate_overlap_stats(object_metrics),
        "object_result_policy_stats": _object_result_policy_stats(object_metrics),
        "object_feature_telemetry_stats": _object_feature_telemetry_stats(
            object_metrics
        ),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "mean_center_drift_px": float(np.mean(drifts)) if drifts else 0.0,
        "max_center_drift_px": float(np.max(drifts)) if drifts else 0.0,
        "edge_refinement_enabled": False,
        "context_expansion": float(_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXPANSION),
        "context_exclusion_margin": float(_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXCLUSION_MARGIN),
        "min_feature_support": int(_SYNTHETIC_MISSING_POLYGON_MIN_FEATURE_SUPPORT),
        "shape_names": sorted(shape_names),
    }


def _html_stats_table(
    title: str,
    stats: dict[str, dict[str, Any]],
) -> str:
    if not stats:
        return ""

    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{int(item.get('total', 0))}</td>"
            f"<td>{int(item.get('passed', 0))}</td>"
            f"<td>{float(item.get('pass_rate', 0.0)):.1f}%</td>"
            f"<td>{int(item.get('unsafe_hidden', 0))}</td>"
            f"<td>{int(item.get('dangerous', 0))}</td>"
            f"<td>{float(item.get('mean_iou', 0.0)):.3f}</td>"
            f"<td>{float(item.get('mean_center_drift_px', 0.0)):.1f}px</td>"
            "</tr>"
        )

    return f"""
      <h2>{html.escape(title)}</h2>
      <table>
        <tr>
          <th>bucket</th>
          <th>objects</th>
          <th>passed</th>
          <th>pass rate</th>
          <th>hidden</th>
          <th>dangerous</th>
          <th>mean IoU</th>
          <th>mean drift</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_anchor_build_stats_table(
    title: str,
    stats: dict[str, dict[str, Any]],
) -> str:
    if not stats:
        return ""
    rows = [
        "<tr><th>reason</th><th>total</th><th>affected objects</th><th>top built source</th><th>top probe</th></tr>"
    ]
    for reason, item in stats.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(reason))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('affected_objects') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_built_source') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_probe') or '—'))}</td>"
            "</tr>"
        )
    return f"""
      <h2>{html.escape(title)}</h2>
      <table>
        {''.join(rows)}
      </table>
    """


def _html_anchor_order_stats_table(
    title: str,
    stats: dict[str, dict[str, Any]],
) -> str:
    if not stats:
        return ""
    rows = [
        "<tr><th>bucket</th><th>total</th><th>runtime anchors</th><th>final anchors</th><th>future anchors</th><th>top runtime</th><th>top final</th></tr>"
    ]
    for name, item in stats.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('runtime_has_any_anchor') or 0)}</td>"
            f"<td>{int(item.get('final_has_any_anchor') or 0)}</td>"
            f"<td>{int(item.get('final_has_future_anchor') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_runtime_source') or item.get('top_runtime_probe') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_final_source') or item.get('top_final_resolved_source') or '—'))}</td>"
            "</tr>"
        )
    return f"""
      <h2>{html.escape(title)}</h2>
      <table>
        {''.join(rows)}
      </table>
    """


def _html_reason_stats_table(
    title: str,
    stats: dict[str, dict[str, Any]],
) -> str:
    if not stats:
        return ""

    rows = []
    for reason, item in list(stats.items())[:16]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{int(item.get('total', 0))}</td>"
            f"<td>{html.escape(str(item.get('top_projection', '—')))}</td>"
            f"<td>{html.escape(str(item.get('top_shape', '—')))}</td>"
            "</tr>"
        )

    return f"""
      <h2>{html.escape(title)}</h2>
      <table>
        <tr>
          <th>reason</th>
          <th>objects</th>
          <th>top projection</th>
          <th>top shape</th>
        </tr>
        {''.join(rows)}
      </table>
    """




def _html_global_translation_rescue_diagnostics_table(
    title: str,
    stats: dict[str, dict[str, Any]],
) -> str:
    if not stats:
        return ""

    rows = []
    for reason, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(reason))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('accepted') or 0)}</td>"
            f"<td>{int(item.get('passed') or 0)}</td>"
            f"<td>{float(item.get('pass_rate') or 0.0):.1f}%</td>"
            f"<td>{html.escape(str(item.get('top_projection') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_shape') or '—'))}</td>"
            f"<td>{float(item.get('mean_local_points') or 0.0):.1f}</td>"
            f"<td>{float(item.get('mean_candidates') or 0.0):.1f}</td>"
            f"<td>{float(item.get('mean_inliers') or 0.0):.1f}</td>"
            f"<td>{float(item.get('mean_inlier_ratio') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_median_error') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_shift_factor') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_local_area_score') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_local_center_factor') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_slot_support') or 0.0):.1f}</td>"
            "</tr>"
        )

    return f"""
      <h2>{html.escape(title)}</h2>
      <table>
        <tr>
          <th>reason</th>
          <th>objects</th>
          <th>accepted</th>
          <th>passed</th>
          <th>pass rate</th>
          <th>top projection</th>
          <th>top shape</th>
          <th>local pts</th>
          <th>candidates</th>
          <th>inliers</th>
          <th>ratio</th>
          <th>median err</th>
          <th>shift</th>
          <th>area</th>
          <th>center</th>
          <th>slot support</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_unsafe_hidden_shadow_table(stats: dict[str, Any]) -> str:
    if not stats or int(stats.get("total_hidden") or 0) <= 0:
        return ""

    rows = [
        "<tr>"
        "<td>overall</td>"
        f"<td>{int(stats.get('total_hidden') or 0)}</td>"
        f"<td>{int(stats.get('shadow_available') or 0)}</td>"
        f"<td>{int(stats.get('would_pass') or 0)}</td>"
        f"<td>{int(stats.get('safe_would_pass') or 0)}</td>"
        f"<td>{int(stats.get('would_be_dangerous') or 0)}</td>"
        f"<td>{float(stats.get('shadow_pass_rate') or 0.0):.1f}%</td>"
        f"<td>{float(stats.get('shadow_dangerous_rate') or 0.0):.1f}%</td>"
        f"<td>{float(stats.get('mean_iou') or 0.0):.3f}</td>"
        f"<td>{float(stats.get('mean_center_drift_px') or 0.0):.1f}px</td>"
        f"<td>{html.escape(str(stats.get('top_shadow_projection') or '—'))}</td>"
        f"<td>{html.escape(str(stats.get('top_hidden_reason') or '—'))}</td>"
        "</tr>"
    ]
    by_projection = stats.get("by_projection")
    if isinstance(by_projection, dict):
        for projection, item in by_projection.items():
            if not isinstance(item, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(projection))}</td>"
                f"<td>{int(item.get('total') or 0)}</td>"
                f"<td>{int(item.get('total') or 0)}</td>"
                f"<td>{int(item.get('would_pass') or 0)}</td>"
                f"<td>{int(item.get('safe_would_pass') or 0)}</td>"
                f"<td>{int(item.get('would_be_dangerous') or 0)}</td>"
                f"<td>{float(item.get('pass_rate') or 0.0):.1f}%</td>"
                f"<td>{float(item.get('dangerous_rate') or 0.0):.1f}%</td>"
                f"<td>{float(item.get('mean_iou') or 0.0):.3f}</td>"
                f"<td>{float(item.get('mean_center_drift_px') or 0.0):.1f}px</td>"
                "<td>—</td>"
                "<td>—</td>"
                "</tr>"
            )

    return f"""
      <h2>Unsafe hidden shadow mode</h2>
      <table>
        <tr>
          <th>bucket</th>
          <th>hidden</th>
          <th>shadow</th>
          <th>would pass</th>
          <th>safe pass</th>
          <th>dangerous</th>
          <th>pass rate</th>
          <th>danger rate</th>
          <th>mean IoU</th>
          <th>mean drift</th>
          <th>top projection</th>
          <th>top reason</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_deep_failure_stats_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('safe_gain_candidates') or 0)}</td>"
            f"<td>{int(item.get('danger_risk_candidates') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_projection') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_shape') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_action') or '—'))}</td>"
            f"<td>{float(item.get('mean_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_center_drift_px') or 0.0):.1f}px</td>"
            f"<td>{float(item.get('mean_support') or 0.0):.1f}</td>"
            f"<td>{html.escape(', '.join(item.get('top_iou_buckets') or []))}</td>"
            f"<td>{html.escape(', '.join(item.get('top_drift_buckets') or []))}</td>"
            f"<td>{html.escape(', '.join(item.get('top_support_buckets') or []))}</td>"
            "</tr>"
        )
    return f"""
      <h2>Deep failure diagnostics</h2>
      <table>
        <tr>
          <th>failure class</th>
          <th>objects</th>
          <th>safe gain candidates</th>
          <th>danger risk</th>
          <th>top projection</th>
          <th>top shape</th>
          <th>top action</th>
          <th>mean IoU</th>
          <th>mean drift</th>
          <th>mean support</th>
          <th>IoU buckets</th>
          <th>drift buckets</th>
          <th>support buckets</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_recovery_opportunity_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('safe_gain_candidates') or 0)}</td>"
            f"<td>{int(item.get('danger_risk_candidates') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_diagnostic') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_action') or '—'))}</td>"
            f"<td>{float(item.get('mean_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_center_drift_px') or 0.0):.1f}px</td>"
            "</tr>"
        )
    return f"""
      <h2>Recovery opportunity map</h2>
      <table>
        <tr>
          <th>recoverability</th>
          <th>objects</th>
          <th>safe gain candidates</th>
          <th>danger risk</th>
          <th>top diagnostic</th>
          <th>top action</th>
          <th>mean IoU</th>
          <th>mean drift</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_method_coverage_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('passed') or 0)}</td>"
            f"<td>{int(item.get('failed') or 0)}</td>"
            f"<td>{float(item.get('pass_rate') or 0.0):.1f}%</td>"
            f"<td>{int(item.get('hidden') or 0)}</td>"
            f"<td>{int(item.get('dangerous') or 0)}</td>"
            f"<td>{float(item.get('mean_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_center_drift_px') or 0.0):.1f}px</td>"
            f"<td>{int(item.get('shadow_available') or 0)}</td>"
            f"<td>{int(item.get('shadow_safe_pass') or 0)}</td>"
            f"<td>{int(item.get('shadow_dangerous') or 0)}</td>"
            f"<td>{int(item.get('global_translation_attempted') or 0)}</td>"
            f"<td>{int(item.get('anchor_release_attempted') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_failure_class') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_global_translation_state') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_anchor_state') or '—'))}</td>"
            "</tr>"
        )
    return f"""
      <h2>Method coverage and overlap</h2>
      <table>
        <tr>
          <th>method/projection</th>
          <th>objects</th>
          <th>passed</th>
          <th>failed</th>
          <th>pass rate</th>
          <th>hidden</th>
          <th>danger</th>
          <th>mean IoU</th>
          <th>mean drift</th>
          <th>shadow</th>
          <th>shadow safe</th>
          <th>shadow danger</th>
          <th>global rescue tries</th>
          <th>anchor tries</th>
          <th>top failure</th>
          <th>top global state</th>
          <th>top anchor state</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_candidate_oracle_stats_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('safe_alternative_objects') or 0)}</td>"
            f"<td>{int(item.get('dangerous_candidate_count') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_projection') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_failure_class') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_best_source') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_safe_gain_source') or '—'))}</td>"
            f"<td>{float(item.get('mean_available_candidates') or 0.0):.1f}</td>"
            f"<td>{float(item.get('mean_safe_candidates') or 0.0):.1f}</td>"
            f"<td>{float(item.get('mean_current_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_best_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_current_drift_px') or 0.0):.1f}px</td>"
            f"<td>{float(item.get('mean_best_drift_px') or 0.0):.1f}px</td>"
            "</tr>"
        )
    return f"""
      <h2>Candidate oracle analysis</h2>
      <table>
        <tr>
          <th>mode</th>
          <th>objects</th>
          <th>safe alternatives</th>
          <th>danger candidates</th>
          <th>top current</th>
          <th>top failure</th>
          <th>top best</th>
          <th>top safe source</th>
          <th>avg candidates</th>
          <th>avg safe</th>
          <th>current IoU</th>
          <th>best IoU</th>
          <th>current drift</th>
          <th>best drift</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_candidate_oracle_matrix_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    )[:50]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('safe_alternative_objects') or 0)}</td>"
            f"<td>{int(item.get('selected_would_pass') or 0)}</td>"
            f"<td>{int(item.get('dangerous_candidate_count') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_failure_class') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_shape') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_safe_gain_source') or '—'))}</td>"
            f"<td>{float(item.get('mean_current_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_best_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_current_drift_px') or 0.0):.1f}px</td>"
            f"<td>{float(item.get('mean_best_drift_px') or 0.0):.1f}px</td>"
            "</tr>"
        )
    return f"""
      <h2>Candidate oracle matrix</h2>
      <table>
        <tr>
          <th>current → oracle best</th>
          <th>objects</th>
          <th>safe alternatives</th>
          <th>selected would pass</th>
          <th>danger candidates</th>
          <th>top failure</th>
          <th>top shape</th>
          <th>top safe source</th>
          <th>current IoU</th>
          <th>best IoU</th>
          <th>current drift</th>
          <th>best drift</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_candidate_overlap_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in list(
        sorted(
            stats.items(),
            key=lambda pair: int(pair[1].get("total", 0)),
            reverse=True,
        )
    )[:64]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('passed') or 0)}</td>"
            f"<td>{int(item.get('failed') or 0)}</td>"
            f"<td>{float(item.get('current_pass_rate') or 0.0):.1f}%</td>"
            f"<td>{int(item.get('shadow_safe_gain') or 0)}</td>"
            f"<td>{int(item.get('shadow_danger_risk') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_shape') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_hidden_reason') or '—'))}</td>"
            f"<td>{float(item.get('mean_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_center_drift_px') or 0.0):.1f}px</td>"
            "</tr>"
        )
    return f"""
      <h2>Candidate overlap matrix</h2>
      <table>
        <tr>
          <th>overlap</th>
          <th>objects</th>
          <th>passed</th>
          <th>failed</th>
          <th>current pass</th>
          <th>shadow safe gain</th>
          <th>shadow danger</th>
          <th>top shape</th>
          <th>top hidden reason</th>
          <th>mean IoU</th>
          <th>mean drift</th>
        </tr>
        {''.join(rows)}
      </table>
    """



def _html_crop_verification_stats_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('failed') or 0)}</td>"
            f"<td>{int(item.get('hidden') or 0)}</td>"
            f"<td>{int(item.get('object_crop_candidate_available') or 0)}</td>"
            f"<td>{int(item.get('object_crop_candidate_would_pass') or 0)}</td>"
            f"<td>{int(item.get('object_crop_candidate_dangerous') or 0)}</td>"
            f"<td>{int(item.get('best_would_pass') or 0)}</td>"
            f"<td>{int(item.get('best_dangerous') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_projection') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_failure_class') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_best_source') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_object_crop_source') or '—'))}</td>"
            f"<td>{float(item.get('mean_best_score') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_best_matches') or 0.0):.1f}</td>"
            f"<td>{float(item.get('mean_object_crop_iou') or 0.0):.3f}</td>"
            "</tr>"
        )
    return f"""
      <h2>Object-crop verification shadow</h2>
      <table>
        <tr>
          <th>assessment</th>
          <th>objects</th>
          <th>failed</th>
          <th>hidden</th>
          <th>object-crop candidates</th>
          <th>object-crop would pass</th>
          <th>object-crop dangerous</th>
          <th>best would pass</th>
          <th>best dangerous</th>
          <th>top projection</th>
          <th>top failure</th>
          <th>top best source</th>
          <th>top object-crop source</th>
          <th>mean score</th>
          <th>mean matches</th>
          <th>object-crop IoU</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_crop_verification_matrix_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for name, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    )[:60]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('object_crop_candidate_available') or 0)}</td>"
            f"<td>{int(item.get('object_crop_candidate_would_pass') or 0)}</td>"
            f"<td>{int(item.get('object_crop_candidate_dangerous') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_failure_class') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_shape') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_oracle_mode') or '—'))}</td>"
            f"<td>{float(item.get('mean_current_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_best_score') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_best_matches') or 0.0):.1f}</td>"
            f"<td>{float(item.get('mean_object_crop_iou') or 0.0):.3f}</td>"
            "</tr>"
        )
    return f"""
      <h2>Object-crop verification matrix</h2>
      <table>
        <tr>
          <th>projection → crop assessment / crop source</th>
          <th>objects</th>
          <th>object-crop candidates</th>
          <th>would pass</th>
          <th>dangerous</th>
          <th>top failure</th>
          <th>top shape</th>
          <th>top oracle mode</th>
          <th>current IoU</th>
          <th>best score</th>
          <th>matches</th>
          <th>object-crop IoU</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_feature_telemetry_table(stats: dict[str, Any]) -> str:
    if not stats:
        return ""

    rows = [
        ("objects", int(stats.get("objects", 0))),
        ("mean reference keypoints", f"{float(stats.get('mean_reference_keypoints_total', 0.0)):.1f}"),
        ("mean frame keypoints", f"{float(stats.get('mean_frame_keypoints_total', 0.0)):.1f}"),
        ("mean frame max keypoints", f"{float(stats.get('mean_frame_max_keypoints', 0.0)):.1f}"),
        ("mean LightGlue ref matches", f"{float(stats.get('mean_lightglue_reference_matches_total', 0.0)):.1f}"),
        ("mean LightGlue frame matches", f"{float(stats.get('mean_lightglue_frame_matches_total', 0.0)):.1f}"),
        ("masked alignment used", int(stats.get("masked_alignment_used", 0))),
        ("mean original reference keypoints", f"{float(stats.get('mean_original_reference_keypoints_total', 0.0)):.1f}"),
        ("mean masked reference keypoints", f"{float(stats.get('mean_masked_reference_keypoints_total', 0.0)):.1f}"),
        ("mean masked LightGlue matches", f"{float(stats.get('mean_masked_lightglue_matches_total', 0.0)):.1f}"),
        ("slot-local attempted", int(stats.get("slot_local_attempted", 0))),
        ("slot-local accepted", int(stats.get("slot_local_accepted", 0))),
        ("slot-local accept rate", f"{float(stats.get('slot_local_accept_rate', 0.0)):.1f}%"),
        ("mean slot-local ref/frame keypoints", f"{float(stats.get('mean_slot_local_reference_keypoints', 0.0)):.1f} / {float(stats.get('mean_slot_local_frame_keypoints', 0.0)):.1f}"),
        ("mean slot-local matches/inliers", f"{float(stats.get('mean_slot_local_raw_matches', 0.0)):.1f} / {float(stats.get('mean_slot_local_inliers', 0.0)):.1f}"),
    ]
    body = "".join(
        "<tr>"
        f"<td>{html.escape(str(name))}</td>"
        f"<td>{html.escape(str(value))}</td>"
        "</tr>"
        for name, value in rows
    )
    return f"""
      <h2>Feature telemetry</h2>
      <table>
        <tr><th>metric</th><th>value</th></tr>
        {body}
      </table>
    """


def _html_result_policy_stats_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for status, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(status))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('passed') or 0)}</td>"
            f"<td>{int(item.get('failed') or 0)}</td>"
            f"<td>{int(item.get('unsafe_hidden') or 0)}</td>"
            f"<td>{int(item.get('dangerous') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_confidence') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_action') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_user_label') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_reason') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_projection') or '—'))}</td>"
            f"<td>{float(item.get('mean_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_center_drift_px') or 0.0):.1f}px</td>"
            "</tr>"
        )
    return f"""
      <h2>Result policy / UI rendering status</h2>
      <table>
        <tr>
          <th>policy</th>
          <th>objects</th>
          <th>passed</th>
          <th>failed</th>
          <th>hidden</th>
          <th>dangerous</th>
          <th>confidence</th>
          <th>UI action</th>
          <th>user label</th>
          <th>top reason</th>
          <th>top projection</th>
          <th>mean IoU</th>
          <th>mean drift</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_yolo_synthetic_feasibility_table(stats: dict[str, dict[str, Any]]) -> str:
    if not stats:
        return ""
    rows = []
    for bucket, item in sorted(
        stats.items(),
        key=lambda pair: int(pair[1].get("total", 0)),
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(bucket))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('failed') or 0)}</td>"
            f"<td>{int(item.get('unsafe_hidden') or 0)}</td>"
            f"<td>{int(item.get('dangerous') or 0)}</td>"
            f"<td>{html.escape(str(item.get('top_expected_role') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_policy') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_projection') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_shape') or '—'))}</td>"
            f"<td>{float(item.get('mean_iou') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_center_drift_px') or 0.0):.1f}px</td>"
            f"<td>{html.escape(str(item.get('limitation') or '—'))}</td>"
            "</tr>"
        )
    return f"""
      <h2>YOLO synthetic feasibility map</h2>
      <table>
        <tr>
          <th>bucket</th>
          <th>objects</th>
          <th>failed</th>
          <th>hidden</th>
          <th>dangerous</th>
          <th>expected YOLO role</th>
          <th>top policy</th>
          <th>top projection</th>
          <th>top shape</th>
          <th>mean IoU</th>
          <th>mean drift</th>
          <th>limitation</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_yolo_synthetic_fixture_table(stats: dict[str, dict[str, Any]]) -> str:
    return _html_detector_fixture_table(
        title="YOLO synthetic detector fixture",
        stats=stats,
    )


def _html_yolo_gt_detector_table(stats: dict[str, dict[str, Any]]) -> str:
    return _html_detector_fixture_table(
        title="YOLO GT detector oracle",
        stats=stats,
    )


def _html_detector_fixture_table(
    *,
    title: str,
    stats: dict[str, dict[str, Any]],
) -> str:
    if not stats:
        return ""
    rows = []
    for profile, item in stats.items():
        by_bucket = item.get("by_bucket")
        if not isinstance(by_bucket, dict):
            by_bucket = {}
        if not by_bucket:
            rows.append(_html_detector_fixture_row(profile, "all", item))
            continue
        first = True
        for bucket, bucket_item in sorted(
            by_bucket.items(),
            key=lambda pair: int(pair[1].get("baseline_failed", 0)),
            reverse=True,
        ):
            if int(bucket_item.get("baseline_failed") or 0) == 0 and bucket != "geometry_baseline_ok":
                continue
            rows.append(
                _html_detector_fixture_row(
                    profile if first else "",
                    bucket,
                    bucket_item,
                )
            )
            first = False
    return f"""
      <h2>{html.escape(title)}</h2>
      <table>
        <tr>
          <th>profile</th>
          <th>bucket</th>
          <th>objects</th>
          <th>baseline failed</th>
          <th>would rescue</th>
          <th>remaining failed</th>
          <th>dangerous if trusted</th>
          <th>top status</th>
          <th>top candidate</th>
          <th>top reason</th>
        </tr>
        {''.join(rows)}
      </table>
    """


def _html_detector_fixture_row(
    profile: str,
    bucket: str,
    item: dict[str, Any],
) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(profile or '—'))}</td>"
        f"<td>{html.escape(str(bucket))}</td>"
        f"<td>{int(item.get('total') or 0)}</td>"
        f"<td>{int(item.get('baseline_failed') or 0)}</td>"
        f"<td>{int(item.get('would_rescue') or 0)}</td>"
        f"<td>{int(item.get('remaining_failed') or 0)}</td>"
        f"<td>{int(item.get('dangerous_if_trusted') or 0)}</td>"
        f"<td>{html.escape(str(item.get('top_status') or '—'))}</td>"
        f"<td>{html.escape(str(item.get('top_candidate_source') or '—'))}</td>"
        f"<td>{html.escape(str(item.get('top_reason') or '—'))}</td>"
        "</tr>"
    )

def _html_failure_microscope_table(rows_data: list[dict[str, Any]]) -> str:
    if not rows_data:
        return ""
    rows = []
    for row in rows_data[:120]:
        rows.append(
            "<tr>"
            f"<td>#{int(row.get('case_index') or 0):03d}</td>"
            f"<td>{html.escape(str(row.get('object_name') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('shape') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('projection') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('result_policy_status') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('result_policy_action') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('yolo_synthetic_test_bucket') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('yolo_fixture_oracle_status') or '—'))}</td>"
            f"<td>{'yes' if row.get('yolo_fixture_oracle_would_gain') else 'no'}</td>"
            f"<td>{html.escape(str(row.get('yolo_fixture_noisy_status') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('yolo_fixture_false_positive_status') or '—'))}</td>"
            f"<td>{'yes' if row.get('yolo_fixture_false_positive_would_be_dangerous') else 'no'}</td>"
            f"<td>{html.escape(str(row.get('yolo_gt_perfect_status') or '—'))}</td>"
            f"<td>{'yes' if row.get('yolo_gt_perfect_would_gain') else 'no'}</td>"
            f"<td>{html.escape(str(row.get('yolo_gt_jitter_status') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('yolo_gt_false_positive_status') or '—'))}</td>"
            f"<td>{'yes' if row.get('yolo_gt_false_positive_would_be_dangerous') else 'no'}</td>"
            f"<td>{html.escape(str(row.get('diagnostic_class') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('recoverability') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('recommended_action') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('candidate_oracle_failure_mode') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('candidate_oracle_best_source') or '—'))}</td>"
            f"<td>{float(row.get('candidate_oracle_best_iou') or 0.0):.3f}</td>"
            f"<td>{float(row.get('candidate_oracle_best_drift_px') or 0.0):.1f}px</td>"
            f"<td>{html.escape(str(row.get('candidate_oracle_safe_gain_source') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('crop_verification_assessment') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('crop_verification_best_source') or '—'))}</td>"
            f"<td>{float(row.get('crop_verification_best_score') or 0.0):.3f}</td>"
            f"<td>{int(row.get('crop_verification_best_matches') or 0)}</td>"
            f"<td>{html.escape(str(row.get('crop_verification_object_crop_source') or '—'))}</td>"
            f"<td>{float(row.get('crop_verification_object_crop_iou') or 0.0):.3f}</td>"
            f"<td>{'yes' if row.get('crop_verification_object_crop_would_pass') else 'no'}</td>"
            f"<td>{'yes' if row.get('crop_verification_object_crop_dangerous') else 'no'}</td>"
            f"<td>{float(row.get('iou') or 0.0):.3f}</td>"
            f"<td>{float(row.get('center_drift_px') or 0.0):.1f}px</td>"
            f"<td>{float(row.get('area_ratio') or 0.0):.2f}x</td>"
            f"<td>{int(row.get('support') or 0)}/{int(row.get('support_total') or 0)}</td>"
            f"<td>{html.escape(str(row.get('hidden_reason') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('hidden_shadow_projection') or '—'))}</td>"
            f"<td>{float(row.get('hidden_shadow_iou') or 0.0):.3f}</td>"
            f"<td>{float(row.get('hidden_shadow_drift_px') or 0.0):.1f}px</td>"
            f"<td>{'yes' if row.get('hidden_shadow_would_pass') else 'no'}</td>"
            f"<td>{'yes' if row.get('hidden_shadow_dangerous') else 'no'}</td>"
            f"<td>{html.escape(str(row.get('global_translation_state') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('anchor_state') or '—'))}</td>"
            "</tr>"
        )
    return f"""
      <h2>Failure microscope</h2>
      <table>
        <tr>
          <th>case</th>
          <th>object</th>
          <th>shape</th>
          <th>projection</th>
          <th>policy</th>
          <th>UI action</th>
          <th>YOLO synthetic bucket</th>
          <th>oracle fixture</th>
          <th>oracle gain</th>
          <th>noisy fixture</th>
          <th>false-positive fixture</th>
          <th>FP dangerous</th>
          <th>GT perfect</th>
          <th>GT gain</th>
          <th>GT jitter</th>
          <th>GT false positive</th>
          <th>GT FP danger</th>
          <th>diagnostic class</th>
          <th>recoverability</th>
          <th>recommended action</th>
          <th>oracle mode</th>
          <th>oracle best</th>
          <th>oracle IoU</th>
          <th>oracle drift</th>
          <th>safe source</th>
          <th>crop assessment</th>
          <th>crop best</th>
          <th>crop score</th>
          <th>crop matches</th>
          <th>object-crop source</th>
          <th>object-crop IoU</th>
          <th>object-crop pass</th>
          <th>object-crop danger</th>
          <th>IoU</th>
          <th>drift</th>
          <th>area</th>
          <th>support</th>
          <th>hidden reason</th>
          <th>shadow projection</th>
          <th>shadow IoU</th>
          <th>shadow drift</th>
          <th>shadow pass</th>
          <th>shadow danger</th>
          <th>global rescue</th>
          <th>anchor</th>
        </tr>
        {''.join(rows)}
      </table>
    """

def _html_fallback_diagnostics_table(
    title: str,
    stats: dict[str, dict[str, Any]],
) -> str:
    if not stats:
        return ""

    rows = []
    for reason, item in stats.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(reason))}</td>"
            f"<td>{int(item.get('total') or 0)}</td>"
            f"<td>{int(item.get('passed') or 0)}</td>"
            f"<td>{float(item.get('pass_rate') or 0.0):.1f}%</td>"
            f"<td>{html.escape(str(item.get('top_shape') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('top_reason_code') or '—'))}</td>"
            f"<td>{float(item.get('mean_area_score') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_center_factor') or 0.0):.3f}</td>"
            f"<td>{float(item.get('mean_slot_support') or 0.0):.1f}</td>"
            "</tr>"
        )

    return f"""
      <h2>{html.escape(title)}</h2>
      <table>
        <tr>
          <th>reason</th>
          <th>objects</th>
          <th>passed</th>
          <th>pass rate</th>
          <th>top shape</th>
          <th>top code</th>
          <th>mean area</th>
          <th>mean center</th>
          <th>mean support</th>
        </tr>
        {''.join(rows)}
      </table>
    """

def _fmt_debug_float(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def _html_case_slot_local_debug(result: SyntheticResult) -> str:
    if not result.object_metrics:
        return ""

    rows = []
    for raw in result.object_metrics:
        metric = _enrich_object_metric(dict(raw))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(metric.get('name') or '—'))}</td>"
            f"<td>{html.escape(str(metric.get('projection') or '—'))}</td>"
            f"<td>{'yes' if bool(metric.get('passed')) else 'no'}</td>"
            f"<td>{int(metric.get('lightglue_reference_matches_total') or 0)}</td>"
            f"<td>{html.escape(str(metric.get('v2_scene_model_source') or '—'))}</td>"
            f"<td>{int(metric.get('v2_scene_match_cells') or 0)} / {int(metric.get('v2_scene_model_inlier_cells') or 0)}</td>"
            f"<td>{int(metric.get('v2_scene_match_bottom_count') or 0)} / {int(metric.get('v2_scene_model_inlier_bottom_count') or 0)}</td>"
            f"<td>{_fmt_debug_float(metric.get('v2_scene_model_inlier_span_x'))} / {_fmt_debug_float(metric.get('v2_scene_model_inlier_span_y'))}</td>"
            f"<td>{_fmt_debug_float(metric.get('v2_scene_model_inlier_max_cell_fraction'))}</td>"
            f"<td>{'yes' if bool(metric.get('none_promoted_to_slot_local_seed')) else 'no'}</td>"
            f"<td>{html.escape(str(metric.get('none_original_reason') or metric.get('none_reason') or '—'))}</td>"
            f"<td>{'yes' if bool(metric.get('slot_local_lightglue_attempted')) else 'no'}</td>"
            f"<td>{'yes' if bool(metric.get('slot_local_lightglue_accepted')) else 'no'}</td>"
            f"<td>{html.escape(str(metric.get('slot_local_lightglue_mode') or '—'))}</td>"
            f"<td>{int(metric.get('slot_local_lightglue_raw_matches') or 0)}</td>"
            f"<td>{int(metric.get('slot_local_lightglue_inliers') or 0)}</td>"
            f"<td>{html.escape(str(metric.get('slot_local_lightglue_reject_reason') or '—'))}</td>"
            "</tr>"
        )

    return f"""
              <h3>Per-object global/local debug</h3>
              <table>
                <tr>
                  <th>object</th>
                  <th>projection</th>
                  <th>pass</th>
                  <th>global matches</th>
                  <th>scene model</th>
                  <th>cells all/inl</th>
                  <th>bottom all/inl</th>
                  <th>inlier span x/y</th>
                  <th>max cell frac</th>
                  <th>seeded none</th>
                  <th>none reason</th>
                  <th>slot tried</th>
                  <th>slot accepted</th>
                  <th>slot mode</th>
                  <th>slot matches</th>
                  <th>slot inliers</th>
                  <th>slot reject</th>
                </tr>
                {''.join(rows)}
              </table>
            """



def _html_decision_legend() -> str:
    return """
      <h2>Как читать один кейс</h2>
      <table>
        <tr><th>Элемент</th><th>Что означает</th></tr>
        <tr><td>Зелёный полигон / GT</td><td>Истинное место отсутствующей детали после синтетической проекции эталона.</td></tr>
        <tr><td>Красный полигон / PRED</td><td>Куда текущий pipeline перенёс expected-зону. Если его нет — зона скрыта как неподтверждённая.</td></tr>
        <tr><td>Белая стрелка drift</td><td>Сдвиг центра PRED относительно GT. Это самый быстрый визуальный индикатор ошибки.</td></tr>
        <tr><td>Оранжевые полигоны</td><td>Похожие distractor-детали. Опасная ошибка — когда PRED ближе к ним, чем к GT.</td></tr>
        <tr><td>PASS</td><td>IoU, drift, area ratio, ось и защита от distractor прошли пороги.</td></tr>
        <tr><td>SAFE FAIL</td><td>Точность не прошла, но проекция не уехала на опасный distractor.</td></tr>
        <tr><td>DANGER</td><td>Полигон лучше скрыть и считать деталь неподтверждённой, чем рисовать уверенную ложную зону.</td></tr>
      </table>
    """

def _write_html_report(
    path: Path,
    *,
    results: list[SyntheticResult],
    summary: dict[str, Any],
) -> None:
    cards = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        safety_status = "SAFE" if result.safety_passed else "DANGER"
        notes = "<br>".join(html.escape(note) for note in result.notes) or "—"
        keypoints_html = ""
        if result.keypoints_image_path:
            keypoints_html = (
                f'<h3>Global keypoints / global LightGlue diagnostics</h3>'
                f'<p class="hint">Панель 4 показывает global LightGlue matches, не локальные пары конкретной детали.</p>'
                f'<img src="{html.escape(result.keypoints_image_path)}" '
                f'alt="Synthetic keypoints {result.index}">'
            )
        object_debug_html = _html_case_slot_local_debug(result)
        cards.append(
            f"""
            <article class="case {'pass' if result.passed else 'fail'}">
              <h2>#{result.index:03d} {html.escape(result.name)} <span>{status}</span></h2>
              <p>{html.escape(result.description)}</p>
              <table>
                <tr><th>kind</th><td>{html.escape(result.scenario_kind)}</td><th>objects</th><td>{result.object_count}</td></tr>
                <tr><th>shape</th><td>{html.escape(result.object_shape if result.object_shape != "multi" else ", ".join(result.object_shapes or []))}</td><th>status</th><td>{html.escape(result.status)}</td></tr>
                <tr><th>projection</th><td>{html.escape(result.projection)}</td><th>safety</th><td>{safety_status}</td></tr>
                <tr><th>unsafe hidden</th><td>{result.unsafe_hidden}</td><th>dangerous</th><td>{result.dangerous_projection}</td></tr>
                <tr><th>IoU</th><td>{result.iou:.3f}</td><th>center drift</th><td>{result.center_drift_px:.1f}px</td></tr>
                <tr><th>area ratio</th><td>{result.area_ratio:.2f}x</td><th>context support</th><td>{result.context_feature_support}/{result.context_feature_total}</td></tr>
                <tr><th>axis angle</th><td>{_format_optional_float(result.axis_angle_error_deg, '°')}</td><th>major length</th><td>{_format_optional_float(result.major_length_ratio, 'x')}</td></tr>
                <tr><th>candidates/inliers</th><td>{result.missing_candidate_count}/{result.missing_inliers}</td><th>closer to distractor</th><td>{result.closer_to_distractor}</td></tr>
                <tr><th>notes</th><td colspan="3">{notes}</td></tr>
              </table>
              <img src="{html.escape(result.image_path)}" alt="Synthetic case {result.index}">
              {keypoints_html}
              {object_debug_html}
            </article>
            """
        )

    document = f"""
    <!doctype html>
    <html lang="ru">
    <head>
      <meta charset="utf-8">
      <title>Synthetic context-ring missing polygon test</title>
      <style>
        body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #111; color: #eee; }}
        h1 {{ margin-bottom: 8px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin: 18px 0 28px; }}
        .metric {{ background: #202020; border-radius: 10px; padding: 14px; }}
        .metric b {{ display: block; font-size: 24px; margin-top: 6px; }}
        .case {{ background: #1b1b1b; border: 1px solid #333; border-radius: 14px; padding: 16px; margin: 18px 0; }}
        .case.pass {{ border-color: #2f9e44; }}
        .case.fail {{ border-color: #e03131; }}
        .case h2 {{ display: flex; justify-content: space-between; gap: 12px; margin-top: 0; }}
        .case.pass h2 span {{ color: #69db7c; }}
        .case.fail h2 span {{ color: #ff8787; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
        th, td {{ border: 1px solid #333; padding: 8px 10px; text-align: left; }}
        th {{ background: #262626; width: 160px; }}
        img {{ width: 100%; border-radius: 10px; border: 1px solid #333; }}
        .hint {{ color: #bbb; }}
        code {{ background: #222; padding: 2px 5px; border-radius: 4px; }}
      </style>
    </head>
    <body>
      <h1>Synthetic context-ring missing polygon test</h1>
      <p>YOLO здесь не запускается. Все detections пустые. Тест проверяет только перенос expected/missing зоны по окружающим feature-точкам.</p>
      <p class="hint">Формы теперь сложные: вогнутые, тонкие, ступенчатые и крючкообразные. Четвёртая панель показывает, куда global alignment поставил context-точки и куда они реально сопоставились.</p>
      <p class="hint">В profile=nightmare по умолчанию сначала идут одиночные кейсы, затем такая же пачка multi-object кейсов: несколько разных/одинаковых объектов в одной сцене.</p>
      <p class="hint">Для каждого кейса пишется отдельная картинка keypoints: первые три панели — global SuperPoint, четвёртая — только global LightGlue pairs. Локальные slot-local пары смотри в таблице под кейсом.</p>
      <p class="hint">Синтетическая камера теперь наклоняет эталон в 8 направлений по seed, а не только одной фиксированной перспективой. Значение tilt на картинке: направление/сила projective-компоненты.</p>
      <div class="summary">
        <div class="metric">Cases<b>{summary['total']}</b></div>
        <div class="metric">Passed<b>{summary['passed']}</b></div>
        <div class="metric">Failed<b>{summary['failed']}</b></div>
        <div class="metric">Pass rate<b>{summary['pass_rate']:.1f}%</b></div>
        <div class="metric">Safety rate<b>{summary['safety_rate']:.1f}%</b></div>
        <div class="metric">Dangerous<b>{summary['dangerous_projection_count']}</b></div>
        <div class="metric">Hidden unsafe<b>{summary['unsafe_hidden_count']}</b></div>
        <div class="metric">Single / Multi<b>{summary['single_cases']} / {summary['multi_cases']}</b></div>
        <div class="metric">Objects<b>{summary['total_expected_objects']}</b></div>
        <div class="metric">Object accuracy<b>{summary['object_accuracy_rate']:.1f}%</b></div>
        <div class="metric">Object safety<b>{summary['object_safety_rate']:.1f}%</b></div>
        <div class="metric">Object hidden<b>{summary['object_unsafe_hidden_count']}</b></div>
        <div class="metric">Mean IoU<b>{summary['mean_iou']:.3f}</b></div>
        <div class="metric">Min IoU<b>{summary['min_iou']:.3f}</b></div>
        <div class="metric">Mean drift<b>{summary['mean_center_drift_px']:.1f}px</b></div>
        <div class="metric">Max drift<b>{summary['max_center_drift_px']:.1f}px</b></div>
      </div>
      <p>Настройки: edge refinement = <code>{summary['edge_refinement_enabled']}</code>, context expansion = <code>{summary['context_expansion']}</code>, min support = <code>{summary['min_feature_support']}</code>.</p>
      {_html_decision_legend()}
      {_html_stats_table('Object projection stats', summary.get('object_projection_stats', {}))}
      {_html_stats_table('Object shape stats', summary.get('object_shape_stats', {}))}
      {_html_reason_stats_table('Top object failure reasons', summary.get('object_failure_reason_stats', {}))}
      {_html_reason_stats_table('Hidden reason stats', summary.get('object_hidden_reason_stats', {}))}
      {_html_reason_stats_table('Anchor release reject stats', summary.get('object_anchor_release_reject_stats', {}))}
      {_html_anchor_build_stats_table('Anchor build reject stats', summary.get('object_anchor_build_reject_stats', {}))}
      {_html_anchor_order_stats_table('Anchor order diagnostics', summary.get('object_anchor_order_stats', {}))}
      {_html_fallback_diagnostics_table('None projection diagnostics', summary.get('object_none_projection_stats', {}))}
      {_html_fallback_diagnostics_table('Global fallback diagnostics', summary.get('object_global_fallback_diagnostics', {}))}
      {_html_global_translation_rescue_diagnostics_table('Global translation rescue diagnostics', summary.get('object_global_translation_rescue_diagnostics', {}))}
      {_html_unsafe_hidden_shadow_table(summary.get('object_unsafe_hidden_shadow_stats', {}))}
      {_html_deep_failure_stats_table(summary.get('object_deep_failure_stats', {}))}
      {_html_recovery_opportunity_table(summary.get('object_recovery_opportunity_stats', {}))}
      {_html_method_coverage_table(summary.get('object_method_coverage_stats', {}))}
      {_html_stats_table('Selective hidden release stats', summary.get('object_selective_hidden_release_stats', {}))}
      {_html_stats_table('Candidate agreement scorer stats', summary.get('object_candidate_agreement_stats', {}))}
      {_html_candidate_oracle_stats_table(summary.get('object_candidate_oracle_stats', {}))}
      {_html_candidate_oracle_matrix_table(summary.get('object_candidate_oracle_matrix', {}))}
      {_html_crop_verification_stats_table(summary.get('object_crop_verification_stats', {}))}
      {_html_crop_verification_matrix_table(summary.get('object_crop_verification_matrix', {}))}
      {_html_candidate_overlap_table(summary.get('object_candidate_overlap_stats', {}))}
      {_html_feature_telemetry_table(summary.get('object_feature_telemetry_stats', {}))}
      {_html_result_policy_stats_table(summary.get('object_result_policy_stats', {}))}
      {_html_yolo_synthetic_feasibility_table(summary.get('object_yolo_synthetic_feasibility_stats', {}))}
      {_html_yolo_synthetic_fixture_table(summary.get('object_yolo_synthetic_fixture_stats', {}))}
      {_html_yolo_gt_detector_table(summary.get('object_yolo_gt_detector_stats', {}))}
      {_html_failure_microscope_table(summary.get('object_failure_microscope', []))}
      {''.join(cards)}
    </body>
    </html>
    """
    path.write_text(document, encoding="utf-8")


def _format_optional_float(value: float | None, suffix: str = "") -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.2f}{suffix}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _polygon_iou(a: PolygonPoints | None, b: PolygonPoints | None) -> float:
    shape_a = _safe_shape(a)
    shape_b = _safe_shape(b)
    if shape_a is None or shape_b is None:
        return 0.0
    try:
        union = shape_a.union(shape_b).area
        if union <= 0:
            return 0.0
        return float(shape_a.intersection(shape_b).area / union)
    except GEOSException:
        return 0.0


def _polygon_center_distance(a: PolygonPoints | None, b: PolygonPoints | None) -> float:
    center_a = _polygon_center(a)
    center_b = _polygon_center(b)
    if center_a is None or center_b is None:
        return float("inf")
    return float(np.linalg.norm(center_a - center_b))


def _polygon_axis_delta(
    predicted: PolygonPoints | None,
    ground_truth: PolygonPoints | None,
) -> tuple[float | None, float | None]:
    predicted_axis = _polygon_axis_metrics(predicted)
    gt_axis = _polygon_axis_metrics(ground_truth)
    if predicted_axis is None or gt_axis is None:
        return None, None

    predicted_angle, predicted_major, _ = predicted_axis
    gt_angle, gt_major, _ = gt_axis
    if predicted_major <= 0 or gt_major <= 0:
        return None, None

    diff = abs(predicted_angle - gt_angle) % 180.0
    angle_error = min(diff, 180.0 - diff)
    major_ratio = predicted_major / gt_major
    return float(angle_error), float(major_ratio)


def _polygon_axis_metrics(
    polygon: PolygonPoints | None,
) -> tuple[float, float, float] | None:
    if not polygon or len(polygon) < 3:
        return None
    points = np.asarray(polygon, dtype=np.float32)[:, :2]
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None

    centered = points - np.mean(points, axis=0, keepdims=True)
    if len(centered) < 2:
        return None

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if vh.shape[0] < 2:
        return None

    major_axis = vh[0]
    minor_axis = vh[1]
    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis
    major_length = float(np.max(major_projection) - np.min(major_projection))
    minor_length = float(np.max(minor_projection) - np.min(minor_projection))
    if major_length <= 0 or not np.isfinite(major_length):
        return None

    angle = math.degrees(math.atan2(float(major_axis[1]), float(major_axis[0]))) % 180.0
    return float(angle), major_length, minor_length


def _shape_aware_projection_ok(
    *,
    object_shape: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
    axis_angle_error_deg: float | None,
    major_length_ratio: float | None,
    weak_context: bool,
) -> bool:
    if axis_angle_error_deg is None or major_length_ratio is None:
        return False
    if area_ratio < 0.55 or area_ratio > 1.75:
        return False

    if object_shape == "thin_fork":
        min_iou = 0.28 if weak_context else 0.34
        max_drift = 24.0 if weak_context else 16.0
        return (
            iou >= min_iou
            and center_drift <= max_drift
            and axis_angle_error_deg <= 13.5
            and 0.70 <= major_length_ratio <= 1.34
        )

    if object_shape == "hook_like_part":
        min_iou = 0.30 if weak_context else 0.38
        max_drift = 25.0 if weak_context else 16.0
        return (
            iou >= min_iou
            and center_drift <= max_drift
            and axis_angle_error_deg <= 17.0
            and 0.68 <= major_length_ratio <= 1.40
        )

    return False


def _polygon_area_ratio(a: PolygonPoints | None, b: PolygonPoints | None) -> float:
    shape_a = _safe_shape(a)
    shape_b = _safe_shape(b)
    if shape_a is None or shape_b is None or shape_b.area <= 0:
        return 0.0
    return float(shape_a.area / shape_b.area)


def _nearest_distractor_distance(
    polygon: PolygonPoints | None,
    distractors: list[PolygonPoints],
) -> float | None:
    center = _polygon_center(polygon)
    if center is None or not distractors:
        return None
    distances = []
    for distractor in distractors:
        distractor_center = _polygon_center(distractor)
        if distractor_center is not None:
            distances.append(float(np.linalg.norm(center - distractor_center)))
    return min(distances) if distances else None


def _polygon_center(polygon: PolygonPoints | None) -> np.ndarray | None:
    if not polygon or len(polygon) < 3:
        return None
    points = np.asarray(polygon, dtype=np.float32)
    if not np.isfinite(points).all():
        return None
    return np.mean(points[:, :2], axis=0)


def _safe_shape(polygon: PolygonPoints | None):
    if not polygon or len(polygon) < 3:
        return None
    try:
        shape = make_valid(Polygon(polygon))
    except (ValueError, GEOSException):
        return None
    if shape.is_empty or shape.area <= 0:
        return None
    return shape


def _bbox_from_polygon(polygon: PolygonPoints) -> tuple[float, float, float, float]:
    points = np.asarray(polygon, dtype=np.float32)
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    factor: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    new_width = width * float(factor)
    new_height = height * float(factor)
    return (
        cx - new_width * 0.5,
        cy - new_height * 0.5,
        cx + new_width * 0.5,
        cy + new_height * 0.5,
    )


def _context_exclusion_margin(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    min_side = max(1.0, min(float(x2 - x1), float(y2 - y1)))
    configured = float(_SYNTHETIC_MISSING_POLYGON_CONTEXT_EXCLUSION_MARGIN)
    return max(0.0, min(configured, min_side * 0.25))


def _point_inside_polygon(point: Point, polygon: np.ndarray) -> bool:
    try:
        return cv2.pointPolygonTest(
            polygon[:, :2].astype(np.float32),
            (float(point[0]), float(point[1])),
            False,
        ) >= 0
    except cv2.error:
        return False


def _point_outside_polygon_margin(
    point: Point,
    polygon: np.ndarray,
    *,
    margin: float,
) -> bool:
    try:
        signed_distance = cv2.pointPolygonTest(
            polygon[:, :2].astype(np.float32),
            (float(point[0]), float(point[1])),
            True,
        )
    except cv2.error:
        return True
    return float(signed_distance) < -max(0.0, float(margin))




def _projection_candidate_registry_metric_fields(debug: dict[str, Any]) -> dict[str, Any]:
    raw_registry = debug.get("missing_polygon_candidate_registry")
    if not isinstance(raw_registry, list):
        return {
            "projection_candidate_count": 0,
            "projection_candidate_sources": [],
            "projection_candidate_selected": None,
            "candidate_agreement_available_count": _int_debug(
                debug.get("missing_polygon_candidate_agreement_available_count")
            ),
            "candidate_agreement_comparison_count": _int_debug(
                debug.get("missing_polygon_candidate_agreement_comparison_count")
            ),
            "candidate_agreement_selected": (
                str(debug.get("missing_polygon_candidate_agreement_selected"))
                if debug.get("missing_polygon_candidate_agreement_selected") is not None
                else None
            ),
            "candidate_agreement_count": _int_debug(
                debug.get("missing_polygon_candidate_agreement_count")
            ),
            "candidate_agreement_level": (
                str(debug.get("missing_polygon_candidate_agreement_level"))
                if debug.get("missing_polygon_candidate_agreement_level") is not None
                else None
            ),
            "candidate_agreement_best_iou": _float_or_none(
                debug.get("missing_polygon_candidate_agreement_best_iou")
            ),
            "candidate_agreement_best_area_score": _float_or_none(
                debug.get("missing_polygon_candidate_agreement_best_area_score")
            ),
            "candidate_agreement_min_center_factor": _float_or_none(
                debug.get("missing_polygon_candidate_agreement_min_center_factor")
            ),
            "candidate_agreement_closest_source": (
                str(debug.get("missing_polygon_candidate_agreement_closest_source"))
                if debug.get("missing_polygon_candidate_agreement_closest_source") is not None
                else None
            ),
            "candidate_confidence": (
                str(debug.get("missing_polygon_candidate_confidence"))
                if debug.get("missing_polygon_candidate_confidence") is not None
                else None
            ),
            "candidate_recommended_action": (
                str(debug.get("missing_polygon_candidate_recommended_action"))
                if debug.get("missing_polygon_candidate_recommended_action") is not None
                else None
            ),
        }

    sources: list[str] = []
    selected: str | None = None
    for raw_item in raw_registry:
        if not isinstance(raw_item, dict):
            continue
        name = raw_item.get("name")
        if name is None:
            continue
        source = str(name)
        sources.append(source)
        if bool(raw_item.get("selected")):
            selected = source

    return {
        "projection_candidate_count": len(sources),
        "projection_candidate_sources": sources,
        "projection_candidate_selected": selected,
        "candidate_agreement_available_count": _int_debug(
            debug.get("missing_polygon_candidate_agreement_available_count")
        ),
        "candidate_agreement_comparison_count": _int_debug(
            debug.get("missing_polygon_candidate_agreement_comparison_count")
        ),
        "candidate_agreement_selected": (
            str(debug.get("missing_polygon_candidate_agreement_selected"))
            if debug.get("missing_polygon_candidate_agreement_selected") is not None
            else None
        ),
        "candidate_agreement_count": _int_debug(
            debug.get("missing_polygon_candidate_agreement_count")
        ),
        "candidate_agreement_level": (
            str(debug.get("missing_polygon_candidate_agreement_level"))
            if debug.get("missing_polygon_candidate_agreement_level") is not None
            else None
        ),
        "candidate_agreement_best_iou": _float_or_none(
            debug.get("missing_polygon_candidate_agreement_best_iou")
        ),
        "candidate_agreement_best_area_score": _float_or_none(
            debug.get("missing_polygon_candidate_agreement_best_area_score")
        ),
        "candidate_agreement_min_center_factor": _float_or_none(
            debug.get("missing_polygon_candidate_agreement_min_center_factor")
        ),
        "candidate_agreement_closest_source": (
            str(debug.get("missing_polygon_candidate_agreement_closest_source"))
            if debug.get("missing_polygon_candidate_agreement_closest_source") is not None
            else None
        ),
        "candidate_confidence": (
            str(debug.get("missing_polygon_candidate_confidence"))
            if debug.get("missing_polygon_candidate_confidence") is not None
            else None
        ),
        "candidate_recommended_action": (
            str(debug.get("missing_polygon_candidate_recommended_action"))
            if debug.get("missing_polygon_candidate_recommended_action") is not None
            else None
        ),
    }


def _fallback_metric_fields(
    debug: dict[str, Any],
    *,
    reason_code: Any,
) -> dict[str, Any]:
    fallback_source = debug.get("missing_polygon_fallback_source")
    fallback_reason = debug.get("missing_polygon_fallback_reason")
    none_reason = debug.get("missing_polygon_none_reason")
    return {
        "reason_code": str(reason_code) if reason_code is not None else None,
        "fallback_source": str(fallback_source) if fallback_source is not None else None,
        "fallback_reason": str(fallback_reason) if fallback_reason is not None else None,
        "fallback_global_available": bool(
            debug.get("missing_polygon_fallback_global_available")
        ),
        "fallback_local_global_disagrees": bool(
            debug.get("missing_polygon_fallback_local_global_disagrees")
        ),
        "fallback_local_global_area_score": _float_or_none(
            debug.get("missing_polygon_fallback_local_global_area_score")
        ),
        "fallback_local_global_center_factor": _float_or_none(
            debug.get("missing_polygon_fallback_local_global_center_factor")
        ),
        "fallback_slot_feature_support": _int_debug(
            debug.get("missing_polygon_fallback_slot_feature_support")
        ),
        "fallback_slot_feature_total": _int_debug(
            debug.get("missing_polygon_fallback_slot_feature_total")
        ),
        "selective_hidden_release": bool(
            debug.get("missing_polygon_selective_hidden_release")
        ),
        "selective_hidden_release_source": (
            str(debug.get("missing_polygon_selective_hidden_release_source"))
            if debug.get("missing_polygon_selective_hidden_release_source") is not None
            else None
        ),
        "selective_hidden_release_hidden_reason": (
            str(debug.get("missing_polygon_selective_hidden_release_hidden_reason"))
            if debug.get("missing_polygon_selective_hidden_release_hidden_reason") is not None
            else None
        ),
        "selective_hidden_release_rejected": bool(
            debug.get("missing_polygon_selective_hidden_release_rejected")
        ),
        "selective_hidden_release_reject_reason": (
            str(debug.get("missing_polygon_selective_hidden_release_reject_reason"))
            if debug.get("missing_polygon_selective_hidden_release_reject_reason") is not None
            else None
        ),
        "selective_hidden_release_slot_support": _int_debug(
            debug.get("missing_polygon_selective_hidden_release_slot_support")
        ),
        "selective_hidden_release_slot_total": _int_debug(
            debug.get("missing_polygon_selective_hidden_release_slot_total")
        ),
        "selective_hidden_release_slot_ratio": _float_or_none(
            debug.get("missing_polygon_selective_hidden_release_slot_ratio")
        ),
        "selective_hidden_release_reference_area_score": _float_or_none(
            debug.get("missing_polygon_selective_hidden_release_reference_area_score")
        ),
        "selective_hidden_release_center_factor": _float_or_none(
            debug.get("missing_polygon_selective_hidden_release_center_factor")
        ),
        "selective_hidden_release_max_other_overlap": _float_or_none(
            debug.get("missing_polygon_selective_hidden_release_max_other_overlap")
        ),
        **_projection_candidate_registry_metric_fields(debug),
        "global_translation_rescue_attempted": bool(
            debug.get("missing_polygon_global_translation_rescue_attempted")
        ),
        "global_translation_rescue_accepted": bool(
            debug.get("missing_polygon_global_translation_rescue_accepted")
        ),
        "global_translation_rescue_reject_reason": (
            str(debug.get("missing_polygon_global_translation_rescue_reject_reason"))
            if debug.get("missing_polygon_global_translation_rescue_reject_reason") is not None
            else None
        ),
        "global_translation_rescue_local_point_count": _int_debug(
            debug.get("missing_polygon_global_translation_rescue_local_point_count")
        ),
        "global_translation_rescue_min_support": _int_debug(
            debug.get("missing_polygon_global_translation_rescue_min_support")
        ),
        "global_translation_rescue_candidate_count": _int_debug(
            debug.get("missing_polygon_global_translation_rescue_candidate_count")
        ),
        "global_translation_rescue_inlier_count": _int_debug(
            debug.get("missing_polygon_global_translation_rescue_inlier_count")
        ),
        "global_translation_rescue_inlier_ratio": _float_or_none(
            debug.get("missing_polygon_global_translation_rescue_inlier_ratio")
        ),
        "global_translation_rescue_median_error": _float_or_none(
            debug.get("missing_polygon_global_translation_rescue_median_error")
        ),
        "global_translation_rescue_shift_factor": _float_or_none(
            debug.get("missing_polygon_global_translation_rescue_shift_factor")
        ),
        "global_translation_rescue_context_spread": _float_or_none(
            debug.get("missing_polygon_global_translation_rescue_context_spread")
        ),
        "global_translation_rescue_search_containment": _float_or_none(
            debug.get("missing_polygon_global_translation_rescue_search_containment")
        ),
        "global_translation_rescue_local_area_score": _float_or_none(
            debug.get("missing_polygon_global_translation_rescue_local_area_score")
        ),
        "global_translation_rescue_local_center_factor": _float_or_none(
            debug.get("missing_polygon_global_translation_rescue_local_center_factor")
        ),
        "global_translation_rescue_slot_feature_support": _int_debug(
            debug.get("missing_polygon_global_translation_rescue_slot_feature_support")
        ),
        "global_translation_rescue_slot_feature_total": _int_debug(
            debug.get("missing_polygon_global_translation_rescue_slot_feature_total")
        ),
        "none_reason": str(none_reason) if none_reason is not None else None,
        "none_has_homography": bool(debug.get("missing_polygon_none_has_homography")),
        "none_projected_point_count": _int_debug(
            debug.get("missing_polygon_none_projected_point_count")
        ),
    }

def _anchor_release_metric_fields(debug: dict[str, Any]) -> dict[str, Any]:
    reject_reason = debug.get("missing_polygon_anchor_release_reject_reason")
    return {
        "anchor_release_attempted": bool(
            debug.get("missing_polygon_anchor_release_attempted")
        ),
        "anchor_release_reject_reason": (
            str(reject_reason) if reject_reason is not None else None
        ),
        "anchor_release_candidate_count": _int_debug(
            debug.get("missing_polygon_anchor_release_candidate_count")
        ),
        "anchor_release_neighbor_candidate_count": _int_debug(
            debug.get("missing_polygon_anchor_release_neighbor_candidate_count")
        ),
        "anchor_release_selected_candidate_count": _int_debug(
            debug.get("missing_polygon_anchor_release_selected_candidate_count")
        ),
        "anchor_release_total_anchors": _int_debug(
            debug.get("missing_polygon_anchor_release_total_anchors")
        ),
        "anchor_release_inliers": _int_debug(
            debug.get("missing_polygon_anchor_release_inliers")
        ),
        "anchor_release_inlier_ratio": _float_or_none(
            debug.get("missing_polygon_anchor_release_inlier_ratio")
        ),
        "anchor_release_median_error": _float_or_none(
            debug.get("missing_polygon_anchor_release_median_error")
        ),
        "anchor_release_dispersion": _float_or_none(
            debug.get("missing_polygon_anchor_release_dispersion")
        ),
        "anchor_release_shift_factor": _float_or_none(
            debug.get("missing_polygon_anchor_release_shift_factor")
        ),
        "anchor_release_build_expected_count": _int_debug(
            debug.get("missing_polygon_anchor_release_build_expected_count")
        ),
        "anchor_release_build_attempt_count": _int_debug(
            debug.get("missing_polygon_anchor_release_build_attempt_count")
        ),
        "anchor_release_build_built_count": _int_debug(
            debug.get("missing_polygon_anchor_release_build_built_count")
        ),
        "anchor_release_build_source_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_build_source_counts")
        ),
        "anchor_release_build_reject_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_build_reject_counts")
        ),
        "anchor_release_build_probe_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_build_probe_counts")
        ),
        "anchor_release_runtime_trusted_anchor_count": _int_debug(
            debug.get("missing_polygon_anchor_release_runtime_trusted_anchor_count")
        ),
        "anchor_release_runtime_trusted_anchor_source_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_runtime_trusted_anchor_source_counts")
        ),
        "anchor_release_runtime_anchor_before_current_count": _int_debug(
            debug.get("missing_polygon_anchor_release_runtime_anchor_before_current_count")
        ),
        "anchor_release_runtime_anchor_after_current_count": _int_debug(
            debug.get("missing_polygon_anchor_release_runtime_anchor_after_current_count")
        ),
        "anchor_release_runtime_processed_expected_count": _int_debug(
            debug.get("missing_polygon_anchor_release_runtime_processed_expected_count")
        ),
        "anchor_release_runtime_future_expected_count": _int_debug(
            debug.get("missing_polygon_anchor_release_runtime_future_expected_count")
        ),
        "anchor_release_runtime_built_count": _int_debug(
            debug.get("missing_polygon_anchor_release_runtime_built_count")
        ),
        "anchor_release_runtime_source_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_runtime_source_counts")
        ),
        "anchor_release_runtime_reject_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_runtime_reject_counts")
        ),
        "anchor_release_runtime_probe_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_runtime_probe_counts")
        ),
        "anchor_release_final_trusted_anchor_count": _int_debug(
            debug.get("missing_polygon_anchor_release_final_trusted_anchor_count")
        ),
        "anchor_release_final_trusted_anchor_source_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_final_trusted_anchor_source_counts")
        ),
        "anchor_release_final_anchor_before_current_count": _int_debug(
            debug.get("missing_polygon_anchor_release_final_anchor_before_current_count")
        ),
        "anchor_release_final_anchor_after_current_count": _int_debug(
            debug.get("missing_polygon_anchor_release_final_anchor_after_current_count")
        ),
        "anchor_release_final_resolved_anchor_count": _int_debug(
            debug.get("missing_polygon_anchor_release_final_resolved_anchor_count")
        ),
        "anchor_release_final_resolved_anchor_source_counts": _dict_int_debug(
            debug.get("missing_polygon_anchor_release_final_resolved_anchor_source_counts")
        ),
    }



def _run_real_yolo_synthetic_pipeline(*, args: argparse.Namespace, output_dir: Path) -> int:
    """Train/load a real YOLO-seg model and evaluate it on synthetic inspection scenes.

    The older report-only YOLO fixtures answer only a theoretical question.  This
    path is intentionally heavier: it exports real images/labels, trains or loads
    Ultralytics YOLO-seg, runs real inference, keeps the existing LightGlue
    transfer as the baseline, and evaluates whether high-confidence YOLO masks
    can be trusted as geometry anchors for rescuing failed polygon transfers.

    Important: a missing YOLO detection is never treated as a missing object in
    this diagnostic.  It only means that YOLO did not provide an anchor.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_dir / "lightglue_baseline"
    baseline_summary = _run_lightglue_synthetic_suite(args=args, output_dir=baseline_dir)

    dataset_dir = output_dir / "real_yolo_synthetic_dataset"
    dataset_summary = _export_real_yolo_synthetic_dataset(
        dataset_dir=dataset_dir,
        train_cases=args.real_yolo_train_cases,
        val_cases=args.real_yolo_val_cases,
        test_cases=args.real_yolo_test_cases or args.cases,
        seed=args.seed,
        profile=args.profile,
        multi_object_cases=args.multi_object_cases,
    )

    if args.real_yolo_model_path:
        model_path = Path(args.real_yolo_model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        train_result = {"source": "provided", "model_path": str(model_path)}
    else:
        train_result = _train_real_yolo_synthetic_model(
            dataset_yaml=dataset_dir / "data.yaml",
            project_dir=output_dir / "real_yolo_train",
            base_model=args.real_yolo_base_model,
            epochs=args.real_yolo_epochs,
            imgsz=args.real_yolo_imgsz,
            batch=args.real_yolo_batch,
            device=args.real_yolo_device,
            keep_training_dir=args.real_yolo_keep_training_dir,
        )
        model_path = Path(str(train_result["model_path"]))

    evaluation = _evaluate_real_yolo_synthetic_model(
        model_path=model_path,
        output_dir=output_dir / "real_yolo_eval",
        seed=args.seed,
        profile=args.profile,
        test_cases=args.real_yolo_test_cases or args.cases,
        multi_object_cases=args.multi_object_cases,
        conf=args.real_yolo_conf,
        iou=args.real_yolo_iou,
        imgsz=args.real_yolo_imgsz,
        device=args.real_yolo_device,
        baseline_results_path=baseline_dir / "results.json",
        anchor_policy=_build_real_yolo_anchor_policy(args),
    )

    comparison = {
        "mode": "real_yolo_anchor_rescue_e2e",
        "baseline_lightglue": baseline_summary,
        "dataset": dataset_summary,
        "training": train_result,
        "real_yolo_anchor_rescue": evaluation,
        "yolo_anchor_rescue": evaluation.get("yolo_anchor_rescue", {}),
        "decision_hint": _real_yolo_decision_hint(evaluation),
    }
    _write_json(output_dir / "real_yolo_synthetic_summary.json", comparison)

    print("real_yolo_anchor_rescue_e2e finished")
    print(f"baseline={baseline_dir / 'summary.json'}")
    print(f"dataset={dataset_dir}")
    print(f"model={model_path}")
    print(f"eval={output_dir / 'real_yolo_eval' / 'summary.json'}")
    print(f"summary={output_dir / 'real_yolo_synthetic_summary.json'}")

    anchor_rescue = evaluation.get("yolo_anchor_rescue", {})
    if args.fail_on_fail and int(anchor_rescue.get("dangerous_missing_anchors") or 0) > 0:
        return 1
    return 0


def _export_real_yolo_synthetic_dataset(
    *,
    dataset_dir: Path,
    train_cases: int,
    val_cases: int,
    test_cases: int,
    seed: int,
    profile: str,
    multi_object_cases: int | None,
) -> dict[str, Any]:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_specs = {
        "train": (max(1, int(train_cases)), seed + 120000, "brutal"),
        "val": (max(1, int(val_cases)), seed + 220000, "brutal"),
        "test": (max(1, int(test_cases)), seed, profile),
    }
    summary: dict[str, Any] = {
        "path": str(dataset_dir),
        "classes": list(_OBJECT_SHAPE_NAMES),
        "splits": {},
    }

    for split, (case_count, split_seed, split_profile) in split_specs.items():
        scenes = _real_yolo_synthetic_scenes(
            total=case_count,
            seed=split_seed,
            profile=split_profile,
            multi_object_cases=(
                _real_yolo_default_dataset_multi_cases(case_count, split)
                if split != "test"
                else _resolve_multi_object_cases(
                    requested=multi_object_cases,
                    profile=profile,
                    base_count=case_count,
                )
            ),
        )
        split_summary = _write_real_yolo_dataset_split(
            dataset_dir=dataset_dir,
            split=split,
            scenes=scenes,
            include_missing_negatives=True,
        )
        summary["splits"][split] = split_summary

    _write_real_yolo_data_yaml(dataset_dir)
    _write_json(dataset_dir / "dataset_summary.json", summary)
    return summary


def _real_yolo_default_dataset_multi_cases(case_count: int, split: str) -> int:
    if split == "train":
        return max(8, int(case_count * 0.45))
    if split == "val":
        return max(4, int(case_count * 0.35))
    return 0


def _real_yolo_synthetic_scenes(
    *,
    total: int,
    seed: int,
    profile: str,
    multi_object_cases: int,
) -> list[_SyntheticScene | _SyntheticMultiScene]:
    scenarios = _build_scenarios(total=total, seed=seed, profile=profile)
    scenes: list[_SyntheticScene | _SyntheticMultiScene] = [
        _build_scene(scenario) for scenario in scenarios
    ]
    if multi_object_cases > 0:
        multi_scenarios = _build_multi_scenarios(
            total=multi_object_cases,
            seed=seed,
            profile=profile,
        )
        scenes.extend(_build_multi_scene(scenario) for scenario in multi_scenarios)
    return scenes


def _write_real_yolo_dataset_split(
    *,
    dataset_dir: Path,
    split: str,
    scenes: Sequence[_SyntheticScene | _SyntheticMultiScene],
    include_missing_negatives: bool,
) -> dict[str, Any]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    image_count = 0
    object_label_count = 0
    distractor_label_count = 0
    empty_label_count = 0

    for index, scene in enumerate(scenes, start=1):
        present_image = _real_yolo_present_target_image(scene)
        present_labels, present_distractors = _real_yolo_scene_labels(
            scene,
            include_expected=True,
            include_distractors=True,
        )
        image_name = f"{split}_{index:05d}_present.png"
        cv2.imwrite(str(image_dir / image_name), present_image)
        _write_yolo_seg_label_file(label_dir / image_name.replace(".png", ".txt"), present_labels)
        image_count += 1
        object_label_count += len(present_labels) - present_distractors
        distractor_label_count += present_distractors

        if include_missing_negatives:
            missing_labels, missing_distractors = _real_yolo_scene_labels(
                scene,
                include_expected=False,
                include_distractors=True,
            )
            missing_name = f"{split}_{index:05d}_missing_negative.png"
            cv2.imwrite(str(image_dir / missing_name), scene.target_image)
            _write_yolo_seg_label_file(label_dir / missing_name.replace(".png", ".txt"), missing_labels)
            image_count += 1
            distractor_label_count += missing_distractors
            if not missing_labels:
                empty_label_count += 1

    return {
        "scenes": len(scenes),
        "images": image_count,
        "expected_object_labels": object_label_count,
        "distractor_labels": distractor_label_count,
        "empty_label_files": empty_label_count,
    }


def _write_real_yolo_data_yaml(dataset_dir: Path) -> None:
    names_lines = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(_OBJECT_SHAPE_NAMES)
    )
    content = (
        f"path: {dataset_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names_lines}\n"
    )
    (dataset_dir / "data.yaml").write_text(content, encoding="utf-8")


def _write_yolo_seg_label_file(path: Path, labels: Sequence[tuple[str, PolygonPoints]]) -> None:
    lines: list[str] = []
    width, height = _CANVAS_SIZE
    for class_name, polygon in labels:
        class_id = _real_yolo_class_id(class_name)
        normalized = _normalize_yolo_polygon(polygon, width=width, height=height)
        if len(normalized) < 6:
            continue
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _real_yolo_class_id(class_name: str) -> int:
    try:
        return list(_OBJECT_SHAPE_NAMES).index(class_name)
    except ValueError:
        return list(_OBJECT_SHAPE_NAMES).index(_DEFAULT_OBJECT_SHAPE)


def _normalize_yolo_polygon(
    polygon: PolygonPoints,
    *,
    width: int,
    height: int,
) -> list[float]:
    shape = _safe_shape(polygon)
    if shape is None or shape.area < 4.0:
        return []
    values: list[float] = []
    for x, y in polygon:
        nx = min(max(float(x) / float(width), 0.0), 1.0)
        ny = min(max(float(y) / float(height), 0.0), 1.0)
        values.extend([nx, ny])
    return values


def _real_yolo_scene_labels(
    scene: _SyntheticScene | _SyntheticMultiScene,
    *,
    include_expected: bool,
    include_distractors: bool,
) -> tuple[list[tuple[str, PolygonPoints]], int]:
    labels: list[tuple[str, PolygonPoints]] = []
    distractor_count = 0
    if isinstance(scene, _SyntheticScene):
        if include_expected:
            labels.append((scene.scenario.object_shape, scene.ground_truth_polygon))
        if include_distractors:
            for polygon in scene.distractor_polygons:
                labels.append((scene.scenario.object_shape, polygon))
                distractor_count += 1
        return labels, distractor_count

    for obj in scene.objects:
        if include_expected:
            labels.append((obj.object_shape, obj.ground_truth_polygon))
        if include_distractors:
            for polygon in obj.distractor_polygons:
                labels.append((obj.object_shape, polygon))
                distractor_count += 1
    return labels, distractor_count


def _real_yolo_present_target_image(scene: _SyntheticScene | _SyntheticMultiScene) -> np.ndarray:
    width, height = _CANVAS_SIZE
    image = _base_scene(width=width, height=height)
    homography = scene.true_homography
    _draw_stable_context(image, homography=homography)

    if isinstance(scene, _SyntheticScene):
        for polygon in scene.distractor_polygons:
            _draw_polygon(image, polygon, (42, 130, 230), fill=True, alpha=0.68)
            _draw_polygon(image, polygon, _DISTRACTOR_COLOR, thickness=2)
        _draw_polygon(image, scene.ground_truth_polygon, (70, 120, 235), fill=True, alpha=0.78)
        _draw_polygon(image, scene.ground_truth_polygon, (20, 40, 160), thickness=2)
        if scene.scenario.occluder:
            occluder_poly = project_polygon(
                [[266.0, 156.0], [430.0, 180.0], [408.0, 264.0], [252.0, 238.0]],
                homography,
            )
            _draw_polygon(image, occluder_poly, (90, 92, 100), fill=True, alpha=0.88)
            _draw_polygon(image, occluder_poly, (180, 180, 190), thickness=2)
        return image

    for obj in scene.objects:
        for polygon in obj.distractor_polygons:
            _draw_polygon(image, polygon, (42, 130, 230), fill=True, alpha=0.60)
            _draw_polygon(image, polygon, _DISTRACTOR_COLOR, thickness=2)
    for obj in scene.objects:
        color = _object_color(obj.index)
        _draw_polygon(image, obj.ground_truth_polygon, color, fill=True, alpha=0.76)
        _draw_polygon(image, obj.ground_truth_polygon, (20, 40, 160), thickness=2)
    if scene.scenario.occluder:
        occluder_poly = project_polygon(
            [[160.0, 140.0], [560.0, 170.0], [532.0, 344.0], [135.0, 312.0]],
            homography,
        )
        _draw_polygon(image, occluder_poly, (88, 90, 98), fill=True, alpha=0.72)
        _draw_polygon(image, occluder_poly, (180, 180, 190), thickness=2)
    return image


def _train_real_yolo_synthetic_model(
    *,
    dataset_yaml: Path,
    project_dir: Path,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    keep_training_dir: bool,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Ultralytics is required for --real-yolo-synthetic. Install with: "
            "python -m pip install ultralytics"
        ) from exc

    if project_dir.exists() and not keep_training_dir:
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_model)
    train_kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": str(project_dir),
        "name": "synthetic_yolo_seg",
        "exist_ok": True,
        "task": "segment",
        "verbose": True,
    }
    device_arg = _real_yolo_device_arg(device)
    if device_arg is not None:
        train_kwargs["device"] = device_arg
    model.train(**train_kwargs)

    best_path = project_dir / "synthetic_yolo_seg" / "weights" / "best.pt"
    last_path = project_dir / "synthetic_yolo_seg" / "weights" / "last.pt"
    model_path = best_path if best_path.exists() else last_path
    if not model_path.exists():
        raise FileNotFoundError("Ultralytics training finished but no best.pt/last.pt was found")
    return {
        "source": "trained",
        "base_model": base_model,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "device": device,
        "model_path": str(model_path),
    }


def _real_yolo_device_arg(device: str) -> str | None:
    normalized = str(device or "auto").strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    if normalized != "cpu" and _real_yolo_cuda_device_requested(normalized):
        try:
            import torch
        except Exception:
            return "cpu"
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            return "cpu"
    return str(device)


def _real_yolo_cuda_device_requested(normalized_device: str) -> bool:
    if normalized_device.startswith("cuda"):
        return True
    return all(part.strip().isdigit() for part in normalized_device.split(","))


def _evaluate_real_yolo_synthetic_model(
    *,
    model_path: Path,
    output_dir: Path,
    seed: int,
    profile: str,
    test_cases: int,
    multi_object_cases: int | None,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    baseline_results_path: Path,
    anchor_policy: dict[str, float],
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Ultralytics is required for real YOLO evaluation") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scenes = _real_yolo_synthetic_scenes(
        total=test_cases,
        seed=seed,
        profile=profile,
        multi_object_cases=_resolve_multi_object_cases(
            requested=multi_object_cases,
            profile=profile,
            base_count=max(1, int(test_cases)),
        ),
    )
    model = YOLO(str(model_path))
    names = _real_yolo_model_names(model)
    baseline_metric_map = _load_real_yolo_baseline_metric_map(baseline_results_path)

    present_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    anchor_present_rows: list[dict[str, Any]] = []
    anchor_missing_rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, start=1):
        present_image = _real_yolo_present_target_image(scene)
        present_path = images_dir / f"case_{index:04d}_present.png"
        cv2.imwrite(str(present_path), present_image)
        present_detections = _predict_real_yolo_detections(
            model=model,
            image_path=present_path,
            names=names,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        present_rows.extend(
            _evaluate_real_yolo_scene(
                scene=scene,
                detections=present_detections,
                target_image=present_image,
                case_index=index,
                variant="present",
            )
        )
        anchor_present_rows.extend(
            _evaluate_real_yolo_anchor_scene(
                scene=scene,
                detections=present_detections,
                case_index=index,
                variant="present",
                baseline_metric_map=baseline_metric_map,
                anchor_policy=anchor_policy,
            )
        )

        missing_path = images_dir / f"case_{index:04d}_missing.png"
        cv2.imwrite(str(missing_path), scene.target_image)
        missing_detections = _predict_real_yolo_detections(
            model=model,
            image_path=missing_path,
            names=names,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )
        missing_rows.extend(
            _evaluate_real_yolo_scene(
                scene=scene,
                detections=missing_detections,
                target_image=scene.target_image,
                case_index=index,
                variant="missing",
            )
        )
        anchor_missing_rows.extend(
            _evaluate_real_yolo_anchor_scene(
                scene=scene,
                detections=missing_detections,
                case_index=index,
                variant="missing",
                baseline_metric_map=baseline_metric_map,
                anchor_policy=anchor_policy,
            )
        )

    present_summary = _summarize_real_yolo_eval_rows(present_rows, variant="present")
    missing_summary = _summarize_real_yolo_eval_rows(missing_rows, variant="missing")
    anchor_rescue_summary = _summarize_real_yolo_anchor_rescue(
        present_rows=anchor_present_rows,
        missing_rows=anchor_missing_rows,
        baseline_metric_map=baseline_metric_map,
    )
    combined = {
        "total_expected_objects": present_summary["total_expected_objects"] + missing_summary["total_expected_objects"],
        "dangerous_expected_matches": present_summary["dangerous_expected_matches"] + missing_summary["dangerous_expected_matches"],
        "accepted_expected_matches": present_summary["accepted_expected_matches"] + missing_summary["accepted_expected_matches"],
        "missing_expected_objects": present_summary["missing_expected_objects"] + missing_summary["missing_expected_objects"],
    }
    summary = {
        "mode": "real_yolo_anchor_rescue_diagnostic",
        "model_path": str(model_path),
        "classes": names,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "anchor_policy": anchor_policy,
        "yolo_anchor_rescue": anchor_rescue_summary,
        "legacy_global_yolo_matching": {
            "note": (
                "Diagnostic only. These values are the old global YOLO matching metrics; "
                "they are not the anchor-rescue decision and must not be used as release criteria."
            ),
            "present": present_summary,
            "missing": missing_summary,
            "combined": combined,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "present_results.json", present_rows)
    _write_json(output_dir / "missing_results.json", missing_rows)
    _write_json(output_dir / "anchor_present_results.json", anchor_present_rows)
    _write_json(output_dir / "anchor_missing_results.json", anchor_missing_rows)
    _write_real_yolo_eval_html(output_dir / "report.html", summary, anchor_present_rows, anchor_missing_rows)
    return summary



def _build_real_yolo_anchor_policy(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_confidence": float(args.real_yolo_anchor_min_conf),
        "max_center_factor": float(args.real_yolo_anchor_max_center_factor),
        "min_slot_iou": float(args.real_yolo_anchor_min_slot_iou),
        "min_detection_containment": float(args.real_yolo_anchor_min_containment),
        "min_slot_coverage": float(args.real_yolo_anchor_min_coverage),
        "min_slot_area_ratio": float(args.real_yolo_anchor_min_area_ratio),
        "max_slot_area_ratio": float(args.real_yolo_anchor_max_area_ratio),
    }


def _load_real_yolo_baseline_metric_map(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw_results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw_results, list):
        return {}

    mapping: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw_case in raw_results:
        if not isinstance(raw_case, dict):
            continue
        case_index = int(raw_case.get("index") or 0)
        metrics = raw_case.get("object_metrics")
        if not isinstance(metrics, list):
            continue
        for raw_metric in metrics:
            if not isinstance(raw_metric, dict):
                continue
            key = _real_yolo_anchor_key(
                case_index=case_index,
                name=str(raw_metric.get("name") or ""),
                object_shape=str(raw_metric.get("object_shape") or ""),
            )
            mapping[key] = dict(raw_metric)
    return mapping


def _real_yolo_anchor_key(*, case_index: int, name: str, object_shape: str) -> tuple[int, str, str]:
    return (int(case_index), str(name), str(object_shape))


def _evaluate_real_yolo_anchor_scene(
    *,
    scene: _SyntheticScene | _SyntheticMultiScene,
    detections: list[YoloDetection],
    case_index: int,
    variant: str,
    baseline_metric_map: dict[tuple[int, str, str], dict[str, Any]],
    anchor_policy: dict[str, float],
) -> list[dict[str, Any]]:
    """Evaluate YOLO as a geometry-anchor source, not as a missing detector.

    In this diagnostic, "no YOLO anchor" never means "object missing".  It only
    means that the detector did not provide a reliable geometry anchor for this
    expected object.  Missing scenes are used only as a safety check: a trusted
    anchor on an intentionally removed target is dangerous and must be tuned out
    before anchors can influence runtime geometry.
    """
    expected, annotation_to_object = _real_yolo_expected_segments(scene)
    assignments = _select_real_yolo_trusted_anchor_assignments(
        expected=expected,
        detections=detections,
        homography=scene.approximate_homography,
        anchor_policy=anchor_policy,
    )
    anchor_transform = _real_yolo_anchor_translation_from_assignments(
        expected=expected,
        detections=detections,
        assignments=assignments,
        homography=scene.approximate_homography,
    )

    rows: list[dict[str, Any]] = []
    for expected_index, expected_item in enumerate(expected):
        obj = annotation_to_object[expected_item.annotation_id]
        key = _real_yolo_anchor_key(
            case_index=case_index,
            name=str(obj["name"]),
            object_shape=str(obj["object_shape"]),
        )
        baseline_metric = baseline_metric_map.get(key, {})
        baseline_passed = bool(baseline_metric.get("passed"))
        baseline_iou = _float_or_none(baseline_metric.get("iou"))
        baseline_center_drift = _float_or_none(baseline_metric.get("center_drift_px"))
        baseline_area_ratio = _float_or_none(baseline_metric.get("area_ratio"))
        baseline_projection = str(baseline_metric.get("projection") or "unknown")

        assignment = assignments.get(expected_index)
        detection = detections[assignment["detection_index"]] if assignment is not None else None
        detected_polygon = _real_yolo_detection_polygon(detection) if detection is not None else None
        direct_iou = _polygon_iou(detected_polygon, obj["ground_truth_polygon"])
        direct_center_drift = _polygon_center_distance(detected_polygon, obj["ground_truth_polygon"])
        direct_area_ratio = _polygon_area_ratio(detected_polygon, obj["ground_truth_polygon"])
        direct_rescue = (
            variant == "present"
            and not baseline_passed
            and detection is not None
            and _real_yolo_present_match_ok(
                object_shape=str(obj["object_shape"]),
                iou=direct_iou,
                center_drift=direct_center_drift,
                area_ratio=direct_area_ratio,
            )
        )

        transformed_polygon = None
        transformed_iou = 0.0
        transformed_center_drift = float("inf")
        transformed_area_ratio = 0.0
        transform_rescue = False
        if anchor_transform is not None and not baseline_passed and assignment is None:
            transformed_polygon = _translate_polygon(
                project_polygon(expected_item.reference_polygon, scene.approximate_homography),
                anchor_transform,
            )
            transformed_iou = _polygon_iou(transformed_polygon, obj["ground_truth_polygon"])
            transformed_center_drift = _polygon_center_distance(transformed_polygon, obj["ground_truth_polygon"])
            transformed_area_ratio = _polygon_area_ratio(transformed_polygon, obj["ground_truth_polygon"])
            transform_rescue = (
                variant == "present"
                and _real_yolo_present_match_ok(
                    object_shape=str(obj["object_shape"]),
                    iou=transformed_iou,
                    center_drift=transformed_center_drift,
                    area_ratio=transformed_area_ratio,
                )
            )

        dangerous_anchor = variant == "missing" and detection is not None
        applied_source = "baseline_geometry" if baseline_passed else "unresolved_baseline_failure"
        applied_iou = baseline_iou if baseline_iou is not None else 0.0
        applied_center_drift = (
            baseline_center_drift if baseline_center_drift is not None else float("inf")
        )
        applied_area_ratio = baseline_area_ratio if baseline_area_ratio is not None else 0.0
        applied_rescue = False

        if variant == "present" and not baseline_passed:
            if direct_rescue:
                applied_source = "trusted_yolo_mask"
                applied_iou = float(direct_iou)
                applied_center_drift = float(direct_center_drift)
                applied_area_ratio = float(direct_area_ratio)
                applied_rescue = True
            elif transform_rescue:
                applied_source = "trusted_anchor_transform"
                applied_iou = float(transformed_iou)
                applied_center_drift = float(transformed_center_drift)
                applied_area_ratio = float(transformed_area_ratio)
                applied_rescue = True

        applied_passed = baseline_passed or applied_rescue
        applied_dangerous_missing_change = variant == "missing" and detection is not None

        rows.append(
            {
                "case_index": case_index,
                "variant": variant,
                "name": obj["name"],
                "object_shape": obj["object_shape"],
                "class_key": expected_item.class_key,
                "baseline_passed": baseline_passed,
                "baseline_projection": baseline_projection,
                "baseline_iou": baseline_iou,
                "baseline_center_drift_px": baseline_center_drift,
                "baseline_area_ratio": baseline_area_ratio,
                "anchor_status": "trusted_anchor" if detection is not None else "no_anchor",
                "anchor_confidence": float(detection.confidence) if detection is not None else None,
                "anchor_detection_index": assignment.get("detection_index") if assignment is not None else None,
                "anchor_score": assignment.get("score") if assignment is not None else None,
                "anchor_slot_iou": assignment.get("slot_iou") if assignment is not None else None,
                "anchor_center_factor": assignment.get("center_factor") if assignment is not None else None,
                "anchor_detection_containment": assignment.get("detection_containment") if assignment is not None else None,
                "anchor_slot_coverage": assignment.get("slot_coverage") if assignment is not None else None,
                "anchor_slot_area_ratio": assignment.get("slot_area_ratio") if assignment is not None else None,
                "direct_anchor_iou": float(direct_iou),
                "direct_anchor_center_drift_px": float(direct_center_drift),
                "direct_anchor_area_ratio": float(direct_area_ratio),
                "direct_anchor_rescue": bool(direct_rescue),
                "anchor_transform_available": anchor_transform is not None,
                "anchor_transform_rescue": bool(transform_rescue),
                "anchor_transform_iou": float(transformed_iou),
                "anchor_transform_center_drift_px": float(transformed_center_drift),
                "anchor_transform_area_ratio": float(transformed_area_ratio),
                "dangerous_missing_anchor": bool(dangerous_anchor),
                "applied_geometry_source": applied_source,
                "applied_geometry_iou": float(applied_iou),
                "applied_geometry_center_drift_px": float(applied_center_drift),
                "applied_geometry_area_ratio": float(applied_area_ratio),
                "applied_geometry_passed": bool(applied_passed),
                "applied_rescued_baseline_failure": bool(applied_rescue),
                "applied_dangerous_missing_change": bool(applied_dangerous_missing_change),
                "semantics": "YOLO anchor helps geometry only; no_anchor is not a missing decision.",
            }
        )
    return rows


def _select_real_yolo_trusted_anchor_assignments(
    *,
    expected: Sequence[ExpectedSegment],
    detections: Sequence[YoloDetection],
    homography: np.ndarray,
    anchor_policy: dict[str, float],
) -> dict[int, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for expected_index, expected_item in enumerate(expected):
        slot_polygon = project_polygon(expected_item.reference_polygon, homography)
        for detection_index, detection in enumerate(detections):
            if detection.class_key != expected_item.class_key:
                continue
            candidate = _score_real_yolo_anchor_candidate(
                expected_index=expected_index,
                detection_index=detection_index,
                slot_polygon=slot_polygon,
                detection=detection,
                anchor_policy=anchor_policy,
            )
            if candidate is not None:
                candidates.append(candidate)

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    used_expected: set[int] = set()
    used_detection: set[int] = set()
    assignments: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        expected_index = int(candidate["expected_index"])
        detection_index = int(candidate["detection_index"])
        if expected_index in used_expected or detection_index in used_detection:
            continue
        assignments[expected_index] = candidate
        used_expected.add(expected_index)
        used_detection.add(detection_index)
    return assignments


def _score_real_yolo_anchor_candidate(
    *,
    expected_index: int,
    detection_index: int,
    slot_polygon: PolygonPoints | None,
    detection: YoloDetection,
    anchor_policy: dict[str, float],
) -> dict[str, Any] | None:
    if float(detection.confidence) < float(anchor_policy["min_confidence"]):
        return None

    detection_polygon = _real_yolo_detection_polygon(detection)
    slot_area = _polygon_area(slot_polygon)
    detection_area = _polygon_area(detection_polygon)
    if slot_area <= 0.0 or detection_area <= 0.0:
        return None

    intersection_area = _polygon_intersection_area(slot_polygon, detection_polygon)
    slot_iou = _polygon_iou(slot_polygon, detection_polygon)
    center_factor = _polygon_center_distance(slot_polygon, detection_polygon) / math.sqrt(max(slot_area, 1.0))
    detection_containment = intersection_area / detection_area
    slot_coverage = intersection_area / slot_area
    slot_area_ratio = detection_area / slot_area

    if center_factor > float(anchor_policy["max_center_factor"]):
        return None
    if slot_iou < float(anchor_policy["min_slot_iou"]):
        return None
    if detection_containment < float(anchor_policy["min_detection_containment"]):
        return None
    if slot_coverage < float(anchor_policy["min_slot_coverage"]):
        return None

    # Trusted anchor means expected geometry and factual YOLO mask agreed.
    # A high-confidence same-class detection that only partially fills the
    # expected slot is still just an unmatched detection, not a geometry anchor.
    if slot_area_ratio < float(anchor_policy["min_slot_area_ratio"]):
        return None
    if slot_area_ratio > float(anchor_policy["max_slot_area_ratio"]):
        return None

    score = (
        float(detection.confidence) * 2.0
        + slot_iou * 2.0
        + detection_containment
        + slot_coverage
        - center_factor
    )
    return {
        "expected_index": int(expected_index),
        "detection_index": int(detection_index),
        "score": float(score),
        "slot_iou": float(slot_iou),
        "center_factor": float(center_factor),
        "detection_containment": float(detection_containment),
        "slot_coverage": float(slot_coverage),
        "slot_area_ratio": float(slot_area_ratio),
    }


def _real_yolo_anchor_translation_from_assignments(
    *,
    expected: Sequence[ExpectedSegment],
    detections: Sequence[YoloDetection],
    assignments: dict[int, dict[str, Any]],
    homography: np.ndarray,
) -> tuple[float, float] | None:
    shifts: list[np.ndarray] = []
    for expected_index, assignment in assignments.items():
        slot_polygon = project_polygon(expected[expected_index].reference_polygon, homography)
        slot_center = _polygon_center(slot_polygon)
        detection_polygon = _real_yolo_detection_polygon(detections[int(assignment["detection_index"])])
        detection_center = _polygon_center(detection_polygon)
        if slot_center is None or detection_center is None:
            continue
        shifts.append(detection_center - slot_center)
    if not shifts:
        return None
    median_shift = np.median(np.asarray(shifts, dtype=np.float32), axis=0)
    return (float(median_shift[0]), float(median_shift[1]))


def _translate_polygon(polygon: PolygonPoints | None, shift: tuple[float, float]) -> PolygonPoints | None:
    if not polygon:
        return None
    dx, dy = shift
    return [[float(x) + dx, float(y) + dy] for x, y in polygon]


def _real_yolo_detection_polygon(detection: YoloDetection | None) -> PolygonPoints | None:
    if detection is None:
        return None
    if detection.polygon:
        return [[float(point[0]), float(point[1])] for point in detection.polygon]
    bbox = detection.bbox or {}
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    w = float(bbox.get("w", 0.0))
    h = float(bbox.get("h", 0.0))
    if w <= 0.0 or h <= 0.0:
        return None
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _polygon_area(polygon: PolygonPoints | None) -> float:
    shape = _safe_shape(polygon)
    return float(shape.area) if shape is not None else 0.0


def _polygon_intersection_area(a: PolygonPoints | None, b: PolygonPoints | None) -> float:
    shape_a = _safe_shape(a)
    shape_b = _safe_shape(b)
    if shape_a is None or shape_b is None:
        return 0.0
    try:
        return float(shape_a.intersection(shape_b).area)
    except GEOSException:
        return 0.0


def _summarize_real_yolo_anchor_rescue(
    *,
    present_rows: Sequence[dict[str, Any]],
    missing_rows: Sequence[dict[str, Any]],
    baseline_metric_map: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    baseline_failed = [metric for metric in baseline_metric_map.values() if not bool(metric.get("passed"))]
    baseline_failed_keys = {
        _real_yolo_anchor_key(
            case_index=int(row.get("case_index") or 0),
            name=str(row.get("name") or ""),
            object_shape=str(row.get("object_shape") or ""),
        )
        for row in present_rows
        if not bool(row.get("baseline_passed"))
    }
    present_trusted = [row for row in present_rows if row.get("anchor_status") == "trusted_anchor"]
    missing_trusted = [row for row in missing_rows if row.get("anchor_status") == "trusted_anchor"]
    direct_rescues = [row for row in present_rows if row.get("direct_anchor_rescue")]
    transform_rescues = [row for row in present_rows if row.get("anchor_transform_rescue")]
    rescued_keys = {
        _real_yolo_anchor_key(
            case_index=int(row.get("case_index") or 0),
            name=str(row.get("name") or ""),
            object_shape=str(row.get("object_shape") or ""),
        )
        for row in [*direct_rescues, *transform_rescues]
    }
    dangerous_missing = [row for row in missing_rows if row.get("dangerous_missing_anchor")]
    applied_runtime_shadow = _summarize_real_yolo_applied_anchor_rescue_shadow(
        present_rows=present_rows,
        missing_rows=missing_rows,
        baseline_metric_map=baseline_metric_map,
    )

    total_failed = len(baseline_failed)
    covered_failed = len(baseline_failed_keys)
    rescued_failed = len(rescued_keys)
    remaining_covered = max(0, covered_failed - rescued_failed)
    not_covered = max(0, total_failed - covered_failed)

    return {
        "semantics": "YOLO is evaluated as a trusted-anchor source for geometry rescue, not as a missing detector.",
        "no_anchor_semantics": "no_anchor means detector did not provide geometry help; it is not a missing decision.",
        "release_criteria": "Use dangerous_missing_anchors == 0 and safety_ready_for_anchor_rescue == true. Ignore legacy_global_yolo_matching for release decisions.",
        "baseline_total_objects": len(baseline_metric_map),
        "baseline_failed_objects": total_failed,
        "baseline_failed_objects_seen_in_real_yolo_test": covered_failed,
        "baseline_failed_objects_covered_by_real_yolo_test": covered_failed,
        "baseline_failed_objects_not_covered_by_real_yolo_test": not_covered,
        "coverage_rate_of_baseline_failures": (covered_failed / total_failed * 100.0) if total_failed else 0.0,
        "present_trusted_anchors": len(present_trusted),
        "present_anchor_rate": (len(present_trusted) / len(present_rows) * 100.0) if present_rows else 0.0,
        "baseline_failures_with_trusted_own_anchor": sum(
            1
            for row in present_rows
            if not bool(row.get("baseline_passed")) and row.get("anchor_status") == "trusted_anchor"
        ),
        "direct_yolo_polygon_rescues": len(direct_rescues),
        "anchor_transform_rescues": len(transform_rescues),
        "rescued_baseline_failures": rescued_failed,
        "remaining_baseline_failures_after_anchor_rescue": remaining_covered,
        "remaining_covered_baseline_failures_after_anchor_rescue": remaining_covered,
        "covered_baseline_failure_rescue_rate": (rescued_failed / covered_failed * 100.0) if covered_failed else 0.0,
        "total_baseline_failure_rescue_rate": (rescued_failed / total_failed * 100.0) if total_failed else 0.0,
        "missing_trusted_anchors": len(missing_trusted),
        "dangerous_missing_anchors": len(dangerous_missing),
        "safety_ready_for_anchor_rescue": len(dangerous_missing) == 0,
        "top_direct_rescue_shapes": _count_rows_by_key(direct_rescues, "object_shape"),
        "top_anchor_transform_rescue_shapes": _count_rows_by_key(transform_rescues, "object_shape"),
        "top_dangerous_anchor_shapes": _count_rows_by_key(dangerous_missing, "object_shape"),
        "applied_runtime_shadow": applied_runtime_shadow,
    }


def _summarize_real_yolo_applied_anchor_rescue_shadow(
    *,
    present_rows: Sequence[dict[str, Any]],
    missing_rows: Sequence[dict[str, Any]],
    baseline_metric_map: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    baseline_total = len(baseline_metric_map)
    baseline_failed = [metric for metric in baseline_metric_map.values() if not bool(metric.get("passed"))]
    baseline_failures = len(baseline_failed)
    applied_rescues = [row for row in present_rows if row.get("applied_rescued_baseline_failure")]
    applied_direct = [row for row in applied_rescues if row.get("applied_geometry_source") == "trusted_yolo_mask"]
    applied_transform = [
        row for row in applied_rescues if row.get("applied_geometry_source") == "trusted_anchor_transform"
    ]
    present_baseline_failed = [row for row in present_rows if not bool(row.get("baseline_passed"))]
    present_remaining_failed = [
        row
        for row in present_baseline_failed
        if not bool(row.get("applied_rescued_baseline_failure"))
    ]
    missing_dangerous_changes = [row for row in missing_rows if row.get("applied_dangerous_missing_change")]

    projected_failures_after_shadow = max(0, baseline_failures - len(applied_rescues))
    projected_accuracy_after_shadow = (
        (baseline_total - projected_failures_after_shadow) / baseline_total * 100.0
        if baseline_total
        else 0.0
    )
    baseline_accuracy = (
        (baseline_total - baseline_failures) / baseline_total * 100.0
        if baseline_total
        else 0.0
    )

    return {
        "semantics": (
            "Runtime shadow keeps LightGlue geometry by default and changes only failed baseline "
            "objects that are rescued by a trusted YOLO anchor."
        ),
        "baseline_total_objects": baseline_total,
        "baseline_failed_objects": baseline_failures,
        "present_covered_baseline_failures": len(present_baseline_failed),
        "applied_rescued_baseline_failures": len(applied_rescues),
        "applied_direct_yolo_mask_rescues": len(applied_direct),
        "applied_anchor_transform_rescues": len(applied_transform),
        "present_remaining_covered_failures_after_apply": len(present_remaining_failed),
        "missing_dangerous_applied_changes": len(missing_dangerous_changes),
        "baseline_object_accuracy_before_shadow": baseline_accuracy,
        "projected_object_accuracy_after_shadow": projected_accuracy_after_shadow,
        "projected_object_accuracy_gain": projected_accuracy_after_shadow - baseline_accuracy,
        "runtime_shadow_ready": len(missing_dangerous_changes) == 0,
        "top_applied_rescue_shapes": _count_rows_by_key(applied_rescues, "object_shape"),
        "top_remaining_failure_shapes": _count_rows_by_key(present_remaining_failed, "object_shape"),
        "top_dangerous_applied_shapes": _count_rows_by_key(missing_dangerous_changes, "object_shape"),
    }


def _count_rows_by_key(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

def _real_yolo_model_names(model: Any) -> list[str]:
    raw_names = getattr(model, "names", None)
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(raw_names)]
    if isinstance(raw_names, (list, tuple)):
        return [str(item) for item in raw_names]
    return list(_OBJECT_SHAPE_NAMES)


def _predict_real_yolo_detections(
    *,
    model: Any,
    image_path: Path,
    names: Sequence[str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> list[YoloDetection]:
    kwargs: dict[str, Any] = {
        "source": str(image_path),
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "verbose": False,
    }
    device_arg = _real_yolo_device_arg(device)
    if device_arg is not None:
        kwargs["device"] = device_arg
    results = model.predict(**kwargs)
    if not results:
        return []
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []

    mask_polygons: list[PolygonPoints | None] = []
    masks = getattr(result, "masks", None)
    if masks is not None and getattr(masks, "xy", None) is not None:
        for xy in masks.xy:
            points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
            mask_polygons.append([[float(x), float(y)] for x, y in points.tolist()] if len(points) >= 3 else None)

    detections: list[YoloDetection] = []
    cls_values = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    conf_values = boxes.conf.detach().cpu().numpy().tolist()
    xyxy_values = boxes.xyxy.detach().cpu().numpy().tolist()
    for index, (class_id, confidence, xyxy) in enumerate(zip(cls_values, conf_values, xyxy_values, strict=False)):
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        polygon = mask_polygons[index] if index < len(mask_polygons) else None
        class_key = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        detections.append(
            YoloDetection(
                class_key=class_key,
                confidence=float(confidence),
                bbox={"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)},
                polygon=polygon,
            )
        )
    return detections


def _evaluate_real_yolo_scene(
    *,
    scene: _SyntheticScene | _SyntheticMultiScene,
    detections: list[YoloDetection],
    target_image: np.ndarray,
    case_index: int,
    variant: str,
) -> list[dict[str, Any]]:
    expected, annotation_to_object = _real_yolo_expected_segments(scene)
    projection_data = _real_feature_projection_data_for_target(scene, target_image)
    matches = match_segments(
        expected,
        detections,
        scene.approximate_homography,
        frame_size=_CANVAS_SIZE,
        projection_data=projection_data,
    )
    rows: list[dict[str, Any]] = []
    for expected_item in expected:
        obj = annotation_to_object[expected_item.annotation_id]
        match = next(
            (
                item for item in matches
                if item.annotation_id == expected_item.annotation_id
                and item.segment_class_id == expected_item.segment_class_id
            ),
            None,
        )
        detected_polygon = match.detected_polygon if match is not None else None
        expected_polygon = match.expected_polygon if match is not None else None
        display_polygon = detected_polygon if detected_polygon is not None else expected_polygon
        iou = _polygon_iou(display_polygon, obj["ground_truth_polygon"])
        center_drift = _polygon_center_distance(display_polygon, obj["ground_truth_polygon"])
        area_ratio = _polygon_area_ratio(display_polygon, obj["ground_truth_polygon"])
        nearest_distractor = _nearest_distractor_distance(display_polygon, obj["distractor_polygons"])
        closer_to_distractor = (
            nearest_distractor is not None
            and math.isfinite(center_drift)
            and nearest_distractor + 1.0 < center_drift
        )
        status = match.status if match is not None else "none"
        expected_ok = status == "ok"
        if variant == "present":
            passed = expected_ok and _real_yolo_present_match_ok(
                object_shape=obj["object_shape"],
                iou=iou,
                center_drift=center_drift,
                area_ratio=area_ratio,
            )
            dangerous = expected_ok and not passed and _is_dangerous_projection(
                predicted_polygon=display_polygon,
                center_drift=center_drift,
                area_ratio=area_ratio,
                closer_to_distractor=closer_to_distractor,
                unsafe_hidden=False,
            )
        else:
            passed = not expected_ok
            dangerous = expected_ok

        rows.append(
            {
                "case_index": case_index,
                "variant": variant,
                "name": obj["name"],
                "object_shape": obj["object_shape"],
                "class_key": expected_item.class_key,
                "status": status,
                "passed": bool(passed),
                "dangerous_expected_match": bool(dangerous),
                "confidence": match.confidence if match is not None else None,
                "iou": float(iou),
                "center_drift_px": float(center_drift),
                "area_ratio": float(area_ratio),
                "closer_to_distractor": bool(closer_to_distractor),
                "detected_class_in_zone": match.detected_class_in_zone if match is not None else None,
                "debug_reason_code": (
                    match.debug.get("reason_code")
                    if match is not None and isinstance(match.debug, dict)
                    else None
                ),
            }
        )
    extra_count = sum(1 for item in matches if item.annotation_id is None and item.status in {"extra", "unmatched"})
    if rows:
        rows[0]["case_extra_or_unmatched_detections"] = extra_count
        rows[0]["case_total_yolo_detections"] = len(detections)
    return rows


def _real_yolo_expected_segments(
    scene: _SyntheticScene | _SyntheticMultiScene,
) -> tuple[list[ExpectedSegment], dict[Any, dict[str, Any]]]:
    expected: list[ExpectedSegment] = []
    mapping: dict[Any, dict[str, Any]] = {}
    if isinstance(scene, _SyntheticScene):
        annotation_id = uuid4()
        segment_class_id = uuid4()
        expected.append(
            ExpectedSegment(
                annotation_id=annotation_id,
                segment_class_id=segment_class_id,
                class_key=scene.scenario.object_shape,
                name=scene.scenario.object_shape,
                hue=0,
                reference_polygon=scene.reference_polygon,
            )
        )
        mapping[annotation_id] = {
            "name": scene.scenario.name,
            "object_shape": scene.scenario.object_shape,
            "ground_truth_polygon": scene.ground_truth_polygon,
            "distractor_polygons": scene.distractor_polygons,
        }
        return expected, mapping

    for obj in scene.objects:
        annotation_id = uuid4()
        segment_class_id = uuid4()
        expected.append(
            ExpectedSegment(
                annotation_id=annotation_id,
                segment_class_id=segment_class_id,
                class_key=obj.object_shape,
                name=obj.name,
                hue=(obj.index * 47) % 360,
                reference_polygon=obj.reference_polygon,
            )
        )
        mapping[annotation_id] = {
            "name": obj.name,
            "object_shape": obj.object_shape,
            "ground_truth_polygon": obj.ground_truth_polygon,
            "distractor_polygons": obj.distractor_polygons,
        }
    return expected, mapping


def _real_yolo_present_match_ok(
    *,
    object_shape: str,
    iou: float,
    center_drift: float,
    area_ratio: float,
) -> bool:
    # Thin and hook-like shapes lose IoU faster from small mask shifts, so keep a
    # shape-aware center guard instead of a single strict IoU threshold.
    iou_floor = 0.36 if object_shape in {"thin_fork", "hook_like_part"} else 0.42
    return (
        iou >= iou_floor
        and center_drift <= 28.0
        and 0.40 <= area_ratio <= 2.20
    )


def _summarize_real_yolo_eval_rows(rows: Sequence[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    dangerous = sum(1 for row in rows if row.get("dangerous_expected_match"))
    accepted = sum(1 for row in rows if row.get("status") == "ok")
    missing = sum(1 for row in rows if row.get("status") != "ok")
    by_shape: dict[str, dict[str, Any]] = {}
    for row in rows:
        shape = str(row.get("object_shape") or "unknown")
        bucket = by_shape.setdefault(shape, {"total": 0, "passed": 0, "dangerous": 0, "accepted": 0})
        bucket["total"] += 1
        bucket["passed"] += int(bool(row.get("passed")))
        bucket["dangerous"] += int(bool(row.get("dangerous_expected_match")))
        bucket["accepted"] += int(row.get("status") == "ok")
    for bucket in by_shape.values():
        bucket["pass_rate"] = (bucket["passed"] / bucket["total"] * 100.0) if bucket["total"] else 0.0
    return {
        "variant": variant,
        "total_expected_objects": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total * 100.0) if total else 0.0,
        "accepted_expected_matches": accepted,
        "missing_expected_objects": missing,
        "dangerous_expected_matches": dangerous,
        "dangerous_rate": (dangerous / total * 100.0) if total else 0.0,
        "mean_iou": float(np.mean([row["iou"] for row in rows if math.isfinite(float(row["iou"]))])) if rows else 0.0,
        "mean_center_drift_px": float(np.mean([row["center_drift_px"] for row in rows if math.isfinite(float(row["center_drift_px"]))])) if rows else 0.0,
        "by_shape": by_shape,
    }


def _real_yolo_decision_hint(evaluation: dict[str, Any]) -> str:
    anchor_rescue = evaluation.get("yolo_anchor_rescue", {})
    dangerous_anchors = int(anchor_rescue.get("dangerous_missing_anchors") or 0)
    rescued = int(anchor_rescue.get("rescued_baseline_failures") or 0)
    if dangerous_anchors > 0:
        return "do_not_use_yolo_anchors_yet: some missing targets still receive trusted YOLO anchors"
    if rescued <= 0:
        return "anchor_pipeline_safe_but_no_rescue_yet: tune anchor assignment or train YOLO better"
    return "candidate_for_yolo_anchor_rescue_experiment: YOLO anchors reduce geometry failures without dangerous missing anchors"


def _write_real_yolo_eval_html(
    path: Path,
    summary: dict[str, Any],
    anchor_present_rows: Sequence[dict[str, Any]],
    anchor_missing_rows: Sequence[dict[str, Any]],
) -> None:
    def value_cell(value: Any, *, digits: int | None = None) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            if not math.isfinite(value):
                return "—"
            if digits is not None:
                return f"{value:.{digits}f}"
        return html.escape(str(value))

    def anchor_rows_html(rows: Sequence[dict[str, Any]], limit: int = 120) -> str:
        lines = []
        for row in rows[:limit]:
            lines.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('case_index')))}</td>"
                f"<td>{html.escape(str(row.get('variant')))}</td>"
                f"<td>{html.escape(str(row.get('name')))}</td>"
                f"<td>{html.escape(str(row.get('object_shape')))}</td>"
                f"<td>{'yes' if row.get('baseline_passed') else 'no'}</td>"
                f"<td>{html.escape(str(row.get('baseline_projection')))}</td>"
                f"<td>{html.escape(str(row.get('anchor_status')))}</td>"
                f"<td>{html.escape(str(row.get('applied_geometry_source')))}</td>"
                f"<td>{'yes' if row.get('applied_rescued_baseline_failure') else 'no'}</td>"
                f"<td>{'yes' if row.get('direct_anchor_rescue') else 'no'}</td>"
                f"<td>{'yes' if row.get('anchor_transform_rescue') else 'no'}</td>"
                f"<td>{'YES' if row.get('dangerous_missing_anchor') else 'no'}</td>"
                f"<td>{value_cell(row.get('anchor_confidence'), digits=3)}</td>"
                f"<td>{value_cell(row.get('anchor_slot_iou'), digits=3)}</td>"
                f"<td>{value_cell(row.get('anchor_slot_coverage'), digits=3)}</td>"
                f"<td>{value_cell(row.get('anchor_slot_area_ratio'), digits=3)}</td>"
                "</tr>"
            )
        return "\n".join(lines)

    anchor_rescue = summary.get("yolo_anchor_rescue", {})
    applied_shadow = anchor_rescue.get("applied_runtime_shadow", {})
    legacy = summary.get("legacy_global_yolo_matching", {})
    summary_for_report = {
        "mode": summary.get("mode"),
        "model_path": summary.get("model_path"),
        "classes": summary.get("classes"),
        "conf": summary.get("conf"),
        "iou": summary.get("iou"),
        "imgsz": summary.get("imgsz"),
        "anchor_policy": summary.get("anchor_policy"),
        "yolo_anchor_rescue": anchor_rescue,
        "legacy_global_yolo_matching_note": legacy.get("note"),
    }
    safe_label = "YES" if anchor_rescue.get("safety_ready_for_anchor_rescue") else "NO"
    content = f"""<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\">
  <title>Real YOLO anchor rescue test</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #333; padding: 8px 10px; text-align: left; }}
    th {{ background: #242424; }}
    .metric {{ display: inline-block; min-width: 180px; margin: 8px; padding: 12px; background: #202020; border-radius: 10px; }}
    .metric b {{ display: block; font-size: 24px; margin-top: 4px; }}
    .hint {{ color: #bbb; max-width: 1100px; }}
    .good {{ color: #69db7c; }}
    .bad {{ color: #ff8787; }}
  </style>
</head>
<body>
  <h1>Real YOLO anchor rescue test</h1>
  <p class="hint">YOLO здесь не является финальным контролёром missing/confirmed. Модель ищет объекты по всему кадру, а самые надёжные detections используются только как anchors для исправления переноса LightGlue. Если YOLO не нашла объект, это считается <code>no_anchor</code>, а не отсутствием детали.</p>
  <p class="hint">Старые метрики global YOLO matching сохранены только в JSON как <code>legacy_global_yolo_matching</code>. Для вывода решения используются только метрики <code>yolo_anchor_rescue</code>.</p>
  <div class="metric">Baseline failures total<b>{anchor_rescue.get('baseline_failed_objects', 0)}</b></div>
  <div class="metric">Covered failures<b>{anchor_rescue.get('baseline_failed_objects_covered_by_real_yolo_test', 0)}</b></div>
  <div class="metric">Rescued covered failures<b>{anchor_rescue.get('rescued_baseline_failures', 0)}</b></div>
  <div class="metric">Remaining covered failures<b>{anchor_rescue.get('remaining_covered_baseline_failures_after_anchor_rescue', 0)}</b></div>
  <div class="metric">Not covered failures<b>{anchor_rescue.get('baseline_failed_objects_not_covered_by_real_yolo_test', 0)}</b></div>
  <div class="metric">Dangerous anchors<b>{anchor_rescue.get('dangerous_missing_anchors', 0)}</b></div>
  <div class="metric">Safety ready<b class="{'good' if anchor_rescue.get('safety_ready_for_anchor_rescue') else 'bad'}">{safe_label}</b></div>
  <div class="metric">Applied rescues<b>{applied_shadow.get('applied_rescued_baseline_failures', 0)}</b></div>
  <div class="metric">Projected accuracy<b>{value_cell(applied_shadow.get('projected_object_accuracy_after_shadow'), digits=2)}%</b></div>
  <div class="metric">Dangerous applied changes<b>{applied_shadow.get('missing_dangerous_applied_changes', 0)}</b></div>
  <h2>Anchor rescue summary</h2>
  <pre>{html.escape(json.dumps(summary_for_report, ensure_ascii=False, indent=2))}</pre>
  <h2>Present anchor rows</h2>
  <table><tr><th>case</th><th>variant</th><th>name</th><th>shape</th><th>baseline passed</th><th>baseline projection</th><th>anchor status</th><th>applied source</th><th>applied rescue</th><th>direct rescue</th><th>transform rescue</th><th>danger</th><th>conf</th><th>slot IoU</th><th>coverage</th><th>area ratio</th></tr>{anchor_rows_html(anchor_present_rows)}</table>
  <h2>Missing negative anchor rows</h2>
  <table><tr><th>case</th><th>variant</th><th>name</th><th>shape</th><th>baseline passed</th><th>baseline projection</th><th>anchor status</th><th>applied source</th><th>applied rescue</th><th>direct rescue</th><th>transform rescue</th><th>danger</th><th>conf</th><th>slot IoU</th><th>coverage</th><th>area ratio</th></tr>{anchor_rows_html(anchor_missing_rows)}</table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")

def _dict_int_debug(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        try:
            result[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result

def _int_debug(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "case"


if __name__ == "__main__":
    raise SystemExit(main())
