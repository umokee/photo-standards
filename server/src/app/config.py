import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = PROJECT_ROOT / "server"
DEFAULT_SAM2_ROOT = PROJECT_ROOT / "storage" / "weights"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "storage" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "storage"))
os.environ.setdefault("YOLO_VERBOSE", "false")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVER_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "PhotoStandards API"
    DEBUG: bool = False

    # Realtime observability.
    PHOTOAPP_REALTIME_PROFILER: bool = False
    PHOTOAPP_REALTIME_PROFILER_WINDOW: int = Field(default=120, ge=1)
    PHOTOAPP_REALTIME_PROFILER_SUMMARY_EVERY: int = Field(default=30, ge=1)

    # Inspection behavior.
    INSPECTION_VERIFICATION_MODE: Literal["alignment", "yolo_count"] = "alignment"

    INSPECTION_SLOT_SEARCH_EXPANSION: float = Field(default=2.35, ge=1.0, le=8.0)
    INSPECTION_SLOT_MIN_SCORE: float = Field(default=0.43, ge=0.0, le=1.0)
    INSPECTION_SLOT_MIN_DETECTION_CONTAINMENT: float = Field(default=0.10, ge=0.0, le=1.0)
    INSPECTION_SLOT_MIN_YOLO_CONFIDENCE: float = Field(default=0.08, ge=0.0, le=1.0)
    INSPECTION_SLOT_MIN_FEATURE_SUPPORT: int = Field(default=4, ge=0, le=64)
    INSPECTION_SLOT_FEATURE_SEARCH_EXPANSION: float = Field(default=2.75, ge=1.0, le=10.0)

    INSPECTION_MISSING_POLYGON_REFINEMENT: bool = True
    INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT: int = Field(default=4, ge=0, le=64)
    INSPECTION_MISSING_POLYGON_MAX_REPROJECTION_ERROR: float = Field(
        default=14.0,
        ge=0.0,
        le=80.0,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_EXPANSION: float = Field(
        default=3.0,
        ge=1.1,
        le=8.0,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_EXCLUSION_MARGIN: float = Field(
        default=8.0,
        ge=0.0,
        le=80.0,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_FRAME_EXCLUSION: bool = False
    INSPECTION_MISSING_POLYGON_MULTI_CONTEXT_EXCLUSION_EXPANSION: float = Field(
        default=1.35,
        ge=1.0,
        le=4.0,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_MIN_INLIER_RATIO: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_MIN_SPREAD: float = Field(
        default=0.16,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_MIN_QUADRANTS: int = Field(
        default=2,
        ge=1,
        le=4,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_MAX_CENTER_DRIFT_FACTOR: float = Field(
        default=0.85,
        ge=0.05,
        le=4.0,
    )
    INSPECTION_MISSING_POLYGON_CONTEXT_MIN_AREA_SCORE: float = Field(
        default=0.38,
        ge=0.01,
        le=1.0,
    )
    INSPECTION_MISSING_FALLBACK_MIN_GLOBAL_AREA_SCORE: float = Field(
        default=0.08,
        ge=0.01,
        le=1.0,
    )
    INSPECTION_MISSING_FALLBACK_MIN_LOCAL_GLOBAL_AREA_SCORE: float = Field(
        default=0.32,
        ge=0.01,
        le=1.0,
    )
    INSPECTION_MISSING_FALLBACK_MAX_LOCAL_GLOBAL_CENTER_FACTOR: float = Field(
        default=0.75,
        ge=0.05,
        le=4.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_DEMOTION_ENABLED: bool = False
    INSPECTION_MISSING_GLOBAL_FALLBACK_DEMOTION_MIN_AREA_SCORE: float = Field(
        default=0.006,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_DEMOTION_MAX_CENTER_FACTOR: float = Field(
        default=4.6,
        ge=0.01,
        le=8.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_DEMOTION_MAX_SLOT_SUPPORT: int = Field(
        default=2,
        ge=0,
        le=128,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_DEMOTION_MAX_SLOT_SUPPORT_RATIO: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_ENABLED: bool = True
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_DIAGNOSTICS_ENABLED: bool = True
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MIN_SUPPORT: int = Field(
        default=3,
        ge=2,
        le=64,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MIN_INLIER_RATIO: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MAX_RESIDUAL_ERROR: float = Field(
        default=6.5,
        ge=0.5,
        le=40.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MAX_SHIFT_FACTOR: float = Field(
        default=0.62,
        ge=0.05,
        le=2.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MIN_LOCAL_AREA_SCORE: float = Field(
        default=0.24,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MAX_LOCAL_CENTER_FACTOR: float = Field(
        default=1.05,
        ge=0.05,
        le=4.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MIN_SEARCH_CONTAINMENT: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_GLOBAL_FALLBACK_TRANSLATION_RESCUE_MAX_OTHER_OVERLAP: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_ENABLED: bool = True
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MIN_SUPPORT: int = Field(
        default=6,
        ge=3,
        le=96,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_NEAREST_POINTS: int = Field(
        default=8,
        ge=3,
        le=32,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MAX_RESIDUAL_ERROR: float = Field(
        default=10.0,
        ge=0.5,
        le=60.0,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MAX_SHIFT_FACTOR: float = Field(
        default=0.58,
        ge=0.05,
        le=2.0,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MIN_SPREAD: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MIN_QUADRANTS: int = Field(
        default=2,
        ge=1,
        le=4,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MIN_SEARCH_CONTAINMENT: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MIN_LOCAL_AREA_SCORE: float = Field(
        default=0.26,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MAX_LOCAL_CENTER_FACTOR: float = Field(
        default=1.08,
        ge=0.05,
        le=4.0,
    )
    INSPECTION_MISSING_LOCAL_DISPLACEMENT_MAX_OTHER_OVERLAP: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_UNSAFE_HIDDEN_SHADOW_ENABLED: bool = True
    INSPECTION_MISSING_CANDIDATE_REGISTRY_DEBUG_ENABLED: bool = True
    INSPECTION_MISSING_SELECTIVE_HIDDEN_RELEASE_ENABLED: bool = True
    INSPECTION_MISSING_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_RELEASE_ENABLED: bool = True
    INSPECTION_MISSING_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_SUPPORT: int = Field(
        default=10,
        ge=0,
        le=128,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_SUPPORT_RATIO: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_REFERENCE_AREA_SCORE: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MAX_OTHER_OVERLAP: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_VISIBLE_FRACTION: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MAX_CENTER_FACTOR: float = Field(
        default=4.20,
        ge=0.1,
        le=8.0,
    )
    INSPECTION_MISSING_CANDIDATE_AGREEMENT_SCORER_ENABLED: bool = True
    INSPECTION_MISSING_CANDIDATE_AGREEMENT_MIN_IOU: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_CANDIDATE_AGREEMENT_MIN_AREA_SCORE: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_CANDIDATE_AGREEMENT_MAX_CENTER_FACTOR: float = Field(
        default=0.55,
        ge=0.0,
        le=4.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_RELEASE_ENABLED: bool = True
    INSPECTION_MISSING_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_CENTER_FACTOR: float = Field(
        default=0.15,
        ge=0.0,
        le=4.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MAX_CENTER_FACTOR: float = Field(
        default=0.55,
        ge=0.0,
        le=4.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_AREA_SCORE: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MAX_OTHER_OVERLAP: float = Field(
        default=1.01,
        ge=0.0,
        le=1.5,
    )
    INSPECTION_MISSING_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_VISIBLE_FRACTION: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_RESCUE_TRANSLATION_ENABLED: bool = True
    INSPECTION_MISSING_RESCUE_TRANSLATION_MIN_SUPPORT: int = Field(
        default=6,
        ge=3,
        le=128,
    )
    INSPECTION_MISSING_RESCUE_TRANSLATION_MIN_INLIER_RATIO: float = Field(
        default=0.46,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_RESCUE_TRANSLATION_MAX_RESIDUAL_ERROR: float = Field(
        default=9.0,
        ge=0.5,
        le=40.0,
    )
    INSPECTION_MISSING_RESCUE_TRANSLATION_MAX_SHIFT_FACTOR: float = Field(
        default=0.48,
        ge=0.05,
        le=2.0,
    )
    INSPECTION_MISSING_SCENE_RESCUE_TRANSLATION_ENABLED: bool = True
    INSPECTION_MISSING_SCENE_RESCUE_CONTEXT_EXPANSION: float = Field(
        default=6.0,
        ge=2.0,
        le=14.0,
    )
    INSPECTION_MISSING_SCENE_RESCUE_MIN_SUPPORT: int = Field(
        default=10,
        ge=4,
        le=256,
    )
    INSPECTION_MISSING_SCENE_RESCUE_MIN_INLIER_RATIO: float = Field(
        default=0.58,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_SCENE_RESCUE_MAX_RESIDUAL_ERROR: float = Field(
        default=8.0,
        ge=0.5,
        le=48.0,
    )
    INSPECTION_MISSING_SCENE_RESCUE_MAX_SHIFT_FACTOR: float = Field(
        default=0.42,
        ge=0.02,
        le=1.5,
    )
    INSPECTION_MISSING_SCENE_RESCUE_MIN_SPREAD: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_SCENE_RESCUE_MAX_OTHER_OVERLAP: float = Field(
        default=0.32,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_MIN_GLOBAL_AREA_SCORE: float = Field(
        default=0.68,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_MAX_GLOBAL_CENTER_FACTOR: float = Field(
        default=0.28,
        ge=0.01,
        le=2.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_GUARDED_RELEASE_ENABLED: bool = False
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_SUPPORT: int = Field(
        default=4,
        ge=0,
        le=128,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_SUPPORT_RATIO: float = Field(
        default=0.16,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_GLOBAL_AREA_SCORE: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_GLOBAL_CENTER_FACTOR: float = Field(
        default=0.34,
        ge=0.01,
        le=4.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_REFERENCE_AREA_SCORE: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_OTHER_OVERLAP: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_GLOBAL_POLYGON_IOU: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_GLOBAL_AXIS_ANGLE: float = Field(
        default=18.0,
        ge=0.0,
        le=90.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_GLOBAL_MAJOR_RATIO: float = Field(
        default=0.65,
        ge=0.01,
        le=4.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MAX_GLOBAL_MAJOR_RATIO: float = Field(
        default=1.55,
        ge=0.01,
        le=4.0,
    )
    INSPECTION_MISSING_MULTI_EXPECTED_SLOT_RELEASE_MIN_VISIBLE_FRACTION: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_ENABLED: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_DEBUG: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_SOURCE_DIAGNOSTICS_ENABLED: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_ORDER_DIAGNOSTICS_ENABLED: bool = True
    INSPECTION_MISSING_FALLBACK_DIAGNOSTICS_ENABLED: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_MIN_ANCHORS: int = Field(
        default=2,
        ge=1,
        le=16,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MIN_INLIER_RATIO: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MAX_ANCHOR_ERROR: float = Field(
        default=9.0,
        ge=0.5,
        le=40.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MAX_SHIFT_FACTOR: float = Field(
        default=0.55,
        ge=0.02,
        le=1.5,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MIN_LOCAL_AREA_SCORE: float = Field(
        default=0.24,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MAX_LOCAL_CENTER_FACTOR: float = Field(
        default=0.95,
        ge=0.02,
        le=3.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MAX_OTHER_OVERLAP: float = Field(
        default=0.32,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_ARBITRATION_ENABLED: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MIN_INLIERS: int = Field(
        default=2,
        ge=1,
        le=16,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MIN_INLIER_RATIO: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MAX_DISPERSION_FACTOR: float = Field(
        default=0.035,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_MAX_SHIFT_FACTOR: float = Field(
        default=0.36,
        ge=0.02,
        le=1.5,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_OVERLAP_CENTER_MARGIN: float = Field(
        default=0.88,
        ge=0.1,
        le=2.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MIN_VISIBLE_FRACTION: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_ENABLED: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MIN_INLIERS: int = Field(
        default=5,
        ge=1,
        le=64,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MIN_INLIER_RATIO: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MAX_ANCHOR_ERROR: float = Field(
        default=7.0,
        ge=0.5,
        le=40.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MAX_SHIFT_FACTOR: float = Field(
        default=0.24,
        ge=0.02,
        le=0.80,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MIN_LOCAL_AREA_SCORE: float = Field(
        default=0.42,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_SINGLE_MAX_LOCAL_CENTER_FACTOR: float = Field(
        default=0.52,
        ge=0.02,
        le=2.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_STRICT_SINGLE_OVERLAP_ENABLED: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_STRICT_SINGLE_OVERLAP_MIN_INLIERS: int = Field(
        default=8,
        ge=1,
        le=96,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_STRICT_SINGLE_OVERLAP_MIN_INLIER_RATIO: float = Field(
        default=0.62,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_STRICT_SINGLE_OVERLAP_MAX_ANCHOR_ERROR: float = Field(
        default=6.5,
        ge=0.5,
        le=40.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_STRICT_SINGLE_OVERLAP_MAX_SHIFT_FACTOR: float = Field(
        default=0.18,
        ge=0.02,
        le=0.80,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_STRICT_SINGLE_OVERLAP_MAX_DISPERSION_FACTOR: float = Field(
        default=0.018,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_STRICT_SINGLE_OVERLAP_MAX_UNRESOLVED: int = Field(
        default=1,
        ge=1,
        le=16,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_LOCAL_MIN_INLIER_RATIO: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MAX_DISPERSION_FACTOR: float = Field(
        default=0.14,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_MAX_NEIGHBOR_DISTANCE_FACTOR: float = Field(
        default=6.0,
        ge=0.5,
        le=20.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_HINT_ENABLED: bool = True
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MIN_INLIERS: int = Field(
        default=3,
        ge=2,
        le=16,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MIN_INLIER_RATIO: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MAX_DISPERSION_FACTOR: float = Field(
        default=0.045,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MIN_FEATURE_SUPPORT: int = Field(
        default=2,
        ge=1,
        le=64,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MIN_FEATURE_RATIO: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MIN_AREA_SCORE: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MAX_CENTER_FACTOR: float = Field(
        default=1.20,
        ge=0.02,
        le=4.0,
    )
    INSPECTION_MISSING_ANCHOR_RELEASE_WEAK_SLOT_MAX_SHIFT_FACTOR: float = Field(
        default=0.36,
        ge=0.02,
        le=1.5,
    )
    INSPECTION_MISSING_MULTI_CONSENSUS_RESCUE_ENABLED: bool = True
    INSPECTION_MISSING_MULTI_CONSENSUS_MIN_NEIGHBORS: int = Field(
        default=2,
        ge=1,
        le=32,
    )
    INSPECTION_MISSING_MULTI_CONSENSUS_MIN_INLIER_RATIO: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_CONSENSUS_MAX_RESIDUAL_ERROR: float = Field(
        default=14.0,
        ge=1.0,
        le=80.0,
    )
    INSPECTION_MISSING_MULTI_CONSENSUS_MAX_SHIFT_FACTOR: float = Field(
        default=0.55,
        ge=0.02,
        le=2.0,
    )
    INSPECTION_MISSING_MULTI_CONSENSUS_MIN_AREA_SCORE: float = Field(
        default=0.42,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_MULTI_CONSENSUS_MAX_CENTER_FACTOR: float = Field(
        default=0.95,
        ge=0.05,
        le=4.0,
    )
    INSPECTION_MISSING_MULTI_CONSENSUS_MAX_FALLBACK_CENTER_FACTOR: float = Field(
        default=0.95,
        ge=0.05,
        le=4.0,
    )

    INSPECTION_MISSING_HIDDEN_CLUSTER_CONSENSUS_ENABLED: bool = True
    INSPECTION_MISSING_HIDDEN_CLUSTER_MIN_CANDIDATES: int = Field(
        default=3,
        ge=2,
        le=32,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MIN_INLIERS: int = Field(
        default=3,
        ge=2,
        le=32,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MIN_INLIER_RATIO: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MAX_RESIDUAL_ERROR: float = Field(
        default=11.0,
        ge=1.0,
        le=80.0,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MAX_CURRENT_ERROR: float = Field(
        default=14.0,
        ge=1.0,
        le=80.0,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MAX_SHIFT_FACTOR: float = Field(
        default=0.38,
        ge=0.02,
        le=1.5,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MIN_AREA_SCORE: float = Field(
        default=0.24,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MAX_CENTER_FACTOR: float = Field(
        default=1.35,
        ge=0.05,
        le=4.0,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MIN_FALLBACK_AREA_SCORE: float = Field(
        default=0.32,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MAX_FALLBACK_CENTER_FACTOR: float = Field(
        default=0.72,
        ge=0.05,
        le=4.0,
    )
    INSPECTION_MISSING_HIDDEN_CLUSTER_MAX_OTHER_OVERLAP: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )

    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_ENABLED: bool = True
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_MAX_SUPPORT: int = Field(
        default=24,
        ge=3,
        le=128,
    )
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_MAX_TOTAL: int = Field(
        default=42,
        ge=3,
        le=256,
    )
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_SPARSE_SUPPORT: int = Field(
        default=8,
        ge=3,
        le=64,
    )
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_SPARSE_CANDIDATES: int = Field(
        default=10,
        ge=3,
        le=128,
    )
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_MIN_MEDIAN_ERROR: float = Field(
        default=3.25,
        ge=0.0,
        le=40.0,
    )
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_MAX_INLIER_RATIO: float = Field(
        default=0.86,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_MIN_CENTER_FACTOR: float = Field(
        default=0.14,
        ge=0.0,
        le=2.0,
    )
    INSPECTION_MISSING_WEAK_CONTEXT_AFFINE_DEMOTION_MIN_AREA_SCORE: float = Field(
        default=0.86,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_POLYGON_EDGE_REFINEMENT: bool = False
    INSPECTION_MISSING_POLYGON_EDGE_SNAP_RADIUS: int = Field(default=14, ge=0, le=64)
    INSPECTION_MISSING_POLYGON_EDGE_BLEND: float = Field(default=0.65, ge=0.0, le=1.0)
    INSPECTION_MISSING_POLYGON_EDGE_DENSIFY_STEP: int = Field(default=8, ge=2, le=32)
    INSPECTION_MISSING_POLYGON_EDGE_SMOOTHING: float = Field(
        default=0.18,
        ge=0.0,
        le=0.45,
    )
    INSPECTION_MISSING_POLYGON_EDGE_SMOOTH_ITERATIONS: int = Field(
        default=2,
        ge=0,
        le=8,
    )
    INSPECTION_MISSING_POLYGON_EDGE_MAX_INWARD_SHIFT_FRACTION: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )
    INSPECTION_MISSING_POLYGON_EDGE_MIN_WIDTH_RATIO: float = Field(
        default=0.55,
        ge=0.05,
        le=1.0,
    )

    MAX_REALTIME_INSPECTIONS: int = Field(default=2, ge=1, le=10)

    # Database connection.
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 5432
    DB_NAME: str = "photo-standards"
    DB_USER: str = "postgres"
    DB_PASS: str = "postgres"  # noqa:S105
    DB_SOCKET_DIR: Path | None = None

    # Filesystem storage.
    STORAGE_ROOT: Path = PROJECT_ROOT / "storage"

    # Interactive segmentation.
    SAM2_DEVICE: str = "cpu"
    SAM2_ROOT: str = f"{STORAGE_ROOT}/weights"
    SAM2_MODEL_CFG: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    SAM2_CHECKPOINT: str = f"{DEFAULT_SAM2_ROOT}/sam2.1_hiera_small.pt"

    # Reference alignment.
    ALIGNMENT_BACKEND: Literal["auto", "torch", "orb"] = "auto"
    ALIGNMENT_DEVICE: Literal["auto", "cuda", "cpu"] = "auto"
    ALIGNMENT_ORB_FALLBACK: bool = True
    ALIGNMENT_IDENTITY_SHORTCUT: bool = True

    # YOLO training and inference.
    YOLO_DEVICE: Literal["auto", "cuda", "cpu"] = "auto"
    YOLO_DEFAULT_IMGSZ: int = Field(default=640, ge=32)
    YOLO_CONF_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)
    YOLO_REALTIME_CONF_THRESHOLD: float = Field(default=0.20, ge=0.0, le=1.0)
    YOLO_NMS_IOU: float = Field(default=0.55, ge=0.0, le=1.0)
    YOLO_EXTRA_CONF_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    YOLO_HALF: bool = False

    # Built-in background task runner.
    TASK_RUNNER_ENABLED: bool = True
    TASK_RUNNER_FAIL_ACTIVE_ON_STARTUP: bool = True
    TASK_RUNNER_POLL_INTERVAL_SEC: float = 1.0
    TASK_RUNNER_CPU_CONCURRENCY: int = 2
    TASK_RUNNER_GPU_CONCURRENCY: int = 2
    TASK_RUNNER_TRAINING_CONCURRENCY: int = 1
    TASK_RUNNER_INSPECTION_CONCURRENCY: int = 2
    TASK_RUNNER_MODEL_IMPORT_CONCURRENCY: int = 1
    TASK_RUNNER_HEARTBEAT_TIMEOUT_SEC: float = 900.0
    TASK_RUNNER_SERVICE_MONITOR_INTERVAL_SEC: float = 2.0
    TASK_RUNNER_SERVICE_RESTART_DELAY_SEC: float = 2.0
    TASK_RUNNER_SERVICE_MAX_RESTART_DELAY_SEC: float = 30.0
    TASK_RUNNER_SERVICE_STOP_TIMEOUT_SEC: float = 10.0
    TASK_RUNNER_SERVICE_PARENT_CHECK_INTERVAL_SEC: float = 3.0
    TASK_RUNNER_SERVICE_HEARTBEAT_INTERVAL_SEC: float = 3.0
    TASK_RUNNER_SERVICE_HEARTBEAT_TIMEOUT_SEC: float = 30.0
    TASK_RUNNER_SERVICE_MEMORY_LIMIT_MB: int = 0
    TASK_RUNNER_SERVICE_MIN_FREE_MEMORY_MB: int = 1024
    TASK_RUNNER_SERVICE_CPU_LIMIT_PERCENT: float = 95.0
    TASK_RUNNER_SERVICE_RESOURCE_GRACE_TICKS: int = 3

    @property
    def _database_host_query(self) -> str:
        if self.DB_SOCKET_DIR:
            return f"host={quote_plus(str(self.DB_SOCKET_DIR))}"

        return f"{self.DB_HOST}:{self.DB_PORT}"

    @property
    def database_url_async(self) -> str:
        if self.DB_SOCKET_DIR:
            return (
                f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}"
                f"@/{self.DB_NAME}?{self._database_host_query}"
            )

        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self._database_host_query}/{self.DB_NAME}"
        )

    @property
    def database_url_sync(self) -> str:
        if self.DB_SOCKET_DIR:
            return (
                f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}"
                f"@/{self.DB_NAME}?{self._database_host_query}"
            )

        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self._database_host_query}/{self.DB_NAME}"
        )

    @property
    def database_url_conninfo(self) -> str:
        if self.DB_SOCKET_DIR:
            return (
                f"postgresql://{self.DB_USER}:{self.DB_PASS}"
                f"@/{self.DB_NAME}?{self._database_host_query}"
            )

        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASS}"
            f"@{self._database_host_query}/{self.DB_NAME}"
        )

    @property
    def database_url_async_for_listen(self) -> str:
        url = self.database_url_async
        return url.replace("postgresql+asyncpg", "postgresql")

    @property
    def standards_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "standards"

    @property
    def inspections_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "inspections"

    @property
    def models_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "models"

    @property
    def logs_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "logs"

    @property
    def current_logs_path(self) -> Path:
        current_date = datetime.now(UTC).date().isoformat()
        return self.logs_storage_path / current_date

    @property
    def worker_logs_path(self) -> Path:
        return self.current_logs_path

    @property
    def server_logs_path(self) -> Path:
        return self.current_logs_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


__all__ = ["settings"]
