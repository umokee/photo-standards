from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

PolygonPoints = list[list[float]]

_MIN_LOCAL_INLIERS = 5


class AlignmentStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_MATCHES = "insufficient_matches"
    INSUFFICIENT_INLIERS = "insufficient_inliers"
    HOMOGRAPHY_FAILED = "homography_failed"


@dataclass(frozen=True, slots=True)
class LocalProjectionData:
    global_homography: np.ndarray | None
    reference_points: np.ndarray | None
    frame_points: np.ndarray | None
    frame_size: tuple[int, int] | None = None
    frame: np.ndarray | None = None
    reference_frame: np.ndarray | None = None
    reference_feature_count: int | None = None
    frame_feature_count: int | None = None
    frame_max_keypoints: int | None = None
    frame_keypoint_grid: tuple[int, int] | None = None
    masked_alignment_used: bool = False
    original_reference_feature_count: int | None = None
    masked_reference_feature_count: int | None = None
    original_reference_keypoints: np.ndarray | None = None
    masked_reference_keypoints: np.ndarray | None = None
    frame_keypoints: np.ndarray | None = None

    @property
    def has_local_points(self) -> bool:
        reference = _as_points(self.reference_points)
        frame = _as_points(self.frame_points)
        return (
            reference is not None
            and frame is not None
            and len(reference) == len(frame)
            and len(reference) >= _MIN_LOCAL_INLIERS
        )


@dataclass(slots=True)
class FrameAlignment:
    status: AlignmentStatus
    homography: np.ndarray | None
    raw_match_count: int
    inlier_count: int
    median_error: float | None = None
    reference_inliers: np.ndarray | None = None
    frame_inliers: np.ndarray | None = None
    reference_matches: np.ndarray | None = None
    frame_matches: np.ndarray | None = None
    method: str = "unknown"
    stage: str | None = None
    reason: str | None = None
    reference_feature_count: int | None = None
    frame_feature_count: int | None = None
    frame_max_keypoints: int | None = None
    frame_keypoint_grid: tuple[int, int] | None = None
    masked_alignment_used: bool = False
    original_reference_feature_count: int | None = None
    masked_reference_feature_count: int | None = None
    reference_size: tuple[int, int] | None = None
    frame_size: tuple[int, int] | None = None
    extra_debug: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        return self.status == AlignmentStatus.SUCCESS and self.homography is not None

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status.value,
            "stage": self.stage,
            "reason": self.reason,
            "raw_match_count": self.raw_match_count,
            "inlier_count": self.inlier_count,
            "local_reference_match_count": (
                int(len(self.reference_matches))
                if self.reference_matches is not None
                else None
            ),
            "local_frame_match_count": (
                int(len(self.frame_matches)) if self.frame_matches is not None else None
            ),
            "median_error": self.median_error,
            "reference_feature_count": self.reference_feature_count,
            "frame_feature_count": self.frame_feature_count,
            "frame_max_keypoints": self.frame_max_keypoints,
            "frame_keypoint_grid": (
                {
                    "rows": self.frame_keypoint_grid[0],
                    "cols": self.frame_keypoint_grid[1],
                }
                if self.frame_keypoint_grid is not None
                else None
            ),
            "masked_alignment_used": self.masked_alignment_used,
            "original_reference_feature_count": self.original_reference_feature_count,
            "masked_reference_feature_count": self.masked_reference_feature_count,
            "reference_size": (
                {
                    "width": self.reference_size[0],
                    "height": self.reference_size[1],
                }
                if self.reference_size is not None
                else None
            ),
            "frame_size": (
                {
                    "width": self.frame_size[0],
                    "height": self.frame_size[1],
                }
                if self.frame_size is not None
                else None
            ),
            "extra_debug": self.extra_debug,
        }


def project_polygon(
    polygon: PolygonPoints,
    homography: np.ndarray,
) -> PolygonPoints:
    if not polygon:
        return []

    source = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, homography)
    return projected.reshape(-1, 2).tolist()
def alignment_message(alignment: FrameAlignment) -> str:
    if alignment.is_success:
        median_error = alignment.median_error or 0.0
        return (
            f"{alignment.method}_success:"
            f"inliers={alignment.inlier_count}/{alignment.raw_match_count},"
            f"err={median_error:.2f}"
        )

    parts = [
        alignment.method or "unknown",
        alignment.status.value,
    ]

    if alignment.stage:
        parts.append(f"stage={alignment.stage}")

    if alignment.reason:
        parts.append(f"reason={alignment.reason}")

    if alignment.frame_feature_count is not None:
        parts.append(f"frame_features={alignment.frame_feature_count}")

    if alignment.reference_feature_count is not None:
        parts.append(f"ref_features={alignment.reference_feature_count}")

    if alignment.raw_match_count:
        parts.append(f"raw={alignment.raw_match_count}")

    if alignment.inlier_count:
        parts.append(f"inliers={alignment.inlier_count}")

    return ":".join(parts[:3]) + ("," + ",".join(parts[3:]) if len(parts) > 3 else "")


def failed_alignment(
    status: AlignmentStatus = AlignmentStatus.INSUFFICIENT_MATCHES,
    *,
    raw_match_count: int = 0,
    inlier_count: int = 0,
) -> FrameAlignment:
    return FrameAlignment(
        status=status,
        homography=None,
        raw_match_count=raw_match_count,
        inlier_count=inlier_count,
        reference_matches=None,
        frame_matches=None,
        method="unknown",
        stage="external_failed_alignment",
        reason=status.value,
    )
def _as_points(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float32).reshape(-1, 2)

    if len(array) == 0:
        return None

    if not np.isfinite(array).all():
        return None

    return array
