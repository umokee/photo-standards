from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import structlog
from app.config import settings
from app.observability import log_event, throttled_log
from infra.storage.file_storage import resolve_storage_path
from modules.core.standards.reference_constants import (
    MIN_INLIERS_FOR_ALIGNMENT,
    MIN_RAW_MATCHES,
    RANSAC_REPROJECTION_THRESHOLD,
    SUPERPOINT_PHOTO_GRID_COLS,
    SUPERPOINT_PHOTO_GRID_ROWS,
    SUPERPOINT_PHOTO_MAX_KEYPOINTS,
    SUPERPOINT_PHOTO_MAX_SIDE,
    SUPERPOINT_REFERENCE_GRID_COLS,
    SUPERPOINT_REFERENCE_GRID_ROWS,
    SUPERPOINT_REFERENCE_MAX_KEYPOINTS,
    SUPERPOINT_REFERENCE_MAX_SIDE,
)
from modules.core.standards.reference_features import (
    ImageFeatures,
    compute_features,
    load_image,
)
from modules.core.standards.reference_runtime import match_feature_arrays
from modules.yolo.inspection.adapters.context import InspectionContext
from modules.yolo.inspection.domain.alignment import (
    AlignmentStatus,
    FrameAlignment,
)
from modules.yolo.inspection.domain.reference_masking import mask_reference_polygons

logger = structlog.get_logger(__name__)

_ORB_N_FEATURES = 3000
_ORB_RATIO_TEST = 0.75
_ORB_MIN_RAW_MATCHES = 20
_ORB_MIN_INLIERS = 12


def align_frame(
    *,
    context: InspectionContext,
    frame: np.ndarray,
    config: AlignmentValidationConfig | None = None,
    max_side: int | None = SUPERPOINT_PHOTO_MAX_SIDE,
    max_keypoints: int = SUPERPOINT_PHOTO_MAX_KEYPOINTS,
    selection_grid: tuple[int, int] | None = (
        SUPERPOINT_PHOTO_GRID_ROWS,
        SUPERPOINT_PHOTO_GRID_COLS,
    ),
) -> FrameAlignment:
    config = config or AlignmentValidationConfig.industrial()

    if settings.ALIGNMENT_IDENTITY_SHORTCUT:
        identity = _try_identity_alignment(context=context, frame=frame)
        if identity is not None:
            return identity

    if settings.ALIGNMENT_BACKEND in {"auto", "torch"}:
        try:
            frame_features = compute_features(
                frame,
                max_side=max_side,
                max_keypoints=max_keypoints,
                selection_grid=selection_grid,
            )
            masked_reference_features = _try_masked_reference_features(
                context=context,
            )
            torch_alignment = align_with_features(
                context=context,
                frame_features=frame_features,
                frame_shape=frame.shape[:2],
                config=config,
                max_keypoints=max_keypoints,
                selection_grid=selection_grid,
                reference_features=masked_reference_features,
                masked_alignment_used=masked_reference_features is not None,
            )

            if torch_alignment.is_success:
                torch_alignment.method = "torch"
                return torch_alignment

            if not settings.ALIGNMENT_ORB_FALLBACK:
                return torch_alignment

        except Exception as exc:
            log_event(
                logger,
                "warning",
                "inspection.alignment.failed",
                standard_id=context.standard.id,
                reference_image_id=context.reference_image.id,
                method="torch",
                status=AlignmentStatus.HOMOGRAPHY_FAILED.value,
                stage="exception",
                reason=str(exc),
                reference_feature_count=context.reference_features.count,
                frame_size=(frame.shape[1], frame.shape[0]),
                error_type=type(exc).__name__,
                exception=exc,
            )

            if not settings.ALIGNMENT_ORB_FALLBACK:
                return _failed(
                    AlignmentStatus.HOMOGRAPHY_FAILED,
                    method="torch_error",
                    stage="exception",
                    reason=str(exc),
                    context=context,
                    frame_shape=frame.shape[:2],
                )

    if (
        settings.ALIGNMENT_BACKEND in {"auto", "orb"}
        and settings.ALIGNMENT_ORB_FALLBACK
    ):
        return align_with_orb(
            context=context,
            frame=frame,
            config=config,
        )

    return _failed(
        AlignmentStatus.INSUFFICIENT_MATCHES,
        method="disabled",
        stage="backend_disabled",
        reason="alignment backend disabled or unavailable",
        context=context,
        frame_shape=frame.shape[:2],
    )


def align_with_features(
    *,
    context: InspectionContext,
    frame_features: ImageFeatures,
    frame_shape: tuple[int, int],
    config: AlignmentValidationConfig,
    max_keypoints: int = SUPERPOINT_PHOTO_MAX_KEYPOINTS,
    selection_grid: tuple[int, int] | None = None,
    reference_features: ImageFeatures | None = None,
    masked_alignment_used: bool = False,
) -> FrameAlignment:
    frame_height, frame_width = frame_shape
    active_reference_features = reference_features or context.reference_features
    original_reference_feature_count = context.reference_features.count
    masked_reference_feature_count = (
        active_reference_features.count if masked_alignment_used else None
    )

    if active_reference_features.count < 10 or frame_features.count < 10:
        reason = "not enough keypoints before matching"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="feature_count",
            status=AlignmentStatus.INSUFFICIENT_MATCHES,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.INSUFFICIENT_MATCHES,
            method="torch",
            stage="feature_count",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            frame_max_keypoints=max_keypoints,
            frame_keypoint_grid=selection_grid,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    matched = match_features(
        active_reference_features,
        frame_features,
        max_keypoints=max_keypoints,
    )
    if matched is None:
        reason = "match_features returned None"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="match",
            status=AlignmentStatus.INSUFFICIENT_MATCHES,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.INSUFFICIENT_MATCHES,
            method="torch",
            stage="match",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            frame_max_keypoints=max_keypoints,
            frame_keypoint_grid=selection_grid,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    reference_points, frame_points = matched

    return _build_alignment_from_points(
        context=context,
        frame_features=frame_features,
        frame_shape=frame_shape,
        frame_width=frame_width,
        frame_height=frame_height,
        reference_points=reference_points,
        frame_points=frame_points,
        config=config,
        method="torch_masked" if masked_alignment_used else "torch",
        max_keypoints=max_keypoints,
        selection_grid=selection_grid,
        reference_features=active_reference_features,
        masked_alignment_used=masked_alignment_used,
        original_reference_feature_count=original_reference_feature_count,
        masked_reference_feature_count=masked_reference_feature_count,
    )


def align_with_orb(
    *,
    context: InspectionContext,
    frame: np.ndarray,
    config: AlignmentValidationConfig,
) -> FrameAlignment:
    reference_image = _load_reference_image(context.reference_image.image_path)

    reference_gray = _gray(reference_image)
    frame_gray = _gray(frame)

    orb = cv2.ORB_create(nfeatures=_ORB_N_FEATURES)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_gray, None)
    frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)

    if (
        ref_descriptors is None
        or frame_descriptors is None
        or ref_keypoints is None
        or frame_keypoints is None
        or len(ref_keypoints) < 10
        or len(frame_keypoints) < 10
    ):
        return _failed(
            AlignmentStatus.INSUFFICIENT_MATCHES,
            method="orb",
            stage="feature_count",
            reason="ORB did not find enough keypoints/descriptors",
            context=context,
            frame_shape=frame.shape[:2],
            frame_feature_count=len(frame_keypoints or []),
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(ref_descriptors, frame_descriptors, k=2)

    good_matches = []
    for pair in knn:
        if len(pair) != 2:
            continue

        first, second = pair
        if first.distance < _ORB_RATIO_TEST * second.distance:
            good_matches.append(first)

    if len(good_matches) < _ORB_MIN_RAW_MATCHES:
        return _failed(
            AlignmentStatus.INSUFFICIENT_MATCHES,
            raw_match_count=len(good_matches),
            method="orb",
            stage="raw_match_count",
            reason="ORB did not find enough raw matches",
            context=context,
            frame_shape=frame.shape[:2],
            frame_feature_count=len(frame_keypoints),
        )

    good_matches = sorted(good_matches, key=lambda item: item.distance)[:512]

    reference_points = np.asarray(
        [ref_keypoints[item.queryIdx].pt for item in good_matches],
        dtype=np.float32,
    )
    frame_points = np.asarray(
        [frame_keypoints[item.trainIdx].pt for item in good_matches],
        dtype=np.float32,
    )

    frame_features_for_debug = ImageFeatures(
        keypoints=frame_points,
        descriptors=np.empty((len(frame_points), 1), dtype=np.float32),
        image_width=frame.shape[1],
        image_height=frame.shape[0],
    )

    return _build_alignment_from_points(
        context=context,
        frame_features=frame_features_for_debug,
        frame_shape=frame.shape[:2],
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        reference_points=reference_points,
        frame_points=frame_points,
        config=config,
        method="orb",
        min_raw_matches=_ORB_MIN_RAW_MATCHES,
        min_inliers=_ORB_MIN_INLIERS,
    )


def _build_alignment_from_points(
    *,
    context: InspectionContext,
    frame_features: ImageFeatures,
    frame_shape: tuple[int, int],
    frame_width: int,
    frame_height: int,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    config: AlignmentValidationConfig,
    method: str,
    max_keypoints: int | None = None,
    selection_grid: tuple[int, int] | None = None,
    min_raw_matches: int = MIN_RAW_MATCHES,
    min_inliers: int = MIN_INLIERS_FOR_ALIGNMENT,
    reference_features: ImageFeatures | None = None,
    masked_alignment_used: bool = False,
    original_reference_feature_count: int | None = None,
    masked_reference_feature_count: int | None = None,
) -> FrameAlignment:
    active_reference_features = reference_features or context.reference_features
    raw_match_count = int(reference_points.shape[0])

    if raw_match_count < min_raw_matches:
        reason = f"{method}: not enough raw matches"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="raw_match_count",
            status=AlignmentStatus.INSUFFICIENT_MATCHES,
            raw_match_count=raw_match_count,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.INSUFFICIENT_MATCHES,
            raw_match_count=raw_match_count,
            method=method,
            stage="raw_match_count",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            reference_matches=reference_points,
            frame_matches=frame_points,
            frame_keypoint_grid=selection_grid,
            frame_max_keypoints=max_keypoints,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    homography, mask, ransac_threshold = _find_best_homography(
        reference_points=reference_points,
        frame_points=frame_points,
    )

    if homography is None or mask is None:
        reason = f"{method}: findHomography returned None"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="ransac",
            status=AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            method=method,
            stage="ransac",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            reference_matches=reference_points,
            frame_matches=frame_points,
            frame_keypoint_grid=selection_grid,
            frame_max_keypoints=max_keypoints,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        reason = f"{method}: non-finite or wrong homography shape"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="ransac",
            status=AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            method=method,
            stage="ransac",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            reference_matches=reference_points,
            frame_matches=frame_points,
            frame_keypoint_grid=selection_grid,
            frame_max_keypoints=max_keypoints,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    inlier_mask = mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())

    if inlier_count < min_inliers:
        reason = f"{method}: not enough inliers"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="inlier_count",
            status=AlignmentStatus.INSUFFICIENT_INLIERS,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            selected_ransac_reprojection_threshold=ransac_threshold,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.INSUFFICIENT_INLIERS,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            method=method,
            stage="inlier_count",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            reference_matches=reference_points,
            frame_matches=frame_points,
            frame_keypoint_grid=selection_grid,
            frame_max_keypoints=max_keypoints,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    reference_inliers = reference_points[inlier_mask]
    frame_inliers = frame_points[inlier_mask]

    inliers_ok, inliers_reason = validate_inliers_distribution(
        reference_inliers,
        width=active_reference_features.image_width,
        height=active_reference_features.image_height,
        config=config,
    )
    if not inliers_ok:
        reason = f"{method}: {inliers_reason}"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="inliers_distribution",
            status=AlignmentStatus.INSUFFICIENT_INLIERS,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.INSUFFICIENT_INLIERS,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            method=method,
            stage="inliers_distribution",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            reference_matches=reference_points,
            frame_matches=frame_points,
            frame_keypoint_grid=selection_grid,
            frame_max_keypoints=max_keypoints,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    reproj_ok, median_error, reproj_reason = validate_reprojection_error(
        reference_inliers,
        frame_inliers,
        homography,
        config=config,
    )
    if not reproj_ok:
        reason = f"{method}: {reproj_reason}"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="reprojection_error",
            status=AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            median_error=median_error,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            median_error=median_error,
            method=method,
            stage="reprojection_error",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            reference_matches=reference_points,
            frame_matches=frame_points,
            frame_keypoint_grid=selection_grid,
            frame_max_keypoints=max_keypoints,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    quad_ok, quad_reason = validate_projected_quad(
        homography,
        reference_width=active_reference_features.image_width,
        reference_height=active_reference_features.image_height,
        frame_width=frame_width,
        frame_height=frame_height,
        config=config,
    )
    if not quad_ok:
        reason = f"{method}: {quad_reason}"
        _log_alignment_failure(
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            config=config,
            stage="projected_quad",
            status=AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            median_error=median_error,
            reason=reason,
        )
        return _failed(
            AlignmentStatus.HOMOGRAPHY_FAILED,
            raw_match_count=raw_match_count,
            inlier_count=inlier_count,
            median_error=median_error,
            method=method,
            stage="projected_quad",
            reason=reason,
            context=context,
            frame_features=frame_features,
            frame_shape=frame_shape,
            reference_matches=reference_points,
            frame_matches=frame_points,
            frame_keypoint_grid=selection_grid,
            frame_max_keypoints=max_keypoints,
            reference_features=active_reference_features,
            masked_alignment_used=masked_alignment_used,
            original_reference_feature_count=original_reference_feature_count,
            masked_reference_feature_count=masked_reference_feature_count,
        )

    return FrameAlignment(
        status=AlignmentStatus.SUCCESS,
        homography=homography.astype(np.float32),
        raw_match_count=raw_match_count,
        inlier_count=inlier_count,
        median_error=median_error,
        reference_inliers=reference_inliers.astype(np.float32),
        frame_inliers=frame_inliers.astype(np.float32),
        reference_matches=_as_match_points(reference_points),
        frame_matches=_as_match_points(frame_points),
        method=method,
        stage="success",
        reason=None,
        reference_feature_count=active_reference_features.count,
        frame_feature_count=frame_features.count,
        frame_max_keypoints=max_keypoints,
        frame_keypoint_grid=selection_grid,
        masked_alignment_used=masked_alignment_used,
        original_reference_feature_count=original_reference_feature_count,
        masked_reference_feature_count=masked_reference_feature_count,
        reference_size=(
            active_reference_features.image_width,
            active_reference_features.image_height,
        ),
        frame_size=(frame_width, frame_height),
    )


def _try_identity_alignment(
    *,
    context: InspectionContext,
    frame: np.ndarray,
) -> FrameAlignment | None:
    try:
        reference_image = _load_reference_image(context.reference_image.image_path)
    except Exception:
        return None

    if reference_image.shape != frame.shape:
        return None

    diff = cv2.absdiff(reference_image, frame)
    mean_diff = float(np.mean(diff))

    if mean_diff > 1.0:
        return None

    height, width = frame.shape[:2]
    homography = np.eye(3, dtype=np.float32)

    reference_inliers = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )

    return FrameAlignment(
        status=AlignmentStatus.SUCCESS,
        homography=homography,
        raw_match_count=4,
        inlier_count=4,
        median_error=0.0,
        reference_inliers=reference_inliers,
        frame_inliers=reference_inliers.copy(),
        reference_matches=reference_inliers.copy(),
        frame_matches=reference_inliers.copy(),
        method="identity",
        stage="identity",
        reason="reference view and frame are almost identical",
        reference_feature_count=context.reference_features.count,
        frame_feature_count=None,
        reference_size=(width, height),
        frame_size=(width, height),
    )


def _failed(
    status: AlignmentStatus,
    *,
    raw_match_count: int = 0,
    inlier_count: int = 0,
    median_error: float | None = None,
    method: str = "unknown",
    stage: str | None = None,
    reason: str | None = None,
    context: InspectionContext | None = None,
    frame_features: ImageFeatures | None = None,
    frame_shape: tuple[int, int] | None = None,
    frame_feature_count: int | None = None,
    frame_max_keypoints: int | None = None,
    frame_keypoint_grid: tuple[int, int] | None = None,
    reference_matches: np.ndarray | None = None,
    frame_matches: np.ndarray | None = None,
    reference_features: ImageFeatures | None = None,
    masked_alignment_used: bool = False,
    original_reference_feature_count: int | None = None,
    masked_reference_feature_count: int | None = None,
) -> FrameAlignment:
    reference_size = None
    reference_feature_count = None

    active_reference_features = reference_features
    if active_reference_features is None and context is not None:
        active_reference_features = context.reference_features

    if active_reference_features is not None:
        reference_feature_count = active_reference_features.count
        reference_size = (
            active_reference_features.image_width,
            active_reference_features.image_height,
        )

    if frame_features is not None:
        frame_feature_count = frame_features.count
        frame_size = (frame_features.image_width, frame_features.image_height)
    elif frame_shape is not None:
        frame_height, frame_width = frame_shape
        frame_size = (frame_width, frame_height)
    else:
        frame_size = None

    return FrameAlignment(
        status=status,
        homography=None,
        raw_match_count=raw_match_count,
        inlier_count=inlier_count,
        median_error=median_error,
        reference_matches=_as_match_points(reference_matches),
        frame_matches=_as_match_points(frame_matches),
        method=method,
        stage=stage,
        reason=reason,
        reference_feature_count=reference_feature_count,
        frame_feature_count=frame_feature_count,
        frame_max_keypoints=frame_max_keypoints,
        frame_keypoint_grid=frame_keypoint_grid,
        masked_alignment_used=masked_alignment_used,
        original_reference_feature_count=original_reference_feature_count,
        masked_reference_feature_count=masked_reference_feature_count,
        reference_size=reference_size,
        frame_size=frame_size,
    )


def _as_match_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None

    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(array) == 0:
        return None

    if not np.isfinite(array).all():
        return None

    return array


def _try_masked_reference_features(
    *,
    context: InspectionContext,
) -> ImageFeatures | None:
    polygons = _selected_reference_polygons(context)
    if not polygons:
        return None

    try:
        reference_image = _load_reference_image(context.reference_image.image_path)
        masked_reference = mask_reference_polygons(reference_image, polygons)
        return compute_features(
            masked_reference,
            max_side=SUPERPOINT_REFERENCE_MAX_SIDE,
            max_keypoints=SUPERPOINT_REFERENCE_MAX_KEYPOINTS,
            selection_grid=(
                SUPERPOINT_REFERENCE_GRID_ROWS,
                SUPERPOINT_REFERENCE_GRID_COLS,
            ),
        )
    except Exception as exc:
        throttled_log(
            logger,
            "warning",
            "inspection.alignment.masked_reference_failed",
            key=f"masked-reference:{context.reference_image.id}",
            standard_id=context.standard.id,
            reference_image_id=context.reference_image.id,
            error_type=type(exc).__name__,
            reason=str(exc),
        )
        return None


def _selected_reference_polygons(context: InspectionContext) -> list[list[list[float]]]:
    selected_ids = {item.id for item in context.selected_classes}
    if not selected_ids:
        return []

    polygons: list[list[list[float]]] = []
    for annotation in context.reference_image.annotations:
        if annotation.segment_class_id not in selected_ids:
            continue
        for polygon in annotation.points or []:
            if len(polygon) >= 3:
                polygons.append(polygon)
    return polygons


@lru_cache(maxsize=128)
def _load_reference_image(image_path: str) -> np.ndarray:
    return load_image(resolve_storage_path(image_path))


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _log_alignment_failure(
    *,
    context: InspectionContext,
    frame_features: ImageFeatures,
    frame_shape: tuple[int, int],
    config: AlignmentValidationConfig,
    stage: str,
    status: AlignmentStatus,
    raw_match_count: int | None = None,
    inlier_count: int | None = None,
    median_error: float | None = None,
    selected_ransac_reprojection_threshold: float | None = None,
    reason: str | None = None,
    frame_max_keypoints: int | None = None,
) -> None:
    frame_height, frame_width = frame_shape

    log_event(
        logger,
        "warning",
        "inspection.alignment.failed",
        stage=stage,
        status=status.value,
        reason=reason,
        standard_id=context.standard.id,
        reference_image_id=context.reference_image.id,
        selected_class_count=len(context.selected_classes),
        frame_size=(frame_width, frame_height),
        reference_feature_count=context.reference_features.count,
        reference_size=(
            context.reference_features.image_width,
            context.reference_features.image_height,
        ),
        frame_feature_count=frame_features.count,
        frame_max_keypoints=frame_max_keypoints,
        raw_match_count=raw_match_count,
        raw_match_threshold=MIN_RAW_MATCHES,
        inlier_count=inlier_count,
        inlier_threshold=MIN_INLIERS_FOR_ALIGNMENT,
        median_error=median_error,
        ransac_reprojection_threshold=RANSAC_REPROJECTION_THRESHOLD,
        selected_ransac_reprojection_threshold=selected_ransac_reprojection_threshold,
        grid_cols=config.grid_cols,
        grid_rows=config.grid_rows,
        min_grid_cells=config.min_grid_cells,
        max_cell_fraction=config.max_cell_fraction,
        max_median_error=config.max_median_error,
        min_quad_area_fraction=config.min_quad_area_fraction,
        min_quad_dim_fraction=config.min_quad_dim_fraction,
        max_off_screen_margin=config.max_off_screen_margin,
    )


@dataclass(slots=True, frozen=True)
class AlignmentValidationConfig:
    grid_cols: int = 4
    grid_rows: int = 4
    min_grid_cells: int = 2
    max_cell_fraction: float = 0.85
    max_median_error: float = 20.0
    min_quad_area_fraction: float = 0.005
    min_quad_dim_fraction: float = 0.05
    max_off_screen_margin: float = 2

    @classmethod
    def industrial(cls) -> AlignmentValidationConfig:
        return cls()

    @classmethod
    def strict(cls) -> AlignmentValidationConfig:
        return cls(
            min_grid_cells=3,
            max_cell_fraction=0.55,
            max_median_error=15.0,
            min_quad_area_fraction=0.03,
            min_quad_dim_fraction=0.15,
            max_off_screen_margin=0.75,
        )

    @classmethod
    def permissive(cls) -> AlignmentValidationConfig:
        return cls(
            min_grid_cells=1,
            max_cell_fraction=1.0,
            max_median_error=30.0,
            min_quad_area_fraction=0.001,
            min_quad_dim_fraction=0.02,
            max_off_screen_margin=4.0,
        )


def _find_best_homography(
    *,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    source = reference_points.reshape(-1, 1, 2)
    destination = frame_points.reshape(-1, 1, 2)
    method_candidates = [cv2.RANSAC]

    usac_magsac = getattr(cv2, "USAC_MAGSAC", None)
    if usac_magsac is not None:
        method_candidates.insert(0, usac_magsac)

    threshold_candidates = (
        RANSAC_REPROJECTION_THRESHOLD,
        RANSAC_REPROJECTION_THRESHOLD * 1.5,
        RANSAC_REPROJECTION_THRESHOLD * 2.0,
        RANSAC_REPROJECTION_THRESHOLD * 3.0,
    )

    best: tuple[int, float, np.ndarray, np.ndarray, float] | None = None

    for method in method_candidates:
        for threshold in threshold_candidates:
            try:
                homography, mask = cv2.findHomography(
                    source,
                    destination,
                    method=method,
                    ransacReprojThreshold=float(threshold),
                )
            except cv2.error:
                continue

            if homography is None or mask is None:
                continue
            if homography.shape != (3, 3) or not np.isfinite(homography).all():
                continue

            inlier_mask = mask.reshape(-1).astype(bool)
            inlier_count = int(inlier_mask.sum())
            if inlier_count < 4:
                continue

            median_error = _median_reprojection_error(
                reference_points[inlier_mask],
                frame_points[inlier_mask],
                homography,
            )
            candidate = (inlier_count, -median_error, homography, mask, float(threshold))

            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is None:
        return None, None, RANSAC_REPROJECTION_THRESHOLD

    return best[2], best[3], best[4]


def match_features(
    reference: ImageFeatures,
    frame: ImageFeatures,
    *,
    max_keypoints: int = SUPERPOINT_PHOTO_MAX_KEYPOINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    if reference.count < 10 or frame.count < 10:
        return None

    try:
        return match_feature_arrays(
            reference_keypoints=reference.keypoints.astype(np.float32, copy=False),
            reference_descriptors=reference.descriptors.astype(np.float32, copy=False),
            reference_size=(reference.image_width, reference.image_height),
            frame_keypoints=frame.keypoints.astype(np.float32, copy=False),
            frame_descriptors=frame.descriptors.astype(np.float32, copy=False),
            frame_size=(frame.image_width, frame.image_height),
            max_keypoints=max_keypoints,
        )
    except Exception as exc:
        throttled_log(
            logger,
            event="inspection.alignment.matcher_failed",
            repeated_event="inspection.alignment.matcher_failed.repeated",
            error=exc,
            throttle_key="inspection.alignment.matcher_failed",
            interval_sec=10.0,
        )
        return None


def validate_inliers_distribution(
    inliers: np.ndarray,
    *,
    width: int,
    height: int,
    config: AlignmentValidationConfig,
) -> tuple[bool, str]:
    cells_count = _count_occupied_cells(
        inliers,
        width=width,
        height=height,
        cols=config.grid_cols,
        rows=config.grid_rows,
    )
    if cells_count < config.min_grid_cells:
        return False, f"only {cells_count}/{config.min_grid_cells} grid cells covered"

    max_fraction = _max_cell_fraction(
        inliers,
        width=width,
        height=height,
        cols=config.grid_cols,
        rows=config.grid_rows,
    )
    if max_fraction > config.max_cell_fraction:
        return False, f"cell fraction {max_fraction:.2f} > {config.max_cell_fraction}"

    return True, ""


def validate_reprojection_error(
    reference_inliers: np.ndarray,
    frame_inliers: np.ndarray,
    homography: np.ndarray,
    *,
    config: AlignmentValidationConfig,
) -> tuple[bool, float, str]:
    median_error = _median_reprojection_error(
        reference_inliers,
        frame_inliers,
        homography,
    )

    if median_error > config.max_median_error:
        return (
            False,
            median_error,
            (f"median reproj error {median_error:.1f}px > {config.max_median_error}"),
        )
    return True, median_error, ""


def _median_reprojection_error(
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    homography: np.ndarray,
) -> float:
    projected = cv2.perspectiveTransform(
        reference_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - frame_points, axis=1)
    return float(np.median(errors))


def validate_projected_quad(
    homography: np.ndarray,
    *,
    reference_width: int,
    reference_height: int,
    frame_width: int,
    frame_height: int,
    config: AlignmentValidationConfig,
) -> tuple[bool, str]:
    if min(reference_width, reference_height, frame_width, frame_height) <= 0:
        return False, "invalid image dimensions"

    corners = np.array(
        [
            [0, 0],
            [reference_width - 1, 0],
            [reference_width - 1, reference_height - 1],
            [0, reference_height - 1],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    try:
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    except cv2.error:
        return False, "perspective transform threw cv2.error"

    if not np.isfinite(projected).all():
        return False, "non-finite values in projection"

    contour = projected.astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False, "projected quad is not convex"

    frame_area = frame_width * frame_height
    quad_area = abs(cv2.contourArea(contour))
    area_fraction = quad_area / frame_area
    if area_fraction < config.min_quad_area_fraction:
        return False, (
            f"quad area {area_fraction:.4f} < {config.min_quad_area_fraction}"
        )

    x, y, w, h = cv2.boundingRect(contour)
    w_fraction = w / frame_width
    h_fraction = h / frame_height
    if w_fraction < config.min_quad_dim_fraction:
        return False, f"quad width {w_fraction:.3f} < {config.min_quad_dim_fraction}"
    if h_fraction < config.min_quad_dim_fraction:
        return False, f"quad height {h_fraction:.3f} < {config.min_quad_dim_fraction}"

    margin_x = frame_width * config.max_off_screen_margin
    margin_y = frame_height * config.max_off_screen_margin
    if x + w < -margin_x or x > frame_width + margin_x:
        return False, "quad too far off-screen horizontally"
    if y + h < -margin_y or y > frame_height + margin_y:
        return False, "quad too far off-screen vertically"

    return True, ""


def _count_occupied_cells(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    cols: int,
    rows: int,
) -> int:
    if len(points) == 0 or width <= 0 or height <= 0:
        return 0
    occupied: set[tuple[int, int]] = set()
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            col = min(cols - 1, int((x / width) * cols))
            row = min(rows - 1, int((y / height) * rows))
            occupied.add((col, row))
    return len(occupied)


def _max_cell_fraction(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    cols: int,
    rows: int,
) -> float:
    if len(points) == 0 or width <= 0 or height <= 0:
        return 1.0
    counts: dict[tuple[int, int], int] = {}
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            col = min(cols - 1, int((x / width) * cols))
            row = min(rows - 1, int((y / height) * rows))
            key = (col, row)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 1.0
    return max(counts.values()) / sum(counts.values())
