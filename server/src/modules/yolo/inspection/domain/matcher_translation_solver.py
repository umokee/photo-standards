from __future__ import annotations

import numpy as np
from modules.yolo.inspection.domain.matcher_geometry import bbox_diag
from modules.yolo.inspection.domain.matcher_structs import BBox, TranslationSolveResult


def solve_translation_from_projected_points(
    *,
    local_frame: np.ndarray,
    projected_reference: np.ndarray,
    min_support: int,
    threshold: float,
    global_bbox: BBox,
) -> tuple[TranslationSolveResult | None, str | None]:
    residuals = local_frame.astype(np.float32) - projected_reference.astype(np.float32)
    if (
        residuals.ndim != 2
        or residuals.shape[1] < 2
        or not np.isfinite(residuals).all()
    ):
        return None, "invalid_residuals"

    median_residual = np.median(residuals[:, :2], axis=0).astype(np.float32)
    if not np.isfinite(median_residual).all():
        return None, "invalid_median_residual"

    residual_errors = np.linalg.norm(
        residuals[:, :2] - median_residual[None, :], axis=1
    )
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        return None, "invalid_residual_errors"

    inlier_mask = residual_errors <= threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(residual_errors))
    if inlier_count < min_support:
        return None, "too_few_inliers"

    inlier_ratio = inlier_count / max(1, candidate_count)
    median_error = float(np.median(residual_errors[inlier_mask]))
    shift_x = float(median_residual[0])
    shift_y = float(median_residual[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / max(1.0, bbox_diag(global_bbox))

    return (
        TranslationSolveResult(
            residuals=residuals,
            residual_errors=residual_errors,
            inlier_mask=inlier_mask,
            median_residual=median_residual,
            candidate_count=candidate_count,
            inlier_count=inlier_count,
            inlier_ratio=float(inlier_ratio),
            median_error=float(median_error),
            shift_x=shift_x,
            shift_y=shift_y,
            shift_factor=float(shift_factor),
        ),
        None,
    )
