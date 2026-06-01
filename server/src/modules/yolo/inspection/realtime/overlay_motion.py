from __future__ import annotations

import cv2
import numpy as np

from .constants import (
    MOTION_MAX_ERROR,
    MOTION_MAX_POINTS,
    MOTION_MAX_SCALE,
    MOTION_MAX_STALE_FRAMES,
    MOTION_MAX_TRANSLATION_FRACTION,
    MOTION_MIN_POINTS,
    MOTION_MIN_SCALE,
    MOTION_RANSAC_THRESHOLD,
    MOTION_REINIT_POINTS,
)


class OverlayMotionTracker:
    def __init__(self) -> None:
        self._base_to_current = np.eye(3, dtype=np.float32)
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self._stale_frames = 0
        self._frame_size: tuple[int, int] | None = None

    def reset(self, frame: np.ndarray) -> None:
        self._base_to_current = np.eye(3, dtype=np.float32)
        self._stale_frames = 0
        self._frame_size = (frame.shape[1], frame.shape[0])
        self._init_points(frame)

    def track(self, frame: np.ndarray) -> np.ndarray | None:
        if self._prev_gray is None or self._prev_points is None:
            self.reset(frame)
            return self._base_to_current.copy()

        if len(self._prev_points) < MOTION_MIN_POINTS:
            self._init_points(frame)
            return self._base_to_current.copy()

        gray = _gray(frame)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        if next_points is None or status is None:
            return self._track_lost()

        prev = self._prev_points.reshape(-1, 2)
        nxt = next_points.reshape(-1, 2)

        good = status.reshape(-1).astype(bool)
        if errors is not None:
            good &= errors.reshape(-1) < MOTION_MAX_ERROR

        prev_good = prev[good]
        nxt_good = nxt[good]

        if len(prev_good) < MOTION_MIN_POINTS:
            return self._track_lost()

        affine, inliers = cv2.estimateAffinePartial2D(
            prev_good,
            nxt_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=MOTION_RANSAC_THRESHOLD,
            maxIters=300,
            confidence=0.98,
        )

        if affine is None or inliers is None:
            return self._track_lost()

        inliers = inliers.reshape(-1).astype(bool)
        if int(inliers.sum()) < MOTION_MIN_POINTS:
            return self._track_lost()

        delta = np.eye(3, dtype=np.float32)
        delta[:2, :] = affine.astype(np.float32)

        candidate = _normalize_transform(delta @ self._base_to_current)
        if candidate is None or not self._is_reasonable_transform(candidate):
            return self._track_lost()

        self._base_to_current = candidate
        self._prev_gray = gray
        self._prev_points = nxt_good[inliers].reshape(-1, 1, 2).astype(np.float32)
        self._stale_frames = 0

        if len(self._prev_points) < MOTION_REINIT_POINTS:
            self._init_points(frame)

        return self._base_to_current.copy()

    def _init_points(self, frame: np.ndarray) -> None:
        gray = _gray(frame)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MOTION_MAX_POINTS,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        self._prev_gray = gray
        self._prev_points = points.astype(np.float32) if points is not None else None
        self._frame_size = (frame.shape[1], frame.shape[0])

    def _track_lost(self) -> np.ndarray | None:
        self._stale_frames += 1

        if self._stale_frames <= MOTION_MAX_STALE_FRAMES:
            return self._base_to_current.copy()

        self._prev_gray = None
        self._prev_points = None
        return None

    def _is_reasonable_transform(self, transform: np.ndarray) -> bool:
        if self._frame_size is None:
            return True

        width, height = self._frame_size
        tx = abs(float(transform[0, 2]))
        ty = abs(float(transform[1, 2]))

        if tx > width * MOTION_MAX_TRANSLATION_FRACTION:
            return False
        if ty > height * MOTION_MAX_TRANSLATION_FRACTION:
            return False

        sx = float(np.linalg.norm(transform[:2, 0]))
        sy = float(np.linalg.norm(transform[:2, 1]))

        return (
            MOTION_MIN_SCALE <= sx <= MOTION_MAX_SCALE
            and MOTION_MIN_SCALE <= sy <= MOTION_MAX_SCALE
        )


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _normalize_transform(transform: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(transform, dtype=np.float32)

    if matrix.shape != (3, 3):
        return None
    if not np.isfinite(matrix).all():
        return None
    if abs(float(matrix[2, 2])) < 1e-9:
        return None

    matrix = matrix / matrix[2, 2]
    if not np.isfinite(matrix).all():
        return None

    return matrix.astype(np.float32)
