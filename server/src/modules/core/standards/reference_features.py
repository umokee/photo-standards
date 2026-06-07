from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from modules.core.standards.reference_constants import (
    SUPERPOINT_INPUT_STRIDE,
    SUPERPOINT_VIDEO_GRID_COLS,
    SUPERPOINT_VIDEO_GRID_ROWS,
    SUPERPOINT_VIDEO_MAX_KEYPOINTS,
    SUPERPOINT_VIDEO_MAX_SIDE,
)
from modules.core.standards.reference_runtime import compute_superpoint_features


@dataclass(slots=True)
class ImageFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray
    image_width: int
    image_height: int

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])


def compute_features(
    image: np.ndarray,
    *,
    max_side: int | None = SUPERPOINT_VIDEO_MAX_SIDE,
    max_keypoints: int = SUPERPOINT_VIDEO_MAX_KEYPOINTS,
    selection_grid: tuple[int, int] | None = (
        SUPERPOINT_VIDEO_GRID_ROWS,
        SUPERPOINT_VIDEO_GRID_COLS,
    ),
) -> ImageFeatures:
    h0, w0 = image.shape[:2]
    tensor, scale_x, scale_y = _preprocess(image, max_side)
    processed_height = int(tensor.shape[-2])
    processed_width = int(tensor.shape[-1])

    features = compute_superpoint_features(
        torch.from_numpy(tensor),
        max_keypoints=max_keypoints,
    )

    keypoints = _as_numpy(features.get("keypoints"))
    descriptors = _as_numpy(features.get("descriptors"))

    scores_raw = features.get("keypoint_scores")
    if scores_raw is None:
        scores_raw = features.get("scores")

    scores = _as_numpy(scores_raw)

    if keypoints is None or descriptors is None:
        raise RuntimeError(
            f"SuperPoint вернул неполный результат: keys={list(features.keys())}"
        )

    if keypoints.ndim == 3:
        keypoints = keypoints[0]

    if descriptors.ndim == 3:
        descriptors = descriptors[0]

    if scores is not None and scores.ndim == 2:
        scores = scores[0]

    keypoints = np.asarray(keypoints, dtype=np.float32)
    descriptors = np.asarray(descriptors, dtype=np.float32)

    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != keypoints.shape[0]:
            scores = None

    if keypoints.shape[0] != descriptors.shape[0]:
        raise RuntimeError(
            "SuperPoint вернул несовместимые keypoints/descriptors: "
            f"{keypoints.shape} / {descriptors.shape}"
        )

    selected_idx = _select_keypoint_indices(
        keypoints=keypoints,
        scores=scores,
        max_keypoints=max_keypoints,
        selection_grid=selection_grid,
        image_width=processed_width,
        image_height=processed_height,
    )
    if selected_idx is not None:
        keypoints = keypoints[selected_idx]
        descriptors = descriptors[selected_idx]

    if scale_x != 1.0 or scale_y != 1.0:
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

    return ImageFeatures(
        keypoints=np.ascontiguousarray(keypoints, dtype=np.float32),
        descriptors=np.ascontiguousarray(descriptors, dtype=np.float32),
        image_width=w0,
        image_height=h0,
    )


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def _select_keypoint_indices(
    *,
    keypoints: np.ndarray,
    scores: np.ndarray | None,
    max_keypoints: int,
    selection_grid: tuple[int, int] | None,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    count = int(keypoints.shape[0])
    if count <= max_keypoints:
        return None

    score_order = _score_order(scores=scores, count=count)
    if selection_grid is None:
        return score_order[:max_keypoints]

    rows, cols = selection_grid
    if rows <= 0 or cols <= 0 or image_width <= 0 or image_height <= 0:
        return score_order[:max_keypoints]

    cell_count = rows * cols
    quota_per_cell = max(1, max_keypoints // cell_count)

    xs = np.clip(keypoints[:, 0], 0.0, float(max(image_width - 1, 0)))
    ys = np.clip(keypoints[:, 1], 0.0, float(max(image_height - 1, 0)))
    cell_x = np.minimum((xs * cols / max(float(image_width), 1.0)).astype(np.int32), cols - 1)
    cell_y = np.minimum((ys * rows / max(float(image_height), 1.0)).astype(np.int32), rows - 1)
    cell_ids = cell_y * cols + cell_x

    selected: list[int] = []
    selected_mask = np.zeros(count, dtype=bool)

    for cell_id in range(cell_count):
        cell_indices = np.flatnonzero(cell_ids == cell_id)
        if cell_indices.size == 0:
            continue

        if scores is not None:
            safe_scores = np.nan_to_num(scores[cell_indices], nan=-np.inf)
            ranked_cell = cell_indices[np.argsort(-safe_scores, kind="stable")]
        else:
            ranked_cell = cell_indices

        take = ranked_cell[:quota_per_cell]
        selected.extend(int(index) for index in take)
        selected_mask[take] = True

    if len(selected) < max_keypoints:
        for index in score_order:
            index_int = int(index)
            if selected_mask[index_int]:
                continue
            selected.append(index_int)
            selected_mask[index_int] = True
            if len(selected) >= max_keypoints:
                break

    if not selected:
        return score_order[:max_keypoints]

    return np.asarray(selected[:max_keypoints], dtype=np.int64)


def _score_order(*, scores: np.ndarray | None, count: int) -> np.ndarray:
    if scores is None:
        return np.arange(count, dtype=np.int64)

    safe_scores = np.nan_to_num(scores, nan=-np.inf)
    return np.argsort(-safe_scores, kind="stable").astype(np.int64, copy=False)


def _round_to_stride(value: int) -> int:
    rounded = (value // SUPERPOINT_INPUT_STRIDE) * SUPERPOINT_INPUT_STRIDE
    return max(SUPERPOINT_INPUT_STRIDE, rounded)


def _preprocess(
    image: np.ndarray,
    max_side: int | None,
) -> tuple[np.ndarray, float, float]:
    h0, w0 = image.shape[:2]

    if max_side is not None and max(h0, w0) > max_side:
        scale = max_side / max(h0, w0)
        target_w = _round_to_stride(int(round(w0 * scale)))
        target_h = _round_to_stride(int(round(h0 * scale)))
    else:
        target_w = _round_to_stride(w0)
        target_h = _round_to_stride(h0)

    if target_w != w0 or target_h != h0:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        scale_x = w0 / target_w
        scale_y = h0 / target_h
    else:
        resized = image
        scale_x = 1.0
        scale_y = 1.0

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    tensor = gray.astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)[np.newaxis, np.newaxis, ...]
    return tensor, scale_x, scale_y


def _as_numpy(value) -> np.ndarray | None:
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)
