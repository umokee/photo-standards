from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from modules.core.standards.reference_constants import (
    SUPERPOINT_INPUT_STRIDE,
    SUPERPOINT_REALTIME_MAX_KEYPOINTS,
    SUPERPOINT_REALTIME_MAX_SIDE,
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
    max_side: int | None = SUPERPOINT_REALTIME_MAX_SIDE,
    max_keypoints: int = SUPERPOINT_REALTIME_MAX_KEYPOINTS,
) -> ImageFeatures:
    h0, w0 = image.shape[:2]
    tensor, scale_x, scale_y = _preprocess(image, max_side)

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

    if keypoints.shape[0] != descriptors.shape[0]:
        raise RuntimeError(
            "SuperPoint вернул несовместимые keypoints/descriptors: "
            f"{keypoints.shape} / {descriptors.shape}"
        )

    if scores is not None and keypoints.shape[0] > max_keypoints:
        top_idx = np.argpartition(-scores, max_keypoints)[:max_keypoints]
        keypoints = keypoints[top_idx]
        descriptors = descriptors[top_idx]
    elif keypoints.shape[0] > max_keypoints:
        keypoints = keypoints[:max_keypoints]
        descriptors = descriptors[:max_keypoints]

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
