from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
from infra.storage.file_storage import resolve_storage_path
from modules.core.standards.reference_features import ImageFeatures

_DESCRIPTOR_DIM = 256


def features_rel_path(standard_id: UUID, image_id: UUID) -> str:
    return (
        Path("standards") / str(standard_id) / "features" / f"{image_id}.npz"
    ).as_posix()


def save_features(features: ImageFeatures, rel_path: str) -> None:
    absolute = resolve_storage_path(rel_path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        absolute,
        keypoints=features.keypoints,
        descriptors=features.descriptors,
        image_width=np.int32(features.image_width),
        image_height=np.int32(features.image_height),
    )


def load_features(rel_path: str) -> ImageFeatures:
    absolute = resolve_storage_path(rel_path)
    with np.load(absolute) as data:
        return _build_image_features(data, rel_path)


def features_file_is_compatible(rel_path: str) -> bool:
    absolute = resolve_storage_path(rel_path)
    if not absolute.is_file():
        return False
    try:
        with np.load(absolute) as data:
            _build_image_features(data, rel_path)
    except Exception:
        return False
    return True


def delete_features(rel_path: str) -> None:
    absolute = resolve_storage_path(rel_path)
    absolute.unlink(missing_ok=True)


def _build_image_features(data: np.lib.npyio.NpzFile, rel_path: str) -> ImageFeatures:
    required = {"keypoints", "descriptors", "image_width", "image_height"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(
            f"Features файл {rel_path} повреждён: отсутствуют поля {sorted(missing)}."
        )

    keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    descriptors = np.asarray(data["descriptors"])
    image_width = int(data["image_width"])
    image_height = int(data["image_height"])

    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise RuntimeError(
            f"Features файл {rel_path} несовместим: keypoints формы (N, 2), "
            f"получено {tuple(keypoints.shape)}."
        )

    if descriptors.ndim != 2 or descriptors.shape[1] != _DESCRIPTOR_DIM:
        raise RuntimeError(
            f"Features файл {rel_path} несовместим: descriptors формы "
            f"(N, {_DESCRIPTOR_DIM}), получено {tuple(descriptors.shape)}."
        )

    if descriptors.dtype != np.float32:
        descriptors = descriptors.astype(np.float32, copy=False)

    if keypoints.shape[0] != descriptors.shape[0]:
        raise RuntimeError(
            f"Features файл {rel_path} несовместим: число keypoints и descriptors "
            f"не совпадает."
        )

    return ImageFeatures(
        keypoints=keypoints,
        descriptors=descriptors,
        image_width=image_width,
        image_height=image_height,
    )
