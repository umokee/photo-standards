from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.exception import ValidationError
from infra.storage.file_storage import resolve_storage_path
from modules.yolo.training.domain.dataset_split import polygons_to_yolo_lines
from modules.yolo.training.domain.types import TrainingData, TrainingSplitPlan

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingPaths:
    base_weights: Path
    base_weights_rel: str

    dataset_root: Path
    dataset_root_rel: str

    run_dir: Path
    run_dir_rel: str

    checkpoint: Path
    checkpoint_rel: str

    best_checkpoint: Path
    best_checkpoint_rel: str

    final_checkpoint: Path
    final_checkpoint_rel: str

    final_weights: Path
    final_weights_rel: str

    @classmethod
    def from_payload(cls, payload: dict) -> TrainingPaths:
        final_checkpoint_rel = payload.get("final_checkpoint_path")
        legacy_final_weights_rel = payload["final_weights_path"]

        if final_checkpoint_rel is None:
            final_checkpoint_rel = (
                legacy_final_weights_rel
                if legacy_final_weights_rel.endswith(".pt")
                else str(Path(legacy_final_weights_rel).with_suffix(".pt"))
            )

        final_weights_rel = legacy_final_weights_rel
        final_weights_path = Path(final_weights_rel)
        if final_weights_path.suffix.lower() != ".pt":
            final_weights_rel = str(final_weights_path.with_suffix(".pt"))

        return cls(
            base_weights=resolve_storage_path(payload["base_weights_path"]),
            base_weights_rel=payload["base_weights_path"],
            dataset_root=resolve_storage_path(payload["dataset_root"]),
            dataset_root_rel=payload["dataset_root"],
            run_dir=resolve_storage_path(payload["run_dir"]),
            run_dir_rel=payload["run_dir"],
            checkpoint=resolve_storage_path(payload["checkpoint_path"]),
            checkpoint_rel=payload["checkpoint_path"],
            best_checkpoint=resolve_storage_path(payload["best_checkpoint_path"]),
            best_checkpoint_rel=payload["best_checkpoint_path"],
            final_checkpoint=resolve_storage_path(final_checkpoint_rel),
            final_checkpoint_rel=final_checkpoint_rel,
            final_weights=resolve_storage_path(final_weights_rel),
            final_weights_rel=final_weights_rel,
        )


@dataclass(slots=True)
class TrainingParams:
    epochs: int
    imgsz: int
    batch_size: int
    train_ratio: int
    val_ratio: int

    @classmethod
    def from_payload(cls, payload: dict) -> TrainingParams:
        return cls(
            epochs=int(payload["epochs"]),
            imgsz=int(payload["imgsz"]),
            batch_size=int(payload["batch_size"]),
            train_ratio=int(payload["train_ratio"]),
            val_ratio=int(payload["val_ratio"]),
        )


@dataclass(slots=True)
class DatasetInfo:
    class_keys: list[str]
    class_meta: list[dict]
    yaml_path: Path


def base_weights_filename(architecture: str) -> str:
    return f"{architecture}.pt"


def base_weights_rel(architecture: str) -> str:
    return f"weights/{base_weights_filename(architecture)}"


def ensure_base_weights_exist(base_weights_path: str) -> None:
    if not resolve_storage_path(base_weights_path).is_file():
        raise ValidationError(f"Базовые веса не найдены: {base_weights_path}")


def build_training_paths(
    *,
    group_id: UUID,
    task_id: UUID,
    version: int,
    base_weights_path: str,
) -> TrainingPaths:
    task_root = Path("models") / str(group_id) / "tasks" / str(task_id)

    dataset_root_rel = (task_root / "dataset").as_posix()
    run_dir_rel = (task_root / "run").as_posix()
    checkpoint_rel = (task_root / "run" / "weights" / "last.pt").as_posix()
    best_checkpoint_rel = (task_root / "run" / "weights" / "best.pt").as_posix()
    final_checkpoint_rel = (
        Path("models") / str(group_id) / f"v{version}.pt"
    ).as_posix()
    final_weights_rel = (Path("models") / str(group_id) / f"v{version}.pt").as_posix()

    return TrainingPaths(
        base_weights=resolve_storage_path(base_weights_path),
        base_weights_rel=base_weights_path,
        dataset_root=resolve_storage_path(dataset_root_rel),
        dataset_root_rel=dataset_root_rel,
        run_dir=resolve_storage_path(run_dir_rel),
        run_dir_rel=run_dir_rel,
        checkpoint=resolve_storage_path(checkpoint_rel),
        checkpoint_rel=checkpoint_rel,
        best_checkpoint=resolve_storage_path(best_checkpoint_rel),
        best_checkpoint_rel=best_checkpoint_rel,
        final_checkpoint=resolve_storage_path(final_checkpoint_rel),
        final_checkpoint_rel=final_checkpoint_rel,
        final_weights=resolve_storage_path(final_weights_rel),
        final_weights_rel=final_weights_rel,
    )


def build_training_payload(
    *,
    model_id: UUID,
    group_id: UUID,
    version: int,
    architecture: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    train_ratio: int,
    val_ratio: int,
    paths: TrainingPaths,
) -> dict:
    return {
        "model_id": str(model_id),
        "group_id": str(group_id),
        "version": version,
        "architecture": architecture,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "base_weights_path": paths.base_weights_rel,
        "dataset_root": paths.dataset_root_rel,
        "run_dir": paths.run_dir_rel,
        "checkpoint_path": paths.checkpoint_rel,
        "best_checkpoint_path": paths.best_checkpoint_rel,
        "final_checkpoint_path": paths.final_checkpoint_rel,
        "final_weights_path": paths.final_weights_rel,
    }


def payload_version(payload: dict) -> int | None:
    raw = payload.get("version")
    if raw is None:
        return None

    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def build_temp_dataset(
    data: TrainingData,
    split: TrainingSplitPlan,
    *,
    dataset_root: Path,
) -> DatasetInfo:
    if dataset_root.exists():
        shutil.rmtree(dataset_root, ignore_errors=True)

    for split_name in ("train", "val", "test"):
        (dataset_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    image_map = {
        image.image_id: image
        for standard in data.standards
        for image in standard.images
    }

    for split_name, image_ids in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        for image_id in image_ids:
            image = image_map[image_id]
            _materialize_one(
                image_id=image_id,
                image_path=image.image_path,
                annotations=image.annotations,
                width=image.width,
                height=image.height,
                dataset_root=dataset_root,
                split_name=split_name,
            )

    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(data.class_keys)}",
                f"names: {json.dumps(data.class_keys, ensure_ascii=False)}",
            ]
        ),
        encoding="utf-8",
    )

    return DatasetInfo(
        class_keys=data.class_keys,
        class_meta=data.class_meta,
        yaml_path=yaml_path,
    )


def cleanup_task_artifacts(
    *,
    paths: TrainingPaths,
    preserve_dataset_and_checkpoint: bool,
) -> None:
    if preserve_dataset_and_checkpoint:
        return

    shutil.rmtree(paths.dataset_root, ignore_errors=True)
    shutil.rmtree(paths.run_dir, ignore_errors=True)


def resolve_model_weights_path(weights_path: str | None) -> Path | None:
    if not weights_path:
        return None
    return resolve_storage_path(weights_path)


def ensure_model_weights_ready(model) -> None:
    if not model.weights_path:
        raise ValidationError("У модели отсутствует путь к весам")

    weights_path = resolve_storage_path(model.weights_path)
    if not weights_path.is_file():
        raise ValidationError("Файл весов не найден")


def delete_model_artifacts(
    *,
    model_id: UUID,
    weights_path: str | None,
) -> None:
    resolved_weights_path = resolve_model_weights_path(weights_path)
    manifest_path = (
        resolved_weights_path.with_suffix(".classes.json")
        if resolved_weights_path is not None
        else None
    )

    for path in (resolved_weights_path, manifest_path):
        if path is None or not path.is_file():
            continue

        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.warning(
                "Failed to delete model file",
                extra={"model_id": str(model_id), "path": str(path)},
                exc_info=True,
            )


def _materialize_one(
    *,
    image_id: UUID,
    image_path: str,
    annotations,
    width: int,
    height: int,
    dataset_root: Path,
    split_name: str,
) -> None:
    source_path = resolve_storage_path(image_path)
    suffix = source_path.suffix or ".jpg"

    dst_image = dataset_root / "images" / split_name / f"{image_id}{suffix}"
    dst_label = dataset_root / "labels" / split_name / f"{image_id}.txt"

    try:
        os.symlink(source_path, dst_image)
    except OSError:
        shutil.copy2(source_path, dst_image)

    lines = polygons_to_yolo_lines(
        annotations,
        width=width,
        height=height,
    )
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
