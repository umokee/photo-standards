from __future__ import annotations

import random

from modules.yolo.training.constants import training as training_constants
from modules.yolo.training.domain.types import (
    TrainingAnnotation,
    TrainingData,
    TrainingSplitPlan,
)


def validate_training_data(data: TrainingData) -> None:
    if not data.standards:
        raise ValueError("В группе нет эталонов")

    if not data.class_keys:
        raise ValueError("В группе нет классов сегментов")

    has_annotated = any(
        image.annotations
        for standard in data.standards
        for image in standard.images
        if image.is_annotated
    )
    if not has_annotated:
        raise ValueError("Нет размеченных изображений для обучения")


def plan_dataset_split(
    data: TrainingData,
    *,
    train_ratio: int,
    val_ratio: int,
) -> TrainingSplitPlan:
    _validate_ratios(train_ratio, val_ratio)

    test_ratio = 100 - train_ratio - val_ratio
    seed = data.group_id.int ^ (train_ratio << 8) ^ (val_ratio << 16)

    train_ids: list = []
    val_ids: list = []
    test_ids: list = []

    for standard in data.standards:
        image_ids = [image.image_id for image in standard.images if image.is_annotated]
        image_count = len(image_ids)

        if image_count == 0:
            continue

        if image_count < training_constants.min_images_to_train:
            raise ValueError(
                f"Минимум {training_constants.min_images_to_train} размеченных фото на эталон "
                f"(«{standard.standard_name}»: {image_count})"
            )

        rng = random.Random(seed ^ standard.standard_id.int)  # noqa: S311
        rng.shuffle(image_ids)

        val_n = _split_count(image_count, val_ratio)
        test_n = _split_count(image_count, test_ratio)
        train_n = image_count - val_n - test_n

        if train_n < 1:
            raise ValueError(
                f"«{standard.standard_name}»: размер train < 1 при {train_ratio}/{val_ratio}/{test_ratio}%"
            )

        train_ids.extend(image_ids[:train_n])
        val_ids.extend(image_ids[train_n : train_n + val_n])
        test_ids.extend(image_ids[train_n + val_n : train_n + val_n + test_n])

    if not train_ids:
        raise ValueError("Не удалось сформировать train split")

    if val_ratio > 0 and not val_ids:
        raise ValueError("Не удалось сформировать val split")

    return TrainingSplitPlan(
        train=train_ids,
        val=val_ids,
        test=test_ids,
    )


def polygons_to_yolo_lines(
    annotations: list[TrainingAnnotation],
    *,
    width: int,
    height: int,
) -> list[str]:
    lines: list[str] = []

    for annotation in annotations:
        if not annotation.points:
            continue

        polygon = annotation.points[0]
        normalized: list[str] = []

        for x, y in polygon:
            normalized.append(str(max(0, min(1, x / width))))
            normalized.append(str(max(0, min(1, y / height))))

        if normalized:
            lines.append(f"{annotation.class_index} " + " ".join(normalized))

    return lines


def _split_count(total: int, ratio: int) -> int:
    if ratio <= 0:
        return 0
    return max(1, int(total * ratio / 100))


def _validate_ratios(train_ratio: int, val_ratio: int) -> None:
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("train и val не могут быть отрицательными")

    if train_ratio > training_constants.train_ratio.max:
        raise ValueError(
            f"train не может превышать {training_constants.train_ratio.max}%"
        )

    if val_ratio > training_constants.val_ratio.max:
        raise ValueError(f"val не может превышать {training_constants.val_ratio.max}%")

    cap = min(training_constants.ratio_sum_max, 100)
    if train_ratio + val_ratio > cap:
        raise ValueError(f"Сумма train и val не может превышать {cap}%")
