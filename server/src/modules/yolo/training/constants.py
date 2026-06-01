from __future__ import annotations

from constants_base import (
    ConstModel,
    IntRange,
    IntValuesWithDefault,
    ValuesCollection,
    ValuesWithDefault,
)


class TrainingStatuses(ValuesCollection[str]):
    active: tuple[str, ...]
    default: str


class TrainingConstants(ConstModel):
    statuses: TrainingStatuses
    architectures: ValuesWithDefault
    image_size: IntValuesWithDefault
    epochs: IntRange
    batch_size: IntRange
    train_ratio: IntRange
    val_ratio: IntRange
    ratio_sum_max: int
    min_images_to_train: int


training = TrainingConstants(
    statuses=TrainingStatuses(
        values=("pending", "preparing", "training", "saving", "done", "failed"),
        active=("pending", "preparing", "training", "saving"),
        default="pending",
    ),
    architectures=ValuesWithDefault(
        values=(
            "yolo26n-seg",
            "yolo26s-seg",
            "yolo26m-seg",
            "yolo26l-seg",
            "yolo26x-seg",
        ),
        default="yolo26n-seg",
    ),
    image_size=IntValuesWithDefault(
        values=(320, 416, 512, 640, 768, 1024, 1280),
        default=640,
    ),
    epochs=IntRange(default=100, min=1, max=1000),
    batch_size=IntRange(default=16, min=1, max=256),
    train_ratio=IntRange(default=35, min=1, max=90),
    val_ratio=IntRange(default=10, min=1, max=50),
    ratio_sum_max=100,
    min_images_to_train=3,
)

__all__ = ["training"]
