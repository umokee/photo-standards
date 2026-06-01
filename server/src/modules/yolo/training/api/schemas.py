from __future__ import annotations

from datetime import datetime
from typing import Self
from uuid import UUID

from modules.yolo.training.constants import training
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrainRequest(BaseModel):
    group_id: UUID

    architecture: str = training.architectures.default
    epochs: int = Field(
        training.epochs.default,
        ge=training.epochs.min,
        le=training.epochs.max,
    )
    imgsz: int = training.image_size.default
    batch_size: int = Field(
        training.batch_size.default,
        ge=training.batch_size.min,
        le=training.batch_size.max,
    )

    train_ratio: int = Field(
        training.train_ratio.default,
        ge=training.train_ratio.min,
        le=training.train_ratio.max,
    )
    val_ratio: int = Field(
        training.val_ratio.default,
        ge=training.val_ratio.min,
        le=training.val_ratio.max,
    )

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, value: str) -> str:
        if value not in training.architectures:
            raise ValueError(f"Неизвестная архитектура: {value}")
        return value

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, value: int) -> int:
        if value not in training.image_size:
            raise ValueError(f"Некорректный размер изображения: {value}")
        return value

    @model_validator(mode="after")
    def validate_ratio_sum(self) -> Self:
        safe = min(training.ratio_sum_max, 100)
        if self.train_ratio + self.val_ratio > safe:
            raise ValueError(f"Сумма train и val должна быть не больше {safe}%")
        return self


class TrainingStartResponse(BaseModel):
    task_id: UUID
    model_id: UUID


class MlModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID

    architecture: str
    weights_path: str | None = None
    version: int | None = None

    epochs: int | None = None
    imgsz: int
    batch_size: int | None = None

    num_classes: int | None = None
    class_keys: list[str] | None = None
    class_meta: list[dict] | None = None
    metrics: dict | None = None

    train_ratio: int | None = None
    val_ratio: int | None = None
    test_ratio: int | None = None

    total_images: int | None = None
    train_count: int | None = None
    val_count: int | None = None
    test_count: int | None = None

    is_active: bool
    trained_at: datetime | None = None
    created_at: datetime
