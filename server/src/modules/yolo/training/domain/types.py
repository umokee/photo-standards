from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True)
class TrainingAnnotation:
    segment_class_id: UUID
    class_key: str
    class_index: int
    points: list[list[list[float]]]


@dataclass(slots=True)
class TrainingImage:
    image_id: UUID
    image_path: str
    width: int
    height: int
    is_annotated: bool
    annotations: list[TrainingAnnotation]


@dataclass(slots=True)
class TrainingStandard:
    standard_id: UUID
    standard_name: str
    angle: str | None
    images: list[TrainingImage]


@dataclass(slots=True)
class TrainingData:
    group_id: UUID
    group_name: str
    class_keys: list[str]
    class_meta: list[dict]
    standards: list[TrainingStandard]


@dataclass(slots=True)
class TrainingSplitPlan:
    train: list[UUID]
    val: list[UUID]
    test: list[UUID]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)
