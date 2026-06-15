from datetime import datetime
from typing import Self
from uuid import UUID

from modules.core.schemas import Name
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

_DESCRIPTION_MAX_LENGTH = 2000


def _normalize_description(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _validate_description_length(value: str | None) -> str | None:
    if value is not None and len(value) > _DESCRIPTION_MAX_LENGTH:
        raise PydanticCustomError(
            "group_description_error",
            f"Укажите описание длиной не более {_DESCRIPTION_MAX_LENGTH} символов"
        )
    return value


class GroupSegmentClassResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    class_group_id: UUID | None
    name: str
    hue: int


class GroupSegmentClassCategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    name: str
    segment_classes: list[GroupSegmentClassResponse] = Field(default_factory=list)


class GroupStatsResponse(BaseModel):
    standards_count: int = 0
    images_count: int = 0
    annotated_images_count: int = 0
    polygons_count: int = 0
    segment_class_groups_count: int = 0
    segment_classes_count: int = 0
    models_count: int = 0
    inspections_count: int = 0


class GroupStandardResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    name: str
    angle: str | None
    is_active: bool
    created_at: datetime
    reference_path: str | None = None
    images_count: int = 0
    annotated_images_count: int = 0


class GroupActiveModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    architecture: str
    version: int | None
    epochs: int | None
    imgsz: int
    batch_size: int | None
    num_classes: int | None
    metrics: dict | None
    class_keys: list[str] | None = None
    class_meta: list[dict] | None = None
    is_active: bool
    trained_at: datetime | None
    created_at: datetime


class GroupListItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    stats: GroupStatsResponse


class GroupDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    stats: GroupStatsResponse
    standards: list[GroupStandardResponse] = Field(default_factory=list)
    active_model: GroupActiveModelResponse | None = None
    segment_class_categories: list[GroupSegmentClassCategoryResponse] = Field(
        default_factory=list
    )
    ungrouped_segment_classes: list[GroupSegmentClassResponse] = Field(
        default_factory=list
    )


class GroupMutationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None = None
    created_at: datetime


class GroupCreate(BaseModel):
    name: Name
    description: str | None = None

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        return _validate_description_length(_normalize_description(value))


class GroupUpdate(BaseModel):
    name: Name | None = None
    description: str | None = None

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        return _validate_description_length(_normalize_description(value))

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if not self.model_dump(exclude_unset=True):
            raise PydanticCustomError(
                "group_update_error", "Укажите хотя бы одно поле"
            )
        return self
