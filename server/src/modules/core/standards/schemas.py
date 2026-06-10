from datetime import datetime
from typing import Self
from uuid import UUID

from modules.core.standards.constants import standards
from modules.core.schemas import AnnotationPoints, Name
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StandardSegmentClassResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    class_group_id: UUID | None
    name: str
    hue: int


class StandardSegmentClassCategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    name: str
    segment_classes: list[StandardSegmentClassResponse] = Field(default_factory=list)


class StandardStatsResponse(BaseModel):
    images_count: int = 0
    annotated_images_count: int = 0
    unannotated_images_count: int = 0
    segment_classes_count: int = 0
    segment_class_categories_count: int = 0
    reference_image_id: UUID | None = None
    reference_path: str | None = None


class StandardImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    standard_id: UUID
    image_path: str
    is_reference: bool
    annotation_count: int = 0
    features_keypoint_count: int | None = None
    features_computed_at: datetime | None = None
    created_at: datetime


class StandardSegmentClassWithPointsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    class_group_id: UUID | None
    name: str
    hue: int
    points: AnnotationPoints


class StandardImageDetailResponse(StandardImageResponse):
    segment_classes: list[StandardSegmentClassWithPointsResponse] = Field(
        default_factory=list
    )


class StandardDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    name: str
    angle: str | None
    is_active: bool
    created_at: datetime
    stats: StandardStatsResponse
    images: list[StandardImageResponse] = Field(default_factory=list)
    segment_class_categories: list[StandardSegmentClassCategoryResponse] = Field(
        default_factory=list
    )
    ungrouped_segment_classes: list[StandardSegmentClassResponse] = Field(
        default_factory=list
    )
    # UI-only view: classes that are actually used by annotations of this standard.
    # The original fields above must keep returning all group classes for the image editor.
    used_segment_class_categories: list[StandardSegmentClassCategoryResponse] = Field(
        default_factory=list
    )
    used_ungrouped_segment_classes: list[StandardSegmentClassResponse] = Field(
        default_factory=list
    )


class StandardMutationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    name: str
    angle: str | None
    is_active: bool
    created_at: datetime


class StandardCreate(BaseModel):
    group_id: UUID
    name: Name
    angle: str | None = None

    @field_validator("angle")
    @classmethod
    def validate_angle(cls, val: str | None) -> str | None:
        if val is not None and val not in standards.angles:
            raise ValueError(
                f"Ракурс должен быть один из {', '.join(standards.angles)}"
            )
        return val


class StandardUpdate(BaseModel):
    name: Name | None = None
    angle: str | None = None
    is_active: bool | None = None

    @field_validator("angle")
    @classmethod
    def validate_angle(cls, val: str | None) -> str | None:
        if val is not None and val not in standards.angles:
            raise ValueError(
                f"Ракурс должен быть один из {', '.join(standards.angles)}"
            )
        return val

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if not self.model_dump(exclude_unset=True):
            raise ValueError("Необходимо передать хотя бы одно поле")
        return self
