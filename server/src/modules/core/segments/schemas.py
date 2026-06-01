from uuid import UUID

from modules.core.segments.constants import segments
from modules.core.schemas import AnnotationPoints, Name
from pydantic import BaseModel, ConfigDict, Field


class SegmentClassDraftItem(BaseModel):
    id: UUID | None = None
    name: Name
    hue: int = Field(
        segments.hue.default,
        ge=segments.hue.min,
        le=segments.hue.max,
    )


class SegmentClassCategoryDraftItem(BaseModel):
    id: UUID | None = None
    name: Name
    segment_classes: list[SegmentClassDraftItem] = Field(default_factory=list)


class SaveSegmentClassesRequest(BaseModel):
    categories: list[SegmentClassCategoryDraftItem] = Field(default_factory=list)
    ungrouped_classes: list[SegmentClassDraftItem] = Field(default_factory=list)
    deleted_category_ids: list[UUID] = Field(default_factory=list)
    deleted_class_ids: list[UUID] = Field(default_factory=list)


class SegmentClassResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    class_group_id: UUID | None
    name: str
    hue: int


class SegmentClassCategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    name: str
    segment_classes: list[SegmentClassResponse] = Field(default_factory=list)


class SaveSegmentClassesResponse(BaseModel):
    group_id: UUID
    categories: list[SegmentClassCategoryResponse] = Field(default_factory=list)
    ungrouped_classes: list[SegmentClassResponse] = Field(default_factory=list)


class SegmentClassWithPointsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    group_id: UUID
    class_group_id: UUID | None
    name: str
    hue: int
    points: AnnotationPoints


class AnnotationSave(BaseModel):
    points: AnnotationPoints
