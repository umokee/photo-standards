import math
from uuid import UUID

from modules.core.segments.constants import segments
from modules.core.schemas import AnnotationPoints, Name
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError


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

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: AnnotationPoints) -> AnnotationPoints:
        for polygon_index, polygon in enumerate(value):
            if len(polygon) < 3:
                raise PydanticCustomError(
                    "annotation_points_error",
                    f"Полигон #{polygon_index + 1} должен содержать минимум 3 точки"
                )

            for point_index, point in enumerate(polygon):
                if len(point) != 2:
                    raise PydanticCustomError(
                        "annotation_points_error",
                        f"Точка #{point_index + 1} в полигоне #{polygon_index + 1} должна содержать 2 координаты"
                    )

                x, y = point
                if not math.isfinite(x) or not math.isfinite(y):
                    raise PydanticCustomError(
                        "annotation_points_error",
                        "Координаты полигона должны быть конечными числами",
                    )

        return value
