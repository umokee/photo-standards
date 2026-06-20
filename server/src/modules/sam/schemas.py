from typing import Literal
from uuid import UUID

from pydantic import BaseModel, field_validator
from pydantic_core import PydanticCustomError


class SamPromptPoint(BaseModel):
    x: float
    y: float
    label: Literal[0, 1] = 1


class SamClickRequest(BaseModel):
    image_id: UUID
    points: list[SamPromptPoint]

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[SamPromptPoint]) -> list[SamPromptPoint]:
        if not value:
            raise PydanticCustomError("sam_points_error", "Добавьте хотя бы одну точку")
        return value


class SamClickResponse(BaseModel):
    points: list[list[float]]
    score: float
