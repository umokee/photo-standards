from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class SamPromptPoint(BaseModel):
    x: float
    y: float
    label: Literal[0, 1] = 1


class SamClickRequest(BaseModel):
    image_id: UUID
    points: list[SamPromptPoint] = Field(min_length=1)


class SamClickResponse(BaseModel):
    points: list[list[float]]
    score: float
