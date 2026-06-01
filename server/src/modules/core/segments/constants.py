from __future__ import annotations

from constants_base import ConstModel


class HueConstants(ConstModel):
    default: int
    min: int
    max: int


class SegmentsConstants(ConstModel):
    hue: HueConstants


segments = SegmentsConstants(
    hue=HueConstants(
        default=210,
        min=0,
        max=359,
    )
)
