from __future__ import annotations

from constants_base import ConstModel, ValuesOnly


class StandardsConstants(ConstModel):
    angles: ValuesOnly


standards = StandardsConstants(
    angles=ValuesOnly(
        values=("front", "top", "left", "right", "back"),
    )
)
