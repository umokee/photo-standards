from typing import Annotated

from pydantic import BeforeValidator
from pydantic_core import PydanticCustomError


def _validate_name(value: str) -> str:
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if not normalized:
        raise PydanticCustomError("name_error", "Укажите название")
    if len(normalized) > 255:
        raise PydanticCustomError(
            "name_error", "Укажите название длиной не более 255 символов"
        )
    return normalized


Name = Annotated[str, BeforeValidator(_validate_name)]

AnnotationPoints = list[list[list[float]]]
