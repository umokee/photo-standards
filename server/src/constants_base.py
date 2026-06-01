from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

ValueT = TypeVar("ValueT")


class ConstModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        frozen=True,
    )


class ValuesCollection(ConstModel, Generic[ValueT]):
    values: tuple[ValueT, ...]

    def __iter__(self) -> Iterator[ValueT]:
        return iter(self.values)

    def __contains__(self, item: object) -> bool:
        return item in self.values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> ValueT:
        return self.values[index]


class ValuesWithDefault(ValuesCollection[str]):
    default: str


class ValuesOnly(ValuesCollection[str]):
    pass


class IntValuesWithDefault(ValuesCollection[int]):
    default: int


class IntRange(ConstModel):
    default: int
    min: int
    max: int
