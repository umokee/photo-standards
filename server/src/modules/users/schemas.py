from datetime import datetime
from typing import Annotated, Self
from uuid import UUID

from modules.users.constants import users
from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

Username = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_.-]+$",
    ),
]

Password = Annotated[
    str,
    StringConstraints(
        min_length=6,
        max_length=255,
    ),
]

FullName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=255,
    ),
]


class UserCreate(BaseModel):
    username: Username
    password: Password
    full_name: FullName
    role: str = users.roles.default

    @field_validator("role")
    @classmethod
    def validate_role(cls, val: str) -> str:
        if val not in users.roles:
            raise ValueError(f"Роль должна быть одна из {', '.join(users.roles)}")
        return val


class UserUpdate(BaseModel):
    username: Username | None = None
    password: Password | None = None
    full_name: FullName | None = None
    role: str | None = None
    is_active: bool | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, val: str | None) -> str | None:
        if val is not None and val not in users.roles:
            raise ValueError(f"Роль должна быть одна из {', '.join(users.roles)}")
        return val

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if not self.model_dump(exclude_unset=True):
            raise ValueError("Необходимо передать хотя бы одно поле")
        return self


class UserResponse(BaseModel):
    id: UUID
    username: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
