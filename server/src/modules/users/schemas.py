from datetime import datetime
import re
from typing import Annotated, Self
from uuid import UUID

from modules.users.constants import users
from pydantic import (
    BeforeValidator,
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _validate_username(value: str) -> str:
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if len(normalized) < 3:
        raise PydanticCustomError(
            "username_error", "Укажите имя пользователя длиной не менее 3 символов"
        )
    if len(normalized) > 100:
        raise PydanticCustomError(
            "username_error",
            "Укажите имя пользователя длиной не более 100 символов",
        )
    if not _USERNAME_RE.fullmatch(normalized):
        raise PydanticCustomError(
            "username_error",
            "Укажите имя пользователя латиницей, цифрами или символами ._-",
        )
    return normalized


def _validate_password(value: str) -> str:
    if not isinstance(value, str):
        return value

    if len(value) < 6:
        raise PydanticCustomError(
            "password_error", "Укажите пароль длиной не менее 6 символов"
        )
    if len(value) > 255:
        raise PydanticCustomError(
            "password_error", "Укажите пароль длиной не более 255 символов"
        )
    return value


def _validate_full_name(value: str) -> str:
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if len(normalized) < 3:
        raise PydanticCustomError(
            "full_name_error", "Укажите имя длиной не менее 3 символов"
        )
    if len(normalized) > 255:
        raise PydanticCustomError(
            "full_name_error", "Укажите имя длиной не более 255 символов"
        )
    return normalized


Username = Annotated[str, BeforeValidator(_validate_username)]
Password = Annotated[str, BeforeValidator(_validate_password)]
FullName = Annotated[str, BeforeValidator(_validate_full_name)]


class UserCreate(BaseModel):
    username: Username
    password: Password
    full_name: FullName
    role: str = users.roles.default

    @field_validator("role")
    @classmethod
    def validate_role(cls, val: str) -> str:
        if val not in users.roles:
            raise PydanticCustomError(
                "user_role_error",
                f"Выберите роль из списка: {', '.join(users.roles)}",
            )
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
            raise PydanticCustomError(
                "user_role_error",
                f"Выберите роль из списка: {', '.join(users.roles)}",
            )
        return val

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if not self.model_dump(exclude_unset=True):
            raise PydanticCustomError("user_update_error", "Укажите хотя бы одно поле")
        return self


class UserResponse(BaseModel):
    id: UUID
    username: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
