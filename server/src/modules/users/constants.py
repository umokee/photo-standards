from __future__ import annotations

from constants_base import ConstModel, ValuesWithDefault


class UsersConstants(ConstModel):
    roles: ValuesWithDefault


users = UsersConstants(
    roles=ValuesWithDefault(
        values=("operator", "admin"),
        default="operator",
    )
)
