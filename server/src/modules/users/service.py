from uuid import UUID

import bcrypt
from app.exception import ConflictError, NotFoundError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User
from .schemas import UserCreate, UserUpdate


async def _ensure_username_unique(
    db: AsyncSession,
    username: str,
    user_id: UUID | None = None,
) -> None:
    query = select(User).where(User.username == username)

    if user_id is not None:
        query = query.where(User.id != user_id)

    existing = await db.scalar(query)
    if existing:
        raise ConflictError(
            "Пользователь уже существует",
            details={
                "entity": "user",
                "entity_label": "Пользователь",
                "field": "username",
                "value": username,
            },
        )


async def get_users(
    db: AsyncSession,
) -> list[User]:
    result = await db.execute(select(User).order_by(User.id))
    return result.scalars().all()


async def get_user(
    db: AsyncSession,
    user_id: UUID,
) -> User:
    user = await db.get(User, user_id)
    if not user:
        raise NotFoundError("Пользователь", user_id)
    return user


async def create(
    db: AsyncSession,
    data: UserCreate,
) -> User:
    await _ensure_username_unique(db, data.username)
    data_dict = data.model_dump()
    data_dict["password_hash"] = bcrypt.hashpw(
        data_dict.pop("password").encode(), bcrypt.gensalt()
    ).decode()
    user = User(**data_dict)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def update(
    db: AsyncSession,
    user_id: UUID,
    data: UserUpdate,
) -> User:
    user = await get_user(db, user_id)

    if data.username is not None:
        await _ensure_username_unique(db, data.username, user_id)

    for key, value in data.model_dump(exclude_unset=True).items():
        if key == "password":
            user.password_hash = bcrypt.hashpw(
                value.encode(), bcrypt.gensalt()
            ).decode()
        else:
            setattr(user, key, value)

    await db.commit()
    await db.refresh(user)
    return user


async def delete(
    db: AsyncSession,
    user_id: UUID,
) -> None:
    user = await get_user(db, user_id)
    await db.delete(user)
    await db.commit()
