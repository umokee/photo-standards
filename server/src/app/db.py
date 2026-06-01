from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import lru_cache

from app.config import settings
from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    return create_async_engine(
        url=settings.database_url_async,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    return create_engine(
        url=settings.database_url_sync,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )


AsyncSessionLocal = async_sessionmaker(
    bind=get_async_engine(),
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

SyncSessionLocal = sessionmaker(
    bind=get_sync_engine(),
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def get_sync_session() -> Session:
    return SyncSessionLocal()


async def dispose_engines() -> None:
    await get_async_engine().dispose()
    get_sync_engine().dispose()


