from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Camera


async def list_cameras(
    db: AsyncSession,
) -> list[Camera]:
    result = await db.execute(select(Camera).order_by(Camera.created_at.desc()))

    return list(result.scalars().all())


async def get_camera(
    db: AsyncSession,
    camera_id: UUID,
) -> Camera | None:
    return await db.get(Camera, camera_id)


async def list_active_cameras_for_healthcheck(
    db: AsyncSession,
) -> list[Camera]:
    result = await db.execute(
        select(Camera)
        .where(Camera.is_active.is_(True))
        .order_by(
            Camera.last_checked_at.asc().nullsfirst(),
            Camera.created_at.asc(),
        )
    )

    return list(result.scalars().all())
