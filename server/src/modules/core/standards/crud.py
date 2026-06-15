from __future__ import annotations

from uuid import UUID

from app.exception import ConflictError, NotFoundError
from modules.core.groups.models import Group
from modules.core.segments.models import (
    SegmentAnnotation,
    SegmentClassGroup,
)
from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Standard, StandardImage


async def get_standard(
    db: AsyncSession,
    *,
    standard_id: UUID,
) -> Standard:
    standard = await db.get(Standard, standard_id)
    if standard is None:
        raise NotFoundError("Эталон", standard_id)
    return standard


async def get_standard_with_relations(
    db: AsyncSession,
    *,
    standard_id: UUID,
) -> Standard:
    result = await db.execute(
        select(Standard)
        .options(
            selectinload(Standard.images).selectinload(StandardImage.annotations),
            selectinload(Standard.group)
            .selectinload(Group.segment_class_groups)
            .selectinload(SegmentClassGroup.segment_classes),
            selectinload(Standard.group).selectinload(Group.segment_classes),
        )
        .where(Standard.id == standard_id)
    )
    standard = result.scalar_one_or_none()
    if standard is None:
        raise NotFoundError("Эталон", standard_id)
    return standard


async def create_standard(
    db: AsyncSession,
    *,
    group_id: UUID,
    name: str,
    angle: str | None,
) -> Standard:
    standard = Standard(
        group_id=group_id,
        name=name,
        angle=angle,
    )
    db.add(standard)
    await db.commit()
    await db.refresh(standard)
    return standard


async def ensure_standard_name_unique(
    db: AsyncSession,
    *,
    group_id: UUID,
    name: str,
    angle: str | None,
    exclude_id: UUID | None = None,
) -> None:
    query = select(Standard.id).where(
        Standard.group_id == group_id,
        Standard.name == name,
    )

    if angle is None:
        query = query.where(Standard.angle.is_(None))
    else:
        query = query.where(Standard.angle == angle)

    if exclude_id is not None:
        query = query.where(Standard.id != exclude_id)

    if await db.scalar(query) is not None:
        raise ConflictError(
            "Эталон уже существует",
            details={
                "entity": "standard",
                "entity_label": "Эталон",
                "field": "name",
                "value": name,
                "group_id": str(group_id),
                "angle": angle,
            },
        )


async def list_standard_image_paths(
    db: AsyncSession,
    *,
    standard_id: UUID,
) -> list[tuple[str, str | None]]:
    result = await db.execute(
        select(StandardImage.image_path, StandardImage.features_path).where(
            StandardImage.standard_id == standard_id
        )
    )
    return list(result.all())


async def create_standard_images(
    db: AsyncSession,
    *,
    standard_id: UUID,
    items: list[tuple[UUID, str]],
) -> None:
    for image_id, image_path in items:
        db.add(
            StandardImage(
                id=image_id,
                standard_id=standard_id,
                image_path=image_path,
                is_reference=False,
            )
        )
    await db.commit()


async def list_standard_images(
    db: AsyncSession,
    *,
    image_ids: list[UUID],
) -> list[StandardImage]:
    if not image_ids:
        return []

    result = await db.execute(
        select(StandardImage)
        .options(selectinload(StandardImage.annotations))
        .where(StandardImage.id.in_(image_ids))
    )
    return list(result.scalars().all())


async def get_image(
    db: AsyncSession,
    *,
    image_id: UUID,
) -> StandardImage:
    image = await db.get(StandardImage, image_id)
    if image is None:
        raise NotFoundError("Фото", image_id)
    return image


async def get_image_with_standard(
    db: AsyncSession,
    *,
    image_id: UUID,
) -> StandardImage:
    result = await db.execute(
        select(StandardImage)
        .options(selectinload(StandardImage.standard))
        .where(StandardImage.id == image_id)
    )
    image = result.scalar_one_or_none()
    if image is None:
        raise NotFoundError("Фото", image_id)
    return image


async def get_image_with_annotations(
    db: AsyncSession,
    *,
    image_id: UUID,
) -> StandardImage:
    result = await db.execute(
        select(StandardImage)
        .options(
            selectinload(StandardImage.standard),
            selectinload(StandardImage.annotations).selectinload(
                SegmentAnnotation.segment_class
            ),
        )
        .where(StandardImage.id == image_id)
    )
    image = result.scalar_one_or_none()
    if image is None:
        raise NotFoundError("Фото", image_id)
    return image


async def set_standard_reference(
    db: AsyncSession,
    *,
    image: StandardImage,
) -> StandardImage:
    # is_reference теперь означает "фото участвует в пуле проверки".
    # Повторное нажатие снимает метку. Остальные фото этого стандарта не трогаем.
    image.is_reference = not bool(image.is_reference)
    await db.commit()
    await db.refresh(image)
    return image


async def delete_standard_image(
    db: AsyncSession,
    *,
    image: StandardImage,
) -> None:
    await db.delete(image)
    await db.commit()
