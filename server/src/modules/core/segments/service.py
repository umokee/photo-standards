from __future__ import annotations

from uuid import UUID

from app.exception import ConflictError, ValidationError
from modules.core.groups import crud as groups_crud
from modules.core.schemas import AnnotationPoints
from modules.core.standards import crud as standards_crud
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud
from .models import SegmentAnnotation, SegmentClass, SegmentClassGroup
from .schemas import (
    AnnotationSave,
    SaveSegmentClassesRequest,
    SaveSegmentClassesResponse,
    SegmentClassCategoryResponse,
    SegmentClassResponse,
    SegmentClassWithPointsResponse,
)


async def list_segment_classes(
    db: AsyncSession,
    group_id: UUID,
) -> SaveSegmentClassesResponse:
    await groups_crud.get_group(db, group_id=group_id)
    categories, ungrouped_classes = await crud.get_group_tree(db, group_id=group_id)
    return _build_save_response(group_id, categories, ungrouped_classes)


async def save_annotation(
    db: AsyncSession,
    segment_class_id: UUID,
    image_id: UUID,
    data: AnnotationSave,
) -> SegmentClassWithPointsResponse:
    segment_class = await crud.get_segment_class(
        db,
        segment_class_id=segment_class_id,
    )
    image = await standards_crud.get_image_with_standard(
        db,
        image_id=image_id,
    )

    if image.standard.group_id != segment_class.group_id:
        raise ValidationError("Класс и фото принадлежат разным группам изделия")

    try:
        annotation = await crud.get_annotation(
            db,
            segment_class_id=segment_class_id,
            image_id=image_id,
        )

        if not data.points:
            if annotation is not None:
                await db.delete(annotation)
        elif annotation is None:
            db.add(
                SegmentAnnotation(
                    segment_class_id=segment_class_id,
                    image_id=image_id,
                    points=data.points,
                )
            )
        else:
            annotation.points = data.points

        await db.commit()
    except IntegrityError as error:
        await db.rollback()
        raise ConflictError("Конфликт при сохранении аннотации") from error
    except Exception:
        await db.rollback()
        raise

    return _build_segment_class_with_points_response(segment_class, data.points)


async def save_segment_classes(
    db: AsyncSession,
    group_id: UUID,
    data: SaveSegmentClassesRequest,
) -> SaveSegmentClassesResponse:
    await groups_crud.get_group(db, group_id=group_id)
    _validate_request_payload(data)

    try:
        for class_id in data.deleted_class_ids:
            segment_class = await crud.get_segment_class_or_none(
                db,
                segment_class_id=class_id,
            )
            if segment_class is not None:
                _ensure_segment_class_belongs_to_group(segment_class, group_id)
                await db.delete(segment_class)

        for category_id in data.deleted_category_ids:
            category = await crud.get_category_or_none(
                db,
                category_id=category_id,
            )
            if category is not None:
                _ensure_category_belongs_to_group(category, group_id)

                attached_classes = await crud.list_category_segment_classes(
                    db,
                    category_id=category.id,
                )
                for item in attached_classes:
                    item.class_group_id = None

                await db.delete(category)

        await db.flush()

        for category_item in data.categories:
            if category_item.id is None:
                await crud.ensure_category_name_unique(
                    db,
                    group_id=group_id,
                    name=category_item.name,
                )
                category = SegmentClassGroup(
                    group_id=group_id,
                    name=category_item.name,
                )
                db.add(category)
                await db.flush()
            else:
                category = await crud.get_category(
                    db,
                    category_id=category_item.id,
                )
                _ensure_category_belongs_to_group(category, group_id)

                if category.name != category_item.name:
                    await crud.ensure_category_name_unique(
                        db,
                        group_id=group_id,
                        name=category_item.name,
                        exclude_category_id=category.id,
                    )

                category.name = category_item.name

            for class_item in category_item.segment_classes:
                if class_item.id is None:
                    await crud.ensure_segment_class_name_unique(
                        db,
                        group_id=group_id,
                        name=class_item.name,
                    )
                    db.add(
                        SegmentClass(
                            group_id=group_id,
                            class_group_id=category.id,
                            name=class_item.name,
                            hue=class_item.hue,
                        )
                    )
                else:
                    segment_class = await crud.get_segment_class(
                        db,
                        segment_class_id=class_item.id,
                    )
                    _ensure_segment_class_belongs_to_group(segment_class, group_id)

                    if segment_class.name != class_item.name:
                        await crud.ensure_segment_class_name_unique(
                            db,
                            group_id=group_id,
                            name=class_item.name,
                            exclude_segment_class_id=segment_class.id,
                        )

                    segment_class.class_group_id = category.id
                    segment_class.name = class_item.name
                    segment_class.hue = class_item.hue

        for class_item in data.ungrouped_classes:
            if class_item.id is None:
                await crud.ensure_segment_class_name_unique(
                    db,
                    group_id=group_id,
                    name=class_item.name,
                )
                db.add(
                    SegmentClass(
                        group_id=group_id,
                        class_group_id=None,
                        name=class_item.name,
                        hue=class_item.hue,
                    )
                )
            else:
                segment_class = await crud.get_segment_class(
                    db,
                    segment_class_id=class_item.id,
                )
                _ensure_segment_class_belongs_to_group(segment_class, group_id)

                if segment_class.name != class_item.name:
                    await crud.ensure_segment_class_name_unique(
                        db,
                        group_id=group_id,
                        name=class_item.name,
                        exclude_segment_class_id=segment_class.id,
                    )

                segment_class.class_group_id = None
                segment_class.name = class_item.name
                segment_class.hue = class_item.hue

        await db.commit()
    except ConflictError:
        raise
    except IntegrityError as error:
        await db.rollback()
        raise ConflictError("Конфликт имен в категориях или классах") from error
    except Exception:
        await db.rollback()
        raise

    categories, ungrouped_classes = await crud.get_group_tree(db, group_id=group_id)
    return _build_save_response(group_id, categories, ungrouped_classes)


def _validate_request_payload(
    data: SaveSegmentClassesRequest,
) -> None:
    seen_category_names: set[str] = set()
    seen_class_names: set[str] = set()

    for category in data.categories:
        category_name = _norm(category.name)
        if category_name in seen_category_names:
            raise ConflictError("Категории не должны дублироваться")
        seen_category_names.add(category_name)

        for item in category.segment_classes:
            class_name = _norm(item.name)
            if class_name in seen_class_names:
                raise ConflictError(
                    "Имена классов должны быть уникальны в пределах группы"
                )
            seen_class_names.add(class_name)

    for item in data.ungrouped_classes:
        class_name = _norm(item.name)
        if class_name in seen_class_names:
            raise ConflictError("Имена классов должны быть уникальны в пределах группы")
        seen_class_names.add(class_name)


def _norm(value: str) -> str:
    return value.strip().casefold()


def _ensure_category_belongs_to_group(
    category: SegmentClassGroup,
    group_id: UUID,
) -> None:
    if category.group_id != group_id:
        raise ValidationError("Категория принадлежит другой группе изделия")


def _ensure_segment_class_belongs_to_group(
    segment_class: SegmentClass,
    group_id: UUID,
) -> None:
    if segment_class.group_id != group_id:
        raise ValidationError("Класс принадлежит другой группе изделия")


def _build_segment_class_response(
    segment_class: SegmentClass,
) -> SegmentClassResponse:
    return SegmentClassResponse.model_validate(segment_class)


def _build_category_response(
    category: SegmentClassGroup,
) -> SegmentClassCategoryResponse:
    items = sorted(category.segment_classes, key=lambda item: item.name.lower())
    return SegmentClassCategoryResponse(
        id=category.id,
        group_id=category.group_id,
        name=category.name,
        segment_classes=[_build_segment_class_response(item) for item in items],
    )


def _build_save_response(
    group_id: UUID,
    categories: list[SegmentClassGroup],
    ungrouped_classes: list[SegmentClass],
) -> SaveSegmentClassesResponse:
    sorted_categories = sorted(categories, key=lambda item: item.name.lower())
    sorted_ungrouped = sorted(ungrouped_classes, key=lambda item: item.name.lower())

    return SaveSegmentClassesResponse(
        group_id=group_id,
        categories=[_build_category_response(item) for item in sorted_categories],
        ungrouped_classes=[
            _build_segment_class_response(item) for item in sorted_ungrouped
        ],
    )


def _build_segment_class_with_points_response(
    segment_class: SegmentClass,
    points: AnnotationPoints,
) -> SegmentClassWithPointsResponse:
    return SegmentClassWithPointsResponse(
        id=segment_class.id,
        group_id=segment_class.group_id,
        class_group_id=segment_class.class_group_id,
        name=segment_class.name,
        hue=segment_class.hue,
        points=points,
    )
