from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter

from .schemas import (
    GroupCreate,
    GroupDetailResponse,
    GroupListItemResponse,
    GroupMutationResponse,
    GroupUpdate,
)
from .service import (
    create_group,
    delete_group,
    get_group,
    get_groups,
    update_group,
)

router = APIRouter(prefix="/groups", tags=["groups"])


@router.get("", response_model=list[GroupListItemResponse])
async def list_groups_route(
    db: DbSession,
) -> list[GroupListItemResponse]:
    return await get_groups(db)


@router.get("/{group_id}", response_model=GroupDetailResponse)
async def get_group_route(
    db: DbSession,
    group_id: UUID,
) -> GroupDetailResponse:
    return await get_group(db, group_id)


@router.post("", response_model=GroupMutationResponse, status_code=201)
async def create_group_route(
    db: DbSession,
    data: GroupCreate,
) -> GroupMutationResponse:
    return await create_group(db, data)


@router.put("/{group_id}", response_model=GroupMutationResponse)
async def update_group_route(
    db: DbSession,
    group_id: UUID,
    data: GroupUpdate,
) -> GroupMutationResponse:
    return await update_group(db, group_id, data)


@router.delete("/{group_id}", status_code=204)
async def delete_group_route(
    db: DbSession,
    group_id: UUID,
) -> None:
    await delete_group(db, group_id)
