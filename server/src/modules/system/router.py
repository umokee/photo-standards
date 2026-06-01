from __future__ import annotations

from fastapi import APIRouter

from .schemas import SystemStatsResponse
from .service import get_system_stats

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats_route() -> SystemStatsResponse:
    return await get_system_stats()
