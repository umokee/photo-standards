from constants import AppConstants, constants
from fastapi import APIRouter
from modules.cameras.router import router as cameras_router
from modules.core.groups.router import router as groups_router
from modules.core.segments.router import router as segment_classes_router
from modules.core.standards.router import router as standards_router
from modules.sam.router import router as sam_router
from modules.system.router import router as system_router
from modules.tasks.router import router as tasks_router
from modules.users.router import router as users_router
from modules.yolo.router import router as yolo_router

api_router = APIRouter()

api_router.include_router(groups_router)
api_router.include_router(standards_router)
api_router.include_router(segment_classes_router)
api_router.include_router(tasks_router)
api_router.include_router(yolo_router)
api_router.include_router(sam_router)
api_router.include_router(cameras_router)
api_router.include_router(system_router)
api_router.include_router(users_router)


@api_router.get(
    "/meta/constants",
    response_model=AppConstants,
    response_model_by_alias=False,
    include_in_schema=False,
)
async def get_constants() -> AppConstants:
    return constants


@api_router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "OK"}
