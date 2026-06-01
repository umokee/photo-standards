from __future__ import annotations

from app.live.constants import RealtimeConstants, realtime
from constants_base import ConstModel
from infra.storage.constants import UploadsConstants, uploads
from modules.core.segments.constants import SegmentsConstants, segments
from modules.core.standards.constants import StandardsConstants, standards
from modules.users.constants import UsersConstants, users
from modules.yolo.inspection.constants import InspectionConstants, inspections
from modules.yolo.training.constants import TrainingConstants, training


class AppConstants(ConstModel):
    realtime: RealtimeConstants
    users: UsersConstants
    standards: StandardsConstants
    inspections: InspectionConstants
    training: TrainingConstants
    segments: SegmentsConstants
    uploads: UploadsConstants


constants = AppConstants(
    realtime=realtime,
    users=users,
    standards=standards,
    inspections=inspections,
    training=training,
    segments=segments,
    uploads=uploads,
)

__all__ = [
    "AppConstants",
    "constants",
]
