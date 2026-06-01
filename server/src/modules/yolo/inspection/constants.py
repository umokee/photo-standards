from __future__ import annotations

from constants_base import ConstModel, ValuesWithDefault


class InspectionStatuses(ValuesWithDefault):
    passed: str
    failed: str


class InspectionModes(ValuesWithDefault):
    photo: str
    snapshot: str
    realtime: str


class InspectionConstants(ConstModel):
    statuses: InspectionStatuses
    modes: InspectionModes


inspections = InspectionConstants(
    statuses=InspectionStatuses(
        passed="passed",
        failed="failed",
        values=("passed", "failed"),
        default="passed",
    ),
    modes=InspectionModes(
        photo="photo",
        snapshot="snapshot",
        realtime="realtime",
        values=("photo", "snapshot", "realtime"),
        default="photo",
    ),
)
