from __future__ import annotations

from typing import Literal


class InspectionSettingsMixin:
    # Runtime inspection behavior. Detailed matcher thresholds live with the matcher.
    INSPECTION_ALIGNMENT_PROFILE: Literal["balanced", "safe", "debug", "experimental"] = "balanced"
    INSPECTION_DEBUG_PAYLOAD: bool = True
    INSPECTION_FAILSAFE_REQUIRE_CONFIRMED_POSE: bool = True
    INSPECTION_VERIFICATION_MODE: Literal["alignment", "yolo_count"] = "alignment"
