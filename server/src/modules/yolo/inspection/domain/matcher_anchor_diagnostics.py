from __future__ import annotations

from typing import Any

from modules.yolo.inspection.domain.matcher_structs import TrustedAnchor


def update_anchor_release_debug(
    reject_debug: dict[str, Any] | None,
    **fields: Any,
) -> None:
    if reject_debug is None:
        return

    for key, value in fields.items():
        reject_debug[f"missing_polygon_anchor_release_{key}"] = value


def set_anchor_release_reject(
    reject_debug: dict[str, Any] | None,
    reason: str,
    **fields: Any,
) -> None:
    if reject_debug is None:
        return

    reject_debug["missing_polygon_anchor_release_reject_reason"] = reason
    update_anchor_release_debug(reject_debug, **fields)


def trusted_anchor_source_counts(anchors: list[TrustedAnchor]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for anchor in anchors:
        counts[anchor.source] = counts.get(anchor.source, 0) + 1
    return counts
