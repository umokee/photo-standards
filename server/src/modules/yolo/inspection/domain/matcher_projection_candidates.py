from __future__ import annotations

from typing import Any

from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    MissingFallbackState,
    MissingLocalDisplacement,
    MissingTranslationRescue,
    ProjectionCandidate,
)


def projection_candidate(
    *,
    source: str,
    polygon: list[list[float]] | None,
    bbox: BBox | None,
    debug: dict[str, Any] | None = None,
    context_rescue_used: bool = False,
) -> ProjectionCandidate:
    return ProjectionCandidate(
        source=source,
        polygon=polygon,
        bbox=bbox,
        debug=dict(debug or {}),
        context_rescue_used=context_rescue_used,
    )


def apply_projection_candidate(
    state: MissingFallbackState,
    candidate: ProjectionCandidate,
) -> None:
    state.debug.update(candidate.debug)
    state.polygon = candidate.polygon
    state.bbox = candidate.bbox
    state.context_rescue_used = candidate.context_rescue_used
    state.source = candidate.source


def translation_rescue_candidate(
    rescue: MissingTranslationRescue,
    *,
    source: str,
    debug: dict[str, Any],
) -> ProjectionCandidate:
    return projection_candidate(
        source=source,
        polygon=rescue.polygon,
        bbox=rescue.bbox,
        debug=debug,
        context_rescue_used=True,
    )


def local_displacement_candidate(
    displacement: MissingLocalDisplacement,
    *,
    debug: dict[str, Any],
) -> ProjectionCandidate:
    return projection_candidate(
        source="local_displacement",
        polygon=displacement.polygon,
        bbox=displacement.bbox,
        debug=debug,
        context_rescue_used=True,
    )


def global_fallback_candidate(
    *,
    polygon: list[list[float]],
    bbox: BBox,
    debug: dict[str, Any],
) -> ProjectionCandidate:
    return projection_candidate(
        source="global_fallback",
        polygon=polygon,
        bbox=bbox,
        debug=debug,
    )
