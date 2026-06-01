from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

EmbeddingState = dict[str, Any]


_EMBEDDING_CACHE_SIZE = 3
_PREDICTOR_SNAPSHOT_ATTRS = ("_features", "_orig_hw", "_is_batch")

_embedding_cache_lock = threading.Lock()
_embedding_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def snapshot_predictor_state(predictor: Any) -> EmbeddingState:
    return {
        attr: getattr(predictor, attr)
        for attr in _PREDICTOR_SNAPSHOT_ATTRS
        if hasattr(predictor, attr)
    }


def restore_predictor_state(predictor: Any, state: EmbeddingState) -> None:
    for key, value in state.items():
        setattr(predictor, key, value)

    predictor._is_image_set = True


def cache_get(image_id: str) -> EmbeddingState | None:
    with _embedding_cache_lock:
        state = _embedding_cache.get(image_id)
        if state is None:
            return None

        _embedding_cache.move_to_end(image_id)
        return state


def cache_put(image_id: str, state: EmbeddingState) -> None:
    with _embedding_cache_lock:
        _embedding_cache[image_id] = state
        _embedding_cache.move_to_end(image_id)

        while len(_embedding_cache) > _EMBEDDING_CACHE_SIZE:
            _embedding_cache.popitem(last=False)
