from __future__ import annotations

from typing import Protocol

import numpy as np


class VideoBackend(Protocol):
    def is_opened(self) -> bool:
        ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    def release(self) -> None:
        ...
