"""
Frame sources: unified interface for reading frames from files and streams.
Supports seeking backwards within a bounded buffer (for streams).
"""

import numpy as np

from abc import ABC, abstractmethod


class BaseFrameSource(ABC):
    """Abstract interface for frame sources (file or stream)."""

    @abstractmethod
    def get(self, frame_id: int) -> np.ndarray | None:
        """Return frame by global index, or None if unavailable."""
        ...

    @abstractmethod
    def set(self, prop_id: int, value: float) -> bool:
        """Emulate cv2.VideoCapture.set (only CAP_PROP_POS_FRAMES is meaningful)."""
        ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Emulate cv2.VideoCapture.read using the current cursor."""
        ...

    @abstractmethod
    def get_prop(self, prop_id: int) -> float:
        """Emulate cv2.VideoCapture.get for WIDTH/HEIGHT/COUNT/FPS."""
        ...

    @abstractmethod
    def frame_count(self) -> int:
        """Total (file) or already-read (stream) frame count."""
        ...

    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height)."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release resources."""
        ...