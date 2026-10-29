import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple

DETECTION = Tuple[float, float, float, float, float, int]
TRACK = Tuple[int, float, float, float, float, float, int]


class BaseTracker(ABC):
    """Abstract base class for all trackers."""

    def __init__(self, **kwargs):
        """
        Initialize the base tracker.

        :param kwargs: Tracker parameters (forwarded by subclasses).
        """
        self._tracker = None
        self._tracker_id = 0

    @abstractmethod
    def update(self, detections: List[DETECTION], frame: np.ndarray) -> List[TRACK]:
        """
        Update the tracker with new detections.

        :param detections: List of detections (x1, y1, x2, y2, confidence, class_id).
        :param frame: Current frame (BGR, uint8); used by some trackers only.
        :return: List of tracks (track_id, x1, y1, x2, y2, confidence, class_id).
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the tracker to its initial state."""
        pass