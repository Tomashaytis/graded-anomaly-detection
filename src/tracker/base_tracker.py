import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple

# Универсальный формат детекции
DETECTION = Tuple[float, float, float, float, float, int]  # x1, y1, x2, y2, confidence, class_id
TRACK = Tuple[int, float, float, float, float, float, int]  # track_id, x1, y1, x2, y2, confidence, class_id


class BaseTracker(ABC):
    """Абстрактный базовый класс для всех трекеров"""

    def __init__(self, **kwargs):
        self.tracker = None
        self.tracker_id = 0

    @abstractmethod
    def update(self, detections: List[DETECTION], frame: np.ndarray) -> List[TRACK]:
        """
        Обновляет трекер новыми детекциями

        Args:
            detections: список детекций в формате (x1, y1, x2, y2, confidence, class_id)
            frame: текущий кадр (для некоторых трекеров)

        Returns:
            список треков в формате (track_id, x1, y1, x2, y2, confidence, class_id)
        """
        pass

    @abstractmethod
    def reset(self):
        """Сбрасывает трекер"""
        pass