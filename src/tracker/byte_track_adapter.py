import numpy as np
import supervision as sv

from typing import List

from .base_tracker import BaseTracker, DETECTION, TRACK


class ByteTrackAdapter(BaseTracker):
    """Адаптер для ByteTrack (через supervision)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracker = sv.ByteTrack(**kwargs)

    def update(self, detections: List[DETECTION], frame: np.ndarray) -> List[TRACK]:
        # Конвертируем в формат supervision
        if not detections:
            return []

        xyxy = np.array([d[:4] for d in detections])
        confidence = np.array([d[4] for d in detections])
        class_id = np.array([d[5] for d in detections])

        # Создаём объект Detections
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Обновляем трекер
        tracked_detections = self.tracker.update_with_detections(sv_detections)

        # Конвертируем обратно в единый формат
        tracks = []
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            conf = tracked_detections.confidence[i]
            cls_id = tracked_detections.class_id[i]
            tracks.append((track_id, x1, y1, x2, y2, conf, cls_id))

        return tracks

    def reset(self):
        self.tracker = sv.ByteTrack()
