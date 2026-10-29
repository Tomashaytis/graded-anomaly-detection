import numpy as np
import supervision as sv

from typing import List
from trackers import ByteTrackTracker

from .base_tracker import BaseTracker, DETECTION, TRACK


class ByteTrackAdapter(BaseTracker):
    """Adapter for ByteTrack (via supervision)."""

    def __init__(self, **kwargs):
        """
        Initialize the ByteTrack tracker.

        :param kwargs: Parameters forwarded to ByteTrackTracker
        """
        super().__init__(**kwargs)
        self.tracker = ByteTrackTracker(**kwargs)

    def update(self, detections: List[DETECTION], frame: np.ndarray) -> List[TRACK]:
        if not detections:
            empty_detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                class_id=np.empty(0, dtype=int),
            )
            tracked_detections = self.tracker.update(empty_detections)
        else:
            xyxy = np.array([d[:4] for d in detections])
            confidence = np.array([d[4] for d in detections])
            class_id = np.array([d[5] for d in detections])

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )

            tracked_detections = self.tracker.update(sv_detections)

        tracks = []
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            conf = tracked_detections.confidence[i]
            cls_id = tracked_detections.class_id[i]
            tracks.append((track_id, x1, y1, x2, y2, conf, cls_id))

        return tracks

    def reset(self):
        self.tracker = ByteTrackTracker()