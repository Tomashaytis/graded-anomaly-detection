import cv2
import numpy as np

from pathlib import Path

from .base_frame_source import BaseFrameSource


class FileFrameSource(BaseFrameSource):
    """Frame source backed by a regular video file."""

    def __init__(self, video_path: str | Path):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open {video_path}")
        self._w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._cursor = 0

    def get(self, frame_id: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = self.cap.read()
        return frame if ok else None

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._cursor = int(value)
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self.get(self._cursor)
        if frame is None:
            return False, None
        self._cursor += 1
        return True, frame

    def get_prop(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._n)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def frame_count(self) -> int:
        return self._n

    def frame_size(self) -> tuple[int, int]:
        return self._w, self._h

    def release(self) -> None:
        self.cap.release()