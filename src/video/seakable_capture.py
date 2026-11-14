import cv2

from .base_frame_source import BaseFrameSource


class SeekableCapture:
    """
    Adapter that makes a BaseFrameSource look like cv2.VideoCapture.
    """

    def __init__(self, source: BaseFrameSource, frame_offset: int = 0):
        """
        Initialize SeekableCapture

        :param source: Frame source.
        :param frame_offset: Frame offset.
        """
        self._source = source
        self._frame_offset = frame_offset
        self._cursor = 0

    def set(self, prop_id: int, value: float) -> bool:
        """Emulate cv2.VideoCapture.set(). Only CAP_PROP_POS_FRAMES is meaningful."""
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._cursor = int(value)
            return True
        return False

    def read(self) -> tuple[bool, "np.ndarray | None"]:
        """Emulate cv2.VideoCapture.read() using the source's current cursor."""
        frame = self._source.get(self._frame_offset + self._cursor)
        if frame is None:
            return False, None
        self._cursor += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        """Emulate cv2.VideoCapture.get() for WIDTH/HEIGHT/COUNT/FPS."""
        return self._source.get_prop(prop_id)

    def isOpened(self) -> bool:
        """Always True: the underlying source is assumed to be valid."""
        return True

    def release(self) -> None:
        """No-op: lifetime of the source is managed by the caller."""
        pass