import os
import cv2
import tempfile
import numpy as np

from pathlib import Path

from .base_frame_source import BaseFrameSource


class StreamFrameSource(BaseFrameSource):
    """
    Frame source backed by a live stream.
    Frames are written to a bounded temp video file so that
    they can be sought backwards later (within the current window).

    The user is responsible for calling `clean_buffer()` once the current
    window is done to free the temp file.
    """

    def __init__(
        self,
        stream: cv2.VideoCapture,
        window_size: int,
        frame_size: tuple[int, int],
        temp_dir: str | Path,
        fps: float = 30.0,
        codec: str = "mp4v",
        ext: str = ".mp4",
    ):
        """
        :param stream: cv2.VideoCapture for the stream (RTSP, webcam, etc.).
        :param window_size: Max number of frames stored in the buffer.
        :param frame_size: (width, height) of frames.
        :param temp_dir: Directory for temporary files ('/dev/shm' for RAM disk).
        :param fps: Frame rate (needed by VideoWriter).
        :param codec: FourCC codec for temp file.
        :param ext: Temp file extension.
        """
        self.stream = stream
        self.window_size = max(1, int(window_size))
        self._w, self._h = frame_size
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = codec
        self.ext = ext

        self._temp_file: Path | None = None
        self._writer: cv2.VideoWriter | None = None
        self._reader: cv2.VideoCapture | None = None

        self._start_id = 0
        self._next_id = 0
        self._cursor = 0

    def advance(self) -> bool:
        """Read next frame from stream, append it to temp file."""
        ok, frame = self.stream.read()
        if not ok:
            return False

        if self._writer is None:
            self._open_writer()

        self._writer.write(frame)
        self._next_id += 1

        if (self._next_id - self._start_id) > self.window_size:
            self._trim_buffer()

        return True

    def get(self, frame_id: int) -> np.ndarray | None:
        """Return frame by global id (advances stream if needed)."""
        if frame_id < self._start_id:
            raise ValueError(
                f"Frame {frame_id} was evicted from the buffer "
                f"(buffer starts at {self._start_id}). "
                f"Increase window_size or reduce lookback."
            )

        while self._next_id <= frame_id:
            if not self.advance():
                return None

        if self._reader is None:
            if self._temp_file is None:
                return None
            self._reader = cv2.VideoCapture(str(self._temp_file))

        local_idx = frame_id - self._start_id
        self._reader.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
        ok, frame = self._reader.read()
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
            return float(self._next_id)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def frame_count(self) -> int:
        return self._next_id

    def frame_size(self) -> tuple[int, int]:
        return self._w, self._h

    def clean_buffer(self) -> None:
        """Release temp file and reader/writer. Safe to call multiple times."""
        if self._reader is not None:
            self._reader.release()
            self._reader = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._temp_file is not None and self._temp_file.exists():
            try:
                self._temp_file.unlink()
            except OSError:
                pass
            self._temp_file = None

    def release(self) -> None:
        self.clean_buffer()
        self.stream.release()

    def _open_writer(self) -> None:
        fd, tmp = tempfile.mkstemp(suffix=self.ext, dir=self.temp_dir)
        os.close(fd)
        self._temp_file = Path(tmp)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self._temp_file), fourcc, self.fps, (self._w, self._h)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {self._temp_file}")

    def _trim_buffer(self) -> None:
        """
        Drop the oldest frames so that buffer_size stays <= window_size.
        Re-creates the temp file (VideoWriter cannot append or truncate
        from the beginning).
        """
        if self._temp_file is None:
            return

        to_drop = (self._next_id - self._start_id) - self.window_size
        if to_drop <= 0:
            return

        if self._reader is not None:
            self._reader.release()
            self._reader = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        old_file = self._temp_file
        old_reader = cv2.VideoCapture(str(old_file))
        old_reader.set(cv2.CAP_PROP_POS_FRAMES, to_drop)

        fd, tmp = tempfile.mkstemp(suffix=self.ext, dir=self.temp_dir)
        os.close(fd)
        new_file = Path(tmp)
        new_writer = cv2.VideoWriter(
            str(new_file),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            (self._w, self._h),
        )

        while True:
            ok, frame = old_reader.read()
            if not ok:
                break
            new_writer.write(frame)

        old_reader.release()
        new_writer.release()
        try:
            old_file.unlink()
        except OSError:
            pass

        self._temp_file = new_file
        self._start_id += to_drop
        self._writer = None

        self._reader = cv2.VideoCapture(str(self._temp_file))
