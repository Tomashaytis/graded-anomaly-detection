import cv2
import warnings
import tempfile

from pathlib import Path

from .base_frame_source import BaseFrameSource
from .file_frame_size import FileFrameSource
from .stream_frame_size import StreamFrameSource

class FrameSourceFactory:
    """Factory for creating frame sources"""

    @staticmethod
    def create_frame_source(
        video_source,
        window_size_sec: float,
        temp_dir: str | Path | None = None,
        fps_fallback: float = 30.0,
        codec: str = "mp4v",
        ext: str = ".mp4",
    ) -> tuple[BaseFrameSource, int]:
        """
        Create a frame source (file or stream) and compute window size in frames.

        :param video_source: str path/URL, or already-opened cv2.VideoCapture.
        :param window_size_sec: Window size in seconds (from config).
        :param temp_dir: Directory for temp files (for streams).
        :param fps_fallback: FPS to use if source reports 0.
        :param codec: Codec for stream temp file.
        :param ext: Extension for stream temp file.
        :return: (source, window_size_frames).
        """
        if isinstance(video_source, (str, Path)):
            video_source = str(video_source)
            is_file = Path(video_source).exists()
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise ValueError(f"Cannot open {video_source}")
        else:
            cap = video_source
            is_file = False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            warnings.warn(
                f"FPS is unknown for the source, using fallback {fps_fallback}",
                category=RuntimeWarning,
                stacklevel=2,
            )
            fps = fps_fallback

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        window_size_frames = max(1, int(round(window_size_sec * fps)))

        if is_file:
            cap.release()
            source = FileFrameSource(video_source)
        else:
            if temp_dir is None:
                temp_dir = "/dev/shm" if Path("/dev/shm").exists() else tempfile.gettempdir()
            source = StreamFrameSource(
                stream=cap,
                window_size=window_size_frames,
                frame_size=(w, h),
                temp_dir=temp_dir,
                fps=fps,
                codec=codec,
                ext=ext,
            )

        return source, window_size_frames