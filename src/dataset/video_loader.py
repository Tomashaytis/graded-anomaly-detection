"""
Video Loader Module
"""

import torch
import cv2
import numpy as np

from pathlib import Path
from typing import Optional, Tuple, List, Generator


class VideoLoader:
    """Class for loading and working with video files."""

    def __init__(self, video_path: str | Path):
        """
        Initialize video loader.

        :param video_path: path to video file
        """
        self.video_path = Path(video_path)
        self._cap = None
        self._total_frames = 0
        self._fps = 0
        self._frame_width = 0
        self._frame_height = 0

        self._load_video_info()

    def _load_video_info(self):
        """Load video information."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._cap.release()
        self._cap = None

    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Video FPS."""
        return self._fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Frame size (width, height)."""
        return self._frame_width, self._frame_height

    def read_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Generator for sequential frame reading.

        :param start_frame: starting frame (0-index)
        :param end_frame: ending frame (exclusive)
        :yield: frames in BGR format
        """
        cap = cv2.VideoCapture(str(self.video_path))

        try:
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_num = start_frame
            while True:
                if end_frame is not None and frame_num >= end_frame:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                yield frame
                frame_num += 1

        finally:
            cap.release()

    def frame(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.

        :param frame_num: frame number (0-index)
        :return: frame in BGR format or None if read failed
        """
        cap = cv2.VideoCapture(str(self.video_path))

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            return frame if ret else None

        finally:
            cap.release()

    def frames_batch(self, frame_nums: List[int]) -> List[Optional[np.ndarray]]:
        """
        Get multiple frames by their numbers.

        :param frame_nums: list of frame numbers
        :return: list of frames (None if frame could not be read)
        """
        frames = []
        cap = cv2.VideoCapture(str(self.video_path))

        try:
            for frame_num in frame_nums:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                frames.append(frame if ret else None)

        finally:
            cap.release()

        return frames

    def info(self) -> dict:
        """
        Get complete video information.

        :return: dictionary with video information
        """
        return {
            'path': str(self.video_path),
            'total_frames': self._total_frames,
            'fps': self._fps,
            'width': self._frame_width,
            'height': self._frame_height,
            'duration_seconds': self._total_frames / self._fps if self._fps > 0 else 0
        }

    def to_tensor(self,
                  max_frames: Optional[int] = None,
                  resize_size: Optional[Tuple[int, int]] = None,
                  normalize: bool = True,
                  sampling_strategy: str = 'first') -> torch.Tensor:
        """
        Load video as tensor with flexible frame sampling.

        :param max_frames: maximum number of frames to load (None = all frames)
        :param resize_size: resize to (height, width) if provided
        :param normalize: normalize pixel values to [0, 1] if True
        :param sampling_strategy: 'first', 'uniform', 'random_segment'
        :return: tensor of shape [C, T, H, W]
        """
        if max_frames is None:
            frames = []
            for i, frame in enumerate(self.read_frames()):
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if resize_size is not None:
                    width, height = resize_size[1], resize_size[0]
                    frame = cv2.resize(frame, (width, height))

                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                if normalize:
                    frame_tensor = frame_tensor / 255.0
                frames.append(frame_tensor)

            if not frames:
                return torch.empty(0)

            return torch.stack(frames, dim=1)

        total_frames = self._total_frames

        if sampling_strategy == 'first':
            # first max_frames of frames
            frame_indices = range(min(max_frames, total_frames))

        elif sampling_strategy == 'uniform':
            # Uniform frames
            frame_indices = np.linspace(0, total_frames - 1, max_frames).astype(int)

        elif sampling_strategy == 'random_segment':
            # Random segment
            if total_frames <= max_frames:
                frame_indices = range(total_frames)
            else:
                start = np.random.randint(0, total_frames - max_frames + 1)
                frame_indices = range(start, start + max_frames)
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

        # Loading frames
        frames = []
        for i in frame_indices:
            frame = self.frame(i)
            if frame is None:
                if frames:
                    frame = frames[-1].clone()
                    continue
                else:
                    frame = np.zeros((self._frame_height, self._frame_width, 3), dtype=np.uint8)

            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if needed
            if resize_size is not None:
                width, height = resize_size[1], resize_size[0]
                frame = cv2.resize(frame, (width, height))

            # To tensor [H, W, C] -> [C, H, W]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()

            if normalize:
                frame_tensor = frame_tensor / 255.0

            frames.append(frame_tensor)

        if not frames:
            return torch.empty(0)

        return torch.stack(frames, dim=1)

    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
            self._cap = None