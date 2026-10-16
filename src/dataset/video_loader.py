"""
Video Loader Module
"""

import cv2
import torch
import numpy as np

from pathlib import Path
from decord import VideoReader, cpu
from typing import Optional, Tuple, List, Generator


class VideoLoader:
    """Class for loading and working with video files."""

    def __init__(self, video_path: str | Path):
        """
        Initialize video loader.

        :param video_path: path to video file
        """
        self.video_path = Path(video_path)
        self._vr = VideoReader(str(self.video_path), ctx=cpu(0))
        self._total_frames = len(self._vr)
        self._fps = self._vr.get_avg_fps()
        h, w = self._vr[0].shape[:2]
        self._frame_width, self._frame_height = w, h

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

    @staticmethod
    def detect_black_borders(frame: np.ndarray,
                             threshold: int = 16) -> Tuple[int, int, int, int]:
        """
        Detect near-black borders around the frame.

        :param frame: BGR frame (uint8)
        :param threshold: max channel value to consider a pixel black (0..255)
        :return: (top, bottom, left, right) — number of pixels to crop from each side
        """
        if frame.size == 0:
            return 0, 0, 0, 0

        black = np.all(frame <= threshold, axis=2)

        h, w = black.shape

        row_is_black = black.all(axis=1)
        col_is_black = black.all(axis=0)

        top = bottom = left = right = 0

        while top < h and row_is_black[top]:
            top += 1

        while bottom < h and row_is_black[h - 1 - bottom]:
            bottom += 1

        while left < w and col_is_black[left]:
            left += 1

        while right < w and col_is_black[w - 1 - right]:
            right += 1

        if top + bottom >= h:
            top = bottom = 0
        if left + right >= w:
            left = right = 0

        return top, bottom, left, right

    @staticmethod
    def reflect_pad_and_resize(frame: np.ndarray,
                               target_size: Tuple[int, int],
                               crop_black: bool = True) -> np.ndarray:
        """
        Crop black borders, then bring the frame to the target aspect ratio:
        - if the frame is wider than the target — center-crop by width
          (no vertical reflection, so people are not flipped upside down);
        - if the frame is taller than the target — reflect left/right
          (bilateral symmetry is preserved, so it looks natural).
        Finally resize to the target size.
        """
        target_h, target_w = target_size

        if crop_black:
            top, bottom, left, right = VideoLoader.detect_black_borders(frame)
            h, w = frame.shape[:2]
            if top + bottom < h and left + right < w and (top or bottom or left or right):
                frame = frame[top:h - bottom, left:w - right]

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target_h, target_w, 3), dtype=frame.dtype)

        target_ratio = target_w / target_h
        current_ratio = w / h

        if current_ratio > target_ratio:
            new_w = min(int(round(h * target_ratio)), w)
            x0 = (w - new_w) // 2
            frame = frame[:, x0:x0 + new_w]
        else:
            new_w = int(round(h * target_ratio))
            pad = new_w - w
            pad_left = pad // 2
            pad_right = pad - pad_left
            if pad_left or pad_right:
                frame = cv2.copyMakeBorder(
                    frame, 0, 0, pad_left, pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )

        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

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
        total_frames = self._total_frames

        if sampling_strategy == 'uniform':
            frame_indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
        elif sampling_strategy == 'first':
            frame_indices = np.arange(min(max_frames, total_frames))
        elif sampling_strategy == 'random_segment':
            if total_frames <= max_frames:
                frame_indices = np.arange(total_frames)
            else:
                start = np.random.randint(0, total_frames - max_frames + 1)
                frame_indices = np.arange(start, start + max_frames)
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

        frames = self._vr.get_batch(frame_indices).asnumpy()

        processed = []
        for frame in frames:
            if resize_size is not None:
                frame = VideoLoader.reflect_pad_and_resize(frame, resize_size)
            t = torch.from_numpy(frame).permute(2, 0, 1).float()
            if normalize:
                t = t / 255.0
            processed.append(t)

        return torch.stack(processed, dim=1)

    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
            self._cap = None