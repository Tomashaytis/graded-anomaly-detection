"""
Video Dataset Module
"""
import torch

from typing import Tuple
from pathlib import Path

from src.utils import create_complex_augmentations

from .lmdb_reader import LMDBReader


class VideoDataset(torch.utils.data.Dataset):
    """
    Dataset for loading videos with augmentations.
    """

    def __init__(
            self,
            lmdb_path: Path,
            class_counts: dict,
            is_train: bool = True,
            video_frame_size: int = 224,
            video_max_frames: int = 64,
            num_optional_augmentations: int = 2,
    ):
        """
        Initialize VideoDataset.

        :param lmdb_path: path to LMDB folder with video data.
        :param class_counts: dict with class index -> number of samples (for adaptive augmentation).
        :param is_train: if True, apply augmentations; if False, only basic transforms.
        :param video_frame_size: video frame size.
        :param video_max_frames: maximum number of frames in video.
        :param num_optional_augmentations: Maximum number of optional augmentations.
        """
        self._reader = LMDBReader(lmdb_path)
        self._class_counts = class_counts
        self._is_train = is_train
        self._video_frame_size = video_frame_size
        self._video_max_frames = video_max_frames
        self._num_optional_augmentations = num_optional_augmentations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Load and augment video by index.

        :param idx: index of the video.
        :return: video (tensor of shape [C, T, H, W] (3, 32, 224, 224)), label (class label as int) and level (0 - 3)
        """
        data = self._reader.get(idx)
        label = data['label']
        level = data['level']
        video = data['video']

        if self._is_train:
            transforms = create_complex_augmentations(
                n_frames=self._video_max_frames,
                size=self._video_frame_size,
                stride=video.shape[1] / self._video_max_frames,
                num_optional_augmentations=self._num_optional_augmentations
            )
            video = transforms(video)

        return video, label, level

    def __len__(self) -> int:
        """
        Get length of dataset.
        :return: length of dataset.
        """
        return len(self._reader)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_reader'] = self._reader.__getstate__()
        return state

    def __setstate__(self, state):
        reader_state = state.pop('_reader')
        self.__dict__.update(state)
        self._reader = LMDBReader.__new__(LMDBReader)
        self._reader.__setstate__(reader_state)
