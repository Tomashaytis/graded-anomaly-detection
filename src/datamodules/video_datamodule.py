"""Video DataModule for Lightning."""

from pathlib import Path
from typing import Optional

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import VideoDataset


class VideoDataModule(L.LightningDataModule):
    """LightningDataModule for video action recognition."""

    def __init__(
            self,
            train_lmdb_path: str,
            val_lmdb_path: str,
            test_lmdb_path: Optional[str],
            annotations_train_path: str,
            batch_size: int = 8,
            num_workers: int = 2,
            video_frame_size: int = 224,
            video_max_frames: int = 32,
            aggressive_aug_threshold: int = 300,
    ):
        """
        Initialize VideoDataModule.

        :param train_lmdb_path: Path to train LMDB database.
        :param val_lmdb_path: Path to validation LMDB database.
        :param test_lmdb_path: Path to test LMDB database (optional).
        :param annotations_train_path: Path to train annotations CSV.
        :param batch_size: Batch size for dataloaders.
        :param num_workers: Number of workers for data loading.
        :param video_frame_size: Size of video frames.
        :param video_max_frames: Maximum number of frames per video.
        :param aggressive_aug_threshold: Threshold for augmentations.
        """
        super().__init__()
        self._train_lmdb_path = Path(train_lmdb_path)
        self._val_lmdb_path = Path(val_lmdb_path)
        self._test_lmdb_path = Path(test_lmdb_path) if test_lmdb_path else None
        self._annotations_train_path = annotations_train_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._video_frame_size = video_frame_size
        self._video_max_frames = video_max_frames
        self._aggressive_aug_threshold = aggressive_aug_threshold

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._class_counts = None

    def prepare_data(self):
        """Download data if needed. Called only on one GPU."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Set up datasets. Called on every GPU."""
        # Load class counts from annotations
        annotations_df = pd.read_csv(self._annotations_train_path)
        self._class_counts = annotations_df['label_id'].value_counts().to_dict()

        if stage == "fit" or stage is None:
            self._train_dataset = VideoDataset(
                self._train_lmdb_path,
                self._class_counts,
                is_train=True,
                video_frame_size=self._video_frame_size,
                video_max_frames=self._video_max_frames,
                aggressive_aug_threshold=self._aggressive_aug_threshold
            )
            self._val_dataset = VideoDataset(
                self._val_lmdb_path,
                self._class_counts,
                is_train=False,
                video_frame_size=self._video_frame_size,
                video_max_frames=self._video_max_frames,
                aggressive_aug_threshold=self._aggressive_aug_threshold
            )

        if stage == "test" or stage is None:
            if self._test_lmdb_path:
                self._test_dataset = VideoDataset(
                    self._test_lmdb_path,
                    self._class_counts,
                    is_train=False,
                    video_frame_size=self._video_frame_size,
                    video_max_frames=self._video_max_frames
                )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        if self._test_dataset:
            return DataLoader(
                self._test_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                pin_memory=True
            )
        return None