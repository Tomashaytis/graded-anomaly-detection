import torch
import random
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torchvision import transforms as TF

from .base_augmentation import BaseAugmentation


class RandomHorizontalFlip(BaseAugmentation):
    """Horizontal flip for video."""

    def __init__(self, p: float = 0.5):
        """
        Initialize horizontal flip augmentation.

        :param p: Probability of applying the flip (0.0 to 1.0)
        """
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            return video.flip(-1)
        return video


class RandomResizedCrop(BaseAugmentation):
    """Random resized crop for video."""

    def __init__(self, size: int | Tuple[int, int], scale: Tuple[float, float] = (0.08, 1.0),
                 ratio: Tuple[float, float] = (3 / 4, 4 / 3), rand: bool = True):
        """
        Initialize random resized crop augmentation.

        :param size: Target output size (height, width). If int, same for both dimensions
        :param scale: Range of scaling factor relative to original area
        :param ratio: Range of aspect ratio
        :param rand: If True, use random crop; if False, use centered crop
        """
        self._size = size if isinstance(size, tuple) else (size, size)
        self._scale = scale
        self._ratio = ratio
        self._rand = rand

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if not self._rand:
            cropped = self.center_crop(video)
            if (cropped.shape[-2], cropped.shape[-1]) != self._size:
                cropped = F.interpolate(
                    cropped, size=self._size, mode='bilinear', align_corners=False
                )
            return cropped

        _, _, h, w = video.shape
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self._scale) * area
            aspect_ratio = random.uniform(*self._ratio)
            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))
            if 0 < new_w <= w and 0 < new_h <= h:
                i = random.randint(0, h - new_h)
                j = random.randint(0, w - new_w)
                cropped = video[:, :, i:i + new_h, j:j + new_w]
                return F.interpolate(cropped, size=self._size, mode='bilinear', align_corners=False)
        return self.center_crop(video)

    def center_crop(self, video: torch.Tensor) -> torch.Tensor:
        """
        Center crop helper method.

        :param video: Input video tensor
        :return: Center cropped video
        """
        _, _, h, w = video.shape
        crop_h, crop_w = self._size
        i = (h - crop_h) // 2
        j = (w - crop_w) // 2
        return video[:, :, i:i + crop_h, j:j + crop_w]


class RandomRotation(BaseAugmentation):
    """Random rotation for video."""

    def __init__(self, degrees: float = 10, p: float = 0.3):
        """
        Initialize random rotation augmentation.

        :param degrees: Maximum rotation angle in degrees (range: -degrees to +degrees)
        :param p: Probability of applying the augmentation
        """
        self._degrees = degrees
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            angle = random.uniform(-self._degrees, self._degrees)
            video = video.permute(1, 0, 2, 3)
            rotated = TF.functional.rotate(video, angle)
            video = rotated.permute(1, 0, 2, 3)
        return video

class Resize(BaseAugmentation):
    """Resize for video."""

    def __init__(self, size: int | Tuple[int, int]):
        """
        Initialize resize augmentation.

        :param size: Target output size (height, width). If int, same for both dimensions
        """
        self._size = size if isinstance(size, tuple) else (size, size)

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        new_h, new_w = self._size

        video_reshaped = video.permute(1, 0, 2, 3).reshape(T * C, H, W)

        video_resized = torch.nn.functional.interpolate(
            video_reshaped.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        video_resized = video_resized.reshape(T, C, new_h, new_w).permute(1, 0, 2, 3)

        return video_resized
