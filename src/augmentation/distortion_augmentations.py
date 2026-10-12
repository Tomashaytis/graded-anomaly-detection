import torch
import random
import numpy as np

from typing import Tuple
from torchvision import transforms as TF

from .base_augmentation import BaseAugmentation


class GaussianNoise(BaseAugmentation):
    """Add Gaussian noise to video."""

    def __init__(self, std: float = 0.05, p: float = 0.3):
        """
        Initialize Gaussian noise augmentation.

        :param std: Standard deviation of Gaussian noise
        :param p: Probability of applying the augmentation
        """
        self._std = std
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            noise = torch.randn_like(video) * self._std
            return torch.clamp(video + noise, 0, 1)
        return video


class GaussianBlur(BaseAugmentation):
    """Gaussian blur for video."""

    def __init__(self, kernel_size: Tuple[int, int] = (3, 3), sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.2):
        """
        Initialize Gaussian blur augmentation.

        :param kernel_size: Size of the Gaussian kernel (height, width)
        :param sigma: Range of sigma values for Gaussian kernel
        :param p: Probability of applying the augmentation
        """
        self._kernel_size = kernel_size
        self._sigma = sigma
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            sigma = random.uniform(*self._sigma)
            video = video.permute(1, 0, 2, 3)
            blurred = TF.functional.gaussian_blur(video, self._kernel_size, [sigma, sigma])
            video = blurred.permute(1, 0, 2, 3)
        return video


class RandomErasing(BaseAugmentation):
    """Randomly erase a 3D patch (spatial-temporal) in video."""

    def __init__(self, p: float = 0.2, scale: Tuple[float, float] = (0.02, 0.1),
                 ratio: Tuple[float, float] = (0.3, 3.0)):
        """
        Initialize random erasing augmentation.

        :param p: Probability of applying the augmentation
        :param scale: Range of erased area relative to total volume (T*H*W)
        :param ratio: Range of aspect ratio for erased patch (height/width)
        """
        self._p = p
        self._scale = scale
        self._ratio = ratio

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            C, T, H, W = video.shape
            area = T * H * W

            for _ in range(10):
                target_area = random.uniform(*self._scale) * area
                aspect_ratio = random.uniform(*self._ratio)

                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))
                t = int(round(target_area / (h * w)))

                t = min(t, T)
                h = min(h, H)
                w = min(w, W)

                if t < T and h < H and w < W:
                    t_start = random.randint(0, T - t)
                    h_start = random.randint(0, H - h)
                    w_start = random.randint(0, W - w)

                    video[:, t_start:t_start + t, h_start:h_start + h, w_start:w_start + w] = 0
                    return video
        return video