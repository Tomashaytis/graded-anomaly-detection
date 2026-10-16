import torch
import random
import torch.nn.functional as F

from typing import Tuple

from .base_augmentation import BaseAugmentation


class TemporalRandomCrop(BaseAugmentation):
    """Random crop along temporal dimension for video."""

    def __init__(self, clip_length: int):
        """
        Initialize temporal random crop augmentation.

        :param clip_length: Target length of video clip after cropping
        """
        self._clip_length = clip_length

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        if T < self._clip_length:
            repeat_factor = (self._clip_length + T - 1) // T
            video = video.repeat(1, repeat_factor, 1, 1)
            T = video.shape[1]

        start = random.randint(0, T - self._clip_length)
        return video[:, start:start + self._clip_length, :, :]


class TemporalFlip(BaseAugmentation):
    """Reverse frame order along temporal dimension for video."""

    def __init__(self, p: float = 0.2):
        """
        Initialize temporal flip augmentation.

        :param p: Probability of applying the flip (0.0 to 1.0)
        """
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            return torch.flip(video, dims=[1])
        return video


class TemporalShift(BaseAugmentation):
    """Cyclic shift along temporal dimension for video."""

    def __init__(self, max_shift: int = 4, p: float = 0.3):
        """
        Initialize temporal shift augmentation.

        :param max_shift: Maximum number of frames to shift (shift range: -max_shift to +max_shift)
        :param p: Probability of applying the augmentation
        """
        self._max_shift = max_shift
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            shift = random.randint(-self._max_shift, self._max_shift)
            return torch.roll(video, shifts=shift, dims=1)
        return video


class TimeMasking(BaseAugmentation):
    """Mask (zero out) a continuous temporal segment for video."""

    def __init__(self, max_mask_ratio: float = 0.3, p: float = 0.5):
        """
        Initialize time masking augmentation.

        :param max_mask_ratio: Maximum ratio of temporal length to mask (0.0 to 1.0)
        :param p: Probability of applying the augmentation
        """
        self._max_mask_ratio = max_mask_ratio
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            C, T, H, W = video.shape
            mask_len = int(T * random.uniform(0.05, self._max_mask_ratio))
            mask_start = random.randint(0, T - mask_len)
            masked = video.clone()
            masked[:, mask_start:mask_start + mask_len] = 0
            return masked
        return video


class TemporalSubsample(BaseAugmentation):
    """Subsample frames with stride and interpolate back to original length."""

    def __init__(self, stride_range: Tuple[int, int] = (1, 3), p: float = 0.3):
        """
        Initialize temporal subsample augmentation.

        :param stride_range: Range of stride values for subsampling (min, max)
        :param p: Probability of applying the augmentation
        """
        self._stride_range = stride_range
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            stride = random.randint(*self._stride_range)
            C, T, H, W = video.shape
            indices = list(range(0, T, stride))
            video = video[:, indices, :, :]
            if video.shape[1] < T:
                video = F.interpolate(video.permute(0, 2, 3, 1),
                                      size=T, mode='linear', align_corners=False)
                video = video.permute(0, 3, 1, 2)
            return video
        return video


class SpeedChange(BaseAugmentation):
    """Change video playback speed by temporal interpolation."""

    def __init__(self, speed_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3):
        """
        Initialize speed change augmentation.

        :param speed_range: Range of speed multipliers (e.g., 0.8 = slower, 1.2 = faster)
        :param p: Probability of applying the augmentation
        """
        self._speed_range = speed_range
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            C, T, H, W = video.shape
            speed = random.uniform(*self._speed_range)
            new_T = max(1, int(T * speed))

            video = video.unsqueeze(0)

            video = F.interpolate(
                video,
                size=(new_T, H, W),
                mode='trilinear',
                align_corners=False
            )

            video = video.squeeze(0)
            return video
        return video

class TemporalStrideSample(BaseAugmentation):
    """Keep every 2nd frame with a random offset (0 or 1), halving temporal length."""

    def __init__(self, stride: int = 2):
        """
        Initialize temporal stride sample augmentation.

        :param stride: Stride value for sampling
        """
        self._stride = stride

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        offset = random.randint(0, self._stride - 1)
        return video[:, offset::self._stride, :, :]

class TemporalStridePick(BaseAugmentation):
    """Replace each stride of consecutive frames with one randomly chosen frame."""

    def __init__(self, stride: int = 2):
        """
        Initialize temporal stride pick augmentation.

        :param stride: Stride value for sampling
        """
        self._stride = stride

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        T_trim = T - (T % self._stride)
        v = video[:, :T_trim, :, :]
        n_groups = T_trim // self._stride
        choices = torch.randint(0, self._stride, (n_groups,), device=v.device)
        idx = torch.arange(0, T_trim, self._stride, device=v.device) + choices
        return v[:, idx, :, :]