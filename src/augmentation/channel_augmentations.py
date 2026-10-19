import torch
import random

from torchvision import transforms as TF

from .base_augmentation import BaseAugmentation


class RandomGrayscale(BaseAugmentation):
    """Grayscale convertion for video."""

    def __init__(self, p: float = 0.5):
        """
        Initialize grayscale augmentation.

        :param p: Probability of applying the flip (0.0 to 1.0)
        """
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            C, T, H, W = video.shape

            weights = torch.tensor(
                [0.299, 0.587, 0.114],
                device=video.device,
                dtype=video.dtype,
            ).view(C, 1, 1, 1)

            gray = (video * weights).sum(dim=0, keepdim=True)

            video = gray.expand(C, T, H, W).contiguous()
        return video


class ColorJitter(BaseAugmentation):
    """Color jitter for video (RGB)."""

    def __init__(
            self,
            brightness: float = 0,
            contrast: float = 0,
            saturation: float = 0,
            hue: float = 0,
            p: float = 0.8
    ):
        """
        Initialize color jitter augmentation.

        :param brightness: Brightness jitter range (0 = disabled)
        :param contrast: Contrast jitter range (0 = disabled)
        :param saturation: Saturation jitter range (0 = disabled)
        :param hue: Hue jitter range (0 = disabled)
        :param p: Probability of applying the augmentation
        """
        self._jitter = TF.ColorJitter(brightness, contrast, saturation, hue)
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            C, T, H, W = video.shape
            video_permuted = video.permute(1, 0, 2, 3)
            jittered_frames = []
            for t in range(T):
                frame = TF.ToPILImage()(video_permuted[t])
                frame = self._jitter(frame)
                jittered_frames.append(TF.ToTensor()(frame))
            video = torch.stack(jittered_frames).permute(1, 0, 2, 3)
        return video
