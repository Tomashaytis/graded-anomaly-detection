import torch
import random

from typing import List

from .base_augmentation import BaseAugmentation


class ComposeVideo(BaseAugmentation):
    """Compose multiple video augmentations into a single pipeline."""

    def __init__(self, transforms: List[BaseAugmentation]):
        """
        Initialize composition of augmentations.

        :param transforms: List of augmentation instances to apply in order
        """
        self._transforms = transforms

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        for transform in self._transforms:
            video = transform(video)
        return video


class RandomApply(BaseAugmentation):
    """Randomly apply a composition of augmentations with given probability."""

    def __init__(self, transform: BaseAugmentation, p: float = 0.5):
        """
        Initialize random apply augmentation.

        :param transform: Composed augmentations to apply
        :param p: Probability of applying the transform
        """
        self._transform = transform
        self._p = p

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self._p:
            return self._transform(video)
        return video


class AdaptiveAugmentation:
    """
    Adaptive augmentations: different for large and small classes.
    """

    def __init__(self,
                 base_transforms: ComposeVideo,
                 aggressive_transforms: ComposeVideo,
                 small_class_threshold: int = 200):
        """
        Initialize adaptive augmentation.

        :param base_transforms: Composed augmentations for large classes
        :param aggressive_transforms: Composed augmentations for small classes
        :param small_class_threshold: Threshold below which class is considered "small"
        """
        self._base_transforms = base_transforms
        self._aggressive_transforms = aggressive_transforms
        self._small_class_threshold = small_class_threshold

    def __call__(self, video: torch.Tensor, class_samples: int) -> torch.Tensor:
        """
        Apply augmentations based on class size.

        :param video: Input video tensor
        :param class_samples: Number of samples in the video's class
        :return: Augmented video tensor
        """
        if class_samples < self._small_class_threshold:
            return self._aggressive_transforms(video)
        else:
            return self._base_transforms(video)