"""
Augmentations utilities.
"""

from typing import List, Optional

from src.augmentation import *

def create_augmentations(class_sample_count: Optional[int] = None, aggressive_threshold: int = 300, size: int = 224) -> ComposeVideo:
    """
    Create augmentations for video.

    :param class_sample_count: Number of samples in the class.
    :param aggressive_threshold: If class_sample_count < aggressive_threshold, aggressive augmentations are added.
    :param size: Image size.
    :return: Composed video augmentations
    """
    base: List[BaseAugmentation] = [
        RandomResizedCrop(size, scale=(0.7, 1.0)),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)
    ]

    aggressive: List[BaseAugmentation] = [
        GaussianNoise(std=0.03, p=0.3),
        RandomErasing(p=0.15),
    ]

    if class_sample_count is not None and class_sample_count < aggressive_threshold:
        base.extend(aggressive)

    return ComposeVideo(base)