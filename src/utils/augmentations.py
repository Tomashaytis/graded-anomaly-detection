"""
Augmentations utilities.
"""

import random

from typing import List, Optional

from src.augmentation import *

def create_simple_augmentations(class_sample_count: Optional[int] = None, aggressive_threshold: int = 300, size: int = 224) -> ComposeVideo:
    """
    Create simple augmentations for video.

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


def create_complex_augmentations(n_frames: int = 32, size: int = 224, stride: int | float = 2, num_optional_augmentations: int = 2) -> ComposeVideo:
    """
    Create complex augmentations for video.
    Optimized version where all spatial transformations are optional.

    :param n_frames: Number of result frames.
    :param size: Target image size for resizing inside crops.
    :param stride: Stride for video.
    :param num_optional_augmentations: Maximum number of optional augmentations.
    :return: Composed video augmentations.
    """
    if isinstance(stride, int):
        temporal_selectors = [
            TemporalRandomCrop(clip_length=n_frames),
            TemporalStrideSample(stride=stride),
            TemporalStridePick(stride=stride)
        ]
    elif stride > 1:
        temporal_selectors = [
            TemporalRandomCrop(clip_length=n_frames),
        ]
    else:
        temporal_selectors = []


    pipeline: List[BaseAugmentation] = [
        random.choice(temporal_selectors),
        RandomHorizontalFlip(p=0.5),
        RandomGrayscale(p=0.2),
    ]

    optional_pool = [
        RandomResizedCrop(size, scale=(0.8, 1.0)),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
        RandomRotation(degrees=10, p=1.0),
        RandomErasing(p=1.0),
        TimeMasking(p=1.0),
        GaussianNoise(std=0.02, p=1.0)
    ]

    current_num_augs = random.randint(0, num_optional_augmentations)

    if current_num_augs > 0:
        selected_augmentations = random.sample(optional_pool, k=current_num_augs)
        pipeline.extend(selected_augmentations)

    return ComposeVideo(pipeline)

