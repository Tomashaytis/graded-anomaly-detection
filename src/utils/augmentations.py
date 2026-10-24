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


def create_complex_augmentations(
        current_n_frames: int,
        current_size: int,
        target_n_frames: int = 32,
        target_size: int = 224,
        num_optional_augmentations: int = 2,
        only_required: bool = False
) -> ComposeVideo:
    """
    Create complex augmentations for video.
    Optimized version where all spatial transformations are optional.

    :param current_n_frames: Number of current frames.
    :param current_size: Current image size.
    :param target_n_frames: Number of result frames.
    :param target_size: Target image size for resizing inside crops.
    :param num_optional_augmentations: Maximum number of optional augmentations.
    :param only_required: If True, return only required augmentations.
    :return: Composed video augmentations.
    """
    if current_n_frames % target_n_frames == 0:
        stride = current_n_frames // target_n_frames
    else:
        stride = None
    required_resized_crop = target_size != current_size

    if only_required:
        required_augs: List[BaseAugmentation] = []
        if stride is not None and stride > 1:
            required_augs.append(TemporalStrideSample(stride=stride, rand=False))
        elif stride != 1:
            required_augs.append(TemporalRandomCrop(clip_length=target_n_frames, rand=False))

        if required_resized_crop:
            required_augs.append(RandomResizedCrop(target_size, scale=(0.8, 1.0), rand=False))

        return ComposeVideo(required_augs)

    if stride is not None and stride > 1:
        temporal_selectors = [
            TemporalRandomCrop(clip_length=target_n_frames),
            TemporalStrideSample(stride=stride),
            TemporalStridePick(stride=stride)
        ]
    elif stride != 1:
        temporal_selectors = [
            TemporalRandomCrop(clip_length=target_n_frames),
        ]
    else:
        temporal_selectors = []

    pipeline: List[BaseAugmentation] = []

    if len(temporal_selectors) > 0:
        pipeline.append(random.choice(temporal_selectors))

    if required_resized_crop:
        pipeline.append(RandomResizedCrop(target_size, scale=(0.8, 1.0)))

    pipeline.extend([
        RandomHorizontalFlip(p=0.5),
        RandomGrayscale(p=0.2),
    ])

    optional_pool: List[BaseAugmentation] = [
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        RandomRotation(degrees=10, p=0.3),
        RandomErasing(p=0.25),
        TimeMasking(p=0.3),
        GaussianNoise(std=0.02, p=0.2),
    ]
    if not required_resized_crop:
        optional_pool.append(RandomResizedCrop(target_size, scale=(0.8, 1.0)),)

    max_k = min(num_optional_augmentations, len(optional_pool))
    current_num_augs = random.randint(0, max_k)

    if current_num_augs > 0:
        selected_augmentations = random.sample(optional_pool, k=current_num_augs)
        pipeline.extend(selected_augmentations)

    return ComposeVideo(pipeline)

