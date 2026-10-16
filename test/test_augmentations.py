import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from src.dataset import VideoLoader
from src.augmentation import *
from src.utils import create_augmentations

VIDEO_PATH = '../data/dataset/overcoming obstacles/turnstile jumping (3)/turnstile jumping (Hand-assembled)/001_01.mp4'
# VIDEO_PATH = "H:\\Павел\\НИРС\\action-marker\\data\\dataset\\strange\\smoking (2)\\smoking (Kinetics-700)\\6hfqkR9EaiA_000008_000018.mp4"

def visualize_video_augmentation(video_tensor: torch.Tensor, transform, interval: int = 0):
    """
    Visualize original and augmented video side by side in real time.
    """
    augmented = transform(video_tensor)

    total_frames = min(video_tensor.shape[1], augmented.shape[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    frame_orig = video_tensor[:, 0].permute(1, 2, 0).numpy()
    frame_orig = np.clip(frame_orig, 0, 1)
    img1 = ax1.imshow(frame_orig)

    frame_aug = augmented[:, 0].permute(1, 2, 0).numpy()
    frame_aug = np.clip(frame_aug, 0, 1)
    img2 = ax2.imshow(frame_aug)

    ax1.set_title('Original', fontsize=10)
    ax2.set_title(f'Augmented ({transform.name})', fontsize=10)
    ax1.axis('off')
    ax2.axis('off')

    def update(frame_idx):
        frame_orig = video_tensor[:, frame_idx % video_tensor.shape[1]].permute(1, 2, 0).numpy()
        frame_orig = np.clip(frame_orig, 0, 1)
        img1.set_array(frame_orig)

        frame_aug = augmented[:, frame_idx % augmented.shape[1]].permute(1, 2, 0).numpy()
        frame_aug = np.clip(frame_aug, 0, 1)
        img2.set_array(frame_aug)

        return img1, img2

    anim = FuncAnimation(fig, update, frames=total_frames * 2,
                         interval=interval, repeat=True, blit=False)

    plt.tight_layout()
    plt.show()
    return anim

def test():
    with VideoLoader(VIDEO_PATH) as loader:
        video_tensor = loader.to_tensor(max_frames=32, resize_size=(224, 224), sampling_strategy='uniform')

        # Spatial augmentations
        visualize_video_augmentation(video_tensor, RandomHorizontalFlip(p=1))
        visualize_video_augmentation(video_tensor, RandomRotation(p=1))
        visualize_video_augmentation(video_tensor, RandomResizedCrop(size=224))
        visualize_video_augmentation(video_tensor, ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1))

        # Temporal augmentations
        visualize_video_augmentation(video_tensor, TemporalRandomCrop(clip_length=16))
        visualize_video_augmentation(video_tensor, TemporalFlip(p=1))
        visualize_video_augmentation(video_tensor, TemporalShift(p=1))
        visualize_video_augmentation(video_tensor, TimeMasking(p=1))
        visualize_video_augmentation(video_tensor, SpeedChange(p=1))
        visualize_video_augmentation(video_tensor, TemporalSubsample(p=1))

        # Distortion augmentations
        visualize_video_augmentation(video_tensor, GaussianNoise(p=1))
        visualize_video_augmentation(video_tensor, GaussianBlur(p=1))
        visualize_video_augmentation(video_tensor, RandomErasing(p=1))

        # Composite augmentations
        visualize_video_augmentation(video_tensor, RandomApply(RandomRotation(p=1), p=1))
        visualize_video_augmentation(video_tensor, create_augmentations())
        visualize_video_augmentation(video_tensor, create_augmentations(class_sample_count=100))


if __name__ == '__main__':
    test()