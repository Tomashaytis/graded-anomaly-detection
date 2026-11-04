import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def visualize_video(video_tensor: torch.Tensor, interval: int = 0):
    """
    Visualize a video tensor of shape (T, C, H, W).
    """
    total_frames = video_tensor.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    frame_orig = video_tensor[0].permute(1, 2, 0).numpy()
    frame_orig = np.clip(frame_orig, 0, 1)
    img = ax.imshow(frame_orig)

    ax.set_title('Loaded', fontsize=10)
    ax.axis('off')

    def update(frame_idx):
        frame_orig = video_tensor[frame_idx % total_frames].permute(1, 2, 0).numpy()
        frame_orig = np.clip(frame_orig, 0, 1)
        img.set_array(frame_orig)
        return (img,)

    anim = FuncAnimation(fig, update, frames=total_frames * 2,
                         interval=interval, repeat=True, blit=False)

    plt.tight_layout()
    plt.show()
    return anim

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