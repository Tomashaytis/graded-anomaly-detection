import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from src.detector import SpectralGroupDetector
from src.utils import GroupCropper


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


def test():
    group_detector = SpectralGroupDetector()
    cropper = GroupCropper(
        group_detector=group_detector,
        n_frames=32,
        size=224,
        n_top=2,
        n_random=1,
    )

    tensors = cropper.get_group_tensors(
        "../data/test_traj/video/017.mp4",
        "../data/test_traj/traj/017.json",
    )

    for gid, t in tensors.items():
        visualize_video(t)


if __name__ == '__main__':
    test()