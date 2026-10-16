import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from matplotlib.animation import FuncAnimation

from src.dataset import LMDBReader

LMDB_PATH = "D:/Projects/actions-data/train_lmdb"


def visualize_lmdb_video(video_tensor: torch.Tensor, interval: int = 0):
    """
    Visualize video from LMDB
    """
    total_frames =video_tensor.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    frame_orig = video_tensor[:, 0].permute(1, 2, 0).numpy()
    frame_orig = np.clip(frame_orig, 0, 1)
    img = ax.imshow(frame_orig)

    ax.set_title('Loaded', fontsize=10)
    ax.axis('off')

    def update(frame_idx):
        frame_orig = video_tensor[:, frame_idx % video_tensor.shape[1]].permute(1, 2, 0).numpy()
        frame_orig = np.clip(frame_orig, 0, 1)
        img.set_array(frame_orig)
        return (img,)

    anim = FuncAnimation(fig, update, frames=total_frames * 2,
                         interval=interval, repeat=True, blit=False)

    plt.tight_layout()
    plt.show()
    return anim

def test():
    reader = LMDBReader(Path(LMDB_PATH))
    data = reader.get(0)
    label = data['label']
    level = data['level']
    video = data['video']

    print('Label:', label)
    print('Level:', level)
    visualize_lmdb_video(video)

    for i in tqdm(range(len(reader)), desc='Testing'):
        _ = reader.get(i)


if __name__ == '__main__':
    test()