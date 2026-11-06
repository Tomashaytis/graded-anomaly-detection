import os
import shutil
import subprocess
import pandas as pd

from tqdm import tqdm

DCSASS_PATH = "E:\\Павел\\НИРС\\action-marker\\data\\DCSASS Dataset"


def get_video_duration(video_path: str) -> float:
    """Return video duration."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


if __name__ == '__main__':
    action_classes = os.listdir(os.path.join(DCSASS_PATH, 'Actions'))
    for action_class in action_classes:
        videos = os.listdir(os.path.join(DCSASS_PATH, 'Actions', action_class))
        labels_path = os.path.join(DCSASS_PATH, 'Labels', f'{action_class}.csv')
        labels_df = pd.read_csv(labels_path, header=None, names=['video_name', 'action_class', 'label'])
        for i in tqdm(range(len(videos)), desc=f'Selecting videos for class {action_class}'):
            segments = os.listdir(os.path.join(DCSASS_PATH, 'Actions', action_class, videos[i]))
            for segment in segments:
                segment_path = os.path.join(DCSASS_PATH, 'Actions', action_class, videos[i], segment)
                segment_name = os.path.splitext(segment)[0]
                match = labels_df.loc[labels_df['video_name'] == segment_name, 'label']
                label = match.iloc[0] if len(match) else None
                if label == 1:
                    dst_path = os.path.join(DCSASS_PATH, 'Anomaly', action_class)
                    os.makedirs(dst_path, exist_ok=True)
                    shutil.copy(segment_path, os.path.join(dst_path, segment))
                elif label == 0:
                    dst_path = os.path.join(DCSASS_PATH, 'Normal', action_class)
                    os.makedirs(dst_path, exist_ok=True)
                    shutil.copy(segment_path, os.path.join(dst_path, segment))
                else:
                    print(segment_name)
                # print(get_video_duration(segment_path))