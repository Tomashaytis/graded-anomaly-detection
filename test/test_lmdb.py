from tqdm import tqdm
from pathlib import Path

from src.dataset import LMDBReader
from src.utils import visualize_video

LMDB_PATH = "F:/Projects/actions-data/train_lmdb"


def test():
    reader = LMDBReader(Path(LMDB_PATH))
    data = reader.get(0)
    label = data['label']
    level = data['level']
    video = data['video']

    print('Label:', label)
    print('Level:', level)
    visualize_video(video)

    for i in tqdm(range(len(reader)), desc='Testing'):
        _ = reader.get(i)


if __name__ == '__main__':
    test()