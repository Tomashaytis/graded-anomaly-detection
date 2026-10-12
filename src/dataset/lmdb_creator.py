"""
LMDB Creator Module
"""

import gc
import lmdb
import torch
import pickle

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from .video_loader import VideoLoader


class LMDBCreator:
    """LMDB database creator for video dataset."""

    def __init__(
            self,
            db_path: Path,
            map_size: int = 50,
            commit_every: int = 100
    ):
        """
        Initialize LMDB creator.

        :param db_path: path to LMDB database file.
        :param map_size: maximum database size in GB.
        :param commit_every: how many frames to commit in one transaction.
        """
        self._db_path = Path(db_path)
        self._map_size = map_size * 1024 ** 3
        self._commit_every = commit_every

    def create(
            self,
            video_paths: List[str],
            labels: List[int],
            max_frames: int = 32,
            resize_size: Tuple[int, int] = (224, 224)
    ) -> dict:
        if len(video_paths) != len(labels):
            raise ValueError(f"Video paths ({len(video_paths)}) and labels ({len(labels)}) length mismatch")

        self._db_path.mkdir(parents=True, exist_ok=True)

        env = lmdb.open(
            str(self._db_path),
            map_size=self._map_size
        )

        stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'failed_videos': []
        }

        idx_correction = 0

        try:
            txn = env.begin(write=True)

            for idx, (video_path, label) in enumerate(
                    tqdm(zip(video_paths, labels), total=len(video_paths), desc="Processing videos")
            ):
                try:
                    with VideoLoader(str(video_path)) as loader:
                        video = loader.to_tensor(
                            max_frames=max_frames,
                            resize_size=resize_size,
                            normalize=True,
                            sampling_strategy='uniform'
                        )
                        video = (video * 255).to(torch.uint8)

                    data = {
                        'video': video,
                        'label': label,
                        'path': video_path,
                        'index': idx - idx_correction
                    }

                    key = f"{idx - idx_correction:010d}".encode()
                    value = pickle.dumps(data)
                    txn.put(key, value)

                    stats['successful'] += 1

                except Exception as e:
                    stats['failed'] += 1
                    stats['failed_videos'].append({'path': video_path, 'error': str(e)})
                    idx_correction += 1

                stats['total_videos'] += 1

                if (idx + 1) % self._commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    gc.collect()

            metadata = {
                'total_videos': stats['successful'],
                'max_frames': max_frames,
                'resize_size': resize_size,
                'original_count': len(video_paths)
            }
            txn.put(b'__meta__', pickle.dumps(metadata))
            txn.commit()

        except Exception as e:
            print(f"Fatal error at index {idx}: {e}")
            txn.abort()
            raise
        finally:
            env.close()

        print(f"\nLMDB created at {self._db_path}")
        print(f"Successful: {stats['successful']}/{stats['total_videos']}")
        if stats['failed'] > 0:
            print(f"Failed: {stats['failed']}")
            for failed in stats['failed_videos'][:5]:
                print(f"  - {failed['path']}: {failed['error']}")

        return stats
