"""
LMDB Reader Module
"""

import lmdb
import pickle

from pathlib import Path
from typing import Optional


class LMDBReader:
    """Class for reading from LMDB database."""

    def __init__(self, db_path: Path):
        """
        Initialize LMDB reader.

        :param db_path: path to LMDB database.
        """
        self._db_path = Path(db_path)
        self._env = None

    def _ensure_env(self):
        """Open LMDB environment if not already open."""
        if self._env is None:
            if not self._db_path.exists():
                raise FileNotFoundError(f"LMDB database not found: {self._db_path}")

            self._env = lmdb.open(
                str(self._db_path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=128,
            )

    def __getstate__(self):
        """Called during pickle serialization for multiprocessing."""
        state = self.__dict__.copy()
        state['_env'] = None
        return state

    def __setstate__(self, state):
        """Called during pickle deserialization in worker process."""
        self.__dict__.update(state)
        self._env = None

    def close(self):
        """Close LMDB environment."""
        if self._env:
            self._env.close()
            self._env = None

    def get(self, idx: int, normalize: bool = True) -> Optional[dict]:
        """
        Get video data by index.

        :param idx: video index.
        :param normalize: normalize video data.
        :return: dictionary with 'video' tensor and 'label'.
        """
        self._ensure_env()

        key = f"{idx:010d}".encode()
        with self._env.begin() as txn:
            value = txn.get(key)
            if value is None:
                return None
            data = pickle.loads(value)
            if normalize:
                data['video'] = data['video'].float() / 255.0
            return data

    def __len__(self) -> int:
        """Get number of videos in database."""
        self._ensure_env()

        with self._env.begin() as txn:
            meta = txn.get(b'__meta__')
            if meta:
                metadata = pickle.loads(meta)
                return metadata['total_videos']
        return 0

    def __del__(self):
        """Close environment on object deletion."""
        self.close()
