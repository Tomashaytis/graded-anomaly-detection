"""
Fix LMDB: sequential keys + update internal index field.
Does NOT modify metadata. Commits every N records to avoid memory issues.
"""

import gc
import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm

from src.dataset import LMDBReader


def fix_lmdb_keys(lmdb_path: str, output_path: str, map_size_gb: int = 250, commit_every: int = 100):
    """
    Renumber LMDB keys and update 'index' field.
    Metadata copied as-is.
    """
    source_path = Path(lmdb_path)
    temp_path = Path(output_path)
    temp_path.mkdir(parents=True, exist_ok=True)

    source_env = lmdb.open(str(source_path), readonly=True, lock=False)
    target_env = lmdb.open(str(temp_path), map_size=map_size_gb * 1024 ** 3)

    with source_env.begin() as src_txn:
        cursor = src_txn.cursor()
        new_idx = 0
        total = src_txn.stat()['entries']

        # Начинаем первую транзакцию
        dst_txn = target_env.begin(write=True)

        for key, value in tqdm(cursor, total=total, desc="Fixing"):
            # Метаданные копируем без изменений
            if key == b'__meta__':
                dst_txn.put(key, value)
                continue

            # Пропускаем None (битые видео)
            if value is None:
                continue

            try:
                # Исправляем только поле index
                data = pickle.loads(value)
                data['index'] = new_idx

                # Новый ключ и значение
                new_key = f"{new_idx:010d}".encode()
                new_value = pickle.dumps(data)

                dst_txn.put(new_key, new_value)
                new_idx += 1

                # Коммит каждые commit_every записей
                if new_idx % commit_every == 0:
                    dst_txn.commit()
                    dst_txn = target_env.begin(write=True)
                    gc.collect()

            except Exception as e:
                print(f"  Skipping {key}: {e}")

        # Финальный коммит
        dst_txn.commit()

    source_env.close()
    target_env.close()

    print(f"\n✅ Fixed {new_idx} records → {temp_path}")


if __name__ == "__main__":
    fix_lmdb_keys(
        lmdb_path='D:/Projects/actions-data/train_lmdb',
        output_path='F:/tmp/train_lmdb',
        map_size_gb=250,
        commit_every=100
    )