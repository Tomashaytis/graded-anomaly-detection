"""
Workers utilities.
"""

import torch
import random

def worker_init_fn(worker_id: int) -> None:
    """
    Resets transforms for worker.

    :param worker_id: worker ID.
    """
    info = torch.utils.data.get_worker_info()
    ds = info.dataset

    seed = torch.initial_seed() % 2**32
    random.seed(seed)

    if hasattr(ds, 'reset_transforms'):
        ds.reset_transforms()