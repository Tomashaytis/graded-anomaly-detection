import json
import numpy as np

from tqdm import tqdm

from src.generator import TrajectoryGenerator

DATASET_LENGTH = 1000
N_FRAMES = 300
SEED = 42
OUTPUT_FILE = "data/traj_dataset.json"

N_GROUPS_RANGE = (1, 4)
GROUP_SIZE_RANGE = (2, 5)
N_SINGLETONS_RANGE = (0, 3)


def scene_to_json(scene):
    return {
        "tracks": {
            str(tid): xy.tolist() for tid, xy in scene["tracks"].items()
        },
        "groups": {
            str(gid): [int(t) for t in members]
            for gid, members in scene["groups"].items()
        },
        "group_sizes": [int(s) for s in scene["group_sizes"]],
        "n_groups": int(scene["n_groups"]),
        "n_singletons": int(scene["n_singletons"]),
    }


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    traj_generator = TrajectoryGenerator(rng)

    dataset: dict[int, dict] = {}

    for scene_id in tqdm(range(DATASET_LENGTH), desc="Generating dataset"):
        n_groups = int(rng.integers(N_GROUPS_RANGE[0], N_GROUPS_RANGE[1] + 1))
        group_sizes_real = tuple(
            int(rng.integers(GROUP_SIZE_RANGE[0], GROUP_SIZE_RANGE[1] + 1))
            for _ in range(n_groups)
        )
        n_singletons = int(
            rng.integers(N_SINGLETONS_RANGE[0], N_SINGLETONS_RANGE[1] + 1)
        )
        group_sizes = group_sizes_real + (1,) * n_singletons

        tracks, groups = traj_generator.make_scene(
            group_sizes=group_sizes,
            n_frames=N_FRAMES,
        )

        all_group_sizes = tuple(len(members) for members in groups.values())

        dataset[scene_id] = {
            "tracks": tracks,
            "groups": groups,
            "group_sizes": all_group_sizes,
            "n_groups": n_groups,
            "n_singletons": n_singletons,
        }

    print(f"Generated {len(dataset)} scenes")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): scene_to_json(v) for k, v in dataset.items()},
            f, indent=2, ensure_ascii=False,
        )