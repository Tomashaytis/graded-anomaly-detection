import json
import numpy as np

from tqdm import tqdm

from src.detector import GroupDetector
from src.utils import PairwiseMetrics, calculate_pairwise_metrics, calculate_ari_nmi

OUTPUT_FILE = "data/traj_dataset.json"
METHOD = "hdbscan"
BOX_SIZE = 50.0


def xy_to_bbox_track(xy: np.ndarray, box_size: float = BOX_SIZE) -> np.ndarray:
    """
    Convert (N, 2) [x, y] centroids to (N, 5) [frame_id, x1, y1, x2, y2].

    :param xy: (N, 2) array of (x, y)
    :param box_size: side length of the synthetic bbox, pixels
    :return: (N, 5) array of [frame_id, x1, y1, x2, y2]
    """
    n = len(xy)
    frame_ids = np.arange(n, dtype=np.float32)
    half = box_size / 2.0
    x1 = xy[:, 0] - half
    y1 = xy[:, 1] - half
    x2 = xy[:, 0] + half
    y2 = xy[:, 1] + half
    return np.stack([frame_ids, x1, y1, x2, y2], axis=1)


def load_dataset(path: str) -> dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset: dict[int, dict] = {}
    for sid, scene in raw.items():
        dataset[int(sid)] = {
            "tracks": {
                int(tid): xy_to_bbox_track(np.asarray(xy, dtype=np.float32))
                for tid, xy in scene["tracks"].items()
            },
            "groups": {
                int(gid): [int(t) for t in members]
                for gid, members in scene["groups"].items()
            },
            "group_sizes": tuple(int(s) for s in scene["group_sizes"]),
            "n_groups": int(scene["n_groups"]),
            "n_singletons": int(scene["n_singletons"]),
        }
    return dataset


if __name__ == '__main__':
    group_detector = GroupDetector(method=METHOD)

    dataset = load_dataset(OUTPUT_FILE)

    pairwise_metrics = PairwiseMetrics()

    ari_scores, nmi_scores = [], []
    for scene_id, scene in tqdm(dataset.items(), desc='Processing scenes'):
        groups_pred = group_detector.process_tracks(tracks=scene["tracks"])
        groups_gt = scene["groups"]
        pairwise_metrics += calculate_pairwise_metrics(groups_gt, groups_pred)
        ari, nmi = calculate_ari_nmi(groups_gt, groups_pred)
        ari_scores.append(ari)
        nmi_scores.append(nmi)
    
    print('Pairwise metrics (macro):')
    print(f"Precision:         {pairwise_metrics.precision:.4f}")
    print(f"Recall:            {pairwise_metrics.recall:.4f}")
    print(f"F1:                {pairwise_metrics.f1:.4f}")
    print(f"Accuracy:          {pairwise_metrics.accuracy:.4f}")
    print(f"Balanced accuracy: {pairwise_metrics.balanced_accuracy:.4f}")
    print(f"MCC:               {pairwise_metrics.mcc:.4f}")
    print(f"Jaccard:           {pairwise_metrics.jaccard:.4f}")
    print()
    print('Others metrics (macro):')
    print(f"ARI: {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")
    print(f"NMI: {np.mean(nmi_scores):.4f} ± {np.std(nmi_scores):.4f}")