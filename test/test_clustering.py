"""
Quick check: spectral clustering of pedestrian trajectories.

Reads a tracks JSON produced by YoloTrailDetector
(format: {track_id: [[frame_id, x1, y1, x2, y2], ...]}),
builds a pairwise similarity matrix, picks the number of clusters
automatically (eigengap for n >= 3, similarity threshold for n = 2),
runs SpectralClustering, and prints groups.
"""

import json
import numpy as np

from pathlib import Path
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import SpectralClustering

TRACKS_PATH = "../data/test_traj/traj/001.mp4"
MIN_LEN = 10
SHOW_AFFINITY = True

SIGMA_SPATIAL = 1.5
SIGMA_STABILITY = 0.5
SIGMA_PARALLEL = 1.5
MIN_COMMON_FRAMES = 5

# Auto-k settings
PAIR_SIM_THRESHOLD = 0.6   # for n == 2: >= this -> one group
EIGENGAP_MIN = 0.01        # for n >= 3: below this -> one group
MAX_K = None               # None = up to n - 1


# ---------- helpers ----------

def load_tracks(path: str) -> dict[int, np.ndarray]:
    raw = json.loads(Path(path).read_text())
    return {int(tid): np.asarray(pts, dtype=np.float32)
            for tid, pts in raw.items()}


def centroids(track: np.ndarray) -> np.ndarray:
    """(N, 5) [f, x1, y1, x2, y2] -> (N, 3) [f, cx, cy]"""
    f = track[:, 0]
    cx = (track[:, 1] + track[:, 3]) / 2
    cy = (track[:, 2] + track[:, 4]) / 2
    return np.stack([f, cx, cy], axis=1)


def pair_similarity(traj_a: np.ndarray, traj_b: np.ndarray,
                    bbox_scale: float) -> float:
    """
    Similarity in [0, 1] between two centroid tracks (N, 3): (f, cx, cy).
    Only frames present in both tracks are compared.
    """
    frames_a = {int(f): (x, y) for f, x, y in traj_a}
    frames_b = {int(f): (x, y) for f, x, y in traj_b}
    common = sorted(set(frames_a) & set(frames_b))
    if len(common) < MIN_COMMON_FRAMES:
        return 0.0

    a = np.array([frames_a[f] for f in common], dtype=np.float32)
    b = np.array([frames_b[f] for f in common], dtype=np.float32)

    d = np.linalg.norm(a - b, axis=1)
    mean_dist = d.mean() / bbox_scale
    std_dist = d.std() / bbox_scale
    hausdorff = max(directed_hausdorff(a, b)[0],
                    directed_hausdorff(b, a)[0]) / bbox_scale

    spatial = np.exp(- (mean_dist ** 2) / (2 * SIGMA_SPATIAL ** 2))
    stability = np.exp(- (std_dist ** 2) / (2 * SIGMA_STABILITY ** 2))
    parallel = np.exp(- (hausdorff ** 2) / (2 * SIGMA_PARALLEL ** 2))
    return float(spatial * stability * parallel)


def build_affinity(tracks: dict[int, np.ndarray]) -> tuple[list[int], np.ndarray]:
    centroids_map = {tid: centroids(pts) for tid, pts in tracks.items()}

    diagonals = []
    for pts in tracks.values():
        w = pts[:, 3] - pts[:, 1]
        h = pts[:, 4] - pts[:, 2]
        diagonals.append(np.hypot(w, h))
    bbox_scale = max(float(np.mean(np.concatenate(diagonals))), 1.0)

    tids = list(tracks.keys())
    n = len(tids)
    A = np.eye(n, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            s = pair_similarity(centroids_map[tids[i]],
                                centroids_map[tids[j]],
                                bbox_scale)
            A[i, j] = A[j, i] = s

    return tids, A


def auto_k(A: np.ndarray) -> int:
    """
    Pick the number of clusters automatically.

    - n == 1: one group
    - n == 2: one group if similarity is high, else two
    - n >= 3: eigengap heuristic on the Laplacian; if the gap is tiny,
              there is no clear structure, so return one group

    :param A: (n, n) similarity matrix with 1s on the diagonal
    :return: number of clusters (>= 1)
    """
    n = A.shape[0]
    if n <= 1:
        return 1
    if n == 2:
        return 1 if A[0, 1] >= PAIR_SIM_THRESHOLD else 2

    # Laplacian of the similarity graph
    W = A.copy()
    np.fill_diagonal(W, 0.0)
    degrees = W.sum(axis=1)
    L = np.diag(degrees) - W

    eigvals = np.sort(np.linalg.eigvalsh(L))

    max_k = min(MAX_K or (n - 1), n - 1)

    best_k = 1
    best_gap = -1.0

    for k in range(2, max_k + 1):
        gap = eigvals[k] - eigvals[k - 1]
        if gap > best_gap:
            best_gap = gap
            best_k = k

    print(f"  Собственные числа: {[f'{x:.4f}' for x in eigvals[:5]]}")
    print(f"  Найденный зазор: best_k={best_k}, best_gap={best_gap:.6f}")

    if best_gap < EIGENGAP_MIN:
        return 1
    return best_k


# ---------- main ----------

def main():
    tracks = load_tracks(TRACKS_PATH)
    tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= MIN_LEN}

    print(f"Tracks loaded: {len(tracks)}")
    if len(tracks) < 2:
        print("Need at least 2 tracks to cluster.")
        return

    tids, A = build_affinity(tracks)

    if SHOW_AFFINITY:
        print("\nAffinity matrix:")
        print("      " + " ".join(f"{t:>6}" for t in tids))
        for i, tid in enumerate(tids):
            row = " ".join(f"{A[i, j]:6.3f}" for j in range(len(tids)))
            print(f"{tid:>5} {row}")

    k = auto_k(A)
    print(f"\nAuto-selected n_clusters: {k}")

    if k == 1:
        groups = {0: [int(t) for t in tids]}
    else:
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = sc.fit_predict(A)
        groups = {}
        for tid, label in zip(tids, labels):
            groups.setdefault(int(label), []).append(int(tid))

    print("\nGroups:")
    for gid, members in sorted(groups.items()):
        print(f"  group {gid}: {members}")


if __name__ == "__main__":
    main()