import json
import numpy as np

from pathlib import Path
from typing import Optional
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import directed_hausdorff


class SpectralGroupDetector:
    """
    Group detection over pedestrian trajectories via spectral clustering.
    """

    def __init__(
            self,
            min_len: int = 10,
            show_affinity: bool = False,
            sigma_spatial: float = 1.5,
            sigma_stability: float = 0.5,
            sigma_parallel: float = 1.5,
            min_common_frames: int = 5,
            pair_sim_threshold: float = 0.6,
            eigengap_min: float = 0.01,
            max_k: Optional[int] = None,
    ):
        """
        Initialize the detector.

        :param min_len: minimum track length to participate in clustering
        :param show_affinity: print the affinity matrix before clustering
        :param sigma_spatial: Gaussian width for mean-distance kernel
        :param sigma_stability: Gaussian width for distance-variance kernel
        :param sigma_parallel: Gaussian width for Hausdorff kernel
        :param min_common_frames: minimum overlapping frames to compare two tracks
        :param pair_sim_threshold: for n == 2, similarity >= this -> one group
        :param eigengap_min: for n >= 3, gap below this -> one group
        :param max_k: upper bound for auto-k (None = up to n - 1)
        """
        self._min_len = min_len
        self._show_affinity = show_affinity

        self._sigma_spatial = sigma_spatial
        self._sigma_stability = sigma_stability
        self._sigma_parallel = sigma_parallel
        self._min_common_frames = min_common_frames

        self._pair_sim_threshold = pair_sim_threshold
        self._eigengap_min = eigengap_min
        self._max_k = max_k

    def process_tracks(self, tracks_path: str) -> dict[int, list[int]]:
        """
        Load tracks, build affinity, cluster, return groups.

        :param tracks_path: path to the tracks JSON file
        :return: mapping group_id -> list of track ids
        """
        tracks = self.load_tracks(tracks_path)
        tracks = {tid: pts for tid, pts in tracks.items()
                  if len(pts) >= self._min_len}

        tids, matrix_a = self.build_affinity(tracks)

        if self._show_affinity:
            print("\nAffinity matrix:")
            print("      " + " ".join(f"{t:>6}" for t in tids))
            for i, tid in enumerate(tids):
                row = " ".join(f"{matrix_a[i, j]:6.3f}" for j in range(len(tids)))
                print(f"{tid:>5} {row}")

        k = self.auto_k(matrix_a)

        if k == 1:
            groups = {0: [int(t) for t in tids]}
        else:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
            )
            labels = sc.fit_predict(matrix_a)
            groups = {}
            for tid, label in zip(tids, labels):
                groups.setdefault(int(label), []).append(int(tid))

        return groups

    @staticmethod
    def load_tracks(path: str) -> dict[int, np.ndarray]:
        """
        Load tracks from a JSON file.

        :param path: path to the JSON file
        :return: mapping track_id -> (N, 5) [frame_id, x1, y1, x2, y2]
        """
        raw = json.loads(Path(path).read_text())
        return {int(tid): np.asarray(pts, dtype=np.float32)
                for tid, pts in raw.items()}

    @staticmethod
    def centroids(track: np.ndarray) -> np.ndarray:
        """
        (N, 5) [f, x1, y1, x2, y2] -> (N, 3) [f, cx, cy]
        """
        f = track[:, 0]
        cx = (track[:, 1] + track[:, 3]) / 2
        cy = (track[:, 2] + track[:, 4]) / 2
        return np.stack([f, cx, cy], axis=1)

    def pair_similarity(self, traj_a: np.ndarray, traj_b: np.ndarray,
                        bbox_scale: float) -> float:
        """
        Similarity in [0, 1] between two centroid tracks (f, cx, cy).

        Product of three Gaussian kernels: spatial (mean distance),
        stability (distance variance), parallel (Hausdorff shape).
        Only frames present in both tracks are compared.

        :param traj_a: first centroid track
        :param traj_b: second centroid track
        :param bbox_scale: normalisation scale (mean bbox diagonal)
        :return: similarity in [0, 1]; 0.0 if too few common frames
        """
        frames_a = {int(f): (x, y) for f, x, y in traj_a}
        frames_b = {int(f): (x, y) for f, x, y in traj_b}
        common = sorted(set(frames_a) & set(frames_b))
        if len(common) < self._min_common_frames:
            return 0.0

        a = np.array([frames_a[f] for f in common], dtype=np.float32)
        b = np.array([frames_b[f] for f in common], dtype=np.float32)

        d = np.linalg.norm(a - b, axis=1)
        mean_dist = d.mean() / bbox_scale
        std_dist = d.std() / bbox_scale
        hausdorff = max(directed_hausdorff(a, b)[0],
                        directed_hausdorff(b, a)[0]) / bbox_scale

        spatial = np.exp(- (mean_dist ** 2) / (2 * self._sigma_spatial ** 2))
        stability = np.exp(- (std_dist ** 2) / (2 * self._sigma_stability ** 2))
        parallel = np.exp(- (hausdorff ** 2) / (2 * self._sigma_parallel ** 2))
        return float(spatial * stability * parallel)

    def build_affinity(self, tracks: dict[int, np.ndarray]
                       ) -> tuple[list[int], np.ndarray]:
        """
        Build the pairwise similarity matrix over all tracks.

        :param tracks: mapping track_id -> (N, 5) [f, x1, y1, x2, y2]
        :return: (list of track ids, (n, n) similarity matrix)
        """
        centroids_map = {tid: self.centroids(pts) for tid, pts in tracks.items()}

        diagonals = []
        for pts in tracks.values():
            w = pts[:, 3] - pts[:, 1]
            h = pts[:, 4] - pts[:, 2]
            diagonals.append(np.hypot(w, h))
        bbox_scale = max(float(np.mean(np.concatenate(diagonals))), 1.0)

        tids = list(tracks.keys())
        n = len(tids)
        matrix_a = np.eye(n, dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                s = self.pair_similarity(centroids_map[tids[i]],
                                         centroids_map[tids[j]],
                                         bbox_scale)
                matrix_a[i, j] = matrix_a[j, i] = s

        return tids, matrix_a

    def auto_k(self, matrix_a: np.ndarray) -> int:
        """
        Pick the number of clusters automatically.

        n == 1: one group;
        n == 2: split by similarity threshold;
        n >= 3: eigengap on the Laplacian.

        :param matrix_a: (n, n) similarity matrix with 1s on the diagonal
        :return: number of clusters (>= 1)
        """
        n = matrix_a.shape[0]
        if n <= 1:
            return 1
        if n == 2:
            return 1 if matrix_a[0, 1] >= self._pair_sim_threshold else 2

        matrix_w = matrix_a.copy()
        np.fill_diagonal(matrix_w, 0.0)
        degrees = matrix_w.sum(axis=1)
        matrix_l = np.diag(degrees) - matrix_w

        eigvals = np.sort(np.linalg.eigvalsh(matrix_l))

        max_k = min(self._max_k or (n - 1), n - 1)

        best_k = 1
        best_gap = -1.0

        for k in range(2, max_k + 1):
            gap = eigvals[k] - eigvals[k - 1]
            if gap > best_gap:
                best_gap = gap
                best_k = k

        if best_gap < self._eigengap_min:
            return 1
        return best_k