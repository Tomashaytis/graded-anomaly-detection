import json
import warnings
import numpy as np

from pathlib import Path
from typing import Optional
from sklearn.cluster import HDBSCAN, DBSCAN, SpectralClustering
from sklearn.exceptions import ConvergenceWarning
from scipy.spatial.distance import directed_hausdorff


class GroupDetector:
    """
    Group detection over pedestrian trajectories.

    Supports two clustering backends:
      - 'spectral': spectral clustering on the affinity matrix
      - 'dbscan': DBSCAN on the precomputed distance matrix (D = 1 - A)
      - 'hdbscan': HDBSCAN on the precomputed distance matrix (D = 1 - A)

    The affinity matrix is built from pairwise trajectory similarity
    (spatial proximity, stability, and shape parallelism).
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
            method: str = "spectral",
            dbscan_eps: float = 0.5,
            dbscan_min_samples: int = 2,
            hdbscan_min_cluster_size: int = 2
    ):
        """
        :param min_len: Minimum track length to participate in clustering.
        :param show_affinity: Print the affinity matrix before clustering.
        :param sigma_spatial: Gaussian width for mean-distance kernel.
        :param sigma_stability: Gaussian width for distance-variance kernel.
        :param sigma_parallel: Gaussian width for Hausdorff kernel.
        :param min_common_frames: Minimum overlapping frames to compare two tracks.
        :param pair_sim_threshold: For n == 2, similarity >= this -> one group.
        :param eigengap_min: For n >= 3, gap below this -> one group (spectral only).
        :param max_k: Upper bound for auto-k (None = up to n - 1, spectral only).
        :param method: Cluster medhod ('spectral', 'dbscan' or 'hdbscan').
        :param dbscan_eps: DBSCAN eps on distance matrix (D = 1 - A).
        :param dbscan_min_samples: DBSCAN min_samples.
        :param hdbscan_min_cluster_size: HDBSCAN min cluster size.
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

        self._method = method
        self._dbscan_eps = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples
        self._hdbscan_min_cluster_size = hdbscan_min_cluster_size

        self._warn_count = 0
        self._warn_details = []

    def process_tracks(self, tracks: dict = None, tracks_path: str = None) -> dict[int, list[int]]:
        """
        Process tracks, build affinity, cluster, return groups.

        :param tracks: Mapping track_id -> (N, 5) array of [frame_id, x1, y1, x2, y2].
        :param tracks_path: Path to the tracks JSON file.
        :return: Mapping group_id -> list of track ids.
        :raises ValueError: If tracks/tracks_path not set.
        """
        if tracks is None and tracks_path is not None:
            tracks = json.loads(Path(tracks_path).read_text())
        elif tracks is None and tracks_path is None:
            raise ValueError("Tracks or tracks path not set")
        tracks = {int(tid): np.asarray(pts, dtype=np.float32) for tid, pts in tracks.items()}
        tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= self._min_len}

        tids, matrix_a = self.build_affinity(tracks)

        if self._show_affinity:
            print("\nAffinity matrix:")
            print("      " + " ".join(f"{t:>6}" for t in tids))
            for i, tid in enumerate(tids):
                row = " ".join(f"{matrix_a[i, j]:6.3f}" for j in range(len(tids)))
                print(f"{tid:>5} {row}")

        if self._method == "dbscan":
            labels = self._cluster_dbscan(matrix_a)
        elif self._method == "hdbscan":
            labels = self._cluster_hdbscan(matrix_a)
        else:
            labels = self._cluster_spectral(matrix_a)

        groups = {}
        for tid, label in zip(tids, labels):
            groups.setdefault(int(label), []).append(int(tid))
        return groups

    def _cluster_spectral(self, matrix_a: np.ndarray) -> np.ndarray:
        """
        Spectral clustering with eigengap-selected k.

        :param matrix_a: Matrix with features.
        :return: Labels.
        """
        k = self.auto_k(matrix_a)
        if k == 1:
            return np.zeros(matrix_a.shape[0], dtype=int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
            )
            labels = sc.fit_predict(matrix_a)

        if caught:
            self._warn_count += 1
            if len(self._warn_details) < 10:
                self._warn_details.append(
                    (matrix_a.shape[0], k, len(set(labels)))
                )

        return labels

    def _cluster_dbscan(self, matrix_a: np.ndarray) -> np.ndarray:
        """
        DBSCAN on the precomputed distance matrix D = 1 - A.

        :param matrix_a: Matrix with features.
        :return: Labels.
        """
        d = 1.0 - matrix_a
        np.fill_diagonal(d, 0.0)
        d = np.clip(d, 0.0, 1.0)
        return DBSCAN(
            eps=self._dbscan_eps,
            min_samples=self._dbscan_min_samples,
            metric="precomputed",
        ).fit_predict(d)

    def _cluster_hdbscan(self, matrix_a: np.ndarray) -> np.ndarray:
        """
        HDBSCAN on the precomputed distance matrix D = 1 - A.

        :param matrix_a: Matrix with features.
        :return: Labels.
        """
        d = 1.0 - matrix_a
        np.fill_diagonal(d, 0.0)
        d = np.clip(d, 0.0, 1.0)
        labels = HDBSCAN(
            min_cluster_size=self._hdbscan_min_cluster_size,
            metric="precomputed",
            allow_single_cluster=False,
            copy=False,
        ).fit_predict(d)
        if np.all(labels == -1):
            return self._cluster_spectral(matrix_a)
        return labels

    @staticmethod
    def centroids(track: np.ndarray) -> np.ndarray:
        """
        Return centroids for track bboxes.

        :param tracks: Tracks.
        :return: Track centroids.
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

        :param traj_a: First centroid track.
        :param traj_b: Second centroid track.
        :param bbox_scale: Normalisation scale (mean bbox diagonal).
        :return: Similarity in [0, 1]; 0.0 if too few common frames.
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

        :param tracks: Mapping track_id -> (N, 5) [f, x1, y1, x2, y2].
        :return: Tuple: (list of track ids, (n, n) similarity matrix).
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
        Pick the number of clusters automatically (spectral only).

        n == 1: one group;
        n == 2: split by similarity threshold;
        n >= 3: eigengap on the Laplacian.

        :param matrix_a: (n, n) similarity matrix with 1s on the diagonal.
        :return: Number of clusters (>= 1).
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