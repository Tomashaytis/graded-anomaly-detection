import cv2
import json
import torch
import random
import numpy as np

from pathlib import Path

from src.detector import GroupDetector
from src.dataset import VideoLoader


class GroupCropper:
    """
    Build fixed-size tensors for pedestrian groups.
    """

    def __init__(
        self,
        group_detector: GroupDetector,
        n_frames: int = 32,
        size: int = 224,
        padding: float = 0.1,
        n_top: int = 2,
        n_random: int = 1,
        global_first: bool = False,
        global_similarity_threshold: float = 0.8,
        seed: int = 42,
    ):
        """
        :param group_detector: SpectralGroupDetector instance.
        :param n_frames: Frames per tensor (resampled uniformly).
        :param size: Spatial size of each frame (size x size).
        :param padding: Bbox padding as a fraction of bbox size.
        :param n_top: Take this many largest groups by member count.
        :param n_random: Take this many random groups from the rest.
        :param global_first: If True, add global crop at first position.
        :param global_similarity_threshold: If global_first=True, drop group crops whose bbox covers
            at least this fraction of the frame area
        :param seed: RNG seed for the random pick.
        """
        self._group_detector = group_detector
        self._n_frames = n_frames
        self._size = size
        self._padding = padding
        self._n_top = n_top
        self._n_random = n_random
        self._global_first = global_first
        self._global_similarity_threshold = global_similarity_threshold
        self._seed = seed

    def get_group_tensors(self, video_path: str, tracks: dict = None, tracks_path: str = None) -> dict[int, torch.Tensor]:
        """
        Build one tensor per selected group.

        :param video_path: Path to the source video.
        :param tracks: Mapping track_id -> (N, 5) array of [frame_id, x1, y1, x2, y2].
        :param tracks_path: Path to the tracks JSON.
        :return: Mapping group_id -> tensor (n_samples, 3, size, size) in RGB, values in [0, 1].
        :raises ValueError: If the video cannot be opened or if tracks/tracks_path not set.
        """
        if tracks is None and tracks_path is not None:
            tracks = json.loads(Path(tracks_path).read_text())
        elif tracks is None and tracks_path is None:
            raise ValueError("Tracks or tracks path not set")
        tracks = {int(tid): np.asarray(pts, dtype=np.float32) for tid, pts in tracks.items()}

        groups = list(self._group_detector.process_tracks(tracks=tracks).values())
        if not groups:
            return {}

        selected = self._select_groups(groups)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result: dict[int, torch.Tensor] = {}
        next_gid = 0

        if self._global_first:
            global_tensor = self._global_to_tensor(cap)
            if global_tensor is not None:
                result[next_gid] = global_tensor
                next_gid += 1

        for members in selected:
            members = [t for t in members if t in tracks]
            if not members:
                continue
            tensor = self._group_to_tensor(cap, tracks, members, frame_w, frame_h)
            if tensor is not None:
                result[next_gid] = tensor
                next_gid += 1

        cap.release()
        return result

    def _select_groups(self, groups: list[list[int]]) -> list[list[int]]:
        """
        Pick groups to crop.

        Sort by member count (descending), take the top n_top,
        then add n_random groups sampled from the rest.

        :param groups: List of groups, each group is a list of track ids.
        :return: List of selected groups.
        """
        candidates = sorted(groups, key=lambda g: -len(g))
        selected = candidates[:self._n_top]
        rest = candidates[self._n_top:]
        if rest and self._n_random > 0:
            rng = random.Random(self._seed)
            selected += rng.sample(rest, min(self._n_random, len(rest)))
        return selected

    @staticmethod
    def _track_bbox_at(track: np.ndarray, frame_id: int):
        """
        Return the bbox of a track at the given frame.

        :param track: (N, 5) array of [frame_id, x1, y1, x2, y2].
        :param frame_id: Frame index.
        :return: (x1, y1, x2, y2) as floats, or None if the track has no point at this frame.
        """
        mask = track[:, 0].astype(int) == frame_id
        if not mask.any():
            return None
        row = track[mask][0]
        return float(row[1]), float(row[2]), float(row[3]), float(row[4])

    def _group_bbox_at(self, members, tracks, frame_id, frame_w, frame_h):
        """
        Group bbox at a given frame: union of member bboxes, with padding,
        clipped to the frame.

        :param members: Track ids of the group.
        :param tracks: Mapping track_id -> (N, 5) array.
        :param frame_id: Frame index.
        :param frame_w: Frame width (for clipping).
        :param frame_h: Frame height (for clipping).
        :return: (x1, y1, x2, y2) as ints, or None if no member is visible.
        """
        boxes = []
        for tid in members:
            bb = self._track_bbox_at(tracks[tid], frame_id)
            if bb is not None:
                boxes.append(bb)
        if not boxes:
            return None

        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - w * self._padding)
        y1 = max(0, y1 - h * self._padding)
        x2 = min(frame_w, x2 + w * self._padding)
        y2 = min(frame_h, y2 + h * self._padding)

        return int(x1), int(y1), int(x2), int(y2)

    def _group_to_tensor(self, cap, tracks, members, frame_w, frame_h) -> torch.Tensor | None:
        """
        Build a single tensor for one group.

        :param cap: Opened cv2.VideoCapture.
        :param tracks: Mapping track_id -> (N, 5) array.
        :param members: Track ids of the group.
        :param frame_w: Frame width.
        :param frame_h: Frame height.
        :return: Tensor (n_samples, 3, size, size) in RGB, values in [0, 1], or None if no frames could be read.
        """
        num_frames = min(len(tracks[tid]) for tid in members)
        if num_frames < 1:
            return None

        if num_frames <= self._n_frames:
            frame_ids = list(range(num_frames))
        else:
            frame_ids = np.linspace(
                0, num_frames - 1, self._n_frames
            ).round().astype(int).tolist()
            seen = set()
            frame_ids = [f for f in frame_ids
                         if not (f in seen or seen.add(f))]

        union = None
        for f in frame_ids:
            bb = self._group_bbox_at(members, tracks, f, frame_w, frame_h)
            if bb is None:
                continue
            if union is None:
                union = list(bb)
            else:
                union[0] = min(union[0], bb[0])
                union[1] = min(union[1], bb[1])
                union[2] = max(union[2], bb[2])
                union[3] = max(union[3], bb[3])
        if union is None:
            return None

        x1, y1, x2, y2 = union
        if x2 <= x1 or y2 <= y1:
            return None

        if self._global_first:
            frame_area = float(frame_w * frame_h)
            group_area = float((x2 - x1) * (y2 - y1))
            if group_area / frame_area >= self._global_similarity_threshold:
                return None

        frames = []
        for f in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = VideoLoader.reflect_pad_and_resize(
                crop, (self._size, self._size),
            )
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames.append(crop)

        if not frames:
            return None

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)

        if tensor.shape[0] < self._n_frames:
            pad = self._n_frames - tensor.shape[0]
            last = tensor[-1:].repeat(pad, 1, 1, 1)
            tensor = torch.cat([tensor, last], dim=0)

        return tensor


    def _global_to_tensor(self, cap) -> torch.Tensor | None:
        """
        Build a tensor of the full frame (no group crop).

        :param cap: Opened cv2.VideoCapture.
        :return: Tensor (n_frames, 3, size, size) in RGB, values in [0, 1],
                 or None if no frames could be read.
        """
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            return None

        if total <= self._n_frames:
            frame_ids = list(range(total))
        else:
            frame_ids = np.linspace(0, total - 1, self._n_frames).round().astype(int).tolist()
            seen = set()
            frame_ids = [f for f in frame_ids if not (f in seen or seen.add(f))]

        frames = []
        for f in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = VideoLoader.reflect_pad_and_resize(frame, (self._size, self._size), ratio_threshold=10)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        if not frames:
            return None

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)

        if tensor.shape[0] < self._n_frames:
            pad = self._n_frames - tensor.shape[0]
            last = tensor[-1:].repeat(pad, 1, 1, 1)
            tensor = torch.cat([tensor, last], dim=0)

        return tensor