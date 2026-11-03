"""
Make group crops from tracks + groups.

Reads:
  - tracks.json: {tid: [[frame_id, x1, y1, x2, y2], ...]}
  - groups.json: {video_name: [[tid, ...], [tid, ...], ...]}

Writes:
  - crops/{video}/{gid}.mp4                — one file per group
  - crops/meta.json                        — list of crop metadata
  - crops/local_crops.json                 — up to 3 selected local crops per group
  - singletons.json                        — {video: [tid, ...]} for GCN
"""

import cv2
import json
import random
import numpy as np

from pathlib import Path

from src.detector import SpectralGroupDetector

TRACKS_PATH = "../data/test_traj/traj/001.json"
VIDEO_PATH = "../data/test_traj/video/001.mp4"

OUTPUT_DIR = "../data/test_traj/crops"

N_SAMPLES = 32
PADDING = 0.1
N_TOP = 2
N_RANDOM = 1
SEED = 42

# --- local crop selection ---
MAX_LOCAL_SLOTS = 3
MAX_AREA_RATIO = 0.8
IOU_THRESH = 0.6


def load_tracks(path: str) -> dict[int, np.ndarray]:
    raw = json.loads(Path(path).read_text())
    return {int(tid): np.asarray(pts, dtype=np.float32)
            for tid, pts in raw.items()}


def select_groups(groups: list[list[int]], n_top: int, n_random: int,
                  seed: int = 42) -> tuple[list[list[int]], list[list[int]]]:
    """
    Split groups into (selected, singletons).

    All groups are eligible for selection (including singletons).
    Singletons are also returned separately for the GCN pipeline.
    """
    singletons = [g for g in groups if len(g) == 1]
    candidates = list(groups)

    if not candidates:
        return [], singletons

    candidates.sort(key=lambda g: -len(g))
    selected = candidates[:n_top]
    rest = candidates[n_top:]

    if rest and n_random > 0:
        rng = random.Random(seed)
        selected += rng.sample(rest, min(n_random, len(rest)))

    return selected, singletons


def track_bbox_at(track: np.ndarray, frame_id: int):
    """Return (x1, y1, x2, y2) at the given frame, or None."""
    mask = track[:, 0].astype(int) == frame_id
    if not mask.any():
        return None
    row = track[mask][0]
    return float(row[1]), float(row[2]), float(row[3]), float(row[4])


def group_bbox_at(members: list[int], tracks: dict[int, np.ndarray],
                  frame_id: int, frame_w: int, frame_h: int,
                  padding: float):
    """Group bbox at the given frame, with padding, clipped to frame."""
    boxes = []
    for tid in members:
        bb = track_bbox_at(tracks[tid], frame_id)
        if bb is not None:
            boxes.append(bb)
    if not boxes:
        return None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * padding)
    y1 = max(0, y1 - h * padding)
    x2 = min(frame_w, x2 + w * padding)
    y2 = min(frame_h, y2 + h * padding)

    return int(x1), int(y1), int(x2), int(y2)


def raw_group_bbox_at(members: list[int], tracks: dict[int, np.ndarray],
                      frame_id: int):
    """Group bbox at the given frame, WITHOUT padding and clipping."""
    boxes = []
    for tid in members:
        bb = track_bbox_at(tracks[tid], frame_id)
        if bb is not None:
            boxes.append(bb)
    if not boxes:
        return None

    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def bbox_iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def select_local_crops(meta: list[dict], frame_w: int, frame_h: int,
                       max_slots: int = MAX_LOCAL_SLOTS,
                       area_ratio_max: float = MAX_AREA_RATIO,
                       iou_thresh: float = IOU_THRESH) -> tuple[list, list]:
    """
    Select up to max_slots local crops from meta.

    Pipeline:
      1. Drop crops covering >= area_ratio_max of the frame ("global").
      2. Greedy NMS by IoU, priority = group_area / crop_area (density).
      3. Take top-max_slots by priority (deterministic, no random).

    Returns:
      (slots, mask) where slots is a list of length max_slots
      (None for empty) and mask is a list of 0/1.
    """
    frame_area = float(frame_w * frame_h)

    # 1. filter out "global" crops
    candidates = []
    for m in meta:
        x1, y1, x2, y2 = m["bbox"]
        crop_area = float(max(0, x2 - x1) * max(0, y2 - y1))
        print(crop_area / frame_area)
        if crop_area <= 0:
            continue
        if crop_area / frame_area >= area_ratio_max:
            continue
        candidates.append(m)

    # priority: density (group_area / crop_area), fallback: smaller crop first
    def priority(m):
        x1, y1, x2, y2 = m["bbox"]
        crop_area = float(max(0, x2 - x1) * max(0, y2 - y1))
        ga = m.get("group_area")
        if ga is not None and crop_area > 0:
            return ga / crop_area
        return -crop_area  # fallback: less global first

    candidates.sort(key=priority, reverse=True)

    # 2. greedy NMS by IoU
    kept = []
    for m in candidates:
        if all(bbox_iou(m["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(m)

    # 3. take top-max_slots by priority (already sorted)
    selected = kept[:max_slots]

    # 4. pad to max_slots with None + mask
    slots = selected + [None] * (max_slots - len(selected))
    mask = [1 if s is not None else 0 for s in slots]
    return slots, mask


def make_crops_for_group(video_path: str, members: list[int],
                         tracks: dict[int, np.ndarray], output_dir: Path,
                         n_samples: int, padding: float,
                         meta: list[dict], video_name: str, gid: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames = min(len(tracks[tid]) for tid in members)
    if num_frames < 2:
        cap.release()
        return

    if num_frames <= n_samples:
        frame_ids = list(range(num_frames))
    else:
        frame_ids = np.linspace(0, num_frames - 1, n_samples).round().astype(int).tolist()
        seen = set()
        frame_ids = [f for f in frame_ids if not (f in seen or seen.add(f))]

    # union bbox (with padding) and raw union bbox (without) over the window
    union = None
    raw_union = None
    for f in frame_ids:
        bb = group_bbox_at(members, tracks, f, frame_w, frame_h, padding)
        rb = raw_group_bbox_at(members, tracks, f)
        if bb is not None:
            if union is None:
                union = list(bb)
            else:
                union[0] = min(union[0], bb[0])
                union[1] = min(union[1], bb[1])
                union[2] = max(union[2], bb[2])
                union[3] = max(union[3], bb[3])
        if rb is not None:
            if raw_union is None:
                raw_union = list(rb)
            else:
                raw_union[0] = min(raw_union[0], rb[0])
                raw_union[1] = min(raw_union[1], rb[1])
                raw_union[2] = max(raw_union[2], rb[2])
                raw_union[3] = max(raw_union[3], rb[3])

    if union is None:
        cap.release()
        return

    crop_w = union[2] - union[0]
    crop_h = union[3] - union[1]
    if crop_w <= 0 or crop_h <= 0:
        cap.release()
        return
    if crop_w % 2:
        crop_w += 1
    if crop_h % 2:
        crop_h += 1

    x1, y1, x2, y2 = union
    out_name = f"{gid:02d}.mp4"
    out_path = output_dir / out_name
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_w, crop_h),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Failed to open writer: {out_path}")

    for f in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            crop = cv2.resize(crop, (crop_w, crop_h))
        writer.write(crop)

    writer.release()
    cap.release()

    group_bbox = None
    group_area = None
    if raw_union is not None:
        group_bbox = [int(v) for v in raw_union]
        group_area = int(max(0, raw_union[2] - raw_union[0]) *
                         max(0, raw_union[3] - raw_union[1]))

    meta.append({
        "video": video_name,
        "group_id": gid,
        "members": [int(t) for t in members],
        "frame_ids": [int(f) for f in frame_ids],
        "n_frames": len(frame_ids),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "group_bbox": group_bbox,
        "group_area": group_area,
        "path": str(out_path),
    })


def main():
    random.seed(SEED)

    video_name = Path(VIDEO_PATH).stem
    tracks = load_tracks(TRACKS_PATH)
    group_detector = SpectralGroupDetector()
    groups = list(group_detector.process_tracks(TRACKS_PATH).values())

    if not groups:
        print("No groups found.")
        return

    selected, singletons = select_groups(groups, N_TOP, N_RANDOM, SEED)

    print(f"Groups total:    {len(groups)}")
    print(f"Selected:        {len(selected)}")
    print(f"Singletons:      {len(singletons)}")
    print(f"Selected groups: {selected}")

    output_dir = Path(OUTPUT_DIR) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: list[dict] = []
    for gid, members in enumerate(selected):
        members = [t for t in members if t in tracks]
        make_crops_for_group(
            VIDEO_PATH, members, tracks, output_dir,
            N_SAMPLES, PADDING, meta, video_name, gid,
        )

    meta_path = Path(OUTPUT_DIR) / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # --- local crop selection: up to MAX_LOCAL_SLOTS per group ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    slots, mask = select_local_crops(meta, frame_w, frame_h)
    local_crops = [{
        "video": video_name,
        "slots": [
            None if s is None else {
                "group_id": s["group_id"],
                "bbox": s["bbox"],
                "path": s["path"],
            }
            for s in slots
        ],
        "mask": mask,
    }]

    local_path = Path(OUTPUT_DIR) / "local_crops.json"
    local_path.write_text(json.dumps(local_crops, indent=2))

    singletons_path = Path(OUTPUT_DIR) / "singletons.json"
    singletons_path.write_text(json.dumps(
        {video_name: singletons}, indent=2,
    ))

    print(f"\nCrops written:   {len(meta)}")
    print(f"Metadata:        {meta_path}")
    print(f"Local crops:     {local_path}")
    print(f"Singletons:      {singletons_path}")


if __name__ == "__main__":
    main()