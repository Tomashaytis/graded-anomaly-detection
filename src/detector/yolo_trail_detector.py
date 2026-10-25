import cv2
import json
import numpy as np

from pathlib import Path
from ultralytics import YOLO


class YoloTrailDetector:
    """
    YOLO-based detector with a tracker that draws motion trails
    """

    def __init__(self,
                 model_path: str,
                 tracker,
                 conf_threshold: float = 0.3,
                 classes=(0,),
                 min_track_length: int = 10,
                 trail_length: int = 60,
                 draw_trail: bool = True,
                 save_json: bool = False):
        """
        Initialize the detector and load the YOLO model.

        :param model_path: path to YOLO weights
        :param tracker: tracker instance exposing update(detections, frame) -> tracks
        :param conf_threshold: minimum detection confidence (0..1)
        :param classes: tuple of class IDs to keep (None = all classes)
        :param min_track_length: minimum track length to be written to JSON
        :param trail_length: number of most recent points to draw per track
        :param draw_trail: whether to draw trails on the output video
        :param save_json: whether to dump trajectories to JSON next to the output video
        """
        self.model_path = model_path
        self.tracker = tracker
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.min_track_length = min_track_length
        self.trail_length = trail_length
        self.draw_trail = draw_trail
        self.save_json = save_json

        self._history: dict[int, list[tuple[float, float]]] = {}
        self._colors: dict[int, tuple] = {}

        self._model = YOLO(str(self.model_path))
        print("YOLO model loaded")

    def process_video(self, video_path: str, output_path: str):
        """
        Process a video: detect, track, draw trails, write the result,
        and optionally export trajectories to JSON.

        :param video_path: path to the input video file
        :param output_path: path to the output video file
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        self._check_video(video_path)

        cap = cv2.VideoCapture(str(video_path))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self._fps,
            (self._frame_width, self._frame_height),
        )
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                dets = self._detect_objects(frame)
                dets = [d for d in dets
                        if d[4] >= self.conf_threshold
                        and (self.classes is None or d[5] in self.classes)]

                if dets:
                    tracks = self.tracker.update(dets, frame)
                    for tid, x1, y1, x2, y2, *_ in tracks:
                        tid = int(tid)
                        if tid < 0:
                            continue
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        self._history.setdefault(tid, []).append((cx, cy))
                        if self.draw_trail:
                            self._draw_one(frame, tid, (x1, y1, x2, y2), cx, cy)

                writer.write(frame)
        finally:
            cap.release()
            writer.release()

        if self.save_json:
            self._dump_json(output_path)

    def _detect_objects(self, frame: np.ndarray) -> list:
        """
        Run YOLO detection on a single frame.

        :param frame: BGR frame (uint8)
        :return: list of detections [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        results = self._model(frame, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, conf, cls_id))
        return detections

    def _color(self, tid: int) -> tuple:
        """
        Get (and lazily generate) a color for a track.

        The color is deterministically derived from the track_id,
        so the same track is always drawn in the same color.

        :param tid: track identifier
        :return: (B, G, R) tuple
        """
        if tid not in self._colors:
            rng = np.random.default_rng(abs(tid))
            self._colors[tid] = tuple(int(c) for c in rng.integers(60, 255, 3))
        return self._colors[tid]

    def _draw_one(self, frame, tid, bbox, cx, cy):
        """
        Draw a single tracked object: bounding box, track ID,
        trail polyline, and the current centroid.

        :param frame: BGR frame to draw on (modified in place)
        :param tid: track identifier
        :param bbox: bounding box as (x1, y1, x2, y2)
        :param cx: centroid x coordinate
        :param cy: centroid y coordinate
        """
        color = self._color(tid)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"id {tid}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        pts = self._history[tid][-self.trail_length:]
        for p, q in zip(pts, pts[1:]):
            cv2.line(frame, (int(p[0]), int(p[1])),
                            (int(q[0]), int(q[1])), color, 2)
        cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)

    def _dump_json(self, output_path: Path):
        """
        Save trajectories to a JSON file next to the output video.

        Only tracks with at least `min_track_length` points are kept.
        Keys are ints (track IDs), values are lists of (x, y) float pairs.

        :param output_path: path to the output video; the JSON path is
                            derived by replacing the suffix with '.json'
        """
        clean = {
            int(tid): [(float(x), float(y)) for x, y in pts]
            for tid, pts in self._history.items()
            if len(pts) >= self.min_track_length
        }
        output_path.with_suffix('.json').write_text(
            json.dumps(clean, indent=2))

    def _check_video(self, video_path: Path):
        """
        Validate the video file and read its basic properties.

        Populates `_fps`, `_frame_width`, `_frame_height`, `_total_frames`.

        :param video_path: path to the video file
        :raises FileNotFoundError: if the file does not exist
        :raises ValueError: if the video cannot be opened
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()