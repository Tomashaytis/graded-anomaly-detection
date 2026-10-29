import cv2
import json
import numpy as np

from pathlib import Path
from typing import Optional

from tqdm import tqdm
from ultralytics import YOLO


Tracks = dict[int, list[tuple[int, float, float, float, float]]]


class YoloTrailDetector:
    """
    YOLO-based detector with a tracker that collects motion trails.

    Detection, storage and rendering are separate steps:
      - process_video()  — detect + track, returns raw trajectories
      - save_tracks()    — dump trajectories to JSON
      - render_video()   — overlay any trajectories onto a video
    """

    def __init__(self,
                 model_path: str,
                 tracker,
                 conf_threshold: float = 0.3,
                 classes=(0,),
                 min_track_length: int = 10,
                 trail_length: int = 60):
        """
        Initialize the detector and load the YOLO model.

        :param model_path: Path to YOLO weights.
        :param tracker: Tracker instance exposing update(detections, frame) -> tracks.
        :param conf_threshold: Minimum detection confidence (0..1).
        :param classes: Tuple of class IDs to keep (None = all classes).
        :param min_track_length: Minimum track length to be written to JSON.
        :param trail_length: How many recent points to draw per track.
        """
        self.model_path = model_path
        self.tracker = tracker
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.min_track_length = min_track_length
        self.trail_length = trail_length

        self._colors: dict[int, tuple[int, int, int]] = {}

        self._model = YOLO(str(self.model_path))
        print("YOLO model loaded")

    def process_video(self, video_path: str, step: int = 1, interpolate: bool = True) -> Tracks:
        """
        Detect and track objects in a video.

        :param video_path: Path to the input video file.
        :param step: Process every N-th frame (1 = every frame).
        :param interpolate: If true, interpolates track gaps.
        :return: Mapping track_id -> list of (frame_id, x1, y1, x2, y2) boxes.
        """
        video_path = Path(video_path)
        self._check_video(video_path)

        history: Tracks = {}

        cap = cv2.VideoCapture(str(video_path))
        frame_id = 0
        try:
            total_rendered = (self._total_frames + step - 1) // step
            with tqdm(total=total_rendered, desc="Processing video", unit="frame") as pbar:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_id % step == 0:
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
                                history.setdefault(tid, []).append(
                                    (frame_id, float(x1), float(y1),
                                     float(x2), float(y2))
                                )

                        pbar.update(1)
                        pbar.set_postfix(tracks=len(history))

                    frame_id += 1
        finally:
            cap.release()

        if interpolate:
            history = self._interpolate(history)

        return history

    def save_tracks(self, output_path: str, tracks: Optional[Tracks] = None):
        """
        Save trajectories to a JSON file.

        :param output_path: Destination JSON path.
        :param tracks: Trajectories to save; if None, raises ValueError.
        """
        if tracks is None:
            raise ValueError("tracks must be provided (process_video returns them)")

        clean = {
            int(tid): [[int(f), float(x1), float(y1), float(x2), float(y2)]
                       for f, x1, y1, x2, y2 in pts]
            for tid, pts in tracks.items()
            if len(pts) >= self.min_track_length
        }
        Path(output_path).write_text(json.dumps(clean, indent=2))

    def render_video(self,
                     video_path: str,
                     output_video_path: str,
                     tracks: Tracks,
                     step: int = 1):
        """
        Overlay trajectories onto a video and write the result.

        :param video_path: Path to the source video.
        :param output_video_path: Path to the output annotated video.
        :param tracks: Mapping track_id -> list of (frame_id, x1, y1, x2, y2).
        :param step: Write every N-th frame (1 = every frame).
        """
        video_path = Path(video_path)
        output_video_path = Path(output_video_path)
        self._check_video(video_path)

        cap = cv2.VideoCapture(str(video_path))
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self._fps / step,
            (self._frame_width, self._frame_height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Failed to open video writer: {output_video_path}")

        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_id % step == 0:
                    for tid, pts in tracks.items():
                        self._draw_one(frame, tid, pts, frame_id)
                    writer.write(frame)

                frame_id += 1
        finally:
            cap.release()
            writer.release()

    def _detect_objects(self, frame: np.ndarray) -> list:
        """
        Run YOLO detection on a single frame.

        :param frame: BGR frame (uint8).
        :return: List of detections [(x1, y1, x2, y2, confidence, class_id), ...].
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

    @staticmethod
    def _interpolate(tracks: Tracks, smooth_window: int = 5) -> Tracks:
        """
        Interpolate tracks with smoothing.

        :param tracks: Tracks.
        :param smooth_window: Number of frames to smooth before/after a gap.
        :return: Interpolated tracks.
        """
        interpolated_tracks: Tracks = {}

        for track_id, track in tracks.items():
            if len(track) == 0:
                continue
            interpolated_track = [track[0]]
            for i in range(1, len(track)):
                if track[i - 1][0] + 1 == track[i][0]:
                    interpolated_track.append(track[i])
                elif track[i][0] <= track[i - 1][0]:
                    continue
                else:
                    # Сглаживаем bbox перед разрывом (последние smooth_window кадров)
                    prev_window = interpolated_track[-smooth_window:]
                    prev_x1 = np.median([p[1] for p in prev_window])
                    prev_y1 = np.median([p[2] for p in prev_window])
                    prev_x2 = np.median([p[3] for p in prev_window])
                    prev_y2 = np.median([p[4] for p in prev_window])

                    # Сглаживаем bbox после разрыва (первые smooth_window кадров)
                    next_window = track[i:i + smooth_window]
                    next_x1 = np.median([p[1] for p in next_window])
                    next_y1 = np.median([p[2] for p in next_window])
                    next_x2 = np.median([p[3] for p in next_window])
                    next_y2 = np.median([p[4] for p in next_window])

                    # Интерполируем между сглаженными bbox
                    for f in range(track[i - 1][0] + 1, track[i][0]):
                        alpha = (f - track[i - 1][0]) / (track[i][0] - track[i - 1][0])
                        x1 = prev_x1 * (1 - alpha) + next_x1 * alpha
                        y1 = prev_y1 * (1 - alpha) + next_y1 * alpha
                        x2 = prev_x2 * (1 - alpha) + next_x2 * alpha
                        y2 = prev_y2 * (1 - alpha) + next_y2 * alpha
                        interpolated_track.append((f, x1, y1, x2, y2))

                    interpolated_track.append(track[i])

            interpolated_tracks[track_id] = interpolated_track

        return interpolated_tracks

    def _color(self, tid: int) -> tuple[int, int, int]:
        """
        Get a BGR color for a track.
        """
        if tid not in self._colors:
            rng = np.random.default_rng(abs(int(tid)))
            self._colors[tid] = tuple(int(c) for c in rng.integers(60, 255, 3))
        return self._colors[tid]

    def _draw_one(self, frame, tid: int, pts, frame_id: int):
        """
        Draw a single track on the frame: bounding box, trail polyline,
        current centroid, and track ID label.

        :param frame: BGR frame to draw on (modified in place).
        :param tid: Track identifier.
        :param pts: List of (frame_id, x1, y1, x2, y2) for this track.
        :param frame_id: Current frame index.
        """
        color = self._color(tid)

        visible = [p for p in pts if p[0] <= frame_id][-self.trail_length:]
        if not visible:
            return

        _, x1, y1, x2, y2 = visible[-1]
        bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, f"id {tid}", (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        centroids = [
            ((p[1] + p[3]) / 2, (p[2] + p[4]) / 2)
            for p in visible
        ]
        for p, q in zip(centroids, centroids[1:]):
            cv2.line(frame, (int(p[0]), int(p[1])),
                            (int(q[0]), int(q[1])), color, 2)
        cx, cy = centroids[-1]
        cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)

    def _check_video(self, video_path: Path):
        """
        Validate the video file and read its basic properties.

        :param video_path: Path to the video file.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the video cannot be opened.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(cap.cv2.CAP_PROP_FRAME_COUNT) if False else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()