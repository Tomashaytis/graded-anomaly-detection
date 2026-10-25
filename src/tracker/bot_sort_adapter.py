import torch
import numpy as np

from typing import List
from pathlib import Path
from boxmot import BotSort

from .base_tracker import BaseTracker, DETECTION, TRACK


class BotSORTAdapter(BaseTracker):
    """Adapter for Bot-SORT from the boxmot library."""

    def __init__(self, **kwargs):
        """
        Initialize the Bot-SORT tracker.

        :param kwargs: tracker parameters. Recognized keys:
            model_weights, fp16, device, with_reid, det_thresh,
            track_high_thresh, track_low_thresh, new_track_thresh,
            match_thresh, track_buffer, frame_rate, per_class
        """
        super().__init__(**kwargs)

        self._init_params = kwargs.copy()

        reid_weights = kwargs.get('model_weights', None)
        half = kwargs.get('fp16', False) and torch.cuda.is_available()
        device = kwargs.get('device', 'cpu')
        with_reid = kwargs.get('with_reid', True)

        if reid_weights and with_reid:
            if not isinstance(reid_weights, Path):
                reid_weights = Path(reid_weights)

            if not reid_weights.exists():
                with_reid = False
                reid_weights = None
            else:
                valid_suffixes = ['.pt', '.pth', '.torchscript', '.onnx', '.engine', '.tflite']
                if reid_weights.suffix not in valid_suffixes:
                    with_reid = False
                    reid_weights = None

        self._create_tracker(
            reid_weights=reid_weights,
            half=half,
            device=device,
            with_reid=with_reid,
            **{k: v for k, v in kwargs.items()
               if k not in ['model_weights', 'fp16', 'device', 'with_reid']}
        )

    def _create_tracker(self, reid_weights, half, device, **kwargs):
        """
        Create the underlying BotSort instance.

        :param reid_weights: path to the ReID model, or None to disable ReID
        :param half: use FP16 inference
        :param device: device string ('cpu', 'cuda', ...)
        :param kwargs: additional tracker parameters
        """
        tracker_params = {
            'half': half,
            'device': device,
            'det_thresh': kwargs.get('det_thresh', 0.3),
            'track_high_thresh': kwargs.get('track_high_thresh', 0.5),
            'track_low_thresh': kwargs.get('track_low_thresh', 0.1),
            'new_track_thresh': kwargs.get('new_track_thresh', 0.6),
            'match_thresh': kwargs.get('match_thresh', 0.8),
            'track_buffer': kwargs.get('track_buffer', 30),
            'frame_rate': kwargs.get('frame_rate', 30),
            'per_class': kwargs.get('per_class', False),
        }

        if reid_weights:
            self.tracker = BotSort(reid_weights=reid_weights, **tracker_params)
        else:
            self.tracker = BotSort(reid=None, **tracker_params)

    def update(self, detections: List[DETECTION], frame: np.ndarray) -> List[TRACK]:
        if detections:
            boxmot_dets = np.array([
                [d[0], d[1], d[2], d[3], d[4], d[5]]
                for d in detections
            ], dtype=np.float32)
        else:
            boxmot_dets = np.empty((0, 6), dtype=np.float32)

        try:
            tracks = self.tracker.update(boxmot_dets, frame)
        except Exception:
            return []

        result = []
        if tracks is not None and len(tracks) > 0:
            for track in tracks:
                if len(track) >= 7:
                    x1, y1, x2, y2 = track[0:4]
                    track_id = int(track[4])
                    conf = float(track[5])
                    cls_id = int(track[6])

                    result.append((
                        track_id,
                        float(x1), float(y1), float(x2), float(y2),
                        conf,
                        cls_id
                    ))

        return result

    def reset(self):
        reid_weights = self._init_params.get('model_weights', None)
        half = self._init_params.get('fp16', False) and torch.cuda.is_available()
        device = self._init_params.get('device', 'cpu')
        with_reid = self._init_params.get('with_reid', True)

        if reid_weights and with_reid:
            if not isinstance(reid_weights, Path):
                reid_weights = Path(reid_weights)
            if not reid_weights.exists():
                with_reid = False
                reid_weights = None

        self._create_tracker(
            reid_weights=reid_weights,
            half=half,
            device=device,
            with_reid=with_reid,
            **{k: v for k, v in self._init_params.items()
               if k not in ['model_weights', 'fp16', 'device', 'with_reid']}
        )