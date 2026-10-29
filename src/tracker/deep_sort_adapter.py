import numpy as np

from typing import List
from boxmot import DeepOcSort

from .base_tracker import BaseTracker, DETECTION, TRACK


class DeepSORTAdapter(BaseTracker):
    """Adapter for DeepOcSort from the boxmot library."""

    def __init__(self, **kwargs):
        """
        Initialize the DeepOcSort tracker.

        :param kwargs: Tracker parameters. Recognized keys:
            model_weights, fp16, device, max_dist, max_iou_dist,
            max_age, n_init, nn_budget, mc_lambda, ema_alpha, per_class
        """
        super().__init__(**kwargs)

        self._init_params = kwargs.copy()

        reid_weights = kwargs.get('model_weights', None)
        half = kwargs.get('fp16', False)
        device = kwargs.get('device', 'cpu')

        max_dist = kwargs.get('max_dist', 0.2)
        max_iou_dist = kwargs.get('max_iou_dist', 0.7)
        max_age = kwargs.get('max_age', 30)
        n_init = kwargs.get('n_init', 3)
        nn_budget = kwargs.get('nn_budget', 100)
        mc_lambda = kwargs.get('mc_lambda', 0.995)
        ema_alpha = kwargs.get('ema_alpha', 0.9)
        per_class = kwargs.get('per_class', False)

        self.tracker = DeepOcSort(
            reid_weights=reid_weights,
            half=half,
            device=device,
            max_dist=max_dist,
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            per_class=per_class,
        )

    def update(self, detections: List[DETECTION], frame: np.ndarray) -> List[TRACK]:
        if detections:
            boxmot_dets = np.array([
                [d[0], d[1], d[2], d[3], d[4], d[5]]
                for d in detections
            ], dtype=np.float32)
        else:
            boxmot_dets = np.empty((0, 6), dtype=np.float32)

        tracks = self.tracker.update(boxmot_dets, frame)

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
        half = self._init_params.get('fp16', False)
        device = self._init_params.get('device', 'cpu')
        max_age = self._init_params.get('max_age', 30)
        n_init = self._init_params.get('n_init', 3)

        self.tracker = DeepOcSort(
            reid_weights=reid_weights,
            half=half,
            device=device,
            max_age=max_age,
            n_init=n_init,
            max_dist=self._init_params.get('max_dist', 0.2),
            max_iou_dist=self._init_params.get('max_iou_dist', 0.7),
            nn_budget=self._init_params.get('nn_budget', 100),
            mc_lambda=self._init_params.get('mc_lambda', 0.995),
            ema_alpha=self._init_params.get('ema_alpha', 0.9),
            per_class=self._init_params.get('per_class', False),
        )