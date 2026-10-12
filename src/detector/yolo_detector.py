import numpy as np

from ultralytics import YOLO

from .base_detector import BaseDetector


class YoloDetector(BaseDetector):
    """Детектор на основе YOLO"""

    def _initialize_detector(self):
        """Инициализация YOLO модели"""
        self._model = YOLO(str(self.model_path))
        print(f"YOLO модель загружена")

    def _detect_objects(self, frame: np.ndarray) -> list:
        """Детекция объектов через YOLO"""
        results = self._model(frame, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, conf, cls_id))

        return detections