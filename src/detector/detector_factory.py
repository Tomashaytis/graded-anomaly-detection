from typing import Dict, Type

from .base_detector import BaseDetector
from .yolo_detector import YoloDetector
from .detr_detector import DETRDetector


class DetectorFactory:
    """Фабрика для создания детекторов"""

    _detectors: Dict[str, Type[BaseDetector]] = {
        'yolo': YoloDetector,
        'detr': DETRDetector,
        # 'efficientdet': EfficientDetDetector,
        # 'opencv_dnn': OpenCVDNNDetector,
        # 'sahi': SAHIDetector,
    }

    @classmethod
    def create_detector(cls, detector_type: str, **kwargs) -> BaseDetector:
        """
        Создание детектора указанного типа

        Args:
            detector_type: тип детектора ('yolo' и т.д.)
            **kwargs: параметры для детектора

        Returns:
            экземпляр детектора
        """
        if detector_type not in cls._detectors:
            available = ', '.join(cls._detectors.keys())
            raise ValueError(f"Неизвестный тип детектора '{detector_type}'. "
                             f"Доступны: {available}")

        detector_class = cls._detectors[detector_type]
        return detector_class(**kwargs)

    @classmethod
    def register_detector(cls, name: str, detector_class: Type[BaseDetector]):
        """Регистрация нового типа детектора"""
        cls._detectors[name] = detector_class