import numpy as np
from typing import List, Optional
from pathlib import Path
import torch

from boxmot import BotSort

from .base_tracker import BaseTracker, DETECTION, TRACK


class BotSORTAdapter(BaseTracker):
    """Адаптер для Bot-SORT из библиотеки boxmot"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Сохраняем все параметры для возможности сброса
        self._init_params = kwargs.copy()

        # Извлекаем параметры
        reid_weights = kwargs.get('model_weights', None)
        half = kwargs.get('fp16', False) and torch.cuda.is_available()
        device = kwargs.get('device', 'cpu')
        with_reid = kwargs.get('with_reid', True)

        # Проверяем модель, если она указана
        if reid_weights and with_reid:
            # Конвертируем в Path если нужно
            if not isinstance(reid_weights, Path):
                reid_weights = Path(reid_weights)

            # Проверяем существование файла
            if not reid_weights.exists():
                print(f"Модель не найдена: {reid_weights}")
                print("Будет использован трекинг без ReID")
                with_reid = False
                reid_weights = None
            else:
                # Проверяем расширение
                valid_suffixes = ['.pt', '.pth', '.torchscript', '.onnx', '.engine', '.tflite']
                if reid_weights.suffix not in valid_suffixes:
                    print(f"Расширение {reid_weights.suffix} может не поддерживаться")
                    print(f"Разрешенные расширения: {valid_suffixes}")

        print(f"Параметры трекера:")
        print(f"  - ReID модель: {reid_weights if reid_weights else 'не используется'}")
        print(f"  - Устройство: {device}")
        print(f"  - FP16: {half}")
        print(f"  - With ReID: {with_reid}")

        # Инициализируем трекер - передаем все параметры один раз
        self._create_tracker(
            reid_weights=reid_weights,
            half=half,
            device=device,
            with_reid=with_reid,
            **{k: v for k, v in kwargs.items() if k not in ['model_weights', 'fp16', 'device', 'with_reid']}
        )

    def _create_tracker(self, reid_weights, half, device, with_reid, **kwargs):
        """Внутренний метод для создания трекера"""
        try:
            self.tracker = BotSort(
                reid_weights=reid_weights,
                half=half,
                device=device,
                det_thresh=kwargs.get('det_thresh', 0.3),
                track_high_thresh=kwargs.get('track_high_thresh', 0.5),
                track_low_thresh=kwargs.get('track_low_thresh', 0.1),
                new_track_thresh=kwargs.get('new_track_thresh', 0.6),
                match_thresh=kwargs.get('match_thresh', 0.8),
                track_buffer=kwargs.get('track_buffer', 30),
                frame_rate=kwargs.get('frame_rate', 30),
                with_reid=with_reid,
                per_class=kwargs.get('per_class', False),
            )
            print("Трекер успешно инициализирован")
        except Exception as e:
            print(f"Ошибка при создании трекера: {e}")
            print("Пробуем создать трекер без ReID...")

            # Пробуем без ReID
            self.tracker = BotSort(
                reid_weights=None,
                half=False,
                device=device,
                det_thresh=kwargs.get('det_thresh', 0.3),
                track_high_thresh=kwargs.get('track_high_thresh', 0.5),
                track_low_thresh=kwargs.get('track_low_thresh', 0.1),
                new_track_thresh=kwargs.get('new_track_thresh', 0.6),
                match_thresh=kwargs.get('match_thresh', 0.8),
                track_buffer=kwargs.get('track_buffer', 30),
                frame_rate=kwargs.get('frame_rate', 30),
                with_reid=False,
                per_class=kwargs.get('per_class', False),
            )
            print("Трекер создан без ReID модели")

    def update(self, detections: List[DETECTION], frame: np.ndarray) -> List[TRACK]:
        """Обновление трекера"""

        # Конвертируем детекции
        if detections:
            boxmot_dets = np.array([
                [d[0], d[1], d[2], d[3], d[4], d[5]]
                for d in detections
            ], dtype=np.float32)
        else:
            boxmot_dets = np.empty((0, 6), dtype=np.float32)

        # Обновляем трекер
        try:
            tracks = self.tracker.update(boxmot_dets, frame)
        except Exception as e:
            print(f"Ошибка при обновлении трекера: {e}")
            return []

        # Конвертируем результат
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
        """Сброс трекера - создаем новый экземпляр с сохраненными параметрами"""
        print("Сброс трекера...")

        # Извлекаем параметры из сохраненных
        reid_weights = self._init_params.get('model_weights', None)
        half = self._init_params.get('fp16', False) and torch.cuda.is_available()
        device = self._init_params.get('device', 'cpu')
        with_reid = self._init_params.get('with_reid', True)

        # Проверяем модель
        if reid_weights and with_reid:
            if not isinstance(reid_weights, Path):
                reid_weights = Path(reid_weights)
            if not reid_weights.exists():
                print(f"Модель не найдена при сбросе: {reid_weights}")
                with_reid = False
                reid_weights = None

        # Создаем новый трекер
        self._create_tracker(
            reid_weights=reid_weights,
            half=half,
            device=device,
            with_reid=with_reid,
            **{k: v for k, v in self._init_params.items()
               if k not in ['model_weights', 'fp16', 'device', 'with_reid']}
        )