import cv2
import torch
import numpy as np

from transformers import DetrImageProcessor, DetrForObjectDetection

from .base_detector import BaseDetector


class DETRDetector(BaseDetector):
    """Детектор на основе DETR от HuggingFace"""

    def _initialize_detector(self):
        """Инициализация DETR модели"""
        # Используем model_path из базового класса
        model_name = str(self.model_path)  # Преобразуем Path в строку

        print(f"Загрузка DETR модели: {model_name}")

        # Загружаем процессор и модель
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)

        # Определяем устройство
        self.device = getattr(self, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        print(f"DETR модель загружена на устройстве: {self.device}")

    def _detect_objects(self, frame: np.ndarray) -> list:
        """Детекция объектов через DETR"""
        # Конвертируем BGR в RGB (DETR обучен на RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Подготовка изображения
        inputs = self.processor(images=frame_rgb, return_tensors="pt").to(self.device)

        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Пост-процессинг
        target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.conf_threshold
        )[0]

        # Формируем результат в нужном формате
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()

            # Фильтрация по классам, если указаны
            if self.classes is not None and int(label) not in self.classes:
                continue

            detections.append((
                float(x1), float(y1), float(x2), float(y2),
                float(score), int(label)
            ))

        return detections