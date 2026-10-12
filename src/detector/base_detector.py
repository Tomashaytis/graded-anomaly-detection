import cv2
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseDetector(ABC):
    """
    Базовый класс для всех детекторов объектов.
    Определяет единый интерфейс для работы с разными детекторами.
    """

    def __init__(self,
                 model_path: str,
                 video_path: str,
                 output_dir: str,
                 tracker,
                 min_track_length: int = 10,
                 padding: int = 30,
                 target_fps: float = 20.0,
                 conf_threshold: float = 0.3,
                 classes: Optional[List[int]] = None):
        """
        Args:
            model_path: путь к модели детектора
            video_path: путь к видеофайлу
            output_dir: директория для сохранения результатов
            tracker: экземпляр трекера
            min_track_length: минимальная длина трека
            padding: отступы при обрезке кадра
            target_fps: целевой FPS для выходных видео
            conf_threshold: порог уверенности детекции
            classes: список классов для детекции (None = все классы)
        """
        self.model_path = model_path
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.tracker = tracker
        self.min_track_length = min_track_length
        self.padding = padding
        self.target_fps = target_fps
        self.conf_threshold = conf_threshold
        self.classes = classes

        # Информация о видео
        self._cap = None
        self._total_frames = 0
        self._fps = 0
        self._frame_width = 0
        self._frame_height = 0

        # Директории для временных файлов и результатов
        self._raw_dir = self.output_dir / 'temp_raw_frames'
        self._final_dir = self.output_dir / 'final_tracks'

        # Структуры для хранения данных
        self._all_tracks = {}
        self._track_metadata = {}
        self._frame_count = 0

        # Проверка наличия видео
        self._check_video()

        # Создание директорий
        self._create_directories()

        # Инициализация детектора (должна быть реализована в наследниках)
        self._initialize_detector()

    def _check_video(self):
        """Проверка видео файла"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap.release()

        print(f"Видео: {self.video_path.name}")
        print(f"  - Кадров: {self._total_frames}")
        print(f"  - FPS: {self._fps:.2f}")
        print(f"  - Размер: {self._frame_width}x{self._frame_height}")

    def _create_directories(self):
        """Создание необходимых директорий"""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._final_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _initialize_detector(self):
        """
        Инициализация модели детектора.
        Должна быть реализована в каждом наследнике.
        """
        pass

    @abstractmethod
    def _detect_objects(self, frame: np.ndarray) -> List[Tuple]:
        """
        Детекция объектов на кадре.

        Args:
            frame: кадр в формате BGR (как в OpenCV)

        Returns:
            список детекций в формате [(x1, y1, x2, y2, conf, class_id), ...]
        """
        pass

    def process_video(self):
        """Основной метод обработки видео"""
        print("\n" + "=" * 50)
        print(f"НАЧАЛО ОБРАБОТКИ ВИДЕО")
        print(f"Детектор: {self.__class__.__name__}")
        print(f"Трекер: {self.tracker.__class__.__name__}")
        print("=" * 50 + "\n")

        # Этап 1: Детекция и сбор кадров
        self.detect_frames()

        # Этап 2: Анализ треков
        self.analyze_tracks()

        # Этап 3: Создание видео
        self.create_videos()

        # Этап 4: Очистка
        self.cleanup()

        print("\n" + "=" * 50)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"Результаты сохранены в: {self._final_dir}")
        print("=" * 50)

    def detect_frames(self):
        """Этап 1: Детекция объектов и сохранение кадров"""
        print("\n[Этап 1/4] Детекция объектов")
        print("-" * 40)

        self._frame_count = 0
        self._all_tracks.clear()
        self.tracker.reset()

        self._cap = cv2.VideoCapture(str(self.video_path))

        try:
            from tqdm import tqdm
            pbar = tqdm(total=self._total_frames, desc="Обработка кадров",
                        unit="кадр", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break

                self._frame_count += 1

                # Детекция объектов
                detections = self._detect_objects(frame)

                # Фильтрация по порогу уверенности
                detections = [d for d in detections if d[4] >= self.conf_threshold]

                # Фильтрация по классам
                if self.classes is not None:
                    detections = [d for d in detections if d[5] in self.classes]

                # Обновление трекера
                if detections:
                    tracks = self.tracker.update(detections, frame)

                    # Сохранение кадров для каждого трека
                    for track in tracks:
                        track_id, x1, y1, x2, y2, conf, cls_id = track
                        self._save_detection_frame(frame, track_id, (x1, y1, x2, y2))

                pbar.update(1)

                # Периодическая информация
                if self._frame_count % 100 == 0:
                    pbar.set_postfix({
                        'треков': len(self._all_tracks)
                    })

            pbar.close()

        finally:
            self._cap.release()

        print(f"\nСбор завершён. Кадров: {self._frame_count}, треков: {len(self._all_tracks)}")

    def _save_detection_frame(self, frame: np.ndarray, track_id: int, bbox: tuple):
        """Сохранение кадра с объектом"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # Добавляем паддинг
        x1_pad = max(0, x1 - self.padding)
        y1_pad = max(0, y1 - self.padding)
        x2_pad = min(w, x2 + self.padding)
        y2_pad = min(h, y2 + self.padding)

        # Вырезаем кадр
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        # Сохраняем только достаточно большие кадры
        if crop.shape[0] > 20 and crop.shape[1] > 20:
            frame_filename = self._raw_dir / f'track_{track_id}_frame_{self._frame_count:04d}.jpg'
            cv2.imwrite(str(frame_filename), crop)

            # Сохраняем метаданные
            if track_id not in self._all_tracks:
                self._all_tracks[track_id] = []

            self._all_tracks[track_id].append({
                'frame_num': self._frame_count,
                'filename': str(frame_filename),
                'coords': (x1_pad, y1_pad, x2_pad, y2_pad),
                'size': (crop.shape[1], crop.shape[0])
            })

    def analyze_tracks(self):
        """Этап 2: Анализ треков"""
        print("\n[Этап 2/4] Анализ треков")
        print("-" * 40)

        self._track_metadata.clear()
        tracks_before = len(self._all_tracks)
        tracks_after = 0

        from tqdm import tqdm
        for track_id, frames in tqdm(self._all_tracks.items(), desc="Анализ треков",
                                     unit="трек"):

            if len(frames) < self.min_track_length:
                continue

            # Находим максимальные размеры
            widths = [f['size'][0] for f in frames]
            heights = [f['size'][1] for f in frames]

            max_width = max(widths)
            max_height = max(heights)

            # Делаем размеры чётными
            if max_width % 2 != 0:
                max_width += 1
            if max_height % 2 != 0:
                max_height += 1

            self._track_metadata[track_id] = {
                'max_width': max_width,
                'max_height': max_height,
                'num_frames': len(frames),
                'frames': sorted(frames, key=lambda x: x['frame_num']),
                'start_frame': frames[0]['frame_num'],
                'end_frame': frames[-1]['frame_num']
            }
            tracks_after += 1

        print(f"\nРезультаты анализа:")
        print(f"  - Всего треков: {tracks_before}")
        print(f"  - Коротких треков: {tracks_before - tracks_after}")
        print(f"  - Треков для обработки: {tracks_after}")

    def create_videos(self):
        """Этап 3: Создание видео"""
        print("\n[Этап 3/4] Создание видео")
        print("-" * 40)

        videos_created = 0

        from tqdm import tqdm
        for track_id, meta in tqdm(self._track_metadata.items(), desc="Создание видео",
                                   unit="трек"):
            if self._create_single_video(track_id, meta):
                videos_created += 1

        print(f"\nСоздано видео: {videos_created}")

    def _create_single_video(self, track_id: int, metadata: dict) -> bool:
        """Создание одного видео для трека"""
        try:
            output_path = self._final_dir / f'person_{track_id:04d}.avi'
            max_w, max_h = metadata['max_width'], metadata['max_height']

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(str(output_path), fourcc, self.target_fps, (max_w, max_h))

            if not out.isOpened():
                return False

            frames_written = 0
            for frame_data in metadata['frames']:
                crop = cv2.imread(frame_data['filename'])

                if crop is not None:
                    final_frame = self._prepare_frame(crop, max_w, max_h)
                    out.write(final_frame)
                    frames_written += 1

            out.release()

            if frames_written == 0 and output_path.exists():
                output_path.unlink()
                return False

            return True

        except Exception as e:
            print(f"Ошибка при создании видео для трека {track_id}: {e}")
            return False

    def _prepare_frame(self, crop: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Подготовка кадра к записи"""
        h, w = crop.shape[:2]

        if w < target_w or h < target_h:
            # Добавляем чёрные границы
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - w) // 2
            y_offset = (target_h - h) // 2
            canvas[y_offset:y_offset + h, x_offset:x_offset + w] = crop
            return canvas
        else:
            # Ресайзим
            return cv2.resize(crop, (target_w, target_h))

    def cleanup(self):
        """Этап 4: Очистка временных файлов"""
        print("\n[Этап 4/4] Очистка")
        print("-" * 40)

        if not self._raw_dir.exists():
            print("Временные файлы отсутствуют")
            return

        temp_files = list(self._raw_dir.glob('*.jpg'))

        if not temp_files:
            print("Временные файлы отсутствуют")
            self._raw_dir.rmdir()
            return

        from tqdm import tqdm
        for f in tqdm(temp_files, desc="Удаление временных файлов", unit="файл"):
            try:
                f.unlink()
            except OSError:
                pass

        try:
            self._raw_dir.rmdir()
            print(f"Удалено {len(temp_files)} временных файлов")
        except OSError:
            print(f"Не удалось удалить директорию {self._raw_dir}")

    def __del__(self):
        """Деструктор для освобождения ресурсов"""
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()