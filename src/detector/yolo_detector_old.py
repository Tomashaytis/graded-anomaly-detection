import os
import cv2
import glob
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict

from src.tracker import BaseTracker


class YoloDetector:
    def __init__(self, model_path: str, video_path: str, output_dir: str, tracker: BaseTracker,
                 min_track_length: int = 10, padding: int = 30, target_fps: float = 20.0):
        # Параметры
        self._model_path = model_path
        self._video_path = video_path
        self._output_dir = output_dir
        self._min_track_length = min_track_length
        self._padding = padding
        self._target_fps = target_fps

        # Загрузка модели
        self._model = YOLO(model_path)
        self._cap = cv2.VideoCapture(video_path)

        # Получаем информацию о видео
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

        # Создаём директории
        self._raw_dir = os.path.join(output_dir, 'temp_raw_frames')
        self._final_dir = os.path.join(output_dir, 'final_tracks')
        os.makedirs(self._raw_dir, exist_ok=True)
        os.makedirs(self._final_dir, exist_ok=True)

        # Трекер (теперь используем универсальный интерфейс)
        self._tracker = tracker

        # Структуры для хранения данных
        self._all_tracks = defaultdict(list)
        self._track_metadata = {}
        self._frame_count = 0

        print(f"Инициализация завершена. Всего кадров в видео: {self._total_frames}")
        print(f"Минимальная длина трека: {min_track_length} кадров")
        print(f"Padding: {padding} пикселей")
        print(f"Трекер: {tracker.__class__.__name__}")

    def process_video(self):
        """Основной метод, объединяющий все этапы обработки"""
        print("\n=== НАЧАЛО ОБРАБОТКИ ВИДЕО ===\n")

        # Этап 1: Детекция и сбор кадров
        self.detect_frames()

        # Этап 2: Анализ треков
        self.analyze_tracks()

        # Этап 3: Создание видео
        self.create_videos()

        # Этап 4: Очистка
        self.cleanup()

        print("\n=== ОБРАБОТКА ЗАВЕРШЕНА ===")
        print(f"Результаты сохранены в: {self._final_dir}")

    def detect_frames(self):
        """Этап 1: Детекция объектов и сохранение кадров"""
        print("\n[Этап 1/4] Детекция объектов и сбор кадров")
        print("-" * 50)

        # Сбрасываем счётчик кадров
        self._frame_count = 0
        self._all_tracks.clear()

        # Сбрасываем трекер
        self._tracker.reset()

        # Открываем видео заново на всякий случай
        self._cap = cv2.VideoCapture(self._video_path)

        with tqdm(total=self._total_frames, desc="Обработка кадров", unit="кадр",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    break

                self._frame_count += 1

                # Детекция объектов через YOLO
                results = self._model(frame, classes=[0], verbose=False)[0]

                # Конвертируем детекции в универсальный формат
                detections = []
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        detections.append((x1, y1, x2, y2, conf, cls_id))

                # Обновляем трекер через универсальный интерфейс
                if detections:
                    tracks = self._tracker.update(detections, frame)

                    # Сохраняем кадры для каждого трека
                    for track in tracks:
                        track_id, x1, y1, x2, y2, conf, cls_id = track
                        self._save_detection_frame(frame, track_id, (x1, y1, x2, y2))

                pbar.update(1)

                # Периодическое обновление информации
                if self._frame_count % 100 == 0:
                    pbar.set_postfix({
                        'треков': len(self._all_tracks),
                        'активных': sum(1 for t in self._all_tracks.values() if t)
                    })

        self._cap.release()

        print(f"\nСбор завершён. Всего кадров: {self._frame_count}")
        print(f"Обнаружено треков: {len(self._all_tracks)}")

    def _save_detection_frame(self, frame: np.ndarray, track_id: int, bbox: tuple):
        """Вспомогательный метод для сохранения кадра обнаружения"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # Добавляем паддинг
        x1_pad = max(0, x1 - self._padding)
        y1_pad = max(0, y1 - self._padding)
        x2_pad = min(w, x2 + self._padding)
        y2_pad = min(h, y2 + self._padding)

        # Вырезаем кадр
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        # Сохраняем только достаточно большие кадры
        if crop.shape[0] > 50 and crop.shape[1] > 50:
            frame_filename = os.path.join(
                self._raw_dir,
                f'track_{track_id}_frame_{self._frame_count:04d}.jpg'
            )
            cv2.imwrite(frame_filename, crop)

            # Сохраняем метаданные
            self._all_tracks[track_id].append({
                'frame_num': self._frame_count,
                'filename': frame_filename,
                'coords': (x1_pad, y1_pad, x2_pad, y2_pad),
                'size': (crop.shape[1], crop.shape[0])  # (width, height)
            })

    def analyze_tracks(self):
        """Этап 2: Анализ собранных треков"""
        print("\n[Этап 2/4] Анализ треков")
        print("-" * 50)

        self._track_metadata.clear()
        tracks_before = len(self._all_tracks)
        tracks_after = 0

        for track_id, frames in tqdm(self._all_tracks.items(), desc="Анализ треков", unit="трек",
                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):

            # Пропускаем слишком короткие треки
            if len(frames) < self._min_track_length:
                continue

            # Находим максимальные размеры для этого трека
            widths = [f['size'][0] for f in frames]
            heights = [f['size'][1] for f in frames]

            max_width = max(widths)
            max_height = max(heights)

            # Делаем размеры чётными для кодеков
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

        print(f"\nАнализ завершён:")
        print(f"  - Всего треков: {tracks_before}")
        print(f"  - Треков короче {self._min_track_length} кадров: {tracks_before - tracks_after}")
        print(f"  - Треков для обработки: {tracks_after}")

    def create_videos(self):
        """Этап 3: Создание финальных видео"""
        print("\n[Этап 3/4] Создание видео")
        print("-" * 50)

        videos_created = 0
        videos_failed = 0

        for track_id, meta in tqdm(self._track_metadata.items(), desc="Создание видео", unit="трек",
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):

            success = self._create_single_video(track_id, meta)

            if success:
                videos_created += 1
            else:
                videos_failed += 1

        print(f"\nВидео создано: {videos_created}")
        if videos_failed > 0:
            print(f"Не удалось создать: {videos_failed}")

    def _create_single_video(self, track_id: int, metadata: dict) -> bool:
        """Вспомогательный метод для создания одного видео"""
        try:
            output_path = f'{self._final_dir}/person_{track_id:04d}.avi'
            max_w, max_h = metadata['max_width'], metadata['max_height']

            # Создаём VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, self._target_fps, (max_w, max_h))

            if not out.isOpened():
                return False

            frames_written = 0

            # Обрабатываем каждый кадр трека
            for frame_data in metadata['frames']:
                # Загружаем сохранённый кадр
                crop = cv2.imread(frame_data['filename'])

                if crop is not None:
                    final_frame = self._prepare_frame(crop, max_w, max_h)
                    out.write(final_frame)
                    frames_written += 1

            out.release()

            # Проверяем, что видео создано успешно
            if frames_written > 0:
                return True
            else:
                # Удаляем пустое видео
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False

        except Exception as e:
            print(f"Ошибка при создании видео для трека {track_id}: {e}")
            return False

    def _prepare_frame(self, crop: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Подготовка кадра к записи (центрирование или ресайз)"""
        h, w = crop.shape[:2]

        # Если кадр меньше целевого размера - добавляем чёрные границы
        if w < target_w or h < target_h:
            # Создаём чёрный холст
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Центрируем оригинальный кадр
            x_offset = (target_w - w) // 2
            y_offset = (target_h - h) // 2

            canvas[y_offset:y_offset + h, x_offset:x_offset + w] = crop
            return canvas
        else:
            # Если кадр больше - ресайзим
            return cv2.resize(crop, (target_w, target_h))

    def cleanup(self):
        """Этап 4: Очистка временных файлов"""
        print("\n[Этап 4/4] Очистка временных файлов")
        print("-" * 50)

        if not os.path.exists(self._raw_dir):
            print("Временная директория не найдена")
            return

        # Подсчитываем файлы для удаления
        temp_files = glob.glob(f'{self._raw_dir}/*.jpg')
        file_count = len(temp_files)

        if file_count == 0:
            print("Временные файлы отсутствуют")
            os.rmdir(self._raw_dir)
            return

        # Удаляем файлы
        for f in tqdm(temp_files, desc="Удаление временных файлов", unit="файл"):
            try:
                os.remove(f)
            except OSError:
                pass

        # Удаляем директорию
        try:
            os.rmdir(self._raw_dir)
            print(f"Удалено {file_count} временных файлов")
        except OSError:
            print(f"Не удалось удалить директорию {self._raw_dir}")
