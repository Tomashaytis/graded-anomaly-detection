from .base_tracker import BaseTracker
from .byte_track_adapter import ByteTrackAdapter
from .bot_sort_adapter import BotSORTAdapter
from .deep_sort_adapter import DeepSORTAdapter


class TrackerFactory:
    """Фабрика для создания трекеров"""

    @staticmethod
    def create_tracker(tracker_type: str, **kwargs) -> BaseTracker:
        """
        Создаёт трекер указанного типа

        Args:
            tracker_type: 'bytetrack', 'botsort', 'deepsort'
            **kwargs: параметры для конкретного трекера

        Returns:
            экземпляр трекера
        """
        trackers = {
            'bytetrack': ByteTrackAdapter,
            'botsort': BotSORTAdapter,
            'deepsort': DeepSORTAdapter,
        }

        if tracker_type not in trackers:
            raise ValueError(f"Неизвестный тип трекера: {tracker_type}. Доступны: {list(trackers.keys())}")

        return trackers[tracker_type](**kwargs)