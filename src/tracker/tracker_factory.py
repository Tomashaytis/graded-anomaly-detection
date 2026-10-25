from .base_tracker import BaseTracker
from .byte_track_adapter import ByteTrackAdapter
from .bot_sort_adapter import BotSORTAdapter
from .deep_sort_adapter import DeepSORTAdapter


class TrackerFactory:
    """Factory for creating tracker instances."""

    @staticmethod
    def create_tracker(tracker_type: str, **kwargs) -> BaseTracker:
        """
        Create a tracker of the given type.

        :param tracker_type: tracker name — 'bytetrack', 'botsort' or 'deepsort'
        :param kwargs: parameters forwarded to the tracker constructor
        :return: tracker instance
        :raises ValueError: if the tracker type is unknown
        """
        trackers = {
            'bytetrack': ByteTrackAdapter,
            'botsort': BotSORTAdapter,
            'deepsort': DeepSORTAdapter,
        }

        if tracker_type not in trackers:
            available = ', '.join(trackers.keys())
            raise ValueError(f"Unknown tracker type: {tracker_type}. "
                             f"Available: {available}")

        return trackers[tracker_type](**kwargs)