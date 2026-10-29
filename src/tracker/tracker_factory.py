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

        :param tracker_type: Tracker name — 'bytetrack', 'botsort' or 'deepsort'.
        :param kwargs: Parameters forwarded to the tracker constructor.
        :return: Tracker instance.
        :raises ValueError: If the tracker type is unknown.
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