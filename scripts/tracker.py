import torch

from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector

if __name__ == '__main__':

    # Вариант 1: ByteTrack
    tracker = TrackerFactory.create_tracker('bytetrack')

    # Вариант 2: BotSORT
    # tracker = TrackerFactory.create_tracker(
    #     'botsort',
    #     model_weights='../models/tracker/reid/hacnn_market_xent.pt',
    #     fp16=torch.cuda.is_available(),
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    #     with_reid=True,
    #     track_buffer=15,
    #     frame_rate=30,
    #     det_thresh=0.3,
    #     track_high_thresh=0.5,
    #     track_low_thresh=0.1,
    #     new_track_thresh=0.6,
    #     match_thresh=0.8
    # )

    detector = YoloTrailDetector(
        model_path='../models/detector/yolo/yolov8n.pt',
        tracker=tracker,
        classes=(0,),
        trail_length=80,
        save_json=True,
    )
    detector.process_video(
        video_path='../data/test.mp4',
        output_path='../data/trajectories.mp4'
    )