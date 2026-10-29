from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector


MODEL_PATH = "../models/detector/yolo/yolov8n.pt"
TEST_VIDEO = "../data/test.mp4"
OUTPUT_VIDEO = "../data/trajectories.mp4"
OUTPUT_TRAJECTORIES = "../data/trajectories.json"


if __name__ == '__main__':
    tracker = TrackerFactory.create_tracker('bytetrack')

    detector = YoloTrailDetector(
        model_path=MODEL_PATH,
        tracker=tracker,
        classes=(0,),
        min_track_length=10,
    )

    tracks = detector.process_video(TEST_VIDEO, step=1)
    detector.render_video(TEST_VIDEO, OUTPUT_VIDEO, tracks)