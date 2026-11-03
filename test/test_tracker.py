from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector


VIDEO_NAME = '001'
MODEL_PATH = "../models/detector/yolo/yolov8n.pt"
TEST_VIDEO = f"../data/test_traj/video/{VIDEO_NAME}.mp4"
OUTPUT_VIDEO = f"../data/test_traj/video_traj/{VIDEO_NAME}.mp4"
OUTPUT_TRAJECTORIES = f"../data/test_traj/traj/{VIDEO_NAME}.json"


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
    detector.save_tracks(OUTPUT_TRAJECTORIES, tracks)