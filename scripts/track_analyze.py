import os

from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector, SpectralGroupDetector


MODEL_PATH = "../models/detector/yolo/yolov8m.pt"
INPUT_VIDEO_DIR = "../data/test_traj/video"
OUTPUT_VIDEO_DIR = "../data/test_traj/video_traj"
OUTPUT_TRAJECTORIES_DIR = "../data/test_traj/traj"
OUTPUT_GROUPS_FILE = "../data/test_traj/groups.txt"


if __name__ == '__main__':
    tracker = TrackerFactory.create_tracker('bytetrack')

    trail_detector = YoloTrailDetector(
        model_path=MODEL_PATH,
        tracker=tracker,
        classes=(0,),
        min_track_length=10,
    )
    group_detector = SpectralGroupDetector()

    video_names = os.listdir(INPUT_VIDEO_DIR)
    groups_lines = []

    for video_name in video_names:
        input_video_path = os.path.join(INPUT_VIDEO_DIR, video_name)
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, video_name)
        name, _ = os.path.splitext(video_name)
        output_json_path = os.path.join(OUTPUT_TRAJECTORIES_DIR, name + ".json")

        tracks = trail_detector.process_video(input_video_path, step=1)
        trail_detector.render_video(input_video_path, output_video_path, tracks)
        trail_detector.save_tracks(output_json_path, tracks)

        groups = group_detector.process_tracks(output_json_path)
        groups_sorted = [sorted(members) for _, members in sorted(groups.items())]
        groups_str = ", ".join(f"[{', '.join(map(str, g))}]" for g in groups_sorted)
        groups_lines.append(f"{video_name}: {groups_str}")

    with open(OUTPUT_GROUPS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(groups_lines) + "\n")