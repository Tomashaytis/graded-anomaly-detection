from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector, GroupDetector
from src.utils import GroupCropper, visualize_video

MODEL_PATH = "../models/detector/yolo/yolov8m.pt"
VIDEO_PATH = "../data/test_traj/video/018.mp4"
GROUP_METHOD = 'hdbscan'
N_FRAMES = 32
SIZE = 224
N_TOP = 2
N_RANDOM = 1
GLOBAL_FIRST = True


def main():
    tracker = TrackerFactory.create_tracker('bytetrack')

    detector = YoloTrailDetector(
        model_path=MODEL_PATH,
        tracker=tracker,
        classes=(0,),
        min_track_length=10,
    )

    tracks = detector.process_video(VIDEO_PATH, step=1)

    group_detector = GroupDetector(method=GROUP_METHOD)
    cropper = GroupCropper(

        group_detector=group_detector,
        n_frames=N_FRAMES,
        size=SIZE,
        n_top=N_TOP,
        n_random=N_RANDOM,
        global_first=GLOBAL_FIRST,
    )

    tensors = cropper.get_group_tensors(
        video_path=VIDEO_PATH,
        tracks=tracks
    )

    for gid, t in tensors.items():
        visualize_video(t)


if __name__ == '__main__':
    main()