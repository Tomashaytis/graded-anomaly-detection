from src.detector import GroupDetector
from src.cropping import GroupCropper
from src.utils import visualize_video


def test():
    group_detector = GroupDetector(method="spectral")
    cropper = GroupCropper(
        group_detector=group_detector,
        n_frames=32,
        size=224,
        n_top=2,
        n_random=1,
        global_first=True,
    )

    tensors = cropper.get_group_tensors(
        video_path="../data/test_traj/video/017.mp4",
        tracks_path="../data/test_traj/traj/017.json",
    )

    for gid, t in tensors.items():
        visualize_video(t)


if __name__ == '__main__':
    test()