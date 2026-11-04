from src.detector import SpectralGroupDetector
from src.utils import GroupCropper, visualize_video


def test():
    group_detector = SpectralGroupDetector()
    cropper = GroupCropper(
        group_detector=group_detector,
        n_frames=32,
        size=224,
        n_top=2,
        n_random=1,
    )

    tensors = cropper.get_group_tensors(
        video_path="../data/test_traj/video/017.mp4",
        tracks_path="../data/test_traj/traj/017.json",
    )

    for gid, t in tensors.items():
        visualize_video(t)


if __name__ == '__main__':
    test()