import os
import yaml
import argparse
import warnings
import rootutils

from pathlib import Path

from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector, GroupDetector
from src.cropping import GroupCropper
from src.video import FrameSourceFactory, StreamFrameSource
from src.utils import visualize_video

root = rootutils.setup_root(__file__, indicator='.git', pythonpath=True, cwd=True)

DEFAULT_CONFIG_PATH = os.path.join('configs', 'inference.yaml')


def inference(config: dict, video_path: str):
    """
    Inference stage for action recognition system.

    :param config: Inference config.
    :param video_path: Video path for inference.
    """
    preprocessing_config = config['preprocessing']
    postprocessing_config = config['postprocessing']
    model_preset = config['model']['preset']
    model_config = config['model_presets'][model_preset]
    danger_levels = config['danger_levels']
    action_categories = config['action_categories']
    contextual_settings = config['context']

    n_crops = (preprocessing_config['n_top']
               + preprocessing_config['n_random']
               + int(preprocessing_config['global_first']))

    if n_crops > model_config['batch_size']:
        warnings.warn(
            f"n_crops ({n_crops}) is larger than batch_size ({model_config['batch_size']}). "
            f"Skipping inference.",
            category=UserWarning,
            stacklevel=2,
        )
        return

    source, window_size = FrameSourceFactory.create_frame_source(
        video_source=video_path,
        window_size_sec=preprocessing_config['window_size_sec'],
        temp_dir="/dev/shm" if Path("/dev/shm").exists() else None,
        fps_fallback=30.0,
        codec="mp4v",
        ext=".mp4",
    )

    if preprocessing_config['tracker'] == 'bytetrack':
        tracker = TrackerFactory.create_tracker(tracker_type=preprocessing_config['tracker'])
    else:
        tracker = TrackerFactory.create_tracker(
            tracker_type=preprocessing_config['tracker'],
            model_path=preprocessing_config['reid_path']
        )

    detector = YoloTrailDetector(
        model_path=preprocessing_config['detector_path'],
        tracker=tracker,
        classes=(0,),
        min_track_length=preprocessing_config['min_track_length'],
    )

    group_detector = GroupDetector(method=preprocessing_config['group_method'])
    group_cropper = GroupCropper(
        group_detector=group_detector,
        n_frames=model_config['video_n_frames'],
        size=model_config['video_frame_size'],
        n_top=preprocessing_config['n_top'],
        n_random=preprocessing_config['n_random'],
        global_first=preprocessing_config['global_first'],
    )

    checkpoint_path = Path(model_config['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Set paths.checkpoint in config to an existing .ckpt file."
        )

    window_start = 0
    while True:
        tracks = detector.process_video(
            frame_source=source,
            step=1,
            frame_offset=window_start,
            max_frames=window_size
        )

        tensors = group_cropper.get_group_tensors(
            frame_source=source,
            tracks=tracks,
            frame_offset=window_start
        )

        for gid, t in tensors.items():
            visualize_video(t)

        if isinstance(source, StreamFrameSource):
            source.clean_buffer()

        window_start += window_size

        if source.frame_count() <= window_start:
            break

    source.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'video_path',
        type=str,
        help='path to video for inference'
    )
    parser.add_argument(
        '-c', '--config',
        required=False,
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help='path to inference configuration'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    inference(config, args.video_path)


if __name__ == '__main__':
    main()