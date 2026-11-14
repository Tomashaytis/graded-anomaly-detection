import os
import json
import yaml
import torch
import argparse
import warnings
import rootutils

from pathlib import Path

from src.models import create_model
from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector, GroupDetector
from src.cropping import GroupCropper
from src.video import FrameSourceFactory, StreamFrameSource

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
    batch_size = model_config['batch_size']

    if n_crops > batch_size:
        warnings.warn(
            f"n_crops ({n_crops}) > batch_size ({batch_size}). "
            f"Will keep only the first {batch_size} crops per window.",
            category=UserWarning,
            stacklevel=2,
        )

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

    with open(model_config['id2label_path'], 'r', encoding='utf-8') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    model = create_model(
        model_name=model_config['name'],
        num_classes=len(id2label),
        dropout=model_config.get('dropout', 0.0),
        weights_path=None,
        freeze_backbone=False,
    )

    checkpoint_path = Path(model_config['checkpoint_path'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Set paths.checkpoint in config to an existing .ckpt file."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {
        k.removeprefix('_model.'): v
        for k, v in state_dict.items()
        if k.startswith('_model.')
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical = [k for k in (missing + unexpected) if 'head1' in k or 'head2' in k]
    if critical:
        raise RuntimeError(f"Head mismatch in checkpoint: {critical}")

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    level_weights = {int(k): v['weight'] for k, v in danger_levels.items()}
    level_names = {int(k): v['name'] for k, v in danger_levels.items()}
    action_to_level = {
        a['name']: a['danger_level']
        for cat in action_categories.values()
        for a in cat
    }

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

        items = list(tensors.items())[:batch_size]

        if items:
            gids = [gid for gid, _ in items]
            batch = torch.stack([t for _, t in items], dim=0).to(device)

            with torch.no_grad():
                logits1, logits2 = model(batch)
                level_probs = torch.sigmoid(logits1.squeeze(-1))

            for i, gid in enumerate(gids):
                level_prob = level_probs[i].item()

                is_global = preprocessing_config['global_first'] and gid == 0
                threshold = (
                    postprocessing_config['global_danger_threshold']
                    if is_global
                    else postprocessing_config['local_danger_threshold']
                )

                if level_prob < threshold:
                    print(
                        f"[window={window_start}, gid={gid}] "
                        f"normal (level_prob={level_prob:.3f}, thr={threshold})"
                    )
                    continue

                pred_level = min(
                    level_weights,
                    key=lambda lvl: abs(level_weights[lvl] - level_prob),
                )
                pred_level_name = level_names[pred_level]

                action_probs = torch.softmax(logits2[i], dim=-1)
                action_id = int(action_probs.argmax().item())
                action_prob = float(action_probs[action_id].item())
                action_name = id2label[action_id]

                action_danger_level = action_to_level.get(action_name)

                effective_level = max(pred_level, action_danger_level or 0)

                if effective_level <= 0:
                    is_anomaly = False
                elif effective_level == 1:
                    is_anomaly = contextual_settings.get(action_name, False)
                else:
                    is_anomaly = True

                print(
                    f"[window={window_start}, gid={gid}] "
                    f"level_prob={level_prob:.3f} (thr={threshold}) → "
                    f"pred_level={pred_level} ({pred_level_name}), "
                    f"action={action_name} ({action_prob:.3f}), "
                    f"action_level={action_danger_level}, "
                    f"effective_level={effective_level} ({level_names[effective_level]}), "
                    f"anomaly={is_anomaly}"
                )

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