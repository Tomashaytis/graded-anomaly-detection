import os
import json
import yaml
import queue
import torch
import argparse
import warnings
import rootutils
import threading
import traceback
import subprocess

from pathlib import Path
from datetime import datetime
from typing import Optional, Iterable

from src.models import create_model
from src.tracker import TrackerFactory
from src.detector import YoloTrailDetector, GroupDetector
from src.cropping import GroupCropper
from src.video import FrameSourceFactory, StreamFrameSource

root = rootutils.setup_root(__file__, indicator='.git', pythonpath=True, cwd=True)

DEFAULT_CONFIG_PATH = os.path.join('configs', 'inference.yaml')
VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

RESULTS_FILE = 'results.json'
PARTIAL_FILE = 'results.partial.jsonl'
ERRORS_FILE = 'errors.jsonl'


def _git_sha() -> Optional[str]:
    """Return GIT head SHA."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _now_iso() -> str:
    """Return current time."""
    return datetime.now().isoformat(timespec='seconds')


def _atomic_write_json(path: Path, obj: object, indent: int = 2):
    """
    Dump object to JSON.

    :param path: Path to JSON file.
    :param obj: Object to dump.
    :param indent: Indent for JSON file.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def _iter_videos(dataset_root: str) -> Iterable[Path]:
    """
    Return iterator for dataset videos.

    :param dataset_root: Path to dataset root.
    :return: Iterator.
    """
    root_path = Path(dataset_root)
    for p in sorted(root_path.rglob('*')):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


class ModelBundle:
    """
    Model bundle.
    """

    def __init__(self, model, device, id2label, model_config):
        """
        Initialize model bundle.

        :param model: Model.
        :param device: Device.
        :param id2label: Mapping class_id -> label.
        :param model_config: Model config.
        """
        self.model = model
        self.device = device
        self.id2label = id2label
        self.model_config = model_config


def build_model(config: dict) -> ModelBundle:
    """
    Build model for action recognition.

    :param config: Inference config.
    :return: Model bundle.
    """
    model_preset = config['model']['preset']
    model_config = config['model_presets'][model_preset]

    with open(config['model']['id2label_path'], 'r', encoding='utf-8') as f:
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
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

    return ModelBundle(model=model, device=device, id2label=id2label, model_config=model_config)


def run_inference_on_video(config: dict, video_path: str, bundle: ModelBundle) -> dict:
    """
    Run inference for a single video.

    :param config: Inference config.
    :param video_path: Video path.
    :param bundle: Model bundle.
    :return: Inference results.
    """
    preprocessing_config = config['preprocessing']
    postprocessing_config = config['postprocessing']
    danger_levels = config['danger_levels']
    action_categories = config['action_categories']
    contextual_settings = config['context']

    model_config = bundle.model_config
    model = bundle.model
    device = bundle.device
    id2label = bundle.id2label

    n_crops = (preprocessing_config['n_top']
               + preprocessing_config['n_random']
               + int(preprocessing_config['global_first']))
    batch_size = model_config['batch_size']
    if n_crops > batch_size:
        warnings.warn(
            f"n_crops ({n_crops}) > batch_size ({batch_size}).",
            category=UserWarning,
        )

    level_weights = {int(k): v['weight'] for k, v in danger_levels.items()}
    level_names = {int(k): v['name'] for k, v in danger_levels.items()}
    action_to_level = {
        a['name']: a['danger_level']
        for cat in action_categories.values()
        for a in cat
    }

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
            model_path=preprocessing_config['reid_path'],
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

    windows_out: list[dict] = []
    batch_queue: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    def safe_put(item):
        """Save put in queue."""
        while not stop_event.is_set():
            try:
                batch_queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        """Data producer."""
        window_start = 0
        try:
            while not stop_event.is_set():
                tracks = detector.process_video(
                    frame_source=source,
                    step=1,
                    frame_offset=window_start,
                    max_frames=window_size,
                )
                tensors = group_cropper.get_group_tensors(
                    frame_source=source,
                    tracks=tracks,
                    frame_offset=window_start,
                )
                items = list(tensors.items())[:batch_size]
                if items:
                    gids = [gid for gid, _ in items]
                    batch = torch.stack([t for _, t in items], dim=0)
                    batch = batch.permute(0, 2, 1, 3, 4)
                    safe_put((window_start, gids, batch))

                if isinstance(source, StreamFrameSource):
                    source.clean_buffer()

                window_start += window_size
                if source.frame_count() <= window_start:
                    break
        finally:
            safe_put(None)
            stop_event.set()

    def consumer():
        """Data consumer."""
        try:
            while True:
                item = batch_queue.get()
                if item is None:
                    break

                window_start, gids, batch = item
                batch = batch.to(device, non_blocking=True)

                with torch.no_grad():
                    logits1, logits2 = model(batch)
                    level_probs = torch.sigmoid(logits1.squeeze(-1))

                groups_out = []
                for i, gid in enumerate(gids):
                    level_prob = float(level_probs[i].item())
                    is_global = preprocessing_config['global_first'] and gid == 0
                    threshold = float(
                        postprocessing_config['global_danger_threshold']
                        if is_global
                        else postprocessing_config['local_danger_threshold']
                    )

                    # Normal-группа: компактная запись
                    if level_prob < threshold:
                        groups_out.append({
                            'gid': int(gid),
                            'level_prob': round(level_prob, 6),
                            'threshold': threshold,
                            'is_anomaly': False,
                        })
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

                    groups_out.append({
                        'gid': int(gid),
                        'level_prob': round(level_prob, 6),
                        'threshold': threshold,
                        'pred_level': int(pred_level),
                        'pred_level_name': pred_level_name,
                        'action': action_name,
                        'action_prob': round(action_prob, 6),
                        'action_danger_level': int(action_danger_level) if action_danger_level is not None else None,
                        'effective_level': int(effective_level),
                        'effective_level_name': level_names[effective_level],
                        'is_anomaly': bool(is_anomaly),
                    })

                windows_out.append({'window_start': int(window_start), 'groups': groups_out})
                batch_queue.task_done()
        except Exception as e:
            print(f"[consumer] crashed on {video_path}: {e}")
            traceback.print_exc()
            stop_event.set()
            try:
                batch_queue.put_nowait(None)
            except queue.Full:
                pass

    producer_thread = threading.Thread(target=producer, daemon=True)
    consumer_thread = threading.Thread(target=consumer, daemon=True)

    try:
        producer_thread.start()
        consumer_thread.start()
        producer_thread.join()
        consumer_thread.join()
    finally:
        source.release()

    all_groups = [g for w in windows_out for g in w['groups']]
    anomalies = [g for g in all_groups if g['is_anomaly']]
    action_counts: dict[str, int] = {}
    for g in all_groups:
        a = g.get('action')
        if a:
            action_counts[a] = action_counts.get(a, 0) + 1
    top_action = max(action_counts, key=action_counts.get) if action_counts else None
    max_level = max((g.get('effective_level', 0) for g in all_groups), default=0)

    anomaly_windows = sorted({
        w['window_start']
        for w in windows_out
        if any(g['is_anomaly'] for g in w['groups'])
    })

    return {
        'video_path': video_path,
        'status': 'ok',
        'n_windows': len(windows_out),
        'summary': {
            'n_events': len(all_groups),
            'n_anomalies': len(anomalies),
            'max_effective_level': int(max_level),
            'top_action': top_action,
            'top_action_count': action_counts.get(top_action, 0) if top_action else 0,
            'anomaly_windows': anomaly_windows,
        },
        'windows': windows_out,
    }


def _append_partial(partial_path: Path, record: dict):
    """
    Dump record to partial JSON file.

    :param partial_path: Path to partial JSON file.
    :param record: Record to dump.
    """
    with open(partial_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _load_done_videos(partial_path: Path) -> set[str]:
    """
    Load set of processed videos.

    :param partial_path: Path to partial JSON file.
    :return: Set of processed videos.
    """
    done = set()
    if not partial_path.exists():
        return done
    with open(partial_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec['video_path'])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def merge_partial_to_results(run_dir: Path, run_meta: dict):
    """
    Merge partial data to results.

    :param run_dir: Run directory.
    :param run_meta: Run metadata.
    """
    partial_path = run_dir / PARTIAL_FILE
    results_path = run_dir / RESULTS_FILE
    tmp_path = results_path.with_suffix('.json.tmp')

    if not partial_path.exists():
        print(f"[merge] no {partial_path.name}, nothing to merge")
        return

    n_videos = 0
    n_ok = 0
    n_failed = 0
    n_events = 0
    n_anomalies = 0
    max_level = 0

    with open(tmp_path, 'w', encoding='utf-8') as out, \
         open(partial_path, 'r', encoding='utf-8') as inp:

        out.write('{\n')
        out.write('  "run_meta": ')
        json.dump(run_meta, out, ensure_ascii=False, indent=2)
        out.write(',\n')

        out.write('  "videos": [\n')
        first = True
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_videos += 1
            if rec.get('status') == 'ok':
                n_ok += 1
            else:
                n_failed += 1
            s = rec.get('summary', {})
            n_events += s.get('n_events', 0)
            n_anomalies += s.get('n_anomalies', 0)
            max_level = max(max_level, s.get('max_effective_level', 0))

            if not first:
                out.write(',\n')
            first = False
            out.write('    ')
            json.dump(rec, out, ensure_ascii=False, indent=2)
        out.write('\n  ],\n')

        out.write('  "summary": ')
        json.dump({
            'n_videos': n_videos,
            'n_videos_ok': n_ok,
            'n_videos_failed': n_failed,
            'n_events': n_events,
            'n_anomalies': n_anomalies,
            'max_effective_level': max_level,
        }, out, ensure_ascii=False, indent=2)
        out.write('\n}\n')

    os.replace(tmp_path, results_path)
    print(f"[merge] wrote {results_path} ({n_videos} videos, {n_events} events, {n_anomalies} anomalies)")


def to_parquet(run_dir: Path) -> None:
    """results.json -> events.parquet + videos.parquet."""
    try:
        import pandas as pd
    except ImportError:
        print("[parquet] pandas not installed, skipping")
        return

    results_path = run_dir / RESULTS_FILE
    if not results_path.exists():
        print(f"[parquet] {results_path} not found, run --merge first")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    events_rows = []
    videos_rows = []
    for v in data['videos']:
        s = v.get('summary', {})
        videos_rows.append({
            'video_path': v['video_path'],
            'status': v.get('status'),
            'n_windows': v.get('n_windows'),
            'n_events': s.get('n_events'),
            'n_anomalies': s.get('n_anomalies'),
            'max_effective_level': s.get('max_effective_level'),
            'top_action': s.get('top_action'),
            'top_action_count': s.get('top_action_count'),
        })
        for w in v.get('windows', []):
            for g in w.get('groups', []):
                events_rows.append({
                    'video_path': v['video_path'],
                    'window_start': w['window_start'],
                    'gid': g['gid'],
                    'level_prob': g.get('level_prob'),
                    'threshold': g.get('threshold'),
                    'pred_level': g.get('pred_level'),
                    'pred_level_name': g.get('pred_level_name'),
                    'action': g.get('action'),
                    'action_prob': g.get('action_prob'),
                    'action_danger_level': g.get('action_danger_level'),
                    'effective_level': g.get('effective_level'),
                    'effective_level_name': g.get('effective_level_name'),
                    'is_anomaly': g.get('is_anomaly'),
                })

    pd.DataFrame(events_rows).to_parquet(run_dir / 'events.parquet', index=False)
    pd.DataFrame(videos_rows).to_parquet(run_dir / 'videos.parquet', index=False)
    print(f"[parquet] wrote events.parquet ({len(events_rows)}) and videos.parquet ({len(videos_rows)})")


def make_run_id(config: dict) -> str:
    """
    Generate run ID.

    :param config: Inference config.
    :return: Run ID.
    """
    preset = config['model']['preset']
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{preset}"


def build_run_meta(run_id: str, config: dict) -> dict:
    """
    Collect run metadata.

    :param run_id: Run ID.
    :param config: Inference config.
    :return: Run metadata.
    """
    preset = config['model']['preset']
    model_config = config['model_presets'][preset]
    return {
        'run_id': run_id,
        'started_at': _now_iso(),
        'config': config,
        'checkpoint': model_config['checkpoint_path'],
        'preset': preset,
        'git_sha': _git_sha(),
        'torch': torch.__version__,
    }


def batch_inference(
    config: dict,
    dataset_root: str,
    output_dir: str,
    resume: bool = True,
    stop_on_error: bool = False,
):
    """
    Run batch inference mode.

    :param config: Inference config.
    :param dataset_root: Path to dataset root.
    :param output_dir: Path to output directory.
    :param resume: If True, resume batch inference mode.
    :param stop_on_error: If True, stop on error.
    """
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_dir.name
    run_meta = build_run_meta(run_id, config)

    partial_path = run_dir / PARTIAL_FILE
    errors_path = run_dir / ERRORS_FILE

    done = _load_done_videos(partial_path) if resume else set()
    videos = list(_iter_videos(dataset_root))
    print(f"[run] {run_id}: {len(videos)} videos, {len(done)} already done")

    bundle = build_model(config)

    with open(errors_path, 'a', encoding='utf-8') as err_f:
        for idx, video in enumerate(videos, 1):
            vpath = str(video)
            if vpath in done:
                print(f"[{idx}/{len(videos)}] skip {vpath}")
                continue

            print(f"[{idx}/{len(videos)}] {vpath}")
            try:
                record = run_inference_on_video(config, vpath, bundle)
            except KeyboardInterrupt:
                print("[run] interrupted")
                raise
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  FAILED: {e}")
                err_f.write(json.dumps({
                    'video_path': vpath,
                    'error': str(e),
                    'traceback': tb,
                    'at': _now_iso(),
                }, ensure_ascii=False) + '\n')
                err_f.flush()
                if stop_on_error:
                    raise
                continue

            _append_partial(partial_path, record)
            s = record['summary']
            print(f"  windows={record['n_windows']}, "
                  f"events={s['n_events']}, anomalies={s['n_anomalies']}")

    merge_partial_to_results(run_dir, run_meta)

    if partial_path.exists():
        partial_path.unlink()
        print(f"[run] removed {partial_path.name}")


def retry_errors(config: dict, output_dir: str):
    """
    Retry errors.

    :param config: Inference config.
    :param output_dir: Output directory.
    """
    run_dir = Path(output_dir)
    errors_path = run_dir / ERRORS_FILE
    if not errors_path.exists():
        print("[retry] no errors file")
        return

    failed: list[str] = []
    with open(errors_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                failed.append(json.loads(line)['video_path'])
            except (json.JSONDecodeError, KeyError):
                continue

    if not failed:
        print("[retry] no failed videos")
        return

    print(f"[retry] {len(failed)} failed videos")
    partial_path = run_dir / PARTIAL_FILE
    done = _load_done_videos(partial_path)
    bundle = build_model(config)

    still_failed = []
    with open(errors_path, 'w', encoding='utf-8') as err_f:
        for idx, vpath in enumerate(failed, 1):
            if vpath in done:
                continue
            print(f"[{idx}/{len(failed)}] retry {vpath}")
            try:
                record = run_inference_on_video(config, vpath, bundle)
                _append_partial(partial_path, record)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  STILL FAILED: {e}")
                still_failed.append(vpath)
                err_f.write(json.dumps({
                    'video_path': vpath,
                    'error': str(e),
                    'traceback': tb,
                    'at': _now_iso(),
                }, ensure_ascii=False) + '\n')
                err_f.flush()

    run_meta = build_run_meta(run_dir.name, config)
    merge_partial_to_results(run_dir, run_meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', nargs='?', type=str, help='dataset folder')
    parser.add_argument('-o', '--output', type=str, help='run output dir (runs/<run_id>)')
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--stop-on-error', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='retry only failed videos')
    parser.add_argument('--merge', action='store_true', help='merge partial -> results.json only')
    parser.add_argument('--to-parquet', action='store_true', help='export results.json to parquet')

    args = parser.parse_args()

    if args.merge or args.to_parquet:
        if not args.output:
            parser.error('--output is required for --merge/--to-parquet')
        run_dir = Path(args.output)
        if args.to_parquet:
            to_parquet(run_dir)
        if args.merge:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            run_meta = build_run_meta(run_dir.name, config)
            merge_partial_to_results(run_dir, run_meta)
            partial = run_dir / PARTIAL_FILE
            if partial.exists():
                partial.unlink()
        return

    if not args.dataset_root:
        parser.error('dataset_root is required')
    if not args.output:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        args.output = os.path.join('runs', make_run_id(config))

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.retry_errors:
        retry_errors(config, args.output)
    else:
        batch_inference(
            config=config,
            dataset_root=args.dataset_root,
            output_dir=args.output,
            resume=not args.no_resume,
            stop_on_error=args.stop_on_error,
        )


if __name__ == '__main__':
    main()