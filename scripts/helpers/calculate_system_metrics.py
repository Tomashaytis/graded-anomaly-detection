import os
import csv
import json
import yaml
import argparse
import numpy as np

from pathlib import Path

DEFAULT_CONFIG_PATH = os.path.join('configs', 'inference.yaml')


def _rule_for(level: int, is_global: bool, rules: list[dict]) -> tuple[int, int, int]:
    if level <= 0:
        return (0, 1, 1)

    target_mode = "global" if is_global else "local"

    for rule in rules:
        if level in rule.get("level", ()) and rule.get("mode") in ("all", target_mode):
            min_lvl = min(rule["level"])
            return (min_lvl, int(rule["k"]), int(rule["n"]))

    return (0, 1, 1)


def apply_persistence_filter(
    video: dict,
    rules: list[dict],
    min_action_prob: float = 0,
    require_same_action: bool = True,
    report: bool = False,
) -> dict:
    windows = video.get('windows', [])
    if not windows:
        return video

    cleaned_windows = []
    for w in windows:
        c_groups = []
        for g in w.get('groups', []):
            g2 = dict(g)
            lvl = int(g2.get('effective_level', 0) or 0)
            pred_lvl = int(g2.get('pred_level', 0) or 0)
            act_prob = float(g2.get('action_prob', 0.0) or 0.0)

            if lvl > pred_lvl and act_prob < min_action_prob:
                g2['effective_level'] = pred_lvl
                if pred_lvl == 0 or g2.get('level_prob', 0.0) < g2.get('threshold', 0.5):
                    g2['is_anomaly'] = False
                    g2['suppressed_by'] = f'low_action_prob({act_prob:.2f}<{min_action_prob})'
            c_groups.append(g2)
        cleaned_windows.append({**w, 'groups': c_groups})

    suppressed_total = 0
    new_windows = []
    for i, w in enumerate(cleaned_windows):
        new_groups = []
        for g in w.get('groups', []):
            g2 = dict(g)
            is_global = (g2.get('gid') == 0)
            lvl = int(g2.get('effective_level', 0) or 0)
            act = g2.get('action')
            min_lvl_for_vote, k, n = _rule_for(lvl, is_global, rules)

            if k > 1 and n > 1 and g2.get('is_anomaly'):
                start = max(0, i - n + 1)
                votes = 0
                for j in range(start, i + 1):
                    matched = any(
                        (cand.get('gid') == 0) == is_global
                        and int(cand.get('effective_level', 0) or 0) >= min_lvl_for_vote
                        and (not require_same_action or cand.get('action') == act)
                        for cand in cleaned_windows[j].get('groups', [])
                    )
                    if matched:
                        votes += 1

                if votes < k:
                    g2['is_anomaly'] = False
                    g2['effective_level'] = 0
                    g2['suppressed_by'] = f'persist(level>={min_lvl_for_vote}:{k}/{n})'
                    suppressed_total += 1
            new_groups.append(g2)
        new_windows.append({**w, 'groups': new_groups})

    all_groups = [g for w in new_windows for g in w['groups']]
    is_anom = [bool(g.get('is_anomaly', False)) for g in all_groups]
    eff_levels = [
        int(g.get('effective_level', 0) or 0) if g.get('is_anomaly') else 0
        for g in all_groups
    ]
    anomaly_windows = sorted({
        w['window_start'] for w in new_windows
        if any(g.get('is_anomaly') for g in w['groups'])
    })
    new_summary = dict(video.get('summary', {}))
    new_summary['n_anomalies'] = int(sum(is_anom))
    new_summary['anomaly_windows'] = anomaly_windows
    new_summary['max_effective_level'] = int(max(eff_levels)) if eff_levels else 0

    if report and suppressed_total:
        print(f"  [persist] {video.get('video_path', '?')}: suppressed {suppressed_total} groups")

    return {**video, 'windows': new_windows, 'summary': new_summary}


def load_results(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_video_features(v: dict, gt: int, score_mode: str) -> dict:
    all_groups = [g for w in v.get('windows', []) for g in w.get('groups', [])]
    level_probs = [g.get('level_prob', 0.0) for g in all_groups]
    eff_levels = [g.get('effective_level', 0) for g in all_groups]
    is_anom = [bool(g.get('is_anomaly', False)) for g in all_groups]

    n_windows = v.get('n_windows', len(v.get('windows', [])))
    anomaly_windows = v.get('summary', {}).get('anomaly_windows', [])
    anomaly_ratio = (len(anomaly_windows) / n_windows) if n_windows else 0.0

    max_level_prob = max(level_probs) if level_probs else 0.0
    max_eff_level = max(eff_levels) if eff_levels else 0
    max_level_x_prob = max(
        (lvl * p for lvl, p in zip(eff_levels, level_probs)),
        default=0.0,
    )

    if score_mode == 'max_level_prob':
        score = max_level_prob
    elif score_mode == 'level_x_prob':
        score = max_level_x_prob
    elif score_mode == 'max_effective_level':
        score = float(max_eff_level)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    return {
        'video_path': v['video_path'],
        'gt': gt,
        'score': float(score),
        'max_level_prob': float(max_level_prob),
        'max_level_x_prob': float(max_level_x_prob),
        'max_effective_level': int(max_eff_level),
        'n_groups': len(all_groups),
        'n_anomaly_groups': sum(is_anom),
        'n_windows': n_windows,
        'anomaly_ratio': float(anomaly_ratio),
        'pred_any': int(any(is_anom)) if is_anom else 0,
        'top_action': v.get('summary', {}).get('top_action'),
    }


def load_split(
    path: str,
    gt: int,
    score_mode: str,
    persistence_rules: list[dict] | None = None,
    min_action_prob: float = 0,
    persist_report: bool = False,
) -> list[dict]:
    data = load_results(path)
    feats = []
    for v in data['videos']:
        if persistence_rules is not None:
            v = apply_persistence_filter(
                v,
                persistence_rules,
                min_action_prob=min_action_prob,
                report=persist_report,
            )
        feats.append(extract_video_features(v, gt, score_mode))
    run_id = data.get('run_meta', {}).get('run_id') or Path(path).stem
    print(f"[load] {run_id}: {len(feats)} videos, gt={'anomaly' if gt else 'normal'}"
          f"{', persistence=on' if persistence_rules is not None else ''}")
    return feats



def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    n_pos = len(pos)
    sum_ranks_pos = ranks[:n_pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * len(neg)))


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float('nan')
    order = np.argsort(-y_score, kind='mergesort')
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def compute_video_metrics(features: list[dict]) -> dict:
    gt = np.array([f['gt'] for f in features])
    pred = np.array([f['pred_any'] for f in features])
    score = np.array([f['score'] for f in features])

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float('nan'))
    accuracy = (tp + tn) / len(gt) if len(gt) else float('nan')
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else float('nan')
    balanced_acc = (tpr + tnr) / 2 if not (np.isnan(tpr) or np.isnan(tnr)) else float('nan')
    far = fp / (fp + tn) if (fp + tn) else float('nan')

    roc_auc = auc_roc(gt, score)
    pr_auc = auc_pr(gt, score)

    normal = [f for f in features if f['gt'] == 0]
    anomaly = [f for f in features if f['gt'] == 1]

    def _mean(xs):
        return float(np.mean(xs)) if xs else float('nan')

    def _median(xs):
        return float(np.median(xs)) if xs else float('nan')

    max_level_normal = [f['max_effective_level'] for f in normal]
    max_level_anomaly = [f['max_effective_level'] for f in anomaly]
    ratio_normal = [f['anomaly_ratio'] for f in normal]
    ratio_anomaly = [f['anomaly_ratio'] for f in anomaly]
    delta = (_mean(max_level_anomaly) - _mean(max_level_normal)) if (max_level_anomaly and max_level_normal) else float('nan')

    return {
        'n_videos': len(gt),
        'n_normal': int((gt == 0).sum()),
        'n_anomaly': int((gt == 1).sum()),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'far': far,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'max_level_normal_mean': _mean(max_level_normal),
        'max_level_normal_median': _median(max_level_normal),
        'max_level_anomaly_mean': _mean(max_level_anomaly),
        'max_level_anomaly_median': _median(max_level_anomaly),
        'max_level_delta': delta,
        'anomaly_ratio_normal_mean': _mean(ratio_normal),
        'anomaly_ratio_anomaly_mean': _mean(ratio_anomaly),
        'max_level_dist_normal': {
            str(lvl): sum(1 for x in max_level_normal if x == lvl) for lvl in range(4)
        },
        'max_level_dist_anomaly': {
            str(lvl): sum(1 for x in max_level_anomaly if x == lvl) for lvl in range(4)
        },
    }


def fmt(x, nd=3):
    if isinstance(x, float):
        return 'nan' if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def print_report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Videos: {m['n_videos']} (normal={m['n_normal']}, anomaly={m['n_anomaly']})")
    print(f"Confusion (video-level): TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"Precision={fmt(m['precision'])}  Recall={fmt(m['recall'])}  F1={fmt(m['f1'])}")
    print(f"Accuracy={fmt(m['accuracy'])}  BalancedAcc={fmt(m['balanced_accuracy'])}  FAR={fmt(m['far'])}")
    print(f"ROC-AUC={fmt(m['roc_auc'])}  PR-AUC={fmt(m['pr_auc'])}")
    print(f"MaxEffLevel: normal mean={fmt(m['max_level_normal_mean'])} "
          f"median={fmt(m['max_level_normal_median'])} | "
          f"anomaly mean={fmt(m['max_level_anomaly_mean'])} "
          f"median={fmt(m['max_level_anomaly_median'])} | "
          f"delta={fmt(m['max_level_delta'])}")
    print(f"MaxEffLevel dist (normal):  {m['max_level_dist_normal']}")
    print(f"MaxEffLevel dist (anomaly): {m['max_level_dist_anomaly']}")
    print(f"Anomaly ratio: normal mean={fmt(m['anomaly_ratio_normal_mean'])} | "
          f"anomaly mean={fmt(m['anomaly_ratio_anomaly_mean'])}")


def print_comparison_table(results: dict[str, dict]) -> None:
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy',
               'far', 'roc_auc', 'pr_auc',
               'max_level_normal_mean', 'max_level_anomaly_mean', 'max_level_delta',
               'anomaly_ratio_normal_mean', 'anomaly_ratio_anomaly_mean']
    header = ['metric'] + list(results.keys())
    width = max(len(h) for h in header) + 2
    print("\n=== Comparison ===")
    print(''.join(h.ljust(width) for h in header))
    print('-' * (width * len(header)))
    for m in metrics:
        row = [m] + [fmt(results[k].get(m)) for k in results]
        print(''.join(str(c).ljust(width) for c in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal', required=True, nargs='+')
    parser.add_argument('--anomaly', required=True, nargs='+')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--score', default='max_level_prob',
                        choices=['max_level_prob', 'level_x_prob', 'max_effective_level'])
    parser.add_argument('--persist', action='store_true')
    parser.add_argument('--csv', default=None)

    args = parser.parse_args()

    if len(args.normal) != len(args.anomaly):
        parser.error(
            f"--normal and --anomaly must have the same number of files "
            f"(got {len(args.normal)} and {len(args.anomaly)})."
        )

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    post_cfg = cfg['postprocessing']
    persistence_rules = post_cfg['filter_rules'] if args.persist else None
    min_action_prob = post_cfg['min_action_prob']

    if args.persist:
        print(f"[persist] on (config={args.config}); min_action_prob={min_action_prob}; rules: {persistence_rules}")

    all_rows = []
    metrics_by_preset = {}

    for normal_path, anomaly_path in zip(args.normal, args.anomaly):
        meta = load_results(normal_path).get('run_meta', {})
        preset = meta.get('preset') or meta.get('run_id') or Path(normal_path).stem

        feats_normal = load_split(normal_path, gt=0, score_mode=args.score,
                                  persistence_rules=persistence_rules,
                                  min_action_prob=min_action_prob)
        feats_anomaly = load_split(anomaly_path, gt=1, score_mode=args.score,
                                   persistence_rules=persistence_rules,
                                   min_action_prob=min_action_prob)
        feats = feats_normal + feats_anomaly

        metrics_by_preset[preset] = compute_video_metrics(feats)
        print_report(preset, metrics_by_preset[preset])

        for f in feats:
            all_rows.append({'preset': preset, **f})

    if len(metrics_by_preset) > 1:
        print_comparison_table(metrics_by_preset)

    if args.csv:
        fields = ['preset', 'video_path', 'gt', 'score', 'max_level_prob',
                  'max_level_x_prob', 'max_effective_level',
                  'n_groups', 'n_anomaly_groups', 'n_windows',
                  'anomaly_ratio', 'pred_any', 'top_action']
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSaved per-video CSV: {args.csv} ({len(all_rows)} rows)")


if __name__ == '__main__':
    main()
