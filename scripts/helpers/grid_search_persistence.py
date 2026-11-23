"""
python -m scripts.helpers.grid_search_persistence `
  --normal data/test_system/results/slowfast/normal.json `
  --anomaly data/test_system/results/slowfast/anomalous.json `
  --max-far 0.15
"""

import sys
import csv
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from calculate_system_metrics import (
    load_results,
    apply_persistence_filter,
    extract_video_features,
    compute_video_metrics,
    fmt,
)


ACTION_PROB_GRID = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70]

RULE_PRESETS = {
    "raw": [],
    "soft": [
        {"level": (1,),    "mode": "all",    "k": 2, "n": 4},
        {"level": (2, 3), "mode": "global", "k": 1, "n": 1},
        {"level": (2, 3), "mode": "local",  "k": 2, "n": 3},
    ],
    "balanced": [
        {"level": (1,),    "mode": "all",    "k": 3, "n": 5},
        {"level": (2, 3), "mode": "global", "k": 1, "n": 1},
        {"level": (2, 3), "mode": "local",  "k": 2, "n": 3},
    ],
    "strict_local": [
        {"level": (1,),    "mode": "all",    "k": 3, "n": 5},
        {"level": (2, 3), "mode": "global", "k": 1, "n": 1},
        {"level": (2, 3), "mode": "local",  "k": 2, "n": 4},
    ],
    "hard": [
        {"level": (1,),    "mode": "all",    "k": 4, "n": 7},
        {"level": (2, 3), "mode": "global", "k": 2, "n": 3},
        {"level": (2, 3), "mode": "local",  "k": 3, "n": 5},
    ],
}


def evaluate_config(
    normal_videos: list[dict],
    anomaly_videos: list[dict],
    rules: list[dict],
    min_action_prob: float,
    require_same_action: bool,
    score_mode: str,
) -> dict:
    feats = []
    for v in normal_videos:
        vf = apply_persistence_filter(
            v,
            rules,
            min_action_prob=min_action_prob,
            require_same_action=require_same_action,
        )
        feats.append(extract_video_features(vf, gt=0, score_mode=score_mode))

    for v in anomaly_videos:
        vf = apply_persistence_filter(
            v,
            rules,
            min_action_prob=min_action_prob,
            require_same_action=require_same_action,
        )
        feats.append(extract_video_features(vf, gt=1, score_mode=score_mode))

    return compute_video_metrics(feats)


def print_grid_table(rows: list[dict]) -> None:
    cols = [
        ("preset", 24),
        ("act_prob", 10),
        ("same_act", 10),
        ("f1", 8),
        ("prec", 8),
        ("rec", 8),
        ("far", 8),
        ("bal_acc", 9),
        ("tp", 5),
        ("fp", 5),
        ("fn", 5),
    ]
    header = "".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        line = "".join(fmt(r[name]).ljust(w) for name, w in cols)
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True, nargs="+")
    parser.add_argument("--anomaly", required=True, nargs="+")
    parser.add_argument(
        "--score",
        default="max_level_prob",
        choices=["max_level_prob", "level_x_prob", "max_effective_level"],
    )
    parser.add_argument("--max-far", type=float, default=1.0, help="Фильтр по максимальному FAR")
    parser.add_argument("--csv", default=None, help="Сохранить все прогоны в CSV")
    args = parser.parse_args()

    normal_videos = []
    for p in args.normal:
        normal_videos.extend(load_results(p)["videos"])

    anomaly_videos = []
    for p in args.anomaly:
        anomaly_videos.extend(load_results(p)["videos"])

    print(f"Loaded {len(normal_videos)} normal and {len(anomaly_videos)} anomaly videos.")

    results = []

    for preset_name, rules in RULE_PRESETS.items():
        for act_prob in ACTION_PROB_GRID:
            for same_act in (True, False):
                if preset_name == "1_raw" and not same_act:
                    continue

                m = evaluate_config(
                    normal_videos=normal_videos,
                    anomaly_videos=anomaly_videos,
                    rules=rules,
                    min_action_prob=act_prob,
                    require_same_action=same_act,
                    score_mode=args.score,
                )
                results.append({
                    "preset": preset_name,
                    "act_prob": act_prob,
                    "same_act": str(same_act),
                    "f1": m["f1"],
                    "prec": m["precision"],
                    "rec": m["recall"],
                    "far": m["far"],
                    "bal_acc": m["balanced_accuracy"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "tn": m["tn"],
                    "fn": m["fn"],
                })

    filtered = [r for r in results if r["far"] <= args.max_far]
    filtered.sort(key=lambda x: (x["f1"], x["bal_acc"]), reverse=True)

    print(f"\n=== Top 20 configs (FAR <= {args.max_far}) sorted by F1 ===")
    print_grid_table(filtered)

    if args.csv and results:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda x: x["f1"], reverse=True))
        print(f"\nSaved full grid results to {args.csv} ({len(results)} runs)")


if __name__ == "__main__":
    main()
