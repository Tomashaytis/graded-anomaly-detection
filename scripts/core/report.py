"""
Visualization of model metrics: both heads on a single figure.
"""

import yaml
import rootutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import pi
from typing import Dict
from pathlib import Path
from matplotlib.patches import Patch

root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True, cwd=True)

CONFIG_PATH = Path("configs") / "report.yaml"


def translate_class_name(class_name: str, lang: str, config) -> str:
    """Translate a class name to the target language."""
    if lang == "ru":
        return config["class_translations_ru"].get(class_name, class_name)
    text = config["class_translations"].get(class_name, class_name)
    return text[0].lower() + text[1:]


def load_head2_category_metrics(config, eval_dir: Path, t: Dict[str, str]) -> pd.DataFrame:
    """
    Load per-class metrics and aggregate F1 by category.

    :param config: Report config.
    :param eval_dir: Path to evaluation directory.
    :param t: Translation dictionary.
    :return: DataFrame with aggregated metrics per category.
    """
    annotations_df = pd.read_csv(config["annotations_path"])
    label_category_mapping = annotations_df[["label_id", "category"]].drop_duplicates()
    mapping_dict = dict(zip(label_category_mapping["label_id"], label_category_mapping["category"]))

    metrics_df = pd.read_csv(eval_dir / "per_class_metrics.csv")
    metrics_df["category"] = metrics_df["class_id"].map(mapping_dict)
    metrics_df = metrics_df.dropna(subset=["category"])

    category_metrics = metrics_df.groupby("category").apply(
        lambda g: pd.Series({
            "total_support": g["support"].sum(),
            "avg_f1": (g["f1_score"] * g["support"]).sum() / g["support"].sum(),
        })
    ).reset_index()

    category_metrics["category_localized"] = category_metrics["category"].map(
        lambda c: t[config["category_keys"][c]] if c in config["category_keys"] else c
    )
    return category_metrics


def plot_model_summary(config):
    """
    Build a single figure with multiple subplots for the selected model.

    :param config: Report config.
    """
    model_key = config["default_model"]
    lang = config["default_lang"]
    figsize = tuple(config["figsize"])
    t = config["texts"][lang]

    model_cfg = config["models"][model_key]
    eval_dir = Path(model_cfg["eval_dir"])
    model_name = model_cfg["name"]
    n_frames = model_cfg["n_frames"]

    with open(eval_dir / "test_results.yaml", "r") as f:
        test_results = yaml.safe_load(f)

    level_cm = np.array(test_results["head1_level_confusion_matrix"], dtype=float)
    category_metrics = load_head2_category_metrics(config, eval_dir, t)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(wspace=2, left=0.08, right=0.95, top=0.85, bottom=0.15)
    ax1, ax2 = axes

    level_names = [t[k] for k in config["level_keys"]]

    cm_norm = level_cm / (level_cm.sum(axis=1, keepdims=True) + 1e-10)
    ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(level_names)))
    ax1.set_yticks(np.arange(len(level_names)))
    ax1.set_xticklabels(level_names, fontsize=10)
    ax1.set_yticklabels(level_names, fontsize=10)
    ax1.set_xlabel(t["xlabel_predicted"], fontsize=11, fontweight="bold")
    ax1.set_ylabel(t["ylabel_true"], fontsize=11, fontweight="bold")
    ax1.set_title(t["subplot_a_title"], fontsize=12, fontweight="bold", pad=10)
    for i in range(len(level_names)):
        for j in range(len(level_names)):
            value = cm_norm[i, j]
            color = "white" if value > 0.5 else "black"
            ax1.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    df = category_metrics.sort_values("avg_f1", ascending=True)
    colors = plt.cm.Blues(df["avg_f1"] / df["avg_f1"].max())
    bars = ax2.barh(df["category_localized"], df["avg_f1"], color=colors,
                    edgecolor="black", linewidth=0.5)
    ax2.set_xlabel(t["xlabel_f1"], fontsize=11, fontweight="bold")
    ax2.set_ylabel(t["ylabel_category"], fontsize=11, fontweight="bold", labelpad=10)
    ax2.set_title(t["subplot_b_title"], fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(0, 1)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, val in zip(bars, df["avg_f1"]):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left", fontsize=9)

    fig.suptitle(
        t["suptitle"].format(name=model_name, n_frames=n_frames),
        fontsize=15, fontweight="bold", y=1.02
    )

    plt.tight_layout()

    suffix = "ru" if lang == "ru" else "en"
    out_path = eval_dir / f"two_heads_summary_{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved: {out_path}")


def load_head2_per_class_metrics(config, eval_dir: Path) -> pd.DataFrame:
    """
    Load per-class metrics for head 2.

    :param config: Report config.
    :param eval_dir: Path to evaluation directory.
    :return: DataFrame with per-class metrics.
    """
    annotations_df = pd.read_csv(config["annotations_path"])
    label_category_mapping = annotations_df[["label_id", "category"]].drop_duplicates()
    mapping_dict = dict(zip(label_category_mapping["label_id"], label_category_mapping["category"]))

    metrics_df = pd.read_csv(eval_dir / "per_class_metrics.csv")
    metrics_df["category"] = metrics_df["class_id"].map(mapping_dict)
    metrics_df = metrics_df.dropna(subset=["category"])
    return metrics_df


def plot_sorted_f1_with_extremes(config):
    """
    Build a paired plot: sorted F1 across classes + top-5/bottom-5.

    :param config: Report config.
    """
    model_key = config["default_model"]
    lang = config["default_lang"]
    t = config["texts"][lang]

    model_cfg = config["models"][model_key]
    eval_dir = Path(model_cfg["eval_dir"])
    model_name = model_cfg["name"]
    n_frames = model_cfg["n_frames"]

    metrics_df = load_head2_per_class_metrics(config, eval_dir)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.95, top=0.85, bottom=0.15)

    df_sorted = metrics_df.sort_values("f1_score", ascending=True).reset_index(drop=True)

    class_indices = range(len(df_sorted))
    f1_values = df_sorted["f1_score"].values

    ax1.plot(class_indices, f1_values, "o-",
             color="#2E86AB", markersize=4, linewidth=1.5, label="F1-score")
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="0.5")
    ax1.axhline(y=0.8, color="green", linestyle="--", alpha=0.7, label="0.8")
    ax1.set_xlabel(t["sorted_f1_xlabel"], fontsize=11, fontweight="bold")
    ax1.set_ylabel(t["sorted_f1_ylabel"], fontsize=11, fontweight="bold")
    ax1.set_title(t["sorted_f1_title"], fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylim(0.3, 1.0)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    top5 = metrics_df.nlargest(5, "f1_score")[["class_name", "f1_score"]].copy()
    bottom5 = metrics_df.nsmallest(5, "f1_score")[["class_name", "f1_score"]].copy()
    top5["class_name"] = top5["class_name"].apply(lambda c: translate_class_name(c, lang, config))
    bottom5["class_name"] = bottom5["class_name"].apply(lambda c: translate_class_name(c, lang, config))
    bottom5 = bottom5.sort_values("f1_score", ascending=True)
    top5 = top5.sort_values("f1_score", ascending=True)

    combined = pd.concat([bottom5, top5])
    colors = ["#CD5C5C"] * len(bottom5) + ["#2E8B57"] * len(top5)
    bars = ax2.barh(combined["class_name"], combined["f1_score"],
                    color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel(t["top_bottom_xlabel"], fontsize=11, fontweight="bold")
    ax2.set_title(t["top_bottom_title"], fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(0, 1)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar, val in zip(bars, combined["f1_score"]):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left", fontsize=9)

    legend_elements = [
        Patch(facecolor="#2E8B57", label=t["top_bottom_legend_best"]),
        Patch(facecolor="#CD5C5C", label=t["top_bottom_legend_worst"]),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.suptitle(
        t["suptitle"].format(name=model_name, n_frames=n_frames),
        fontsize=15, fontweight="bold", y=1.02
    )

    plt.tight_layout()

    suffix = "ru" if lang == "ru" else "en"
    out_path = eval_dir / f"sorted_f1_extremes_{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved: {out_path}")


def plot_radar_by_category(config):
    """
    Build a radar chart with Precision, Recall, and F1-score by category.

    :param config: Report config.
    """
    model_key = config["default_model"]
    lang = config["default_lang"]
    t = config["texts"][lang]

    model_cfg = config["models"][model_key]
    eval_dir = Path(model_cfg["eval_dir"])
    model_name = model_cfg["name"]
    n_frames = model_cfg["n_frames"]

    metrics_df = load_head2_per_class_metrics(config, eval_dir)

    category_metrics = metrics_df.groupby("category").apply(
        lambda g: pd.Series({
            "avg_precision": (g["precision"] * g["support"]).sum() / g["support"].sum(),
            "avg_recall": (g["recall"] * g["support"]).sum() / g["support"].sum(),
            "avg_f1": (g["f1_score"] * g["support"]).sum() / g["support"].sum(),
        })
    ).reset_index()

    category_metrics["category_localized"] = category_metrics["category"].map(
        lambda c: t[config["category_keys"][c]] if c in config["category_keys"] else c
    )

    categories = category_metrics["category_localized"].tolist()
    metrics_names = ["avg_precision", "avg_recall", "avg_f1"]
    labels = [t["radar_precision"], t["radar_recall"], t["radar_f1"]]

    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")

    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    markers = ["o-", "s--", "^-.", "D:", "v-"]

    for i, (metric, label) in enumerate(zip(metrics_names, labels)):
        values = category_metrics[metric].tolist()
        values += values[:1]
        ax.plot(angles, values, markers[i % len(markers)],
                linewidth=1.8, markersize=5,
                color=colors[i % len(colors)], label=label)
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    for angle, label in zip(angles[:-1], categories):
        if 0 <= angle < pi / 4:
            ha, va = "left", "center"
        elif pi / 4 <= angle < 3 * pi / 4:
            ha, va = "center", "bottom"
        elif 3 * pi / 4 <= angle < 5 * pi / 4:
            ha, va = "right", "center"
        else:
            ha, va = "center", "top"
        ax.text(angle, 1.02, label, ha=ha, va=va, fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(
        t["radar_title"].format(name=model_name, n_frames=n_frames),
        fontsize=14, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    suffix = "ru" if lang == "ru" else "en"
    out_path = eval_dir / f"radar_categories_{suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    plot_model_summary(config)
    plot_sorted_f1_with_extremes(config)
    plot_radar_by_category(config)