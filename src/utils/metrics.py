"""
Metrics utilities.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm_normalized, class_names, num_classes, save_dir):
    """Plot and save confusion matrix."""
    class_ids = range(num_classes)

    plt.figure(figsize=(max(12, num_classes * 0.4), max(10, num_classes * 0.3)))
    labels = [class_names.get(i, str(i))[:15] for i in class_ids]

    sns.heatmap(
        cm_normalized,
        annot=True if num_classes <= 10 else False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Normalized Accuracy'}
    )

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_accuracy(per_class_df, save_dir):
    """Plot per-class accuracy as horizontal bar chart."""
    df_sorted = per_class_df.sort_values('accuracy', ascending=True)

    fig, ax = plt.subplots(figsize=(14, max(10, len(df_sorted) * 0.25)))

    colors = plt.cm.RdYlGn(df_sorted['accuracy'].values)
    ax.barh(range(len(df_sorted)), df_sorted['accuracy'], color=colors)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['class_name'].str[:30], fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_f1(per_class_df, save_dir):
    """Plot per-class F1 score as horizontal bar chart."""
    df_sorted = per_class_df.sort_values('f1_score', ascending=True)

    fig, ax = plt.subplots(figsize=(14, max(10, len(df_sorted) * 0.25)))

    colors = plt.cm.RdYlGn(df_sorted['f1_score'].values)
    ax.barh(range(len(df_sorted)), df_sorted['f1_score'], color=colors)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['class_name'].str[:30], fontsize=8)
    ax.set_xlabel('F1 Score')
    ax.set_title('Per-Class F1 Score')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_f1.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(per_class_df, save_dir):
    """Plot distribution of metrics across all classes."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [per_class_df[m].dropna().values for m in metrics]
    bp = ax.boxplot(data, patch_artist=True)

    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Score')
    ax.set_title('Distribution of Metrics Across Classes')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_support_vs_f1(per_class_df, save_dir):
    """Plot F1 score vs number of samples per class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        per_class_df['support'],
        per_class_df['f1_score'],
        c=per_class_df['accuracy'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        s=80
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label('Accuracy')

    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Number of Samples')
    ax.set_xlim(0, per_class_df['support'].max() * 1.1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Annotate outliers
    low_f1 = per_class_df.nsmallest(3, 'f1_score')
    for _, row in low_f1.iterrows():
        ax.annotate(
            row['class_name'][:15],
            (row['support'], row['f1_score']),
            xytext=(10, -10),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )

    plt.tight_layout()
    plt.savefig(save_dir / 'support_vs_f1.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_heatmap(per_class_df, save_dir):
    """Plot heatmap of all metrics for top and bottom classes."""
    # Select top-10 and bottom-10 by F1
    df_sorted = per_class_df.sort_values('f1_score', ascending=False)
    top_bottom = pd.concat([df_sorted.head(10), df_sorted.tail(10)])

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    heatmap_data = top_bottom.set_index('class_name')[metrics]

    fig, ax = plt.subplots(figsize=(8, max(6, int(len(top_bottom) * 0.4))))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5
    )

    ax.set_title('Metrics Heatmap (Top-10 and Bottom-10 by F1)')
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_results(results_dict, eval_dir):
    """Print evaluation results."""
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")

    print("\nOVERALL METRICS:")
    print(f"  Accuracy:           {results_dict.get('accuracy', 0):.4f}")
    print(f"  Top-3 Accuracy:     {results_dict.get('top3_accuracy', 0):.4f}")
    print(f"  Top-5 Accuracy:     {results_dict.get('top5_accuracy', 0):.4f}")
    if results_dict.get('roc_auc_ovr'):
        print(f"  ROC AUC (OvR):      {results_dict['roc_auc_ovr']:.4f}")

    print("\nMACRO AVERAGES:")
    print(f"  Precision:          {results_dict.get('precision_macro', 0):.4f}")
    print(f"  Recall:             {results_dict.get('recall_macro', 0):.4f}")
    print(f"  F1-Score:           {results_dict.get('f1_macro', 0):.4f}")

    print("\nWEIGHTED AVERAGES:")
    print(f"  Precision:          {results_dict.get('precision_weighted', 0):.4f}")
    print(f"  Recall:             {results_dict.get('recall_weighted', 0):.4f}")
    print(f"  F1-Score:           {results_dict.get('f1_weighted', 0):.4f}")

    if 'per_class_metrics' in results_dict:
        per_class = pd.DataFrame(results_dict['per_class_metrics'])

        print("\nTOP-5 BEST CLASSES:")
        for _, row in per_class.head(5).iterrows():
            print(f"  {row['class_name']:20s} F1={row['f1_score']:.3f}  Acc={row['accuracy']:.3f}  "
                  f"P={row['precision']:.3f}  R={row['recall']:.3f}  (n={int(row['support'])})")

        print("\nTOP-5 WORST CLASSES:")
        for _, row in per_class.tail(5).iterrows():
            print(f"  {row['class_name']:20s} F1={row['f1_score']:.3f}  Acc={row['accuracy']:.3f}  "
                  f"P={row['precision']:.3f}  R={row['recall']:.3f}  (n={int(row['support'])})")

    print(f"\nResults saved to: {eval_dir}")
    print(f"  - test_results.yaml")
    print(f"  - per_class_metrics.csv")
    print(f"  - confusion_matrix.png")
    print(f"{'=' * 80}")