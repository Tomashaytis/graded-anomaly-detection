import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import pi
from pathlib import Path
from matplotlib.patches import Patch

DEFAULT_CONFIG_PATH = os.path.join('configs', 'default.yaml')
EVAL_FOLDER = 'eval_i3d_r50_best-epoch=08-val_acc=0.787'
MODEL_NAME = 'I3D-8x8-R50'


if __name__ == '__main__':
    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    eval_dir = Path(config['paths']['output_dir']) / 'results' / EVAL_FOLDER

    annotations_df = pd.read_csv(config['paths']['annotations_train'])
    label_category_mapping = annotations_df[['label_id', 'category']].drop_duplicates()
    mapping_dict = dict(zip(label_category_mapping['label_id'], label_category_mapping['category']))

    metrics_df = pd.read_csv(eval_dir / 'per_class_metrics.csv')

    metrics_df['category'] = metrics_df['class_id'].map(mapping_dict)

    missing_categories = metrics_df[metrics_df['category'].isna()]
    if not missing_categories.empty:
        print("Внимание! Для следующих классов не найдены категории:")
        print(missing_categories[['class_id', 'class_name']])

    category_metrics = metrics_df.groupby('category').apply(
        lambda g: pd.Series({
            'num_classes': len(g),
            'total_support': g['support'].sum(),
            'avg_precision': (g['precision'] * g['support']).sum() / g['support'].sum(),
            'avg_recall': (g['recall'] * g['support']).sum() / g['support'].sum(),
            'avg_f1': (g['f1_score'] * g['support']).sum() / g['support'].sum(),
            'min_f1': g['f1_score'].min(),
            'max_f1': g['f1_score'].max(),
        })
    ).reset_index()

    category_metrics = category_metrics.sort_values('avg_f1', ascending=False)

    print("\n" + "="*80)
    print("АГРЕГИРОВАННЫЕ МЕТРИКИ ПО КАТЕГОРИЯМ (взвешенные по support)")
    print("="*80)
    print(category_metrics.to_string(index=False))

    output_path = eval_dir / 'category_metrics.csv'
    category_metrics.to_csv(output_path, index=False)
    print(f"\nРезультат сохранён: {output_path}")

    print("\n" + "="*80)
    print("ТОП-5 ЛУЧШИХ КЛАССОВ ПО F1")
    print("="*80)
    top5 = metrics_df.nlargest(5, 'f1_score')[['class_name', 'category', 'f1_score', 'support']]
    print(top5.to_string(index=False))

    print("\n" + "="*80)
    print("ТОП-5 ХУДШИХ КЛАССОВ ПО F1")
    print("="*80)
    bottom5 = metrics_df.nsmallest(5, 'f1_score')[['class_name', 'category', 'f1_score', 'support']]
    print(bottom5.to_string(index=False))

    # Двойной график
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # F1-score по категориям
    category_metrics_sorted = category_metrics.sort_values('avg_f1', ascending=True)

    colors = plt.cm.Blues(category_metrics_sorted['avg_f1'] / category_metrics_sorted['avg_f1'].max())
    bars = ax1.barh(category_metrics_sorted['category'], category_metrics_sorted['avg_f1'],
                    color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Average F1-score', fontsize=11, labelpad=8)
    ax1.set_ylabel('Category', fontsize=11, labelpad=8)
    ax1.set_title('(a) Average F1-score by Action Category', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar, val in zip(bars, category_metrics_sorted['avg_f1']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='left', fontsize=9)

    # Лучшие и худшие классы
    top5 = metrics_df.nlargest(5, 'f1_score')[['class_name', 'f1_score']].copy().iloc[::-1]
    bottom5 = metrics_df.nsmallest(5, 'f1_score')[['class_name', 'f1_score']].copy()

    top5['group'] = 'Best 5 classes'
    bottom5['group'] = 'Worst 5 classes'

    combined = pd.concat([bottom5, top5])
    combined['color'] = combined['group'].map({'Best 5 classes': '#2E8B57', 'Worst 5 classes': '#CD5C5C'})

    bars2 = ax2.barh(combined['class_name'], combined['f1_score'],
                     color=combined['color'], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('F1-score', fontsize=11, labelpad=8)
    ax2.set_ylabel('Class', fontsize=11, labelpad=8)
    ax2.set_title('(b) Best and Worst Performing Classes', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, val in zip(bars2, combined['f1_score']):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='left', fontsize=9)

    legend_elements = [Patch(facecolor='#2E8B57', label='Best 5 classes'),
                       Patch(facecolor='#CD5C5C', label='Worst 5 classes')]
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=False, fontsize=9)

    fig.suptitle(f'Model Performance Analysis: {MODEL_NAME}', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(eval_dir / 'performance_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Conclusion Matrix
    test_results_path = eval_dir / 'test_results.yaml'
    with open(test_results_path, 'r') as f:
        test_results = yaml.safe_load(f)

    cm_raw = test_results['confusion_matrix']
    cm_raw = np.array(cm_raw)

    class_ids = metrics_df['class_id'].values
    class_categories = metrics_df['category'].values

    categories = category_metrics.sort_values('avg_f1', ascending=False)['category'].tolist()
    n_cat = len(categories)

    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    cm_cat = np.zeros((n_cat, n_cat))

    for i in range(73):
        true_cat = class_categories[i]
        i_cat = cat_to_idx[true_cat]
        for j in range(73):
            pred_cat = class_categories[j]
            j_cat = cat_to_idx[pred_cat]
            cm_cat[i_cat, j_cat] += cm_raw[i, j]

    cm_cat_norm = cm_cat / cm_cat.sum(axis=1, keepdims=True)

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm_cat_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=categories, yticklabels=categories,
                square=True, cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted Category', fontsize=11)
    plt.ylabel('True Category', fontsize=11)
    plt.title(f'Confusion Matrix by Category: {MODEL_NAME}', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(eval_dir / 'confusion_matrix_categories.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\n" + "=" * 60)
    print("ТОЧНОСТЬ ПО КАТЕГОРИЯМ (доля правильных предсказаний)")
    print("=" * 60)
    for i, cat in enumerate(categories):
        acc = cm_cat_norm[i, i]
        total = cm_cat[i].sum()
        correct = cm_cat[i, i]
        print(f"{cat:25s}: {acc:.3f} ({correct:.0f}/{total:.0f})")

    # График F1
    category_metrics_sorted = category_metrics.sort_values('avg_f1', ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(category_metrics_sorted['category'], category_metrics_sorted['avg_f1'])
    plt.xlabel('Average F1-score')
    plt.ylabel('Category')
    plt.title('Model Performance by Action Category')
    plt.xlim(0, 1)

    for bar, val in zip(bars, category_metrics_sorted['avg_f1']):
        plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(eval_dir / 'f1_by_category.png', dpi=150)
    plt.show()

    # F1-Score gist
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['f1_score'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('F1-score')
    plt.ylabel('Number of classes')
    plt.title('Distribution of F1-score across 73 Action Classes')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.axvline(x=0.8, color='green', linestyle='--', label='Threshold 0.8')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    high_f1 = (metrics_df['f1_score'] >= 0.8).sum()
    mid_f1 = ((metrics_df['f1_score'] >= 0.5) & (metrics_df['f1_score'] < 0.8)).sum()
    low_f1 = (metrics_df['f1_score'] < 0.5).sum()
    plt.text(0.05, plt.ylim()[1] * 0.9,
             f'F1 ≥ 0.8: {high_f1} classes\nF1 0.5–0.8: {mid_f1} classes\nF1 < 0.5: {low_f1} classes',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(eval_dir / 'f1_distribution.png', dpi=150)
    plt.show()

    categories = category_metrics['category'].tolist()
    metrics_names = ['avg_precision', 'avg_recall', 'avg_f1']

    normalized_data = category_metrics[metrics_names].copy()
    for col in metrics_names:
        normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (
                    normalized_data[col].max() - normalized_data[col].min())

    # Радар
    normalized_data = category_metrics[metrics_names].copy()

    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    markers = ['o-', 's--', '^-.']

    for i, metric in enumerate(metrics_names):
        values = normalized_data[metric].tolist()
        values += values[:1]
        label = metric.replace('avg_', '').capitalize()
        if label == 'F1':
            label = 'F1-score'
        ax.plot(angles, values, markers[i % len(markers)],
                linewidth=1.5,
                markersize=4,
                color=colors[i % len(colors)],
                label=label)
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    for angle, label in zip(angles[:-1], categories):
        if 0 <= angle < pi / 4:
            radius_offset = 1.02
            ha, va = 'left', 'center'
        elif pi / 4 <= angle < 3 * pi / 4:
            radius_offset = 1.02
            ha, va = 'center', 'bottom'
        elif 3 * pi / 4 <= angle < 5 * pi / 4:
            radius_offset = 1.02
            ha, va = 'right', 'center'
        else:
            radius_offset = 1.02
            ha, va = 'center', 'top'

        ax.text(angle, radius_offset, label,
                ha=ha, va=va, fontsize=10,
                weight='bold',
                rotation=0)

    ax.set_ylim(0.4, 1)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.4', '0.5', '0.6', '0.7', '', '0.9', '1.0'], fontsize=8)

    ax.set_title(f'Category Performance Radar Chart: {MODEL_NAME}', size=12, pad=20, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), frameon=True, fancybox=True, shadow=False)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(eval_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()




