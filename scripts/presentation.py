import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import pi
from pathlib import Path

MODEL_EVAL_DIRS = {
    'I3D': Path('runs') / 'model=I3D-8x8-R50' / 'results' / 'eval_i3d_r50_best-epoch=08-val_acc=0.787',
    'SlowFast': Path('runs') / 'model=SlowFast-8x8-R50' / 'results' / 'eval_slowfast_r50_best-epoch=10-val_acc=0.774',
    'X3D-M': Path('runs') / 'model=X3D-M' / 'results' / 'eval_x3d_m_best-epoch=18-val_acc=0.775',
}
METRICS_DATA = {
    'I3D-8x8-R50': {
        'Top-1 Acc': 0.7749,
        'Top-3 Acc': 0.9005,
        'Top-5 Acc': 0.9305,
        'Precision (macro)': 0.7351,
        'Recall (macro)': 0.7416,
        'F1 (macro)': 0.7383,
        'ROC AUC (OVR)': 0.9766,
    },
    'SlowFast-8x8-R50': {
        'Top-1 Acc': 0.7657,
        'Top-3 Acc': 0.8865,
        'Top-5 Acc': 0.9133,
        'Precision (macro)': 0.7230,
        'Recall (macro)': 0.7215,
        'F1 (macro)': 0.7222,
        'ROC AUC (OVR)': 0.9660,
    },
    'X3D-M': {
        'Top-1 Acc': 0.7601,
        'Top-3 Acc': 0.8960,
        'Top-5 Acc': 0.9277,
        'Precision (macro)': 0.7000,
        'Recall (macro)': 0.7267,
        'F1 (macro)': 0.7131,
        'ROC AUC (OVR)': 0.9779,
    }
}

TRANSLATION = {
        'emergency': 'Чрезвычайная ситуация',
        'inadequate condition': 'Неадекватное состояние',
        'overcoming obstacles': 'Преодоление препятствий',
        'strange': 'Странные действия',
        'using items and weapons': 'Использование предметов/оружия',
        'vandalism': 'Вандализм',
        'violence': 'Проявление жестокости',
        'computer usage': 'Использование ЭВМ',
        'normal': 'Нормальное поведение'
    }

SAVE_DIR = Path('docs') / 'Рисунки'


if __name__ == '__main__':
    # Load data
    model_data = {}
    for model_name, eval_dir in MODEL_EVAL_DIRS.items():
        df = pd.read_csv(eval_dir / 'category_metrics.csv')
        model_data[model_name] = df.set_index('category')['avg_f1']

    # F1-score для 3 моделей по категориям
    plot_df = pd.DataFrame(model_data).sort_index()

    plot_df.index = plot_df.index.map(TRANSLATION)

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(19, 8))

    plot_df.plot(kind='barh', ax=ax, width=0.7,
                 color=['#2E86AB', '#A23B72', '#6A994E'],
                 edgecolor='black', linewidth=0.5)

    for i in range(len(plot_df)):
        for j, model in enumerate(plot_df.columns):
            val = plot_df.iloc[i, j]
            if val > 0.05:
                ax.text(val + 0.01, i - 0.25 + j * 0.23, f'{val:.3f}',
                        va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel('Средний F1-score', fontsize=14)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title('Средний F1-score обученных моделей по категориям действий', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=14)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Радарная диаграмма
    categories = list(METRICS_DATA['I3D-8x8-R50'].keys())
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')

    colors = {'I3D-8x8-R50': '#2E86AB', 'SlowFast-8x8-R50': '#A23B72', 'X3D-M': '#6A994E'}

    for model_name, metrics in METRICS_DATA.items():
        values = list(metrics.values())
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[model_name])
        ax.fill(angles, values, alpha=0.1, color=colors[model_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.5, 1)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=8)
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Группированная столбчатая диаграмма
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')

    metrics_to_plot = ['Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc', 'F1 (macro)', 'Precision (macro)', 'Recall (macro)', 'ROC AUC (OVR)']
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for i, (model_name, metrics) in enumerate(METRICS_DATA.items()):
        values = [metrics[m] for m in metrics_to_plot]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors[model_name], edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metrics', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, ha='center')
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'grouped_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Тепловая карта сравнения
    comparison_df = pd.DataFrame(METRICS_DATA).T

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(comparison_df.values, cmap='RdYlGn', aspect='auto', vmin=0.1, vmax=1.0)

    ax.set_xticks(np.arange(len(comparison_df.columns)))
    ax.set_yticks(np.arange(len(comparison_df.index)))
    ax.set_xticklabels(comparison_df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(comparison_df.index, fontsize=10)

    for i in range(len(comparison_df.index)):
        for j in range(len(comparison_df.columns)):
            text = ax.text(j, i, f'{comparison_df.iloc[i, j]:.3f}',
                           ha='center', va='center', color='black' if comparison_df.iloc[i, j] < 0.85 else 'white',
                           fontsize=9)

    ax.set_title('Model Performance Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
