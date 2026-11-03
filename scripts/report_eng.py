import os
import yaml
import random
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

DEFAULT_CONFIG_PATH = os.path.join('..', 'configs', 'default.yaml')
EVAL_FOLDER = 'eval_i3d_r50_best-epoch=08-val_acc=0.787'
MODEL_NAME = 'I3D-8x8-R50'


def get_annotation_style():
    """Automatically determines annotation direction"""
    side = random.choice(['right', 'left'])

    if side == 'right':
        x_factor = 1.15
        y_offset = 0.04
        ha = 'left'
    else:
        x_factor = 0.7
        y_offset = 0.06
        ha = 'right'

    return x_factor, y_offset, ha


if __name__ == '__main__':
    random.seed(48)

    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    eval_dir = Path('..') / Path(config['paths']['output_dir']) / 'results' / 'sh' / EVAL_FOLDER

    metrics_df = pd.read_csv(eval_dir / 'per_class_metrics.csv')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8

    fig, ax = plt.subplots(figsize=(19, 7))
    fig.patch.set_facecolor('white')

    # Data for the plot
    plot_df = metrics_df[['class_name', 'support', 'f1_score']].copy()
    plot_df = plot_df.sort_values('support')

    # Scatter plot
    sc = ax.scatter(plot_df['support'], plot_df['f1_score'],
                    c=plot_df['f1_score'], cmap='RdYlGn',
                    s=80, alpha=0.7, edgecolor='black', linewidth=0.8,
                    vmin=0, vmax=1)

    # Trend line (LOWESS)
    lowess_result = lowess(plot_df['f1_score'], plot_df['support'],
                           frac=0.6,
                           return_sorted=True)
    ax.plot(lowess_result[:, 0], lowess_result[:, 1],
            color='#D62728', linewidth=3, label='LOWESS trend')

    # Axis setup
    ax.set_xscale('log')
    tick_positions = [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500]
    tick_labels = ['10', '15', '20', '30', '40', '50', '60', '80', '100', '150', '200', '300', '400', '500']

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, ha='center')
    ax.set_xlabel('Number of examples in class (test set)', fontsize=14, labelpad=10)
    ax.set_ylabel('F1-score', fontsize=14, labelpad=10)
    ax.set_title(f'{MODEL_NAME}: F1-score vs. Number of Training Examples',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linestyle='--', which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('F1-score', fontsize=14, rotation=270, labelpad=15)

    # Mean F1 horizontal line
    mean_f1 = metrics_df['f1_score'].mean()
    ax.axhline(y=mean_f1, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Mean F1-score = {mean_f1:.3f}')
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=14)

    interesting_classes = [
        'fighting',
        'arson',
        'burglary',
        'computer use',
        'taking photo',
        'looking at phone',
        'talking on cell phone',
        'opening door',
        'assembling computer',
        'shooting',
        'smoking',
        'assault',
        'closing door',
        'explosion',
        'vandalism',
    ]

    for class_name in interesting_classes:
        class_data = plot_df[plot_df['class_name'] == class_name]
        if not class_data.empty:
            row = class_data.iloc[0]
            x_factor, y_offset, ha = get_annotation_style()

            ax.annotate(row['class_name'],
                        xy=(row['support'], row['f1_score']),
                        xytext=(row['support'] * x_factor, row['f1_score'] + y_offset),
                        arrowprops=dict(arrowstyle='->', lw=0.8, color='gray', alpha=1),
                        fontsize=12, alpha=1.0,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='none', alpha=0.8),
                        ha=ha, va='center')

    plt.tight_layout()
    plt.savefig(eval_dir / 'f1_vs_support_scatter_trend_en.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.show()