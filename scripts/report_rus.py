import os
import yaml
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import pi
from pathlib import Path
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess

DEFAULT_CONFIG_PATH = os.path.join('configs', 'default.yaml')
EVAL_FOLDER = 'eval_i3d_r50_best-epoch=08-val_acc=0.787'
MODEL_NAME = 'I3D-8x8-R50'


CATEGORY_TRANSLATION = {
    'normal': 'Обычные действия',
    'overcoming obstacles': 'Преодоление препятствий',
    'using items and weapons': 'Использование предметов и оружия',
    'violence': 'Насилие',
    'strange': 'Странные действия',
    'computer usage': 'Использование компьютера',
    'vandalism': 'Вандализм',
    'inadequate condition': 'Неадекватное состояние',
    'emergency': 'Чрезвычайная ситуация'
}

CLASS_TRANSLATION = {
    "abseiling": "спуск по канату",
    "abuse": "жестокое обращение",
    "arguing": "спор",
    "arson": "поджог",
    "assault": "нападение",
    "assembling computer": "сборка компьютера",
    "breaking glass": "разбивание стекла",
    "burglary": "кража со взломом",
    "carrying weight": "переноска тяжестей",
    "chiseling": "работа зубилом",
    "cleaning windows": "мытьё окон",
    "climbing a rope": "лазание по канату",
    "climbing ladder": "лазание по лестнице",
    "climbing tree": "лазание по дереву",
    "closing door": "закрывание двери",
    "collecting garbage": "сбор мусора",
    "computer use": "использование компьютера",
    "counting money": "пересчёт денег",
    "crawling": "ползание",
    "dinning": "приём пищи",
    "drop kicking": "пинок в падении",
    "explosion": "взрыв",
    "extinguishing fire": "тушение пожара",
    "faceplanting": "падение лицом вниз",
    "falling off chair": "падение со стула",
    "fighting": "драка",
    "folding clothes": "складывание одежды",
    "graffiti": "граффити",
    "headbutting": "удар головой",
    "huddling": "обсуждение в группе",
    "hugging": "объятия",
    "jogging": "бег трусцой",
    "jumping": "прыжки",
    "kicking": "пинок",
    "lighting fire": "разведение огня",
    "lock picking": "вскрытие замков",
    "looking at phone": "просмотр телефона",
    "moving furniture": "перемещение мебели",
    "opening door": "открывание двери",
    "packing": "упаковка вещей",
    "parkour": "элементы паркура",
    "polishing furniture": "полировка мебели",
    "pushing cart": "толкание тележки",
    "reading": "чтение",
    "robbery": "ограбление",
    "rock climbing": "скалолазание",
    "sawing wood": "распиловка дерева",
    "shaking hands": "рукопожатие",
    "shooting": "стрельба",
    "shredding paper": "измельчение бумаги",
    "sipping cup": "питьё из чашки",
    "slapping": "пощёчина",
    "smashing": "разбивание объектов",
    "smoking": "курение",
    "staring": "пристальный взгляд",
    "sweeping floor": "подметание пола",
    "sword fighting": "фехтование",
    "tackling": "захват",
    "taking photo": "фотографирование",
    "talking on cell phone": "разговор по телефону",
    "texting": "набор текста в телефоне",
    "throwing something": "бросание предметов",
    "using a power drill": "работа дрелью",
    "using a sledge hammer": "работа кувалдой",
    "using a wrench": "работа гаечным ключом",
    "using circular saw": "работа циркулярной пилой",
    "using remote controller": "использование контроллера/пульта",
    "vandalism": "вандализм",
    "waiting in line": "ожидание в очереди",
    "washing hands": "мытьё рук",
    "welding": "сварка",
    "wrestling": "борьба",
    "writing": "письмо"
}

def translate_class_name(class_name_en: str) -> str:
    """Переводит английское название класса на русский"""
    return CLASS_TRANSLATION.get(class_name_en, class_name_en)


random.seed(48)


def get_annotation_style():
    """Автоматически определяет направление подписи"""
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
    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    eval_dir = Path(config['paths']['output_dir']) / 'results' / EVAL_FOLDER

    annotations_df = pd.read_csv(config['paths']['annotations_train'])
    label_category_mapping = annotations_df[['label_id', 'category']].drop_duplicates()
    mapping_dict = dict(zip(label_category_mapping['label_id'], label_category_mapping['category']))

    metrics_df = pd.read_csv(eval_dir / 'per_class_metrics.csv')

    metrics_df['category'] = metrics_df['class_id'].map(mapping_dict)

    # Переводим названия классов на русский
    metrics_df['class_name_ru'] = metrics_df['class_name'].apply(translate_class_name)

    missing_categories = metrics_df[metrics_df['category'].isna()]
    if not missing_categories.empty:
        print("Внимание! Для следующих классов не найдены категории:")
        print(missing_categories[['class_id', 'class_name_ru']])

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

    # Переводим названия категорий
    category_metrics['category_ru'] = category_metrics['category'].map(CATEGORY_TRANSLATION)
    category_metrics = category_metrics.sort_values('avg_f1', ascending=False)

    print("\n" + "="*80)
    print("АГРЕГИРОВАННЫЕ МЕТРИКИ ПО КАТЕГОРИЯМ (взвешенные по support)")
    print("="*80)
    print(category_metrics[['category_ru', 'num_classes', 'total_support', 'avg_precision', 'avg_recall', 'avg_f1']].to_string(index=False))

    output_path = eval_dir / 'category_metrics.csv'
    category_metrics.to_csv(output_path, index=False)
    print(f"\nРезультат сохранён: {output_path}")

    print("\n" + "="*80)
    print("ТОП-5 ЛУЧШИХ КЛАССОВ ПО F1")
    print("="*80)
    top5 = metrics_df.nlargest(5, 'f1_score')[['class_name_ru', 'category', 'f1_score', 'support']].copy()
    top5['category_ru'] = top5['category'].map(CATEGORY_TRANSLATION)
    print(top5[['class_name_ru', 'category_ru', 'f1_score', 'support']].to_string(index=False))

    print("\n" + "="*80)
    print("ТОП-5 ХУДШИХ КЛАССОВ ПО F1")
    print("="*80)
    bottom5 = metrics_df.nsmallest(5, 'f1_score')[['class_name_ru', 'category', 'f1_score', 'support']].copy()
    bottom5['category_ru'] = bottom5['category'].map(CATEGORY_TRANSLATION)
    print(bottom5[['class_name_ru', 'category_ru', 'f1_score', 'support']].to_string(index=False))

    # Двойной график F1-score
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # График (а): Средний F1 по категориям
    category_metrics_sorted = category_metrics.sort_values('avg_f1', ascending=True)

    colors = plt.cm.Blues(category_metrics_sorted['avg_f1'] / category_metrics_sorted['avg_f1'].max())
    bars = ax1.barh(category_metrics_sorted['category_ru'], category_metrics_sorted['avg_f1'],
                    color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Средний F1-score', fontsize=11, labelpad=8)
    ax1.set_ylabel('Категория', fontsize=11, labelpad=8)
    ax1.set_title('(а) Средний F1-score по категориям действий', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar, val in zip(bars, category_metrics_sorted['avg_f1']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='left', fontsize=9)

    # График (б): Лучшие и худшие классы
    top5_ru = metrics_df.nlargest(5, 'f1_score')[['class_name_ru', 'f1_score']].copy().iloc[::-1]
    bottom5_ru = metrics_df.nsmallest(5, 'f1_score')[['class_name_ru', 'f1_score']].copy()

    top5_ru['group'] = '5 лучших классов'
    bottom5_ru['group'] = '5 худших классов'

    combined = pd.concat([bottom5_ru, top5_ru])
    combined['color'] = combined['group'].map({'5 лучших классов': '#2E8B57', '5 худших классов': '#CD5C5C'})

    bars2 = ax2.barh(combined['class_name_ru'], combined['f1_score'],
                     color=combined['color'], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('F1-score', fontsize=11, labelpad=8)
    ax2.set_ylabel('Класс', fontsize=11, labelpad=8)
    ax2.set_title('(б) Лучшие и худшие классы', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for bar, val in zip(bars2, combined['f1_score']):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='left', fontsize=9)

    legend_elements = [Patch(facecolor='#2E8B57', label='5 лучших классов'),
                       Patch(facecolor='#CD5C5C', label='5 худших классов')]
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=False, fontsize=9)

    fig.suptitle(f'Анализ качества работы модели: {MODEL_NAME}', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(eval_dir / 'performance_summary_ru.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # F1 vs число примеров
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8

    fig3, ax = plt.subplots(figsize=(19, 7))
    fig3.patch.set_facecolor('white')

    # Данные для графика
    plot_df = metrics_df[['class_name_ru', 'support', 'f1_score']].copy()
    plot_df = plot_df.sort_values('support')

    # Точечный график
    sc = ax.scatter(plot_df['support'], plot_df['f1_score'],
                    c=plot_df['f1_score'], cmap='RdYlGn',
                    s=80, alpha=0.7, edgecolor='black', linewidth=0.8,
                    vmin=0, vmax=1)

    # Линия тренда (LOWESS)
    lowess_result = lowess(plot_df['f1_score'], plot_df['support'],
                           frac=0.6,
                           return_sorted=True)
    ax.plot(lowess_result[:, 0], lowess_result[:, 1],
            color='#D62728', linewidth=3, label='LOWESS тренд')

    # Настройка осей
    ax.set_xscale('log')
    tick_positions = [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500]
    tick_labels = ['10', '15', '20', '30', '40', '50', '60', '80', '100', '150', '200', '300', '400', '500']

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, ha='center')
    ax.set_xlabel('Число примеров в классе (тестовая выборка)', fontsize=14, labelpad=10)
    ax.set_ylabel('F1-score', fontsize=14, labelpad=10)
    ax.set_title(f'{MODEL_NAME}: Зависимость F1-score от объёма обучающих примеров',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linestyle='--', which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Цветовая шкала
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('F1-score', fontsize=14, rotation=270, labelpad=15)

    # Легенда
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=10)

    # Добавляем горизонтальную линию на уровне среднего F1
    mean_f1 = metrics_df['f1_score'].mean()
    ax.axhline(y=mean_f1, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Средний F1-score = {mean_f1:.3f}')
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=14)

    interesting_classes = [
        'драка',
        'поджог',
        'кража со взломом',
        'использование компьютера',
        'фотографирование',
        'просмотр телефона',
        'разговор по телефону',
        'открывание двери',
        'сборка компьютера',
        'стрельба',
        'курение',
        'нападение',
        'закрывание двери',
        'взрыв',
        'вандализм',
    ]

    for class_name in interesting_classes:
        class_data = plot_df[plot_df['class_name_ru'] == class_name]
        if not class_data.empty:
            row = class_data.iloc[0]
            x_factor, y_offset, ha = get_annotation_style()

            ax.annotate(row['class_name_ru'],
                        xy=(row['support'], row['f1_score']),
                        xytext=(row['support'] * x_factor, row['f1_score'] + y_offset),
                        arrowprops=dict(arrowstyle='->', lw=0.8, color='gray', alpha=1),
                        fontsize=12, alpha=1.0,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='none', alpha=0.8),
                        ha=ha, va='center')

    plt.tight_layout()
    plt.savefig(eval_dir / 'f1_vs_support_scatter_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Добавьте диагностику перед interesting_classes
    print("\n" + "=" * 80)
    print("ПРОВЕРКА КЛАССА 'разговор по телефону':")
    print("=" * 80)

    # Ищем точное совпадение
    print(list(plot_df['class_name_ru']))
