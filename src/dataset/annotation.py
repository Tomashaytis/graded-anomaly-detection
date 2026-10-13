"""
Annotation Generator Module

This module provides functionality to generate annotations for video action recognition datasets
with hierarchical folder structure. It creates CSV annotation files for training, validation,
and testing splits of video data.

Typical usage example:
    generator = AnnotationGenerator(Path("dataset"))
    generator.create_dataset(Path("annotations"), train_ratio=0.7, val_ratio=0.15)
"""

import re
import random
import pandas as pd

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class AnnotationGenerator:
    """
    A class for generating annotation files from hierarchical video dataset structure.

    The class scans a dataset directory organized in the following structure:
        dataset/
            category/
                action_class/
                    source_class/
                        video1.mp4
                        video2.mp4

    It generates CSV annotation files containing paths to videos and their corresponding
    labels, which can be used for training 3D-CNN and other video classification modules.
    """

    def __init__(self, dataset_path: Path = None):
        """
        Initialize the AnnotationGenerator with a dataset path.

        :param dataset_path: Path to the root directory of the video dataset.
        :raises ValueError: If the provided dataset path does not exist.
        """
        if dataset_path is not None:
            self._dataset_path = Path(dataset_path)

            if not self._dataset_path.exists():
                raise ValueError(f"Directory {dataset_path} does not exist!")

        self._video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        self._annotations = pd.DataFrame()
        self._id2label = {}

    @property
    def id2label(self) -> Dict[int, str]:
        """
        :return: id to label dictionary.
        """
        return self._id2label

    def get_video_files(self) -> List[Path]:
        """
        Recursively collect all video files from the dataset directory.

        :return: List of paths to all video files found in the dataset directory.
        """
        video_files = []

        for ext in self._video_extensions:
            video_files.extend(self._dataset_path.rglob(f'*{ext}'))

        return video_files

    def parse_directory_structure(self, video_path: Path) -> Tuple[str, str, int, str]:
        """
        Extract information from the directory structure.

        The expected structure is: category/action_class/source_class/video_file.mp4

        :param video_path: Full path to the video file.
        :return: Tuple containing (category, action_class, source_class).
        """
        rel_path = video_path.relative_to(self._dataset_path)
        parts = rel_path.parts

        category = parts[0] if len(parts) > 0 else "unknown"
        action_class_with_level = parts[1] if len(parts) > 1 else "unknown"
        m = re.match(r'^(.*?)\s*\((\d)\)\s*$', action_class_with_level)
        action_class, level = m.group(1), int(m.group(2))
        source_class = parts[2] if len(parts) > 2 else "unknown"

        return category, action_class, level, source_class

    def generate_annotations(self):
        """
        Generate annotations for all video files.
        """
        video_files = self.get_video_files()
        data = []

        all_labels = set()
        temp_data = []

        for video_file in video_files:
            try:
                category, action_class, level, source_class = self.parse_directory_structure(video_file)
                rel_path = str(video_file)
                temp_data.append([rel_path, action_class, level, category, source_class])
                if action_class != 'normal':
                    all_labels.add(action_class)
            except Exception as e:
                print(f"Processing error for video '{video_file}': {e}")

        label2id = { label: idx for idx, label in enumerate(sorted(all_labels)) }

        for video_path, label, level, category, source_class in temp_data:
            label_int = -1 if label == 'normal' else label2id[label]
            data.append([video_path, label, label_int, level, category, source_class, None])

        self._annotations = pd.DataFrame(
            data,
            columns=['video_path', 'label', 'label_id', 'level', 'category', 'source_class', 'subset']
        )

        self._id2label = { v: k for k, v in label2id.items() }

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                      stratify_by_class: bool = True, random_seed: int = 42):
        """
        Split annotations into train, validation, and test sets.

        :param train_ratio: Proportion of data for training set (default: 0.7).
        :param val_ratio: Proportion of data for validation set (default: 0.15).
        :param stratify_by_class: If True, maintains class proportions across splits (default: True).
        :param random_seed: Random seed for reproducibility (default: 42).
        :raises ValueError: If sum of train_ratio and val_ratio exceeds 1.0.
        """
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError(f"Sum of train_ratio ({train_ratio}) + val_ratio ({val_ratio}) > 1.0")

        if stratify_by_class:
            train_indices = []
            val_indices = []
            test_indices = []

            for label in self._annotations['label'].unique():
                label_mask = self._annotations['label'] == label
                label_indices = self._annotations[label_mask].index.tolist()

                random.seed(random_seed)
                random.shuffle(label_indices)

                n = len(label_indices)
                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)

                train_indices.extend(label_indices[:train_end])
                val_indices.extend(label_indices[train_end:val_end])
                test_indices.extend(label_indices[val_end:])

            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)

            self._annotations.loc[train_indices, 'subset'] = 'train'
            self._annotations.loc[val_indices, 'subset'] = 'validation'
            self._annotations.loc[test_indices, 'subset'] = 'test'

        else:
            indices = self._annotations.index.tolist()
            random.seed(random_seed)
            random.shuffle(indices)

            n = len(indices)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            self._annotations.loc[indices[:train_end], 'subset'] = 'train'
            self._annotations.loc[indices[train_end:val_end], 'subset'] = 'validation'
            self._annotations.loc[indices[val_end:], 'subset'] = 'test'

    def save_annotations(
            self,
            output_path: Path,
            subset: Optional[str] = None,
    ):
        """
        Save annotations to a CSV file.

        :param output_path: Path where the CSV file will be saved.
        :param subset: Subset name to assign to the subset column (e.g., 'train', 'validation', 'test').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if subset is not None:
            self._annotations[self._annotations['subset'] == subset].to_csv(output_path, index=False)
        else:
            self._annotations.to_csv(output_path, index=False)

    def load_annotations(self, file_path: Path):
        """
        Load existing annotations from a CSV file.

        :param file_path: Path to the CSV annotation file.
        :raises FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file {file_path} does not exist!")

        self._annotations = pd.read_csv(file_path)
        print(f"Loaded {len(self._annotations)} annotations from {file_path}")

    def get_statistics(self) -> Dict:
        """
        Calculate statistics from annotations.

        :return: Dictionary containing statistics for total videos, categories, classes, sources, and subsets.
        """
        stats = {
            'total_videos': len(self._annotations),
            'categories': defaultdict(int),
            'classes': defaultdict(int),
            'sources': defaultdict(int),
            'subsets': defaultdict(int)
        }

        for _, row in self._annotations.iterrows():
            stats['categories'][row['category']] += 1
            stats['classes'][row['label']] += 1
            stats['sources'][row['source_class']] += 1

            subset_value = row['subset']
            if pd.isna(subset_value) or subset_value is None:
                subset_value = 'unassigned'
            stats['subsets'][subset_value] += 1

        return stats

    def print_statistics(self):
        """
        Print dataset statistics in human-readable format with hierarchical structure.
        """
        stats = self.get_statistics()

        levels = {
            0: '-',
            1: '!',
            2: '!!',
            3: '!!!'
        }

        print(f"\nDataset statistics:")
        print(f"Total videos: {stats['total_videos']}")

        category_class_counts = defaultdict(lambda: defaultdict(int))
        for _, row in self._annotations.iterrows():
            key = (row['label'], row['level'])
            category_class_counts[row['category']][key] += 1

        category_totals = {cat: sum(classes.values()) for cat, classes in category_class_counts.items()}

        print(f"\nCategories ({len(category_totals)}):")
        for category, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category} ({total})")
            for (cls, level), count in sorted(
                    category_class_counts[category].items(),
                    key=lambda x: x[1],
                    reverse=True
            ):
                print(f"    [{levels[level]}] {cls} ({count})")

        print(f"\nSources ({len(stats['sources'])}):")
        for src, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {src}: {count}")

        dataset_total = defaultdict(int)
        dataset_unique_sources = defaultdict(set)

        for _, row in self._annotations.iterrows():
            src = row['source_class']
            m = re.search(r'\(([^)]+)\)$', src)
            dataset = m.group(1) if m else 'unknown'

            dataset_total[dataset] += 1
            dataset_unique_sources[dataset].add(src)

        print(f"\nSources by dataset:")
        for dataset, total in sorted(dataset_total.items(), key=lambda x: x[1], reverse=True):
            unique = len(dataset_unique_sources[dataset])
            print(f"  {dataset}: {total} videos, {unique} unique sources")

        print(f"\nSplit distribution:")
        for subset, count in sorted(stats['subsets'].items()):
            print(f"  {subset}: {count}")

    def create_dataset(
            self,
            output_dir: Path,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            stratify_by_class: bool = True,
            seed: int = 42
    ):
        """
        Main method to create the dataset annotations.

        This method performs three steps:
        1. Generates annotations from the dataset directory.
        2. Splits annotations into train/validation/test sets.
        3. Saves the splits to CSV files.

        :param output_dir: Directory where annotation CSV files will be saved.
        :param train_ratio: Proportion of data for training set (default: 0.7).
        :param val_ratio: Proportion of data for validation set (default: 0.15).
        :param stratify_by_class: If True, maintains class proportions across splits (default: True).
        :param seed: Random seed for reproducibility (default: 42).
        """
        print(f"Scanning directory {self._dataset_path}...")

        self.generate_annotations()

        if len(self._annotations) == 0:
            print("No video files found!")
            return

        print(f"Found {len(self._annotations)} video files")
        self.print_statistics()

        test_ratio = 1.0 - train_ratio - val_ratio

        print(f"\nSplitting data (train={train_ratio}, val={val_ratio}, test={test_ratio:.2f})...")

        self.split_dataset(train_ratio, val_ratio, stratify_by_class, seed)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.save_annotations(output_dir / "annotations_train.csv", subset="train")
        self.save_annotations(output_dir / "annotations_val.csv", subset="validation")
        self.save_annotations(output_dir / "annotations_test.csv", subset="test")
        self.save_annotations(output_dir / "annotations_all.csv")

        print(f"\nAnnotations created in {output_dir}:")
        print(f"  Train: {len(self._annotations[self._annotations['subset'] == 'train'])} videos")
        print(f"  Test: {len(self._annotations[self._annotations['subset'] == 'test'])} videos")
        print(f"  Validation: {len(self._annotations[self._annotations['subset'] == 'validation'])} videos")
        print(f"  All data: {len(self._annotations)} videos")
