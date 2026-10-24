"""
Main CLI Module
"""

import os
import sys
import json

import numpy as np
import yaml
import typer
import torch
import warnings
import pandas as pd
import lightning as L

from pathlib import Path
from datetime import datetime
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)

from src.utils import *
from src.modules import VideoModule
from src.callbacks import MetricsCallback
from src.datamodules import VideoDataModule
from src.dataset import AnnotationGenerator, LMDBCreator
from src.models import create_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CONFIG_PATH = os.path.join('configs', 'default.yaml')

app = typer.Typer()


@app.command()
def create_lmdb(
        config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to YAML config file"),
):
    """Create LMDB for train, validate and test dataset."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Train LMDB
    if config['lmdb']['create_train']:
        print('Creating LMDB for train dataset...')
        annotations_df = pd.read_csv(config['paths']['annotations_train'])
        annotations_df = annotations_df.sample(frac=1).reset_index(drop=True)
        video_paths = annotations_df['video_path'].tolist()
        labels = annotations_df['label_id'].tolist()
        levels = annotations_df['level'].tolist()

        creator = LMDBCreator(Path(config['paths']['train_lmdb']), map_size=config['lmdb']['train_size'])
        _ = creator.create(
            video_paths=video_paths,
            labels=labels,
            levels=levels,
            max_frames=config['data']['video_max_frames'],
            resize_size=(config['data']['video_frame_size'], config['data']['video_frame_size'])
        )

    # Val LMDB
    if config['lmdb']['create_val']:
        print('Creating LMDB for val dataset...')
        annotations_df = pd.read_csv(config['paths']['annotations_val'])
        annotations_df = annotations_df.sample(frac=1).reset_index(drop=True)
        video_paths = annotations_df['video_path'].tolist()
        labels = annotations_df['label_id'].tolist()
        levels = annotations_df['level'].tolist()

        creator = LMDBCreator(Path(config['paths']['val_lmdb']), map_size=config['lmdb']['val_size'])
        _ = creator.create(
            video_paths=video_paths,
            labels=labels,
            levels=levels,
            max_frames=config['data']['video_max_frames'],
            resize_size=(config['data']['video_frame_size'], config['data']['video_frame_size'])
        )

    # Test LMDB
    if config['lmdb']['create_test']:
        print('Creating LMDB for test dataset...')
        annotations_df = pd.read_csv(config['paths']['annotations_test'])
        annotations_df = annotations_df.sample(frac=1).reset_index(drop=True)
        video_paths = annotations_df['video_path'].tolist()
        labels = annotations_df['label_id'].tolist()
        levels = annotations_df['level'].tolist()

        creator = LMDBCreator(Path(config['paths']['test_lmdb']), map_size=config['lmdb']['test_size'])
        _ = creator.create(
            video_paths=video_paths,
            labels=labels,
            levels=levels,
            max_frames=config['data']['video_max_frames'],
            resize_size=(config['data']['video_frame_size'], config['data']['video_frame_size'])
        )


@app.command()
def annotate(
        config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to YAML config file"),
):
    """Annotate video dataset."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    generator = AnnotationGenerator(Path(config['paths']['dataset']))
    generator.create_dataset(
        output_dir=Path(config['paths']['dataset']),
        train_ratio=config['annotation']['train_ratio'],
        val_ratio=config['annotation']['val_ratio'],
        stratify_by_class=config['annotation']['stratify_by_class'],
        seed=config['annotation']['seed']
    )
    with open(Path(config['paths']['dataset']) / 'id2label.json', 'w', encoding='utf-8') as fp:
        json.dump(generator.id2label, fp, indent=4, ensure_ascii=False)


@app.command()
def dataset_overview(
        config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to YAML config file"),
):
    """Show video dataset overview."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    generator = AnnotationGenerator()
    generator.load_annotations(Path(config['paths']['annotations_all']))
    generator.print_statistics()


@app.command()
def train(
        config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to YAML config file"),
):
    """Train video classification model using PyTorch Lightning."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get model preset configuration
    preset_name = config['model']['preset']
    if preset_name not in config.get('model_presets', {}):
        raise ValueError(
            f"Unknown model preset: {preset_name}. "
            f"Available: {list(config.get('model_presets', {}).keys())}"
        )

    model_config = config['model_presets'][preset_name]
    print(f"Using model preset: {preset_name}")
    print(f"Model config: {model_config}")

    # Set random seed
    L.seed_everything(config['training']['seed'], workers=True)

    # Create output directory with model name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(config['paths']['output_dir']) / f"{preset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Override batch_size and accumulate_grad_batches from preset
    batch_size = model_config.get('batch_size', config['training']['batch_size'])
    accumulate_grad_batches = model_config.get(
        'accumulate_grad_batches',
        config['training']['accumulate_grad_batches']
    )

    # DataModule
    datamodule = VideoDataModule(
        train_lmdb_path=config['paths']['train_lmdb'],
        val_lmdb_path=config['paths']['val_lmdb'],
        test_lmdb_path=config['paths']['test_lmdb'],
        annotations_train_path=config['paths']['annotations_train'],
        batch_size=batch_size,
        num_workers=config['training']['num_workers'],
        video_frame_size=config['data']['video_frame_size'],
        video_max_frames=config['data']['video_max_frames'],
        num_optional_augmentations=config['training']['num_optional_augmentations'],
    )

    # Load annotations for num_classes
    annotations_train_df = pd.read_csv(config['paths']['annotations_train'])
    num_classes = len(
        annotations_train_df[annotations_train_df['label_id'] != -1]['label_id'].unique()
    )
    class_counts = annotations_train_df['label_id'].value_counts().to_dict()

    # Create model using factory with preset config
    model = create_model(
        model_name=model_config['name'],
        num_classes=num_classes,
        dropout=model_config['dropout'],
        weights_path=model_config.get('weights_path'),
        freeze_backbone=model_config['freeze_backbone'],
    )

    # Print model info
    model_info = model.get_info()
    print(f"\n{'=' * 60}")
    print(f"Model: {model_info['name']}")
    print(f"Total params: {model_info['total_params']:,}")
    print(f"Trainable params: {model_info['trainable_params']:,}")
    print(f"Number of classes: {model_info['num_classes']}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"{'=' * 60}\n")

    # Lightning Module
    lightning_module = VideoModule(
        model=model,
        num_classes=num_classes,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer_name=config['training']['optimizer'],
        scheduler_name=config['training']['scheduler'],
        num_epochs=config['training']['num_epochs'],
        label_smoothing=config['training']['label_smoothing'],
        class_counts=class_counts,
        multi_head_mode=model_config['multi_head_mode'],
    )

    # Callbacks
    if model_config['multi_head_mode']:
        callbacks = [
            ModelCheckpoint(
                dirpath=output_dir / 'checkpoints',
                filename='best-{epoch:02d}-{val_acc2_mh:.3f}',
                monitor='val_acc2_mh',
                mode='max',
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor='val_acc2_mh',
                patience=config['early_stopping']['patience'],
                min_delta=config['early_stopping']['min_delta'],
                mode='max',
            ),
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar(refresh_rate=1)
        ]
    else:
        callbacks = [
            ModelCheckpoint(
                dirpath=output_dir / 'checkpoints',
                filename='best-{epoch:02d}-{val_acc_sh:.3f}',
                monitor='val_acc_sh',
                mode='max',
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor='val_acc_sh',
                patience=config['early_stopping']['patience'],
                min_delta=config['early_stopping']['min_delta'],
                mode='max',
            ),
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar(refresh_rate=1)
        ]

    # Logger
    loggers = []
    if config['logging']['tensorboard']:
        loggers.append(L.pytorch.loggers.TensorBoardLogger(save_dir=output_dir, name='tensorboard'))
    if config['logging']['csv']:
        loggers.append(L.pytorch.loggers.CSVLogger(save_dir=output_dir, name='csv'))

    # Trainer
    trainer = L.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        deterministic=False,
        precision='16-mixed',
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Train
    trainer.fit(lightning_module, datamodule=datamodule)

    # Test (optional)
    if config.get('test', False):
        trainer.test(lightning_module, datamodule=datamodule, ckpt_path='best')

    print(f"\n{'=' * 80}")
    print(f"Training Complete")
    print(f"Model preset: {preset_name}")
    print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
    print(f"TensorBoard logs: {output_dir / 'tensorboard'}")
    print(f"{'=' * 80}")


@app.command()
def evaluate(
        config_path: str = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to YAML config file"),
):
    """Evaluate trained model on test set with comprehensive metrics."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get model preset configuration
    preset_name = config['model']['preset']
    if preset_name not in config.get('model_presets', {}):
        raise ValueError(
            f"Unknown model preset: {preset_name}. "
            f"Available: {list(config.get('model_presets', {}).keys())}"
        )

    model_config = config['model_presets'][preset_name]
    checkpoint_path = Path(config['paths']['checkpoint'])

    L.seed_everything(config['training']['seed'], workers=True)

    mode_alias = 'mh' if model_config['multi_head_mode'] else 'sh'
    eval_dir = Path(config['paths']['output_dir']) / 'results' / mode_alias / f'eval_{preset_name}_{checkpoint_path.stem}'
    eval_dir.mkdir(parents=True, exist_ok=True)

    # DataModule
    datamodule = VideoDataModule(
        train_lmdb_path=config['paths']['train_lmdb'],
        val_lmdb_path=config['paths']['val_lmdb'],
        test_lmdb_path=config['paths']['test_lmdb'],
        annotations_train_path=config['paths']['annotations_train'],
        batch_size=model_config.get('batch_size', config['training']['batch_size']),
        num_workers=config['training']['num_workers'],
        video_frame_size=config['data']['video_frame_size'],
        video_max_frames=config['data']['video_max_frames'],
        num_optional_augmentations=config['training']['num_optional_augmentations'],
    )

    # Load annotations for class information
    annotations_train_df = pd.read_csv(config['paths']['annotations_train'])
    num_classes = len(
        annotations_train_df[annotations_train_df['label_id'] != -1]['label_id'].unique()
    )
    class_counts = annotations_train_df['label_id'].value_counts().to_dict()

    with open(config['paths']['id2label'], 'r', encoding='utf-8') as fp:
        class_names = {int(k): v for k, v in json.load(fp).items()}

    # Create model using factory with preset config
    print(f"Creating model architecture: {model_config['name']}...")
    model = create_model(
        model_name=model_config['name'],
        num_classes=num_classes,
        dropout=model_config['dropout'],
        weights_path=None,
        freeze_backbone=False,
    )

    # LightningModule
    lightning_module = VideoModule(
        model=model,
        num_classes=num_classes,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer_name=config['training']['optimizer'],
        scheduler_name=config['training']['scheduler'],
        num_epochs=config['training']['num_epochs'],
        label_smoothing=config['training']['label_smoothing'],
        class_counts=class_counts,
        multi_head_mode=model_config['multi_head_mode'],
    )

    print(f"Loading weights from checkpoint: {checkpoint_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('state_dict', checkpoint)
    missing_keys, unexpected_keys = lightning_module.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")

    # Callbacks including metrics callback
    metrics_callback = MetricsCallback(num_classes, class_names)
    callbacks = [
        RichProgressBar(refresh_rate=1),
        metrics_callback,
    ]

    # Loggers
    loggers = []
    if config['logging']['tensorboard']:
        loggers.append(L.pytorch.loggers.TensorBoardLogger(save_dir=eval_dir, name='tensorboard'))
    if config['logging']['csv']:
        loggers.append(L.pytorch.loggers.CSVLogger(save_dir=eval_dir, name='csv'))

    # Trainer
    trainer = L.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        precision='16-mixed',
    )

    # Single pass evaluation
    print("Starting evaluation on test set...")
    results = trainer.test(lightning_module, datamodule=datamodule)
    results_dict = results[0] if results else {}

    # Compute detailed metrics from collected predictions
    detailed_metrics = metrics_callback.compute_metrics()

    all_results = {**results_dict, **detailed_metrics}

    if model_config['multi_head_mode']:
        per_class_key = 'head2_per_class_metrics'
        cm_key = 'head2_confusion_matrix_normalized'
    else:
        per_class_key = 'per_class_metrics'
        cm_key = 'confusion_matrix_normalized'

    # Save per-class metrics
    if detailed_metrics.get(per_class_key):
        per_class_df = pd.DataFrame(detailed_metrics[per_class_key])
        per_class_df.to_csv(eval_dir / 'per_class_metrics.csv', index=False)

        plot_per_class_accuracy(per_class_df, eval_dir)
        plot_per_class_f1(per_class_df, eval_dir)
        plot_metrics_distribution(per_class_df, eval_dir)
        plot_support_vs_f1(per_class_df, eval_dir)
        plot_metrics_heatmap(per_class_df, eval_dir)

        if detailed_metrics.get(cm_key):
            plot_confusion_matrix(
                np.array(detailed_metrics[cm_key]),
                class_names,
                num_classes,
                eval_dir
            )

    # Sanitizing
    def sanitize(obj):
        if isinstance(obj, float):
            if np.isnan(obj):
                return "nan"
            if np.isinf(obj):
                return "inf" if obj > 0 else "-inf"
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    all_results = sanitize(all_results)

    # Save all results
    with open(eval_dir / 'test_results.yaml', 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False, allow_unicode=True)

    # Print results
    print_results(all_results, eval_dir)


if __name__ == "__main__":
    app()