"""
Video Module for Lightning.
"""

import lightning as L
import torch
import torch.nn as nn

from typing import Tuple

from src.modules.uncertainty_weighting import UncertaintyWeighting


class VideoModule(L.LightningModule):
    """LightningModule for video action recognition."""

    def __init__(
            self,
            model: nn.Module,
            num_classes: int,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            optimizer_name: str = "adamw",
            scheduler_name: str = "cosine",
            num_epochs: int = 20,
            label_smoothing: float = 0.1,
            class_counts: dict = None,
            multi_head_mode: bool = False,
    ):
        """
        Initialize VideoModule.

        :param model: PyTorch model with two heads (returns tuple).
        :param num_classes: Number of classes for head 2.
        :param learning_rate: Learning rate for optimizer.
        :param weight_decay: Weight decay for optimizer.
        :param optimizer_name: Name of optimizer ('adamw' or 'sgd').
        :param scheduler_name: Name of scheduler ('cosine', 'step', 'plateau').
        :param num_epochs: Number of training epochs.
        :param label_smoothing: Label smoothing factor.
        :param class_counts: Class counts.
        :param multi_head_mode: If true, enable multihead mode.
        """
        super().__init__()
        self._model = model
        self._num_classes = num_classes
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._optimizer_name = optimizer_name
        self._scheduler_name = scheduler_name
        self._num_epochs = num_epochs
        self._multi_head_mode = multi_head_mode

        if self._multi_head_mode:
            # Loss for head 1 (BCE)
            self._loss_fn1 = nn.BCEWithLogitsLoss()

            # Loss for head 2 (CE)
            if class_counts is not None:
                total_samples = sum(class_counts.values())
                class_weights = torch.zeros(num_classes)
                for class_id, count in class_counts.items():
                    if 0 <= class_id < num_classes:
                        class_weights[class_id] = total_samples / (num_classes * count)
                class_weights = class_weights / class_weights.mean()
                self._loss_fn2 = nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing,
                    weight=class_weights,
                    ignore_index=-1,
                )
            else:
                self._loss_fn2 = nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing,
                    ignore_index=-1,
                )

            # Uncertainty weighting
            self._uncertainty_weighting = UncertaintyWeighting(num_tasks=2)
        else:
            if class_counts is not None:
                total_samples = sum(class_counts.values())
                class_weights = torch.zeros(num_classes)
                for class_id, count in class_counts.items():
                    if 0 <= class_id < num_classes:
                        class_weights[class_id] = total_samples / (num_classes * count)

                class_weights = class_weights / class_weights.mean()
                self._loss = nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing,
                    weight=class_weights,
                    ignore_index=-1,
                )
            else:
                self._loss = nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing,
                    ignore_index=-1,
                )

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        :param x: Input video tensor of shape (B, C, T, H, W).
        :return: If multi_head_mode enabled - Tuple of (severity_logits, action_logits). Else - action logits.
        """
        return self._model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """
        Lighting shared step for train/test/val stage

        :param batch: Video batch.
        :return: Total loss value.
        """
        videos, labels, level = batch

        if self._multi_head_mode:
            # Model forward pass
            logits1, logits2 = self(videos)

            # Loss for head 1 (severity regressor)
            loss1 = self._loss_fn1(logits1.squeeze(-1), level)

            # Loss for head 2 (action classifier)
            loss2 = self._loss_fn2(logits2, labels)

            # Uncertainty weighting forward pass
            total_loss = self._uncertainty_weighting([loss1, loss2])

            # Metrics
            with torch.no_grad():
                # Head 1 metrics
                preds1 = (torch.sigmoid(logits1.squeeze(-1)) > 0.5).float()
                target1 = (level > 0.5).float()
                acc1 = (preds1 == target1).float().mean()

                # Head 2 metrics
                mask = labels != -1
                if mask.any():
                    _, preds2 = torch.max(logits2[mask], 1)
                    acc2 = (preds2 == labels[mask]).float().mean()
                else:
                    acc2 = torch.tensor(0.0, device=logits2.device)

            # Logging
            self.log(f'{stage}_loss_mh', total_loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=True)
            self.log(f'{stage}_loss1_mh', loss1, on_step=False, on_epoch=True)
            self.log(f'{stage}_loss2_mh', loss2, on_step=False, on_epoch=True)
            self.log(f'{stage}_acc1_mh', acc1, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_acc2_mh', acc2, on_step=False, on_epoch=True, prog_bar=True)
            return total_loss
        else:
            # Model forward pass
            outputs = self(videos)

            # Loss
            loss = self._loss(outputs, labels)

            # Metrics
            with torch.no_grad():
                mask = labels != -1
                if mask.any():
                    _, preds = torch.max(outputs[mask], 1)
                    acc = (preds == labels[mask]).float().mean()
                else:
                    acc = torch.tensor(0.0, device=outputs.device)

            # Logging
            self.log(f'{stage}_loss_sh', loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=True)
            self.log(f'{stage}_acc_sh', acc, on_step=(stage == 'train'), on_epoch=True, prog_bar=True)

            return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lighting train step.

        :param batch: Video batch.
        :param batch_idx: Video batch idx.
        :return: Total loss value.
        """
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lighting validation step.

        :param batch: Video batch.
        :param batch_idx: Video batch idx.
        :return: Total loss value.
        """
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lighting test step.

        :param batch: Video batch.
        :param batch_idx: Video batch idx.
        :return: Total loss value.
        """
        return self._shared_step(batch, 'test')

    def configure_optimizers(self) -> dict:
        """
        Configure optimizer and scheduler.

        :return: Optimizer and scheduler.
        """
        params = list(self._model.parameters())
        if self._multi_head_mode:
            params += list(self._uncertainty_weighting.parameters())

        if self._optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=self._learning_rate,
                weight_decay=self._weight_decay
            )
        elif self._optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=self._learning_rate,
                momentum=0.9,
                weight_decay=self._weight_decay
            )
        else:
            raise ValueError(f'Unknown optimizer: {self._optimizer_name}')

        if self._scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self._num_epochs
            )
        elif self._scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self._scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=3,
                factor=0.5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_acc2_mh' if self._multi_head_mode else 'val_acc_sh'
                }
            }
        else:
            raise ValueError(f'Unknown scheduler: {self._scheduler_name}')

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}