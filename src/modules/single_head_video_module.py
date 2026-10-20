"""
Single Head Video Module for Lightning.
"""

import lightning as L
import torch
import torch.nn as nn


class SingleHeadVideoModule(L.LightningModule):
    """LightningModule for video action recognition with one head."""

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
    ):
        """
        Initialize SingleHeadVideoModule.

        :param model: PyTorch model.
        :param num_classes: Number of classes.
        :param learning_rate: Learning rate for optimizer.
        :param weight_decay: Weight decay for optimizer.
        :param optimizer_name: Name of optimizer ('adamw' or 'sgd').
        :param scheduler_name: Name of scheduler ('cosine', 'step', 'plateau').
        :param num_epochs: Number of training epochs.
        :param label_smoothing: Label smoothing factor.
        :param class_counts: Class counts.
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.num_epochs = num_epochs

        if class_counts is not None:
            total_samples = sum(class_counts.values())
            class_weights = torch.zeros(num_classes)
            for class_id, count in class_counts.items():
                if 0 <= class_id < num_classes:
                    class_weights[class_id] = total_samples / (num_classes * count)

            class_weights = class_weights / class_weights.mean()
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=class_weights,
                ignore_index=-1,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                ignore_index=-1,
            )

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through the model.

        :param x: Input video tensor of shape (B, C, T, H, W).
        :return: Logits.
        """
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """
        Lighting shared step for train/test/val stage

        :param batch: Video batch.
        :return: Loss value.
        """
        videos, labels, _ = batch

        # Model forward pass
        outputs = self(videos)

        # Loss
        loss = self.criterion(outputs, labels)

        # Metrics
        with torch.no_grad():
            mask = labels != -1
            if mask.any():
                _, preds = torch.max(outputs[mask], 1)
                acc = (preds == labels[mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=outputs.device)

        # Logging
        self.log(f'{stage}_loss', loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=True)
        self.log(f'{stage}_acc', acc, on_step=(stage == 'train'), on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lighting train step.

        :param batch: Video batch.
        :param batch_idx: Video batch idx.
        :return: Loss value.
        """
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lighting validation step.

        :param batch: Video batch.
        :param batch_idx: Video batch idx.
        :return: Loss value.
        """
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lighting test step.

        :param batch: Video batch.
        :param batch_idx: Video batch idx.
        :return: Loss value.
        """
        return self._shared_step(batch, 'test')

    def configure_optimizers(self) -> dict:
        """
        Configure optimizer and scheduler.

        :return: Optimizer and scheduler.
        """
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer_name}')

        if self.scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs
            )
        elif self.scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.scheduler_name == 'plateau':
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
                    'monitor': 'val_acc'
                }
            }
        else:
            raise ValueError(f'Unknown scheduler: {self.scheduler_name}')

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}