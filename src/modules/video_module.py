"""
Video Module for Lighting.
"""

import lightning as L
import torch
import torch.nn as nn


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
    ):
        """
        Initialize VideoLightningModule.

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
                class_weights[class_id] = total_samples / (num_classes * count)

            class_weights = class_weights / class_weights.mean()

            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        # Log metrics (prog_bar=True shows in progress bar)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
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