"""
Base class for all video classification models.
"""

import torch
import warnings
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseVideoModel(nn.Module, ABC):
    """Abstract base class for video classification models."""

    def __init__(
            self,
            num_classes: int,
            dropout: float = 0.5,
            pretrained: bool = False,
            weights_path: Optional[str] = None
    ):
        """
        Initialize base video classification model.

        :param num_classes: Number of output classes.
        :param dropout: Dropout probability for classifier head.
        :param pretrained: Whether to load pretrained weights from torch.hub.
        :param weights_path: Path to custom weights file.
        """
        super().__init__()
        self._num_classes = num_classes
        self._dropout = dropout
        self._pretrained = pretrained
        self._weights_path = weights_path

        # Build the model
        self._backbone = self._create_backbone()
        self._load_weights()
        self._replace_head()

    @abstractmethod
    def _create_backbone(self) -> nn.Module:
        """
        Create and return the backbone architecture.

        :return: Backbone model from pytorchvideo hub or custom source.
        """
        pass

    @abstractmethod
    def _get_head_in_features(self) -> int:
        """
        Get the number of input features for classifier head.

        :return: Input features dimension extracted from backbone's last layer.
        """
        pass

    @abstractmethod
    def _get_finetune_block_indices(self) -> list:
        """
        Get indices of blocks to unfreeze during fine-tuning.

        :return: List of block indices (e.g. [3, 4, 5] for ResNet-based models).
        """
        pass

    def _load_weights(self) -> None:
        """Load pretrained weights from file if specified in config."""
        if self._weights_path is not None:
            print(f"Loading weights from {self._weights_path}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                checkpoint = torch.load(
                    self._weights_path,
                    map_location='cpu',
                    weights_only=False
                )
            # Support different checkpoint formats
            state_dict = checkpoint.get(
                'model_state',
                checkpoint.get('state_dict', checkpoint)
            )
            self._backbone.load_state_dict(state_dict, strict=False)

    def _replace_head(self) -> None:
        """Replace classifier head with new one adapted to target classes."""
        in_features = self._get_head_in_features()
        print(f"Replacing classifier head: {in_features} -> {self._num_classes} classes")

        if self._dropout > 0:
            new_head = nn.Sequential(
                nn.Dropout(self._dropout),
                nn.Linear(in_features, self._num_classes)
            )
        else:
            new_head = nn.Linear(in_features, self._num_classes)

        self._set_head(new_head)

    @abstractmethod
    def _set_head(self, new_head: nn.Module) -> None:
        """
        Set the new classifier head on the model.

        :param new_head: New classifier head module (Linear or Sequential).
        """
        pass

    def freeze_backbone(self) -> None:
        """Freeze backbone layers for transfer learning, unfreezing only specified blocks and head."""
        print(f"Freezing backbone for {self.__class__.__name__}...")

        # Step 1: Freeze all parameters
        for param in self._backbone.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze specified blocks for fine-tuning
        finetune_indices = self._get_finetune_block_indices()
        self._unfreeze_blocks(finetune_indices)

        # Step 3: Always unfreeze the classifier head
        self._unfreeze_head()

        # Log trainable parameters summary
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    @abstractmethod
    def _unfreeze_blocks(self, block_indices: list) -> None:
        """
        Unfreeze specific blocks by their indices.

        :param block_indices: List of block indices to unfreeze for fine-tuning.
        """
        pass

    @abstractmethod
    def _unfreeze_head(self) -> None:
        """Unfreeze classifier head parameters to allow training."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        :param x: Input video tensor of shape (B, C, T, H, W).
        :return: Logits tensor of shape (B, num_classes).
        """
        return self._backbone(x)

    def get_info(self) -> Dict[str, Any]:
        """
        Get model information summary.

        :return: Dictionary with model name, parameter counts, and configuration.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'name': self.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_classes': self._num_classes,
            'dropout': self._dropout,
        }