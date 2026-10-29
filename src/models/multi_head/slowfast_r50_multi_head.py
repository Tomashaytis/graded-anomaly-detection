"""
SlowFast model with two heads for video classification.
"""

import torch
import torch.nn as nn

from typing import List, Tuple

from src.models.base_model import BaseVideoModel


class SlowFastR50MultiHead(BaseVideoModel):
    """SlowFast model with ResNet-50 backbone and two heads for video classification."""

    def _create_backbone(self) -> nn.Module:
        """
        Create SlowFast R50 backbone from pytorchvideo hub.

        :return: SlowFast ResNet-50 model.
        """
        model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'slowfast_r50',
            pretrained=False
        )
        return model

    def _get_head_in_features(self) -> int:
        """
        Get input features from the head projection layer.

        :return: Number of input features for classifier head.
        :raises AttributeError: If classifier head cannot be found.
        """
        if hasattr(self._backbone, 'blocks') and hasattr(self._backbone.blocks[-1], 'proj'):
            return self._backbone.blocks[-1].proj.in_features
        raise AttributeError("Cannot find classifier head in SlowFast model")

    def _get_finetune_block_indices(self) -> list:
        """
        Get indices of ResNet blocks to unfreeze (res4, res5).

        :return: List of block indices [4, 5].
        """
        return [4, 5]

    def _replace_head(self) -> None:
        """
        Override: skip default single-head replacement.
        Heads are created in _set_head.
        """
        self._set_head(None)

    def _set_head(self, new_head: nn.Module | None) -> None:
        """
        Create two heads for multi-head model.

        :param new_head: Ignored (kept for compatibility with BaseVideoModel).
        """
        in_features = self._get_head_in_features()

        self._backbone.blocks[-1].proj = nn.Identity()

        # Head 1: severity regressor
        self.head1 = nn.Sequential(
            nn.Dropout(self._dropout),
            nn.Linear(in_features, 1)
        )

        # Head 2: action classifier
        self.head2 = nn.Sequential(
            nn.Dropout(self._dropout),
            nn.Linear(in_features, self._num_classes)
        )

    def _unfreeze_blocks(self, block_indices: list) -> None:
        """
        Unfreeze specified ResNet blocks by their indices.

        :param block_indices: List of block indices to unfreeze.
        """
        for idx in block_indices:
            if idx < len(self._backbone.blocks):
                for param in self._backbone.blocks[idx].parameters():
                    param.requires_grad = True

    def _unfreeze_head(self) -> None:
        """Unfreeze head parameters for training."""
        for param in self.head1.parameters():
            param.requires_grad = True
        for param in self.head2.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor | List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass handling both single tensor and list of tensors.

        :param x: Single tensor (B, C, T, H, W) or list of [slow_path, fast_path]
        :return: Logits tensor of shape (B, num_classes)
        """
        if isinstance(x, torch.Tensor):
            x = self._split_to_pathways(x)
        features = self._backbone(x)
        logits1 = self.head1(features)
        logits2 = self.head2(features)
        return logits1, logits2

    @staticmethod
    def _split_to_pathways(x: torch.Tensor) -> list:
        """
        Split single video tensor into slow and fast pathways.

        Slow pathway: fewer frames (e.g., every 8th frame)
        Fast pathway: all frames or every other frame

        :param x: Video tensor of shape (B, C, T, H, W)
        :return: List of [slow_pathway, fast_pathway]
        """
        batch_size, channels, num_frames, height, width = x.shape

        alpha = 4
        slow_indices = torch.arange(0, num_frames, alpha, device=x.device)
        slow_pathway = x[:, :, slow_indices, :, :]  # (B, C, T/4, H, W)

        fast_pathway = x  # (B, C, T, H, W)

        return [slow_pathway, fast_pathway]