"""
TimeSformer with two heads for graded anomaly detection.
"""

import torch
import torch.nn as nn

from typing import Tuple
from transformers import TimesformerModel

from src.models.base_model import BaseVideoModel


class TimeSformerMultiHead(BaseVideoModel):
    """
    TimeSformer backbone from Hugging Face with two heads.
    """

    def _create_backbone(self) -> nn.Module:
        """Load TimeSformer encoder (without classification head)."""
        model = TimesformerModel.from_pretrained(self._weights_path)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _get_head_in_features(self) -> int:
        """TimeSformer-base hidden size is 768."""
        return self._backbone.config.hidden_size

    def _replace_head(self) -> None:
        """Skip default replacement; heads are created in _set_head."""
        self._set_head(None)

    def _set_head(self, new_head: nn.Module | None) -> None:
        """Create two heads on top of [CLS] token features."""
        in_features = self._get_head_in_features()

        # Head 1: severity regressor (sigmoid)
        self.head1 = nn.Sequential(
            nn.Dropout(self._dropout),
            nn.Linear(in_features, 1)
        )

        # Head 2: action classifier (softmax)
        self.head2 = nn.Sequential(
            nn.Dropout(self._dropout),
            nn.Linear(in_features, self._num_classes)
        )

    def _load_weights(self) -> None:
        """Skip: weights already loaded in _create_backbone via from_pretrained."""
        pass

    def _get_finetune_block_indices(self) -> list:
        """
        Indices of encoder layers to unfreeze.

        TimeSformer-base has 12 layers (0..11).
        Last 2 layers = [10, 11].
        """
        return [10, 11]

    def _unfreeze_blocks(self, block_indices: list) -> None:
        """
        Unfreeze specified encoder layers.

        :param block_indices: List of layer indices to unfreeze.
        """
        for idx in block_indices:
            if 0 <= idx < len(self._backbone.encoder.layer):
                for param in self._backbone.encoder.layer[idx].parameters():
                    param.requires_grad = True

    def _unfreeze_head(self) -> None:
        """Unfreeze head parameters for training."""
        for param in self.head1.parameters():
            param.requires_grad = True
        for param in self.head2.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: Input tensor of shape (B, T, C, H, W) — TimeSformer format.
        :return: Tuple (severity_logits, action_logits).
        """
        if x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        outputs = self._backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]

        logits1 = self.head1(features)
        logits2 = self.head2(features)
        return logits1, logits2