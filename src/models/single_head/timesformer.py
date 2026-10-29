"""
TimeSformer single-head for video classification.
"""

import torch
import torch.nn as nn

from transformers import TimesformerModel

from src.models.base_model import BaseVideoModel


class TimeSformer(BaseVideoModel):
    """
    TimeSformer backbone from Hugging Face with a single classifier head.
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

    def _set_head(self, new_head: nn.Module) -> None:
        """
        Store the single classifier head.

        :param new_head: New classifier head module (Linear or Sequential).
        """
        self.head = new_head

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
        """Unfreeze classifier head parameters for training."""
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Input tensor of shape (B, C, T, H, W) from VideoDataset.
        :return: Logits tensor of shape (B, num_classes).
        """
        if x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        outputs = self._backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]

        return self.head(features)