"""
VideoMAE with two heads for graded anomaly detection.
"""

import torch
import torch.nn as nn

from typing import Tuple
from transformers import VideoMAEModel, VideoMAEConfig

from src.models.base_model import BaseVideoModel


class VideoMAEMultiHead(BaseVideoModel):
    """
    VideoMAE backbone from Hugging Face with two heads.
    """

    def _create_backbone(self) -> nn.Module:
        """Load VideoMAE encoder (without classification head)."""
        if self._weights_path is None:
            config = VideoMAEConfig()
            model = VideoMAEModel(config)
        else:
            model = VideoMAEModel.from_pretrained(self._weights_path)

        for param in model.parameters():
            param.requires_grad = False
        return model

    def _get_head_in_features(self) -> int:
        """VideoMAE-base hidden size is 768."""
        return self._backbone.config.hidden_size

    def _replace_head(self) -> None:
        """Skip default replacement; heads are created in _set_head."""
        self._set_head(None)

    def _set_head(self, new_head: nn.Module | None) -> None:
        """
        Create two heads on top of pooled video features.

        :param new_head: Ignored (kept for compatibility with BaseVideoModel).
        """
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

        VideoMAE-base has 12 layers (0..11).
        Last 4 layers = [8, 9, 10, 11].
        """
        return [8, 9, 10, 11]

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

        :param x: Single tensor (B, C, T, H, W)
        :return: logits1 (severity), logits2 (action)
        """
        if x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 1, 3, 1, 1)
        x = (x - mean) / std

        # NaN check for inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[INPUT] nan={torch.isnan(x).sum()}, inf={torch.isinf(x).sum()}, "
                  f"min={x.min():.4f}, max={x.max():.4f}")

        outputs = self._backbone(pixel_values=x)

        features = outputs.last_hidden_state.mean(dim=1)

        # NaN check for features
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"[FEATURES] nan={torch.isnan(features).sum()}, inf={torch.isinf(features).sum()}, "
                  f"min={features.min():.4f}, max={features.max():.4f}")

        logits1 = self.head1(features)
        logits2 = self.head2(features)

        # NaN check for logits
        if torch.isnan(logits1).any() or torch.isinf(logits1).any():
            print(f"[LOGITS1] nan={torch.isnan(logits1).sum()}, min={logits1.min():.4f}, max={logits1.max():.4f}")
        if torch.isnan(logits2).any() or torch.isinf(logits2).any():
            print(f"[LOGITS2] nan={torch.isnan(logits2).sum()}, min={logits2.min():.4f}, max={logits2.max():.4f}")

        return logits1, logits2