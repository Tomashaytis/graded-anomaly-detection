"""
X3D model variants (XS, S, M, L) for video classification.
"""

import torch
import torch.nn as nn

from src.models.base_model import BaseVideoModel


class X3D(BaseVideoModel):
    """X3D model for video classification.

    Supports variants: 'x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l'.
    """

    VARIANTS = {
        'x3d_xs': 'x3d_xs',
        'x3d_s': 'x3d_s',
        'x3d_m': 'x3d_m',
        'x3d_l': 'x3d_l',
    }

    FINETUNE_INDICES = {
        'x3d_xs': [3, 4],
        'x3d_s': [4, 5],
        'x3d_m': [4, 5],
        'x3d_l': [4, 5],
    }

    def __init__(self, variant: str = 'x3d_m', **kwargs):
        """
        Initialize X3D model with specified variant.

        :param variant: X3D variant name.
        :param kwargs: Additional arguments passed to BaseVideoModel.
        """
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown X3D variant: {variant}. "
                f"Available: {list(self.VARIANTS.keys())}"
            )
        self.variant = variant
        super().__init__(**kwargs)

    def _create_backbone(self) -> nn.Module:
        """
        Create X3D backbone from pytorchvideo hub.

        :return: X3D model with specified variant.
        """
        model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            self.variant,
            pretrained=self._pretrained
        )
        return model

    def _get_head_in_features(self) -> int:
        """
        Get input features from the last block's head layer.

        :return: Number of input features for classifier head.
        :raises AttributeError: If classifier head cannot be found.
        """
        last_block = self._backbone.blocks[-1]
        if hasattr(last_block, 'proj'):
            return last_block.proj.in_features
        elif hasattr(last_block, 'fc'):
            return last_block.fc.in_features
        raise AttributeError("Cannot find classifier head in X3D model")

    def _get_finetune_block_indices(self) -> list:
        """
        Get indices of X3D stages to unfreeze based on variant.

        :return: List of stage indices.
        """
        return self.FINETUNE_INDICES.get(self.variant, [4, 5])

    def _set_head(self, new_head: nn.Module) -> None:
        """
        Replace the classifier head in the last X3D block.

        :param new_head: New classifier head module.
        """
        last_block = self._backbone.blocks[-1]
        if hasattr(last_block, 'proj'):
            last_block.proj = new_head
        elif hasattr(last_block, 'fc'):
            last_block.fc = new_head

    def _unfreeze_blocks(self, block_indices: list) -> None:
        """
        Unfreeze specified X3D stages by their indices.

        :param block_indices: List of stage indices to unfreeze.
        """
        for stage_idx in block_indices:
            if stage_idx < len(self._backbone.blocks):
                for param in self._backbone.blocks[stage_idx].parameters():
                    param.requires_grad = True

    def _unfreeze_head(self) -> None:
        """Unfreeze classifier head parameters for training."""
        last_block = self._backbone.blocks[-1]
        head = getattr(last_block, 'proj', None) or getattr(last_block, 'fc', None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True