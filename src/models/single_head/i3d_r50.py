"""
I3D with ResNet-50 backbone model.
"""

import torch
import torch.nn as nn

from src.models.base_model import BaseVideoModel


class I3DR50(BaseVideoModel):
    """I3D model with ResNet-50 backbone for video classification."""

    def _create_backbone(self) -> nn.Module:
        """
        Create I3D R50 backbone from pytorchvideo hub.

        :return: I3D ResNet-50 model.
        """
        model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'i3d_r50',
            pretrained=False
        )
        return model

    def _get_head_in_features(self) -> int:
        """
        Get input features from the last block's projection layer.

        :return: Number of input features for classifier head.
        """
        return self._backbone.blocks[-1].proj.in_features

    def _get_finetune_block_indices(self) -> list:
        """
        Get indices of ResNet blocks to unfreeze (res4, res5).

        :return: List of block indices [4, 5].
        """
        return [4, 5]

    def _set_head(self, new_head: nn.Module) -> None:
        """
        Replace the projection layer in the last ResNet block.

        :param new_head: New classifier head module.
        """
        self._backbone.blocks[-1].proj = new_head

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
        """Unfreeze classifier head parameters for training."""
        for param in self._backbone.blocks[-1].proj.parameters():
            param.requires_grad = True