"""
Model registry and factory for video classification models.
"""

from .base_model import BaseVideoModel
from .i3d_r50 import I3DR50
from .x3d import X3D
from .slowfast_r50 import SlowFastR50


MODEL_REGISTRY = {
    'i3d_r50': I3DR50,
    'x3d_xs': X3D,
    'x3d_s': X3D,
    'x3d_m': X3D,
    'x3d_l': X3D,
    'slow_r50': SlowFastR50,
    'slowfast_r50': SlowFastR50,
}


def create_model(
        model_name: str,
        num_classes: int,
        dropout: float = 0.5,
        pretrained: bool = False,
        weights_path: str = None,
        freeze_backbone: bool = True,
) -> BaseVideoModel:
    """
    Factory function to create a video classification model.

    :param model_name: Name of the model (e.g. 'i3d_r50', 'x3d_m').
    :param num_classes: Number of output classes.
    :param dropout: Dropout probability for classifier head.
    :param pretrained: Whether to use pretrained weights from torch.hub.
    :param weights_path: Path to custom weights file (overrides pretrained).
    :param freeze_backbone: Whether to freeze backbone layers for transfer learning.
    :return: Instance of BaseVideoModel ready for training or evaluation.
    :raises ValueError: If model_name is not found in MODEL_REGISTRY.
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]

    # Handle X3D variants which require variant parameter
    if model_class == X3D:
        model = model_class(
            variant=model_name,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            weights_path=weights_path
        )
    else:
        model = model_class(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            weights_path=weights_path
        )

    # Freeze backbone if requested
    if freeze_backbone:
        model.freeze_backbone()

    return model


def list_available_models() -> dict:
    """
    List all available models with their class names.

    :return: Dictionary mapping model names to their class names.
    """
    return {name: cls.__name__ for name, cls in MODEL_REGISTRY.items()}