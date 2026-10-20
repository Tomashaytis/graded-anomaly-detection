"""
Model registry and factory for video classification models.
"""

from .base_model import BaseVideoModel
from src.models.single_head import I3DR50, SlowFastR50, X3D
from src.models.multi_head import I3DR50MultiHead, SlowFastR50MultiHead, X3DMultiHead


MODEL_REGISTRY = {
    'i3d_r50': (I3DR50, None),
    'slowfast_r50': (SlowFastR50, None),
    'x3d_xs': (X3D, 'x3d_xs'),
    'x3d_s': (X3D, 'x3d_s'),
    'x3d_m': (X3D, 'x3d_m'),
    'x3d_l': (X3D, 'x3d_l'),

    'i3d_r50_mh': (I3DR50MultiHead, None),
    'slowfast_r50_mh': (SlowFastR50MultiHead, None),
    'x3d_xs_mh': (X3DMultiHead, 'x3d_xs'),
    'x3d_s_mh': (X3DMultiHead, 'x3d_s'),
    'x3d_m_mh': (X3DMultiHead, 'x3d_m'),
    'x3d_l_mh': (X3DMultiHead, 'x3d_l'),
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

    model_class, variant = MODEL_REGISTRY[model_name]

    kwargs = {
        'num_classes': num_classes,
        'dropout': dropout,
        'pretrained': pretrained,
        'weights_path': weights_path,
    }
    if variant is not None:
        kwargs['variant'] = variant

    model = model_class(**kwargs)

    if freeze_backbone:
        model.freeze_backbone()

    return model


def list_available_models() -> dict:
    """
    List all available models with their class names.

    :return: Dictionary mapping model names to their class names.
    """
    return { name: cls.__name__ for name, (cls, _) in MODEL_REGISTRY.items() }