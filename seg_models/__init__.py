"""seg_models — unified semantic segmentation model zoo.

Quick start::

    from seg_models import create_segmentation_model

    model = create_segmentation_model(
        model_name="unet",
        backbone_name="resnet50",
        in_channels=3,
        out_channels=19,
    )
"""

from .catalogue import (
    get_model_info,
    get_recommended_configs,
    list_available_backbones,
    list_available_models,
)
from .factory import create_segmentation_model

__all__ = [
    "create_segmentation_model",
    "list_available_models",
    "list_available_backbones",
    "get_model_info",
    "get_recommended_configs",
]
