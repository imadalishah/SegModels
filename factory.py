"""Universal factory for semantic segmentation models.

Supports both `segmentation_models_pytorch` (SMP) architectures and
HuggingFace Transformers models behind a single ``create_segmentation_model``
entry-point.

Key behaviours
──────────────
* ``pretrained=False`` → SMP ``encoder_weights`` forced to ``None``;
  HF models instantiated from config (random weights).
* ``in_channels != 3`` → SMP: handled natively.  HF: ``config.num_channels``
  is set so the stem / patch-embed conv is created with the correct fan-in.
* Arbitrary H×W → Every model is wrapped so that inputs are padded to a
  safe resolution and outputs are cropped / interpolated back.
"""

from __future__ import annotations

import warnings
from typing import Optional

import segmentation_models_pytorch as smp
import torch.nn as nn

from .wrappers import (
    HFUpsampleWrapper,
    Mask2FormerSemanticWrapper,
    PadCropWrapper,
)

# Optional HuggingFace imports
try:
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        SegformerConfig,
        SegformerForSemanticSegmentation,
    )

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn(
        "HuggingFace transformers not available. "
        "Install with: pip install transformers"
    )


# ── SMP model look-up table ────────────────────────────────────────────────

_SMP_MAP: dict[str, type] = {
    "unet":              smp.Unet,
    "unet++":            smp.UnetPlusPlus,
    "unetplusplus":      smp.UnetPlusPlus,
    "fpn":               smp.FPN,
    "linknet":           smp.Linknet,
    "pspnet":            smp.PSPNet,
    "pan":               smp.PAN,
    "deeplabv3":         smp.DeepLabV3,
    "deeplabv3+":        smp.DeepLabV3Plus,
    "deeplabv3plus":     smp.DeepLabV3Plus,
    "manet":             smp.MAnet,
    # Specialised aliases — all map to Unet with a specific backbone
    "unetresnet":        smp.Unet,
    "unetefficientnet":  smp.Unet,
    "unetmit":           smp.Unet,
    "unetswin":          smp.Unet,
}


# ── Public factory ──────────────────────────────────────────────────────────

def create_segmentation_model(
    model_name: str,
    backbone_name: str = "resnet50",
    in_channels: int = 3,
    out_channels: int = 1,
    encoder_weights: str = "imagenet",
    activation: Optional[str] = None,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Instantiate any supported segmentation model.

    Returns a module whose forward pass accepts ``(B, C, H, W)`` tensors of
    *arbitrary* spatial size and produces either a plain ``Tensor`` or a dict
    with a ``"logits"`` key, both shaped ``(B, num_classes, H, W)``.
    """

    raw_name = model_name
    model_name = model_name.lower().strip().replace("_", "").replace("-", "")

    if not pretrained:
        encoder_weights = None

    # ── SMP models ──────────────────────────────────────────────────────
    if model_name in _SMP_MAP:
        cls = _SMP_MAP[model_name]
        model = cls(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=activation,
            **kwargs,
        )
        # PSPNet needs H,W divisible by 48 (output_stride 8 × pool_size 6)
        if model_name == "pspnet":
            model = PadCropWrapper(model, divisor=48)
        else:
            model = PadCropWrapper(model, divisor=32)
        return model

    # ── HuggingFace models ──────────────────────────────────────────────
    if not HF_AVAILABLE:
        raise ImportError(
            f"HuggingFace transformers is required for model \'{raw_name}\'. "
            "Run: pip install transformers"
        )

    # -- SegFormer -------------------------------------------------------
    if model_name == "segformerhf":
        if backbone_name == "resnet50":
            backbone_name = "nvidia/segformer-b3-finetuned-ade-512-512"

        config = SegformerConfig.from_pretrained(backbone_name)
        config.num_labels = out_channels
        config.num_channels = in_channels

        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                config=config,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
        else:
            model = SegformerForSemanticSegmentation(config)
        return HFUpsampleWrapper(model)

    # -- Mask2Former -----------------------------------------------------
    if model_name == "mask2former":
        if backbone_name == "resnet50":
            backbone_name = "facebook/mask2former-swin-large-cityscapes-semantic"

        config = Mask2FormerConfig.from_pretrained(backbone_name)
        config.num_labels = out_channels
        if hasattr(config, "backbone_config") and config.backbone_config is not None:
            config.backbone_config.num_channels = in_channels
        else:
            config.num_channels = in_channels

        if pretrained:
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                backbone_name,
                config=config,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
        else:
            model = Mask2FormerForUniversalSegmentation(config)
        return Mask2FormerSemanticWrapper(model, num_classes=out_channels)

    # -- UPerNet + Swin --------------------------------------------------
    if model_name in ("upernetswin", "upernet"):
        from transformers import UperNetConfig, UperNetForSemanticSegmentation

        if backbone_name == "resnet50":
            backbone_name = "openmmlab/upernet-swin-large"

        config = UperNetConfig.from_pretrained(backbone_name)
        config.num_labels = out_channels
        if hasattr(config, "backbone_config") and config.backbone_config is not None:
            config.backbone_config.num_channels = in_channels
        else:
            config.num_channels = in_channels

        if pretrained:
            model = UperNetForSemanticSegmentation.from_pretrained(
                backbone_name,
                config=config,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
        else:
            model = UperNetForSemanticSegmentation(config)
        return HFUpsampleWrapper(model)

    # -- DPT -------------------------------------------------------------
    if model_name == "dpt":
        from transformers import DPTConfig, DPTForSemanticSegmentation

        if backbone_name == "resnet50":
            backbone_name = "Intel/dpt-large-ade"

        config = DPTConfig.from_pretrained(backbone_name)
        config.num_labels = out_channels
        config.num_channels = in_channels

        if pretrained:
            model = DPTForSemanticSegmentation.from_pretrained(
                backbone_name,
                config=config,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
        else:
            model = DPTForSemanticSegmentation(config)
        # DPT\'s ViT backbone assumes a square patch grid → pad to square
        return HFUpsampleWrapper(PadCropWrapper(model, divisor=16, square=True))

    # ── Unknown model ───────────────────────────────────────────────────
    raise ValueError(
        f"Unknown model \'{raw_name}\'. Available: {list(_SMP_MAP.keys())} "
        "+ [segformer_hf, mask2former, upernet_swin, dpt]"
    )
