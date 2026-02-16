"""Model / backbone catalogues, metadata, and recommended configurations."""

from typing import Any, Dict


def list_available_models() -> Dict[str, list]:
    """Return every supported model grouped by category."""
    return {
        "baseline_smp": [
            "unet", "unet++", "fpn", "linknet", "pspnet", "pan",
        ],
        "cnn_sota_smp": [
            "deeplabv3", "deeplabv3+", "manet",
        ],
        "transformer_sota_hf": [
            "segformer_hf", "mask2former", "upernet_swin", "dpt",
        ],
        "specialized_smp": [
            "unet_resnet", "unet_efficientnet", "unet_mit",
        ],
    }


def list_available_backbones() -> Dict[str, list]:
    """Return every supported backbone / checkpoint grouped by family."""
    return {
        "smp_lightweight": [
            "mobilenet_v2", "mobileone_s0", "mobileone_s1",
            "efficientnet-b0", "timm-efficientnet-b0", "resnet18", "resnet34",
        ],
        "smp_standard": [
            "resnet50", "resnet101", "efficientnet-b3",
            "timm-efficientnet-b3", "densenet121", "se_resnet50",
        ],
        "smp_heavy": [
            "resnet152", "efficientnet-b7", "timm-efficientnet-b7",
            "timm-efficientnet-b8", "densenet201", "resnext101_32x8d",
            "resnext101_32x16d", "se_resnet152", "se_resnext101_32x4d",
        ],
        "smp_transformer": [f"mit_b{i}" for i in range(6)],
        "hf_segformer": [
            f"nvidia/segformer-b{i}-finetuned-ade-512-512" for i in range(5)
        ] + ["nvidia/segformer-b5-finetuned-ade-640-640"],
        "hf_mask2former": [
            "facebook/mask2former-swin-tiny-cityscapes-semantic",
            "facebook/mask2former-swin-base-cityscapes-semantic",
            "facebook/mask2former-swin-large-cityscapes-semantic",
            "facebook/mask2former-swin-base-coco-panoptic",
            "facebook/mask2former-swin-large-coco-panoptic",
        ],
        "hf_upernet": [
            "openmmlab/upernet-swin-tiny",
            "openmmlab/upernet-swin-large",
        ],
        "hf_dpt": [
            "Intel/dpt-large-ade",
            "Intel/dpt-hybrid-midas",
        ],
    }


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Return metadata (type, source, year, complexity, param range) for a model."""
    key = model_name.lower().replace("_", "").replace("-", "")
    _INFO = {
        "unet":        {"type": "baseline",  "source": "SMP", "year": 2015, "complexity": "low-medium",  "params_range": "7M-35M"},
        "unet++":      {"type": "baseline",  "source": "SMP", "year": 2018, "complexity": "medium",      "params_range": "9M-45M"},
        "fpn":         {"type": "baseline",  "source": "SMP", "year": 2017, "complexity": "medium",      "params_range": "11M-40M"},
        "linknet":     {"type": "baseline",  "source": "SMP", "year": 2017, "complexity": "low",         "params_range": "11M-25M"},
        "pspnet":      {"type": "baseline",  "source": "SMP", "year": 2017, "complexity": "medium-high", "params_range": "46M-70M"},
        "pan":         {"type": "baseline",  "source": "SMP", "year": 2018, "complexity": "medium-high", "params_range": "23M-50M"},
        "deeplabv3":   {"type": "cnn_sota",  "source": "SMP", "year": 2017, "complexity": "medium-high", "params_range": "39M-68M"},
        "deeplabv3+":  {"type": "cnn_sota",  "source": "SMP", "year": 2018, "complexity": "high",        "params_range": "41M-70M"},
        "manet":       {"type": "cnn_sota",  "source": "SMP", "year": 2020, "complexity": "high",        "params_range": "25M-55M"},
        "segformerhf": {"type": "transformer_sota", "source": "HuggingFace", "year": 2021, "complexity": "medium-high", "params_range": "3.7M-84M"},
        "mask2former": {"type": "transformer_sota", "source": "HuggingFace", "year": 2022, "complexity": "very high",   "params_range": "47M-223M"},
        "upernetswin": {"type": "transformer_sota", "source": "HuggingFace", "year": 2021, "complexity": "very high",   "params_range": "200M+"},
        "dpt":         {"type": "transformer_sota", "source": "HuggingFace", "year": 2021, "complexity": "high",        "params_range": "120M-340M"},
    }
    return _INFO.get(key, {"type": "unknown"})


def get_recommended_configs(dataset_type: str = "cityscapes") -> Dict[str, Dict]:
    """Return a handful of battle-tested (model, backbone, num_classes) combos."""
    num_classes = {
        "cityscapes": 19, "ade20k": 150, "pascal_voc": 21,
        "coco": 133, "custom": 10,
    }.get(dataset_type, 10)

    return {
        "fast_baseline": {
            "model_name": "unet",
            "backbone_name": "resnet34",
            "out_channels": num_classes,
        },
        "standard_baseline": {
            "model_name": "unet",
            "backbone_name": "resnet50",
            "out_channels": num_classes,
        },
        "cnn_sota": {
            "model_name": "deeplabv3+",
            "backbone_name": "resnet101",
            "out_channels": num_classes,
        },
        "transformer_efficient": {
            "model_name": "segformer_hf",
            "backbone_name": "nvidia/segformer-b2-finetuned-ade-512-512",
            "out_channels": num_classes,
        },
        "transformer_sota": {
            "model_name": "mask2former",
            "backbone_name": "facebook/mask2former-swin-large-cityscapes-semantic",
            "out_channels": num_classes,
        },
    }
