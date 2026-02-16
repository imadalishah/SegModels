# SegModels
![Status](https://img.shields.io/badge/status-active_development-orange)
![Release](https://img.shields.io/badge/version-0.1.0--alpha-blue)

![Python Version](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)

![Stars](https://img.shields.io/github/stars/imadalishah/SegModels?style=social)
[![Model Inference Tests](https://github.com/imadalishah/SegModels/actions/workflows/tests.yml/badge.svg)](https://github.com/imadalishah/SegModels/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/imadalishah/SegModels)


> [!WARNING]
> **Work in Progress**: This library is currently in development.
> The API is subject to breaking changes, and some models may not be fully tested yet. 



A unified factory for semantic segmentation models — wrapping both
[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
(SMP) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
behind a single `create_segmentation_model()` call.

## Features

- **One entry-point** for 16+ architectures (UNet, FPN, DeepLabV3+, SegFormer, Mask2Former, UPerNet-Swin, DPT, …).
- **Arbitrary input size** — every model is wrapped so that non-standard H×W (and non-square) inputs just work.
- **Arbitrary input channels** — pass `in_channels=15` (or any value) and the stem convolution is adjusted automatically.
- **Pretrained / from-scratch** toggle via a single `pretrained` flag.
- **Consistent output** — all models return `(B, num_classes, H, W)` tensors (or a dict with a `"logits"` key at the same shape).

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
from seg_models import create_segmentation_model

model = create_segmentation_model(
    model_name="deeplabv3+",
    backbone_name="resnet101",
    in_channels=3, # or for HSI-Drive: 25
    out_channels=19,
    pretrained=True,
)

# Works with any spatial size
import torch
x = torch.randn(1, 3, 209, 416)  # or for HSI-Drive: 209x416
out = model(x)  # (1, 10, 209, 416)
```

## Supported models

| Category | Models | Source |
|---|---|---|
| CNN | UNet, UNet++, FPN, LinkNet, PSPNet, PAN, DeepLabV3, DeepLabV3+, MAnet | SMP |
| Transformer | SegFormer, Mask2Former, UPerNet-Swin, DPT | HuggingFace |

```python
from seg_models import list_available_models, list_available_backbones
print(list_available_models())
print(list_available_backbones())
```

## Running the smoke test

```bash
python -m tests.test_inference
```

## Project structure

```
seg_models/
├── __init__.py      # Public API re-exports
├── factory.py       # create_segmentation_model()
├── wrappers.py      # PadCropWrapper, HFUpsampleWrapper, Mask2FormerSemanticWrapper
├── dataloader.py
├── metrics.py
├── train.py.py
└── catalogue.py     # Model/backbone listings, metadata, recommended configs
tests/
├── __init__.py
└── test_inference.py
requirements.txt
README.md
```

