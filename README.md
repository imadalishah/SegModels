# SegModels
![Status](https://img.shields.io/badge/status-active_development-orange)
![Release](https://img.shields.io/badge/version-0.1.0--alpha-blue)

![Python Version](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c?logo=pytorch&logoColor=white)
[![HF Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Library-HF%20Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![SMP](https://img.shields.io/badge/Library-SMP-blue?logo=pytorch)](https://github.com/qubvel-org/segmentation_models.pytorch)

![Stars](https://img.shields.io/github/stars/imadalishah/SegModels?style=social)
[![Inference Tests](https://github.com/imadalishah/SegModels/actions/workflows/tests.yml/badge.svg)](https://github.com/imadalishah/SegModels/actions/workflows/tests.yml)
![License](https://img.shields.io/badge/License-MIT-blue?logo=mit)

A unified factory for semantic segmentation models — wrapping both
[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
(SMP) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
behind a single `create_segmentation_model()` call.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imadalishah/SegModels/blob/main/colab.ipynb)

## Features

- **One entry-point** for 16+ architectures (UNet, FPN, DeepLabV3+, SegFormer, Mask2Former, UPerNet-Swin, DPT, …).
- **Arbitrary input size** — every model is wrapped so that non-standard H×W (and non-square) inputs just work.
- **Arbitrary input channels** — pass `in_channels=15` (or any value) and the stem convolution is adjusted automatically.
- **Pretrained / from-scratch** toggle via a single `pretrained` flag.
- **Consistent output** — all models return `(B, num_classes, H, W)` tensors (or a dict with a `"logits"` key at the same shape).

## Supported models

| Category | Models | Source |
|---|---|---|
| CNN | UNet, UNet++, FPN, LinkNet, PSPNet, PAN, DeepLabV3, DeepLabV3+, MAnet | SMP |
| Transformer | SegFormer, Mask2Former, UPerNet-Swin, DPT | HuggingFace |

To inspect supported configurations programmatically:

```python
from seg_models import list_available_models, list_available_backbones

print(list_available_models())
print(list_available_backbones())

```

## Installation

```bash
git clone https://github.com/imadalishah/SegModels.git
cd SegModels
pip install -r requirements.txt
pip install -e .

```

## Quick start

```python
import torch
from seg_models import create_segmentation_model

model = create_segmentation_model(
    model_name="deeplabv3+",
    backbone_name="resnet101",
    in_channels=25, # or for Hyperspectral City: 128
    out_channels=19,
    pretrained=True,
)

# Works natively with custom, non-square spatial sizes
x = torch.randn(1, 25, 209, 416)  # Shape: (B, C, H, W)
out = model(x)                    # Shape: (1, 19, 209, 416)

```

## Advanced Usage

### Handling Outputs (Tensors vs. Dicts)

Depending on your downstream training loop framework, you can request a raw tensor or a Hugging Face-style dictionary output.

```python
# Default: Returns a raw Tensor of shape (B, out_channels, H, W)
logits = model(x) 

# Force a dictionary output for Hugging Face Trainer compatibility
model = create_segmentation_model(..., return_dict=True)
outputs = model(x)
logits = outputs["logits"]

```

---

## Important Gotchas & Behavior

* **Mask2Former Post-Processing:** Hugging Face's `Mask2Former` natively outputs class and mask queries rather than a standard pixel map. `SegModels` automatically wraps this architecture to interpolate and map queries into a standard semantic segmentation tensor, but keep in mind that peak memory consumption may spike for exceptionally high target resolutions.
* **Backbone Availability:** Not all backbones are cross-compatible between CNN (SMP) and Transformer (HF) architectures. Use `list_available_backbones(model_name="...")` to verify valid pairings.

---

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
├── train.py
└── catalogue.py     # Model/backbone listings, metadata, recommended configs
tests/
├── __init__.py
└── test_inference.py
requirements.txt
README.md

```

> [!WARNING]
> **Work in Progress**: This library is currently in active alpha development.
> Execution errors and breaking API changes may occur as edge-case model combinations/functionalities are thoroughly tested. Thank you for your patience!

## Contributing

Contributions are welcome! If you'd like to add support for a new Hugging Face architecture, optimize the wrapper layer, or fix an edge-case bug, please open an issue or submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

