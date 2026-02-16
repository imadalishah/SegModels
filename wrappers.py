"""Lightweight wrappers that handle padding, cropping, and output normalisation
so that every model in the zoo accepts arbitrary (B, C, H, W) inputs and
returns a consistent ``{"logits": Tensor[B, num_classes, H, W]}`` output.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Padding / cropping
# ---------------------------------------------------------------------------

class PadCropWrapper(nn.Module):
    """Pad H, W to the nearest multiple of *divisor* (optionally forcing a
    square), forward through the wrapped model, then crop back to the
    original spatial size.

    Parameters
    ----------
    model : nn.Module
        The model to wrap.
    divisor : int
        Both H and W of the padded tensor will be divisible by this value.
    square : bool
        If ``True`` the shorter side is first expanded to match the longer
        side *before* rounding up to *divisor*.  Required for models whose
        internals assume a square patch grid (e.g. DPT / ViT).
    """

    def __init__(self, model: nn.Module, divisor: int = 1, square: bool = False):
        super().__init__()
        self.model = model
        self.divisor = divisor
        self.square = square

    def forward(self, x: torch.Tensor, **kwargs):
        _, _, orig_h, orig_w = x.shape
        h, w = orig_h, orig_w

        if self.square:
            h = w = max(h, w)

        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        target_h, target_w = h + pad_h, w + pad_w
        total_pad_h = target_h - orig_h
        total_pad_w = target_w - orig_w

        if total_pad_h or total_pad_w:
            x = F.pad(x, (0, total_pad_w, 0, total_pad_h), mode="reflect")

        out = self.model(x, **kwargs)

        # Crop back to original spatial dims
        if isinstance(out, torch.Tensor):
            return out[:, :, :orig_h, :orig_w]
        if isinstance(out, dict):
            return {
                k: (
                    v[:, :, :orig_h, :orig_w]
                    if isinstance(v, torch.Tensor) and v.dim() == 4
                    else v
                )
                for k, v in out.items()
            }
        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            out.logits = out.logits[:, :, :orig_h, :orig_w]
        return out


# ---------------------------------------------------------------------------
# HuggingFace output normalisation
# ---------------------------------------------------------------------------

class HFUpsampleWrapper(nn.Module):
    """Bilinearly upsample the ``logits`` field of a HuggingFace model output
    so that it matches the input spatial resolution, and return a plain dict.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, **kwargs):
        _, _, h, w = x.shape
        out = self.model(x, **kwargs)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False,
            )
        return {"logits": logits}


class Mask2FormerSemanticWrapper(nn.Module):
    """Convert Mask2Former query-based outputs into dense per-pixel
    ``[B, num_classes, H, W]`` logits.
    """

    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        out = self.model(pixel_values=pixel_values, **kwargs)
        cls_logits = out.class_queries_logits        # [B, Q, num_labels+1]
        mask_logits = out.masks_queries_logits        # [B, Q, Hp, Wp]

        cls_probs = cls_logits.softmax(dim=-1)[..., :-1]   # drop no-object
        mask_probs = mask_logits.sigmoid()
        sem_logits = torch.einsum("bqc,bqhw->bchw", cls_probs, mask_probs)

        _, _, h, w = pixel_values.shape
        if sem_logits.shape[-2:] != (h, w):
            sem_logits = F.interpolate(
                sem_logits, size=(h, w), mode="bilinear", align_corners=False,
            )
        return {"logits": sem_logits}
