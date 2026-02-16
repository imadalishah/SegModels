"""seg_models.metrics — Segmentation metrics computed on-the-fly.

All metrics operate on raw logits ``(B, C, H, W)`` and integer masks
``(B, H, W)`` and are designed to be accumulated over batches, then
finalised at epoch end.

Usage::

    from seg_models.metrics import MetricTracker

    tracker = MetricTracker(num_classes=19, device="cuda")
    for batch in dataloader:
        preds = model(batch["image"])
        tracker.update(preds, batch["mask"])

    results = tracker.compute()
    print(results)
    tracker.reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Functional helpers
# ═══════════════════════════════════════════════════════════════════════════


def _to_pred_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw model output to ``(pred, target)`` long tensors on the
    same device, both shaped ``(N,)`` with ignored pixels removed.
    """
    # Handle dict output from HF wrappers
    if isinstance(logits, dict):
        logits = logits["logits"]
    if hasattr(logits, "logits"):
        logits = logits.logits

    # Argmax → class indices
    if logits.dim() == 4 and logits.shape[1] > 1:
        pred = logits.argmax(dim=1)  # (B, H, W)
    elif logits.dim() == 4 and logits.shape[1] == 1:
        pred = (logits.squeeze(1) > 0).long()
    else:
        pred = logits

    pred = pred.view(-1)
    target = target.view(-1)

    # Remove ignored pixels
    valid = target != ignore_index
    return pred[valid], target[valid]


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Overall pixel accuracy."""
    return (pred == target).sum().item() / max(target.numel(), 1)


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute a ``(num_classes, num_classes)`` confusion matrix.

    ``cm[i, j]`` = number of pixels with true class *i* predicted as class *j*.
    """
    mask = (target >= 0) & (target < num_classes)
    indices = target[mask] * num_classes + pred[mask]
    cm = torch.bincount(indices, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes).float()


def iou_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """Per-class IoU from a confusion matrix."""
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return iou


def dice_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """Per-class Dice / F1 from a confusion matrix."""
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = 2 * tp + fp + fn
    dice = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return dice


# ═══════════════════════════════════════════════════════════════════════════
# Metric tracker (accumulates over batches)
# ═══════════════════════════════════════════════════════════════════════════


class MetricTracker:
    """Accumulate a confusion matrix over batches and compute standard
    segmentation metrics at epoch end.

    Parameters
    ----------
    num_classes : int
        Number of semantic classes (excluding ignore).
    ignore_index : int
        Label value to ignore (default 255, the Cityscapes convention).
    class_names : list[str] or None
        Optional human-readable class names for pretty-printing.
    device : str or torch.device
        Device for the internal confusion matrix accumulator.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_names: Optional[List[str]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.device = torch.device(device)
        self._cm = torch.zeros(
            num_classes, num_classes, dtype=torch.float64, device=self.device,
        )
        self._total_pixels = 0
        self._correct_pixels = 0

    # ── accumulation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Accumulate one batch of predictions."""
        pred, tgt = _to_pred_mask(logits, target, self.ignore_index)
        pred, tgt = pred.to(self.device), tgt.to(self.device)

        self._cm += confusion_matrix(pred, tgt, self.num_classes).to(self._cm)
        self._correct_pixels += (pred == tgt).sum().item()
        self._total_pixels += tgt.numel()

    # ── finalisation ─────────────────────────────────────────────────────

    def compute(self) -> Dict[str, Any]:
        """Return all metrics as a flat dictionary.

        Keys
        ----
        pixel_accuracy : float
        mean_iou : float
        mean_dice : float
        per_class_iou : dict[str, float]
        per_class_dice : dict[str, float]
        frequency_weighted_iou : float
        """
        cm = self._cm.float()
        iou = iou_from_cm(cm)
        dice = dice_from_cm(cm)

        # Class frequencies (for frequency-weighted IoU)
        class_pixels = cm.sum(dim=1)
        total = class_pixels.sum()
        freq = class_pixels / total.clamp(min=1)

        # Only average over classes that are present in the ground truth
        present = class_pixels > 0
        mean_iou = iou[present].mean().item() if present.any() else 0.0
        mean_dice = dice[present].mean().item() if present.any() else 0.0
        fw_iou = (freq * iou).sum().item()

        pa = self._correct_pixels / max(self._total_pixels, 1)

        per_class_iou = {
            name: round(iou[i].item(), 5)
            for i, name in enumerate(self.class_names)
        }
        per_class_dice = {
            name: round(dice[i].item(), 5)
            for i, name in enumerate(self.class_names)
        }

        return {
            "pixel_accuracy": round(pa, 5),
            "mean_iou": round(mean_iou, 5),
            "mean_dice": round(mean_dice, 5),
            "frequency_weighted_iou": round(fw_iou, 5),
            "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice,
        }

    def reset(self) -> None:
        """Zero all accumulators."""
        self._cm.zero_()
        self._total_pixels = 0
        self._correct_pixels = 0

    # ── pretty-print ─────────────────────────────────────────────────────

    def summary(self, prefix: str = "") -> str:
        """Return a formatted multi-line summary string."""
        r = self.compute()
        lines = [
            f"{prefix}Pixel Accuracy : {r['pixel_accuracy']:.4f}",
            f"{prefix}Mean IoU       : {r['mean_iou']:.4f}",
            f"{prefix}Mean Dice      : {r['mean_dice']:.4f}",
            f"{prefix}FW IoU         : {r['frequency_weighted_iou']:.4f}",
            f"{prefix}{'─' * 40}",
        ]
        for name in self.class_names:
            iou_val = r["per_class_iou"][name]
            dice_val = r["per_class_dice"][name]
            lines.append(f"{prefix}  {name:<20s}  IoU={iou_val:.4f}  Dice={dice_val:.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════════


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Soft Dice loss (1 − Dice) averaged over classes.

    Parameters
    ----------
    logits : (B, C, H, W)
    target : (B, H, W) int64
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # One-hot encode target
    target_flat = target.clone()
    target_flat[target_flat == ignore_index] = 0
    one_hot = F.one_hot(target_flat, num_classes)  # (B, H, W, C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Mask out ignored pixels
    valid = (target != ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
    probs = probs * valid
    one_hot = one_hot * valid

    # Per-class Dice
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)

    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    ignore_index: int = 255,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy + Dice loss (weighted sum)."""
    ce = F.cross_entropy(
        logits, target,
        weight=class_weights,
        ignore_index=ignore_index,
    )
    dl = dice_loss(logits, target, ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dl
