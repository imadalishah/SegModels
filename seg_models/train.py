"""seg_models.train — Training loop for semantic segmentation.

A self-contained, single-file trainer that ties together the model factory,
dataloaders, metrics, and standard PyTorch training utilities.

Usage::

    python -m seg_models.train \\
        --model deeplabv3+ --backbone resnet101 \\
        --image_dir data/images --mask_dir data/masks \\
        --num_classes 19 --epochs 50 --batch_size 8

Or programmatically::

    from seg_models.train import Trainer, TrainConfig
    cfg = TrainConfig(model_name="unet", backbone_name="resnet50", ...)
    trainer = Trainer(cfg)
    trainer.fit()
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from .factory import create_segmentation_model
from .dataloader import create_dataloaders
from .metrics import MetricTracker, combined_loss, dice_loss


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    """All training hyper-parameters in one place."""

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "unet"
    backbone_name: str = "resnet50"
    in_channels: int = 3
    num_classes: int = 19
    pretrained: bool = True

    # ── Data ─────────────────────────────────────────────────────────────
    image_dir: str = "data/images"
    mask_dir: str = "data/masks"
    img_size: tuple = (512, 512)
    batch_size: int = 8
    val_split: float = 0.2
    num_workers: int = 4

    # ── Optimiser / scheduler ────────────────────────────────────────────
    optimizer: str = "adamw"          # "adam" | "adamw" | "sgd"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9            # SGD only
    scheduler: str = "cosine"        # "cosine" | "step" | "plateau" | "none"
    step_size: int = 10              # StepLR
    gamma: float = 0.1               # StepLR / ReduceLROnPlateau
    warmup_epochs: int = 0
    min_lr: float = 1e-7

    # ── Loss ─────────────────────────────────────────────────────────────
    loss: str = "ce+dice"            # "ce" | "dice" | "ce+dice"
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    ignore_index: int = 255

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 50
    amp: bool = True                 # mixed-precision
    grad_clip: float = 1.0           # 0 → disabled
    accumulation_steps: int = 1      # gradient accumulation

    # ── Checkpointing / logging ──────────────────────────────────────────
    output_dir: str = "runs"
    experiment_name: str = "seg_exp"
    save_best: bool = True
    save_every: int = 0              # 0 → only save best
    log_every: int = 10              # batches
    early_stopping_patience: int = 0 # 0 → disabled

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "auto"             # "auto" | "cuda" | "cpu"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════


class Trainer:
    """End-to-end segmentation trainer.

    Parameters
    ----------
    config : TrainConfig
        Full training configuration.
    model : nn.Module or None
        If ``None``, the model is created from ``config`` via the factory.
    """

    def __init__(
        self,
        config: TrainConfig,
        model: Optional[nn.Module] = None,
    ):
        self.cfg = config
        self.device = config.resolve_device()

        # ── Output directory ─────────────────────────────────────────────
        self.run_dir = Path(config.output_dir) / config.experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ── Model ────────────────────────────────────────────────────────
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = create_segmentation_model(
                model_name=config.model_name,
                backbone_name=config.backbone_name,
                in_channels=config.in_channels,
                out_channels=config.num_classes,
                pretrained=config.pretrained,
            ).to(self.device)

        # ── Data ─────────────────────────────────────────────────────────
        self.loaders = create_dataloaders(
            image_dir=config.image_dir,
            mask_dir=config.mask_dir,
            img_size=config.img_size,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=config.num_workers,
        )

        # ── Optimiser ────────────────────────────────────────────────────
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # ── Loss ─────────────────────────────────────────────────────────
        self.loss_fn = self._build_loss()

        # ── AMP ──────────────────────────────────────────────────────────
        self.scaler = GradScaler(enabled=config.amp)

        # ── Metrics ──────────────────────────────────────────────────────
        self.train_metrics = MetricTracker(
            config.num_classes, config.ignore_index, device=self.device,
        )
        self.val_metrics = MetricTracker(
            config.num_classes, config.ignore_index, device=self.device,
        )

        # ── State ────────────────────────────────────────────────────────
        self.best_miou = 0.0
        self.epochs_no_improve = 0
        self.history: List[Dict[str, Any]] = []

    # ── builders ─────────────────────────────────────────────────────────

    def _build_optimizer(self) -> optim.Optimizer:
        cfg = self.cfg
        params = self.model.parameters()
        if cfg.optimizer == "adam":
            return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw":
            return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "sgd":
            return optim.SGD(
                params, lr=cfg.lr,
                momentum=cfg.momentum, weight_decay=cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _build_scheduler(self):
        cfg = self.cfg
        if cfg.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr,
            )
        if cfg.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.step_size, gamma=cfg.gamma,
            )
        if cfg.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=cfg.gamma, patience=5,
            )
        return None

    def _build_loss(self):
        cfg = self.cfg

        if cfg.loss == "ce":
            return lambda logits, target: nn.functional.cross_entropy(
                logits, target, ignore_index=cfg.ignore_index,
            )
        if cfg.loss == "dice":
            return lambda logits, target: dice_loss(
                logits, target, ignore_index=cfg.ignore_index,
            )
        # default: ce + dice
        return lambda logits, target: combined_loss(
            logits, target,
            ce_weight=cfg.ce_weight,
            dice_weight=cfg.dice_weight,
            ignore_index=cfg.ignore_index,
        )

    # ── extract logits helper ────────────────────────────────────────────

    @staticmethod
    def _extract_logits(out) -> torch.Tensor:
        """Normalise model output to a plain ``(B, C, H, W)`` tensor."""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        if hasattr(out, "logits"):
            return out.logits
        raise TypeError(f"Cannot extract logits from {type(out)}")

    # ── single epoch ─────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(self.loaders["train"]):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.amp):
                out = self.model(images)
                logits = self._extract_logits(out)
                loss = self.loss_fn(logits, masks) / self.cfg.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.cfg.accumulation_steps == 0:
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.cfg.accumulation_steps
            n_batches += 1
            self.train_metrics.update(logits.detach(), masks)

            if self.cfg.log_every and (i + 1) % self.cfg.log_every == 0:
                avg = total_loss / n_batches
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [batch {i + 1:>4d}]  loss={avg:.4f}  lr={lr:.2e}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> Dict[str, Any]:
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        n_batches = 0

        for batch in self.loaders["val"]:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.amp):
                out = self.model(images)
                logits = self._extract_logits(out)
                loss = self.loss_fn(logits, masks)

            total_loss += loss.item()
            n_batches += 1
            self.val_metrics.update(logits, masks)

        val_loss = total_loss / max(n_batches, 1)
        metrics = self.val_metrics.compute()
        metrics["val_loss"] = round(val_loss, 5)
        return metrics

    # ── checkpointing ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: Dict, tag: str = "best"):
        path = self.run_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.cfg.__dict__,
            },
            path,
        )
        print(f"  💾 Saved {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> Dict:
        """Load a checkpoint and return its metadata."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  📦 Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt

    # ── main loop ────────────────────────────────────────────────────────

    def fit(self) -> List[Dict[str, Any]]:
        """Run the full training loop.  Returns the per-epoch history."""
        cfg = self.cfg
        print("=" * 70)
        print(f"Training  {cfg.model_name} ({cfg.backbone_name})")
        print(f"  classes={cfg.num_classes}  epochs={cfg.epochs}  "
              f"batch={cfg.batch_size}  lr={cfg.lr}  device={self.device}")
        print("=" * 70)

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2, default=str)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            print(f"\n{'─' * 60}")
            print(f"Epoch {epoch}/{cfg.epochs}")

            # ── Train ────────────────────────────────────────────────────
            train_loss = self._train_one_epoch(epoch)
            train_metrics = self.train_metrics.compute()

            # ── Validate ─────────────────────────────────────────────────
            val_metrics = self._validate()
            elapsed = time.time() - t0

            # ── Scheduler step ───────────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["mean_iou"])
                else:
                    self.scheduler.step()

            # ── Logging ──────────────────────────────────────────────────
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_miou": train_metrics["mean_iou"],
                "val_loss": val_metrics["val_loss"],
                "val_miou": val_metrics["mean_iou"],
                "val_dice": val_metrics["mean_dice"],
                "val_pa": val_metrics["pixel_accuracy"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_s": round(elapsed, 1),
            }
            self.history.append(record)

            print(
                f"  train_loss={record['train_loss']:.4f}  "
                f"val_loss={record['val_loss']:.4f}  "
                f"val_mIoU={record['val_miou']:.4f}  "
                f"val_dice={record['val_dice']:.4f}  "
                f"val_pAcc={record['val_pa']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # ── Checkpointing ────────────────────────────────────────────
            improved = val_metrics["mean_iou"] > self.best_miou
            if improved:
                self.best_miou = val_metrics["mean_iou"]
                self.epochs_no_improve = 0
                if cfg.save_best:
                    self._save_checkpoint(epoch, val_metrics, tag="best")
            else:
                self.epochs_no_improve += 1

            if cfg.save_every and epoch % cfg.save_every == 0:
                self._save_checkpoint(epoch, val_metrics, tag=f"epoch_{epoch}")

            # ── Early stopping ───────────────────────────────────────────
            if (
                cfg.early_stopping_patience > 0
                and self.epochs_no_improve >= cfg.early_stopping_patience
            ):
                print(
                    f"\n⏹  Early stopping after {cfg.early_stopping_patience} "
                    f"epochs without improvement."
                )
                break

        # ── Final summary ────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print(f"Training complete — best val mIoU = {self.best_miou:.4f}")
        print("=" * 70)

        # Save history
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a segmentation model")

    # Model
    p.add_argument("--model", default="unet", dest="model_name")
    p.add_argument("--backbone", default="resnet50", dest="backbone_name")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--no_pretrained", action="store_true")

    # Data
    p.add_argument("--image_dir", default="data/images")
    p.add_argument("--mask_dir", default="data/masks")
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)

    # Optimiser
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="cosine")

    # Loss
    p.add_argument("--loss", default="ce+dice")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accumulation_steps", type=int, default=1)

    # Output
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--experiment_name", default="seg_exp")
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--early_stopping", type=int, default=0)

    args = p.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        backbone_name=args.backbone_name,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_size=(args.img_h, args.img_w),
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        loss=args.loss,
        epochs=args.epochs,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping,
    )


if __name__ == "__main__":
    cfg = parse_args()
    trainer = Trainer(cfg)
    trainer.fit()
"""seg_models.train — Training loop for semantic segmentation.

A self-contained, single-file trainer that ties together the model factory,
dataloaders, metrics, and standard PyTorch training utilities.

Usage::

    python -m seg_models.train \\
        --model deeplabv3+ --backbone resnet101 \\
        --image_dir data/images --mask_dir data/masks \\
        --num_classes 19 --epochs 50 --batch_size 8

Or programmatically::

    from seg_models.train import Trainer, TrainConfig
    cfg = TrainConfig(model_name="unet", backbone_name="resnet50", ...)
    trainer = Trainer(cfg)
    trainer.fit()
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from .factory import create_segmentation_model
from .dataloader import create_dataloaders
from .metrics import MetricTracker, combined_loss, dice_loss


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    """All training hyper-parameters in one place."""

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "unet"
    backbone_name: str = "resnet50"
    in_channels: int = 3
    num_classes: int = 19
    pretrained: bool = True

    # ── Data ─────────────────────────────────────────────────────────────
    image_dir: str = "data/images"
    mask_dir: str = "data/masks"
    img_size: tuple = (512, 512)
    batch_size: int = 8
    val_split: float = 0.2
    num_workers: int = 4

    # ── Optimiser / scheduler ────────────────────────────────────────────
    optimizer: str = "adamw"          # "adam" | "adamw" | "sgd"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9            # SGD only
    scheduler: str = "cosine"        # "cosine" | "step" | "plateau" | "none"
    step_size: int = 10              # StepLR
    gamma: float = 0.1               # StepLR / ReduceLROnPlateau
    warmup_epochs: int = 0
    min_lr: float = 1e-7

    # ── Loss ─────────────────────────────────────────────────────────────
    loss: str = "ce+dice"            # "ce" | "dice" | "ce+dice"
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    ignore_index: int = 255

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 50
    amp: bool = True                 # mixed-precision
    grad_clip: float = 1.0           # 0 → disabled
    accumulation_steps: int = 1      # gradient accumulation

    # ── Checkpointing / logging ──────────────────────────────────────────
    output_dir: str = "runs"
    experiment_name: str = "seg_exp"
    save_best: bool = True
    save_every: int = 0              # 0 → only save best
    log_every: int = 10              # batches
    early_stopping_patience: int = 0 # 0 → disabled

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "auto"             # "auto" | "cuda" | "cpu"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════


class Trainer:
    """End-to-end segmentation trainer.

    Parameters
    ----------
    config : TrainConfig
        Full training configuration.
    model : nn.Module or None
        If ``None``, the model is created from ``config`` via the factory.
    """

    def __init__(
        self,
        config: TrainConfig,
        model: Optional[nn.Module] = None,
    ):
        self.cfg = config
        self.device = config.resolve_device()

        # ── Output directory ─────────────────────────────────────────────
        self.run_dir = Path(config.output_dir) / config.experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ── Model ────────────────────────────────────────────────────────
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = create_segmentation_model(
                model_name=config.model_name,
                backbone_name=config.backbone_name,
                in_channels=config.in_channels,
                out_channels=config.num_classes,
                pretrained=config.pretrained,
            ).to(self.device)

        # ── Data ─────────────────────────────────────────────────────────
        self.loaders = create_dataloaders(
            image_dir=config.image_dir,
            mask_dir=config.mask_dir,
            img_size=config.img_size,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=config.num_workers,
        )

        # ── Optimiser ────────────────────────────────────────────────────
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # ── Loss ─────────────────────────────────────────────────────────
        self.loss_fn = self._build_loss()

        # ── AMP ──────────────────────────────────────────────────────────
        self.scaler = GradScaler(enabled=config.amp)

        # ── Metrics ──────────────────────────────────────────────────────
        self.train_metrics = MetricTracker(
            config.num_classes, config.ignore_index, device=self.device,
        )
        self.val_metrics = MetricTracker(
            config.num_classes, config.ignore_index, device=self.device,
        )

        # ── State ────────────────────────────────────────────────────────
        self.best_miou = 0.0
        self.epochs_no_improve = 0
        self.history: List[Dict[str, Any]] = []

    # ── builders ─────────────────────────────────────────────────────────

    def _build_optimizer(self) -> optim.Optimizer:
        cfg = self.cfg
        params = self.model.parameters()
        if cfg.optimizer == "adam":
            return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw":
            return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "sgd":
            return optim.SGD(
                params, lr=cfg.lr,
                momentum=cfg.momentum, weight_decay=cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _build_scheduler(self):
        cfg = self.cfg
        if cfg.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr,
            )
        if cfg.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.step_size, gamma=cfg.gamma,
            )
        if cfg.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=cfg.gamma, patience=5,
            )
        return None

    def _build_loss(self):
        cfg = self.cfg

        if cfg.loss == "ce":
            return lambda logits, target: nn.functional.cross_entropy(
                logits, target, ignore_index=cfg.ignore_index,
            )
        if cfg.loss == "dice":
            return lambda logits, target: dice_loss(
                logits, target, ignore_index=cfg.ignore_index,
            )
        # default: ce + dice
        return lambda logits, target: combined_loss(
            logits, target,
            ce_weight=cfg.ce_weight,
            dice_weight=cfg.dice_weight,
            ignore_index=cfg.ignore_index,
        )

    # ── extract logits helper ────────────────────────────────────────────

    @staticmethod
    def _extract_logits(out) -> torch.Tensor:
        """Normalise model output to a plain ``(B, C, H, W)`` tensor."""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        if hasattr(out, "logits"):
            return out.logits
        raise TypeError(f"Cannot extract logits from {type(out)}")

    # ── single epoch ─────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(self.loaders["train"]):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.amp):
                out = self.model(images)
                logits = self._extract_logits(out)
                loss = self.loss_fn(logits, masks) / self.cfg.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.cfg.accumulation_steps == 0:
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.cfg.accumulation_steps
            n_batches += 1
            self.train_metrics.update(logits.detach(), masks)

            if self.cfg.log_every and (i + 1) % self.cfg.log_every == 0:
                avg = total_loss / n_batches
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [batch {i + 1:>4d}]  loss={avg:.4f}  lr={lr:.2e}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> Dict[str, Any]:
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        n_batches = 0

        for batch in self.loaders["val"]:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.amp):
                out = self.model(images)
                logits = self._extract_logits(out)
                loss = self.loss_fn(logits, masks)

            total_loss += loss.item()
            n_batches += 1
            self.val_metrics.update(logits, masks)

        val_loss = total_loss / max(n_batches, 1)
        metrics = self.val_metrics.compute()
        metrics["val_loss"] = round(val_loss, 5)
        return metrics

    # ── checkpointing ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: Dict, tag: str = "best"):
        path = self.run_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.cfg.__dict__,
            },
            path,
        )
        print(f"  💾 Saved {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> Dict:
        """Load a checkpoint and return its metadata."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  📦 Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt

    # ── main loop ────────────────────────────────────────────────────────

    def fit(self) -> List[Dict[str, Any]]:
        """Run the full training loop.  Returns the per-epoch history."""
        cfg = self.cfg
        print("=" * 70)
        print(f"Training  {cfg.model_name} ({cfg.backbone_name})")
        print(f"  classes={cfg.num_classes}  epochs={cfg.epochs}  "
              f"batch={cfg.batch_size}  lr={cfg.lr}  device={self.device}")
        print("=" * 70)

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2, default=str)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            print(f"\n{'─' * 60}")
            print(f"Epoch {epoch}/{cfg.epochs}")

            # ── Train ────────────────────────────────────────────────────
            train_loss = self._train_one_epoch(epoch)
            train_metrics = self.train_metrics.compute()

            # ── Validate ─────────────────────────────────────────────────
            val_metrics = self._validate()
            elapsed = time.time() - t0

            # ── Scheduler step ───────────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["mean_iou"])
                else:
                    self.scheduler.step()

            # ── Logging ──────────────────────────────────────────────────
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_miou": train_metrics["mean_iou"],
                "val_loss": val_metrics["val_loss"],
                "val_miou": val_metrics["mean_iou"],
                "val_dice": val_metrics["mean_dice"],
                "val_pa": val_metrics["pixel_accuracy"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_s": round(elapsed, 1),
            }
            self.history.append(record)

            print(
                f"  train_loss={record['train_loss']:.4f}  "
                f"val_loss={record['val_loss']:.4f}  "
                f"val_mIoU={record['val_miou']:.4f}  "
                f"val_dice={record['val_dice']:.4f}  "
                f"val_pAcc={record['val_pa']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # ── Checkpointing ────────────────────────────────────────────
            improved = val_metrics["mean_iou"] > self.best_miou
            if improved:
                self.best_miou = val_metrics["mean_iou"]
                self.epochs_no_improve = 0
                if cfg.save_best:
                    self._save_checkpoint(epoch, val_metrics, tag="best")
            else:
                self.epochs_no_improve += 1

            if cfg.save_every and epoch % cfg.save_every == 0:
                self._save_checkpoint(epoch, val_metrics, tag=f"epoch_{epoch}")

            # ── Early stopping ───────────────────────────────────────────
            if (
                cfg.early_stopping_patience > 0
                and self.epochs_no_improve >= cfg.early_stopping_patience
            ):
                print(
                    f"\n⏹  Early stopping after {cfg.early_stopping_patience} "
                    f"epochs without improvement."
                )
                break

        # ── Final summary ────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print(f"Training complete — best val mIoU = {self.best_miou:.4f}")
        print("=" * 70)

        # Save history
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a segmentation model")

    # Model
    p.add_argument("--model", default="unet", dest="model_name")
    p.add_argument("--backbone", default="resnet50", dest="backbone_name")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--no_pretrained", action="store_true")

    # Data
    p.add_argument("--image_dir", default="data/images")
    p.add_argument("--mask_dir", default="data/masks")
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)

    # Optimiser
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="cosine")

    # Loss
    p.add_argument("--loss", default="ce+dice")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accumulation_steps", type=int, default=1)

    # Output
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--experiment_name", default="seg_exp")
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--early_stopping", type=int, default=0)

    args = p.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        backbone_name=args.backbone_name,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_size=(args.img_h, args.img_w),
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        loss=args.loss,
        epochs=args.epochs,
        amp=not args.no_amp,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping,
    )


if __name__ == "__main__":
    cfg = parse_args()
    trainer = Trainer(cfg)
    trainer.fit()
