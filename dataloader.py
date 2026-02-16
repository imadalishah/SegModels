"""seg_models.dataloader — Generic segmentation dataset & dataloader utilities.

Supports:
  • Folder-based datasets (image dir + mask dir)
  • Multi-channel inputs (e.g. stacked .npy arrays)
  • On-the-fly augmentations via albumentations (optional)
  • Configurable train / val / test splits

Usage::

    from seg_models.dataloader import create_dataloaders

    train_dl, val_dl = create_dataloaders(
        image_dir="data/images",
        mask_dir="data/masks",
        img_size=(512, 512),
        batch_size=8,
        val_split=0.2,
    )
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Optional imports — gracefully degrade
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBU_AVAILABLE = True
except ImportError:
    ALBU_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════


class SegmentationDataset(Dataset):
    """A flexible dataset for semantic segmentation.

    Supports two input modes:

    1. **Image files** (png / jpg / tif) — loaded via PIL and converted to
       ``(C, H, W)`` float tensors in ``[0, 1]``.
    2. **NumPy arrays** (``.npy`` / ``.npz``) — loaded directly, expected
       shape ``(H, W, C)`` or ``(H, W)`` for single-channel.

    Masks are always loaded as ``int64`` tensors of shape ``(H, W)`` with
    class indices.

    Parameters
    ----------
    image_paths : list[str | Path]
        Ordered list of image file paths.
    mask_paths : list[str | Path]
        Ordered list of mask file paths (same length & order as *image_paths*).
    img_size : (int, int) or None
        Target ``(H, W)`` to resize to.  ``None`` → keep original size.
    transform : callable or None
        An albumentations ``Compose`` (or any callable that accepts
        ``image=np.ndarray, mask=np.ndarray`` and returns a dict).
    normalize : bool
        If ``True`` (default), image values are scaled to ``[0, 1]``.
    """

    IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    NP_EXTENSIONS = {".npy", ".npz"}

    def __init__(
        self,
        image_paths: Sequence[Union[str, Path]],
        mask_paths: Sequence[Union[str, Path]],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        assert len(image_paths) == len(mask_paths), (
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        )
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize

    # ── loaders ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Return image as ``(H, W, C)`` uint8 or float array."""
        ext = path.suffix.lower()
        if ext in SegmentationDataset.NP_EXTENSIONS:
            arr = np.load(path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[arr.files[0]]
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            return arr
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to load image files.")
        img = Image.open(path).convert("RGB")
        return np.array(img)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        """Return mask as ``(H, W)`` int array of class indices."""
        ext = path.suffix.lower()
        if ext in SegmentationDataset.NP_EXTENSIONS:
            arr = np.load(path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[arr.files[0]]
            return arr.astype(np.int64)
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to load mask files.")
        mask = Image.open(path)
        return np.array(mask).astype(np.int64)

    # ── core ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self._load_image(self.image_paths[idx])  # (H, W, C)
        mask = self._load_mask(self.mask_paths[idx])      # (H, W)

        # Resize (before augmentation so augmentations see the target size)
        if self.img_size is not None and ALBU_AVAILABLE:
            resizer = A.Resize(height=self.img_size[0], width=self.img_size[1])
            resized = resizer(image=image, mask=mask)
            image, mask = resized["image"], resized["mask"]
        elif self.img_size is not None:
            # Fallback: PIL resize
            if PIL_AVAILABLE:
                h, w = self.img_size
                image = np.array(
                    Image.fromarray(image.astype(np.uint8)).resize((w, h))
                )
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8)).resize(
                        (w, h), resample=Image.NEAREST
                    )
                )

        # Augmentations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # To tensor
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[np.newaxis]  # (1, H, W)
            else:
                image = image.transpose(2, 0, 1)  # (C, H, W)
            image = torch.from_numpy(image.copy()).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.copy()).long()

        # Normalize to [0, 1]
        if self.normalize and image.dtype == torch.float32 and image.max() > 1.0:
            image = image / 255.0

        return {"image": image, "mask": mask}


# ═══════════════════════════════════════════════════════════════════════════
# Default augmentations
# ═══════════════════════════════════════════════════════════════════════════


def get_train_augmentations(img_size: Tuple[int, int]) -> Any:
    """Return a sensible default training augmentation pipeline.

    Requires ``albumentations``.  Returns ``None`` if not installed.
    """
    if not ALBU_AVAILABLE:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4,
            border_mode=0,
        ),
        A.GaussNoise(p=0.2),
    ])


def get_val_augmentations(img_size: Tuple[int, int]) -> Any:
    """Return a minimal validation augmentation pipeline (no-op)."""
    return None


# ═══════════════════════════════════════════════════════════════════════════
# File discovery
# ═══════════════════════════════════════════════════════════════════════════


def discover_files(
    directory: Union[str, Path],
    extensions: Optional[set] = None,
) -> List[Path]:
    """Recursively find files in *directory* matching *extensions*.

    Returns a sorted list of ``Path`` objects.
    """
    directory = Path(directory)
    if extensions is None:
        extensions = SegmentationDataset.IMG_EXTENSIONS | SegmentationDataset.NP_EXTENSIONS
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in extensions
    )


def pair_images_masks(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    image_extensions: Optional[set] = None,
    mask_extensions: Optional[set] = None,
) -> Tuple[List[Path], List[Path]]:
    """Match images to masks by stem (filename without extension).

    Returns two aligned lists ``(image_paths, mask_paths)``.
    Raises ``ValueError`` if any image has no matching mask.
    """
    images = discover_files(image_dir, image_extensions)
    masks = discover_files(mask_dir, mask_extensions)

    mask_map = {p.stem: p for p in masks}
    paired_imgs, paired_masks = [], []
    missing = []

    for img_path in images:
        stem = img_path.stem
        if stem in mask_map:
            paired_imgs.append(img_path)
            paired_masks.append(mask_map[stem])
        else:
            missing.append(stem)

    if missing:
        raise ValueError(
            f"{len(missing)} images have no matching mask. "
            f"First 5: {missing[:5]}"
        )
    return paired_imgs, paired_masks


# ═══════════════════════════════════════════════════════════════════════════
# Dataloader factory
# ═══════════════════════════════════════════════════════════════════════════


def create_dataloaders(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    img_size: Optional[Tuple[int, int]] = (512, 512),
    batch_size: int = 8,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    normalize: bool = True,
) -> Dict[str, DataLoader]:
    """Build train / val (/ test) ``DataLoader`` objects from a folder pair.

    Parameters
    ----------
    image_dir, mask_dir : path-like
        Directories containing images and masks (matched by filename stem).
    img_size : (H, W) or None
        Resize target.  ``None`` keeps original sizes (requires batch_size=1
        or a custom collate_fn).
    val_split, test_split : float
        Fraction of data held out for validation / test.
    train_transform, val_transform : callable or None
        Albumentations pipelines.  ``None`` → use built-in defaults.
    normalize : bool
        Scale pixel values to ``[0, 1]``.

    Returns
    -------
    dict[str, DataLoader]
        Keys: ``"train"``, ``"val"``, and optionally ``"test"``.
    """
    image_paths, mask_paths = pair_images_masks(image_dir, mask_dir)
    n = len(image_paths)
    assert n > 0, f"No image–mask pairs found in {image_dir} / {mask_dir}"

    # Deterministic shuffle
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test
    assert n_train > 0, "Not enough data for training after splits."

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
    }
    if n_test > 0:
        splits["test"] = indices[n_train + n_val :]

    # Resolve transforms
    if train_transform is None and img_size is not None:
        train_transform = get_train_augmentations(img_size)
    if val_transform is None and img_size is not None:
        val_transform = get_val_augmentations(img_size)

    # Build datasets
    full_train_ds = SegmentationDataset(
        image_paths, mask_paths,
        img_size=img_size, transform=train_transform, normalize=normalize,
    )
    full_val_ds = SegmentationDataset(
        image_paths, mask_paths,
        img_size=img_size, transform=val_transform, normalize=normalize,
    )

    loaders: Dict[str, DataLoader] = {}
    for split_name, split_indices in splits.items():
        ds = full_train_ds if split_name == "train" else full_val_ds
        subset = Subset(ds, split_indices)
        loaders[split_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train"),
        )

    print(f"📂 Dataloaders ready — "
          f"train={len(splits['train'])}  val={len(splits['val'])}"
          + (f"  test={len(splits.get('test', []))}" if "test" in splits else ""))

    return loaders
