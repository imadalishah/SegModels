#!/usr/bin/env python
"""Smoke-test: run every model config through a single forward pass and report
pass / fail.

Usage::

    python -m tests.test_inference          # from repo root
    python tests/test_inference.py          # also works
"""

import gc
import time
import traceback

import torch

from seg_models import create_segmentation_model

# ── Configuration ────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CHANNELS = 4

# Test matrix
IN_CHANNELS_LIST = [15]            # arbitrary C
IMG_SIZES = [(209, 416)]           # arbitrary H×W
PRETRAINED_FLAGS = [False]

# (model_name, backbone, extra_kwargs)
MODEL_CONFIGS = [
    # === SMP Baseline ===
    # ("unet",        "resnet50",  {}),
    # ("unet++",      "resnet50",  {}),
    # ("fpn",         "resnet50",  {}),
    # ("linknet",     "resnet50",  {}),
    # ("pspnet",      "resnet50",  {}),
    # ("pan",         "resnet50",  {}),

    # === SMP CNN SOTA ===
    # ("deeplabv3",   "resnet50",  {}),
    # ("deeplabv3+",  "resnet50",  {}),
    # ("manet",       "resnet50",  {}),

    # === Specialised SMP (different backbones) ===
    # ("unet_resnet",       "resnet34",        {}),
    # ("unet_efficientnet", "efficientnet-b0", {}),
    # ("unet_mit",          "mit_b0",          {}),

    # === HuggingFace Transformers ===
    ("segformer_hf", "nvidia/segformer-b0-finetuned-ade-512-512",         {}),
    ("mask2former",  "facebook/mask2former-swin-tiny-cityscapes-semantic", {}),
    ("upernet_swin", "openmmlab/upernet-swin-tiny",                       {}),
    ("dpt",          "Intel/dpt-large-ade",                               {}),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _output_shape(out) -> str:
    """Human-readable shape string from model output."""
    if isinstance(out, torch.Tensor):
        return str(out.shape)
    if isinstance(out, dict):
        key = next(iter(out))
        val = out[key]
        return f"dict[\'{key}\'] → {val.shape}" if hasattr(val, "shape") else str(type(val))
    if hasattr(out, "logits"):
        return f"logits → {out.logits.shape}"
    return str(type(out))


# ── Main ─────────────────────────────────────────────────────────────────────

def run_inference_test():
    results = []
    combos = [
        (mc, ic, sz, pt)
        for mc in MODEL_CONFIGS
        for ic in IN_CHANNELS_LIST
        for sz in IMG_SIZES
        for pt in PRETRAINED_FLAGS
    ]
    total = len(combos)

    print("=" * 100)
    print(
        f"INFERENCE TEST — {len(MODEL_CONFIGS)} models × "
        f"{len(IN_CHANNELS_LIST)} C × {len(IMG_SIZES)} HW × "
        f"{len(PRETRAINED_FLAGS)} pretrained = {total} runs | device={DEVICE}"
    )
    print("=" * 100)

    for idx, ((model_name, backbone, kwargs), in_ch, (img_h, img_w), pt) in enumerate(combos, 1):
        label = f"{model_name} ({backbone})  C={in_ch}  {img_h}×{img_w}  pretrained={pt}"
        print(f"\n[{idx}/{total}] {label}")

        model = dummy = out = None
        status = "FAIL"
        out_shape = None
        t0 = time.time()

        try:
            model = create_segmentation_model(
                model_name=model_name,
                backbone_name=backbone,
                in_channels=in_ch,
                out_channels=OUT_CHANNELS,
                pretrained=pt,
                **kwargs,
            )
            model = model.to(DEVICE).eval()

            dummy = torch.randn(1, in_ch, img_h, img_w, device=DEVICE)
            with torch.no_grad():
                out = model(dummy)

            out_shape = _output_shape(out)
            status = "PASS"
            elapsed = time.time() - t0
            print(f"  ✅ PASS | output={out_shape} | {elapsed:.2f}s")

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ❌ FAIL | {type(exc).__name__}: {exc}")
            traceback.print_exc()

        finally:
            del model, dummy, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append({
            "model": model_name,
            "backbone": backbone,
            "in_channels": in_ch,
            "img_size": f"{img_h}×{img_w}",
            "pretrained": pt,
            "status": status,
            "output_shape": out_shape,
            "time_s": round(elapsed, 2),
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    hdr = (
        f"{\'Model\':<22} {\'Backbone\':<52} {\'C\':>3} {\'HxW\':<10} "
        f"{\'PT\':>5} {\'Status\':<6} {\'Output\':<30} {\'Time\'}"
    )
    print(f"\n{hdr}")
    for r in results:
        print(
            f"{r[\'model\']:<22} {r[\'backbone\']:<52} {r[\'in_channels\']:>3} "
            f"{r[\'img_size\']:<10} {str(r[\'pretrained\']):>5} {r[\'status\']:<6} "
            f"{str(r[\'output_shape\']):<30} {r[\'time_s\']}s"
        )

    print(f"\n✅ Passed: {passed}/{total}  |  ❌ Failed: {failed}/{total}")
    return results


if __name__ == "__main__":
    run_inference_test()
