"""
scripts/viz_augment.py
16-variant augmentation grid annotated with azimuth and label.

Usage:
    python scripts/viz_augment.py --policy canonical --image-id train_00000.png
    python scripts/viz_augment.py --policy canonical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# ── Allow running from repo root ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical import canonicalize
from src.transforms import (
    CanonicalHorizontalFlip,
    CanonicalVerticalFlipLabelSwap,
    PhotometricNegationLabelSwap,
    Sample,
    build_augmentation_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize augmentation policy on an image."
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="canonical",
        choices=["canonical", "identity", "config"],
        help="Augmentation policy: 'canonical' (E4 policy), 'identity', or 'config'.",
    )
    parser.add_argument(
        "--image-id",
        type=str,
        default=None,
        help="Target image_id (e.g. train_00000.png). Defaults to first Depth (Class 0) image.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp/e4_canonical_convnext.yaml",
        help="Path to experiment config if policy='config' or for loading frozen calibration.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/figures",
        help="Output directory for saved figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable augmentations.",
    )
    return parser.parse_args()


def load_sample_by_id(
    data_dir: str | Path,
    image_id: str | None = None,
) -> tuple[np.ndarray, float, int, str]:
    """Load image, azimuth, label, and image_id from cache."""
    cache_dir = Path(data_dir) / "processed"
    index_path = cache_dir / "index.json"
    img_path = cache_dir / "train_images.npy"

    if not index_path.exists() or not img_path.exists():
        raise FileNotFoundError(
            f"Cache files not found in {cache_dir}. Run `make cache` first."
        )

    with open(index_path, encoding="utf-8") as f:
        index_data = json.load(f)
    train_records = index_data["train"]
    images = np.load(img_path, mmap_mode="r")

    target_idx = None
    if image_id is not None:
        for idx, rec in enumerate(train_records):
            if rec["image_id"] == image_id:
                target_idx = idx
                break
        if target_idx is None:
            raise ValueError(f"image_id '{image_id}' not found in train set.")
    else:
        # Default to first crater (label == 0) for clear top/bottom contrast
        for idx, rec in enumerate(train_records):
            if rec.get("label") == 0:
                target_idx = idx
                break
        if target_idx is None:
            target_idx = 0

    rec = train_records[target_idx]
    img = np.array(images[target_idx], dtype=np.uint8)
    azimuth = float(rec["sun_azimuth_angle"])
    label = int(rec["label"])
    img_id = str(rec["image_id"])
    return img, azimuth, label, img_id


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load base config for frozen calibration
    base_cfg_path = Path("configs/config.yaml")
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Missing {base_cfg_path}")
    base_cfg = OmegaConf.load(base_cfg_path)
    cal = base_cfg.frozen.calibration
    delta = float(cal.delta)
    s = int(cal.s)

    # Load raw sample
    raw_img, azimuth, label, img_id = load_sample_by_id(args.data_dir, args.image_id)

    # Canonicalize sample
    canonical_img = canonicalize(raw_img, azimuth, delta, s)

    # Build policy
    if args.policy == "canonical":
        # E4 policy: vflip (0.25), neg (0.15), hflip (0.50)
        # To ensure visual variety across 16 variants, we can use slightly higher p or standard E4 p
        policy_ops = [
            CanonicalVerticalFlipLabelSwap(p=0.40),
            PhotometricNegationLabelSwap(p=0.30),
            CanonicalHorizontalFlip(p=0.50),
        ]
    elif args.policy == "config":
        exp_cfg = OmegaConf.merge(base_cfg, OmegaConf.load(args.config))
        policy_ops = build_augmentation_policy(
            OmegaConf.to_container(exp_cfg, resolve=True), training=True
        )
    else:
        policy_ops = []

    # Generate 16 variants: variant 0 is original canonical; 1-15 are augmented
    variants: list[Sample] = []
    # Variant 0: unmodified canonical image
    variants.append(Sample(image=canonical_img.copy(), azimuth=azimuth, label=label))

    for _ in range(15):
        s_curr = Sample(image=canonical_img.copy(), azimuth=azimuth, label=label)
        for op in policy_ops:
            s_curr = op(s_curr)
        variants.append(s_curr)

    # Plot 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 13))
    class_names = {0: "Depth (0)", 1: "Rise (1)"}

    for i, (ax, var) in enumerate(zip(axes.flat, variants)):
        ax.imshow(var.image, cmap="gray", vmin=0, vmax=255)
        # Compute top vs bottom asymmetry to display
        top_half = np.mean(var.image[:128, :])
        bot_half = np.mean(var.image[128:, :])
        asym = top_half - bot_half

        lbl_str = class_names.get(var.label, f"Label {var.label}")
        prefix = "Orig" if i == 0 else f"Var #{i}"

        # Color title based on label: Blue for Depth, Red/Orange for Rise
        color = "darkblue" if var.label == 0 else "darkred"
        ax.set_title(
            f"{prefix}: {lbl_str}\nTop-Bot: {asym:+.1f} | Az: {var.azimuth:.1f}°",
            fontsize=9,
            fontweight="bold" if i == 0 else "normal",
            color=color,
        )
        ax.axis("off")

    fig.suptitle(
        f"Augmentation Grid (4x4) — Image: {img_id}\n"
        f"Policy: '{args.policy}' (delta={delta:.1f}°, s={s})\n"
        f"Blue = Depth (0), Red = Rise (1) | Label flips with vflip/negation",
        fontsize=12,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_id = img_id.replace(".png", "")
    out_file = out_dir / f"augment_{args.policy}_{clean_id}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Generated 16-variant augmentation grid -> {out_file}")


if __name__ == "__main__":
    main()
