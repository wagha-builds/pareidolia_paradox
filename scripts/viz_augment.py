"""
Visualizes a 16-variant augmentation grid annotated with solar azimuth and label.
Adheres strictly to AGENTS.md §4 and §5 ("Look at pictures: the augmentation grid").
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from src.transforms import get_canonical_train_transforms, get_raw_train_transforms


def draw_sun_arrow(ax, azimuth_deg: float, length: float = 25.0):
    """Draws an arrow pointing toward the sun (theta = azimuth)."""
    h, w = 256, 256
    cx, cy = w / 2, h / 2
    rad = np.radians(azimuth_deg)
    # Azimuth in standard coordinates (from +x counter-clockwise)
    dx = length * np.cos(rad)
    dy = -length * np.sin(rad)  # Inverted y for image coordinates
    ax.annotate(
        "",
        xy=(cx + dx, cy + dy),
        xytext=(cx, cy),
        arrowprops=dict(facecolor="yellow", edgecolor="black", width=1.5, headwidth=6),
    )


def visualize_augmentations(
    policy: str = "raw",
    image_id: str = "",
    index_path: str = "data/processed/index.json",
    cache_path: str = "data/processed/train_images.npy",
    output_dir: str = "reports/figures",
):
    with open(index_path, "r") as f:
        idx_data = json.load(f)

    train_rows = idx_data["train"]
    if image_id:
        matches = [(i, r) for i, r in enumerate(train_rows) if r["image_id"] == image_id]
        if not matches:
            raise ValueError(f"image_id '{image_id}' not found in train set!")
        idx, target_row = matches[0]
    else:
        idx = 0
        target_row = train_rows[0]
        image_id = target_row["image_id"]

    images_mmap = np.load(cache_path, mmap_mode="r")
    base_img = images_mmap[idx]
    base_az = target_row["sun_azimuth_angle"]
    base_label = target_row["label"]

    from src.transforms import Sample
    if policy == "canonical":
        transform = get_canonical_train_transforms()
    else:
        transform = get_raw_train_transforms(delta=46.7, s=-1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 13))
    label_map = {0: "Depth (0)", 1: "Rise (1)"}

    for i in range(16):
        ax = axes[i // 4, i % 4]
        if i == 0:
            # First panel is original unaugmented image
            img, az, lbl = base_img.copy(), base_az, base_label
            title_prefix = "Original"
        else:
            sample = Sample(image=base_img.copy(), azimuth=base_az, label=base_label)
            sample = transform(sample)
            img, az, lbl = sample.image, sample.azimuth, sample.label
            title_prefix = f"Var {i}"

        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        draw_sun_arrow(ax, az)
        ax.set_title(
            f"{title_prefix}\n{label_map.get(lbl, lbl)} | az={az:5.1f}°",
            fontsize=9,
            fontweight="bold",
            color="navy" if lbl == 1 else "darkred",
        )
        ax.axis("off")

    plt.suptitle(
        f"16-Variant Augmentation Grid: {image_id} (Policy: {policy})\n"
        f"Base: {label_map[base_label]} @ Azimuth {base_az:.1f}°",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"augment_{policy}_{image_id}.png")
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved augmentation grid to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Augmentation Grid")
    parser.add_argument("--policy", choices=["raw", "canonical"], default="raw")
    parser.add_argument("--image-id", type=str, default="")
    args = parser.parse_args()

    visualize_augmentations(policy=args.policy, image_id=args.image_id)


if __name__ == "__main__":
    main()
