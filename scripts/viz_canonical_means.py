"""
scripts/viz_canonical_means.py
Generate per-class canonical mean images for both handedness options.

Usage:
    python scripts/viz_canonical_means.py
    python scripts/viz_canonical_means.py --s -1 --delta 46.70   # frozen config params
    python scripts/viz_canonical_means.py --s 1  --delta 135.18  # alternative

Output (saved to reports/figures/):
    canonical_means_s{s}_delta{delta:.1f}.png  — side-by-side: Class 0 | Class 1

MANUAL step: inspect both outputs.
  - The CORRECT (s, delta) is the one where Class 0 and Class 1 means look DIFFERENT
    (e.g. Class 0 bright-top, Class 1 dark-top — or vice versa).
  - If NEITHER separates → escalate per AGENTS.md §11.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Allow running from repo root ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.canonical import canonicalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize canonical per-class mean images."
    )
    p.add_argument(
        "--s",
        type=int,
        choices=[-1, 1],
        default=None,
        help="Handedness: -1 or +1. If omitted, runs both.",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Azimuth offset in degrees. If omitted, uses config values.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="reports/figures",
        help="Output directory for figures (default: reports/figures).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit to first N images per class (default: all).",
    )
    return p.parse_args()


# Default parameter pairs: (s, delta)
_PARAM_PAIRS = [
    (-1, 46.70),  # frozen config parameters
    (+1, 135.18),  # alternative handedness
]


def load_cache(data_dir: str) -> tuple[np.ndarray, list[dict]]:
    """Load train images and index.json."""
    cache_dir = Path(data_dir) / "processed"
    index_path = cache_dir / "index.json"
    img_path = cache_dir / "train_images.npy"

    if not img_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_dir}. Run `python -m src.dataset --mode cache` first."
        )
    with index_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    images = np.load(img_path, mmap_mode="r")
    return images, idx["train"]


def compute_canonical_means(
    images: np.ndarray,
    records: list[dict],
    s: int,
    delta: float,
    max_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute per-class canonical mean images.

    Returns:
        mean_0: float32 (256, 256) mean canonical image for class 0 (Depth)
        mean_1: float32 (256, 256) mean canonical image for class 1 (Rise)
        n0: number of class-0 images used
        n1: number of class-1 images used
    """
    acc0 = np.zeros((256, 256), dtype=np.float64)
    acc1 = np.zeros((256, 256), dtype=np.float64)
    n0 = n1 = 0

    for rec in records:
        lbl = int(rec["label"])
        if lbl == 0 and (max_images is None or n0 < max_images):
            img = np.array(images[records.index(rec)])
            az = float(rec["sun_azimuth_angle"])
            canon = canonicalize(img, az, delta, s)
            acc0 += canon.astype(np.float64)
            n0 += 1
        elif lbl == 1 and (max_images is None or n1 < max_images):
            img = np.array(images[records.index(rec)])
            az = float(rec["sun_azimuth_angle"])
            canon = canonicalize(img, az, delta, s)
            acc1 += canon.astype(np.float64)
            n1 += 1

    mean0 = (acc0 / max(n0, 1)).astype(np.float32)
    mean1 = (acc1 / max(n1, 1)).astype(np.float32)
    return mean0, mean1, n0, n1


def compute_canonical_means_fast(
    images: np.ndarray,
    records: list[dict],
    s: int,
    delta: float,
    max_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Vectorised version using integer indices (faster)."""
    acc0 = np.zeros((256, 256), dtype=np.float64)
    acc1 = np.zeros((256, 256), dtype=np.float64)
    n0 = n1 = 0

    for i, rec in enumerate(records):
        lbl = int(rec["label"])
        if lbl == 0 and (max_images is None or n0 < max_images):
            img = np.array(images[i])
            az = float(rec["sun_azimuth_angle"])
            acc0 += canonicalize(img, az, delta, s).astype(np.float64)
            n0 += 1
        elif lbl == 1 and (max_images is None or n1 < max_images):
            img = np.array(images[i])
            az = float(rec["sun_azimuth_angle"])
            acc1 += canonicalize(img, az, delta, s).astype(np.float64)
            n1 += 1

    mean0 = (acc0 / max(n0, 1)).astype(np.float32)
    mean1 = (acc1 / max(n1, 1)).astype(np.float32)
    return mean0, mean1, n0, n1


def save_side_by_side(
    mean0: np.ndarray,
    mean1: np.ndarray,
    s: int,
    delta: float,
    out_dir: str,
    n0: int,
    n1: int,
) -> Path:
    """Save side-by-side Class 0 | Class 1 mean image as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mean0, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Class 0 — Depth (Crater)\nn={n0}", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(mean1, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Class 1 — Rise (Mound)\nn={n1}", fontsize=13)
        axes[1].axis("off")

        s_str = "m1" if s == -1 else "p1"
        fig.suptitle(
            f"Canonical mean images  |  s={s:+d}, δ={delta:.2f}°\n"
            f"Bright top = sun at top (expected for correct handedness)",
            fontsize=12,
        )
        plt.tight_layout()
        out_path = Path(out_dir) / f"canonical_means_s{s_str}_delta{delta:.1f}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    except ImportError:
        # Fallback: save raw npy if matplotlib unavailable
        out_dir_path = Path(out_dir)
        np.save(out_dir_path / f"canonical_mean0_s{s}_delta{delta:.1f}.npy", mean0)
        np.save(out_dir_path / f"canonical_mean1_s{s}_delta{delta:.1f}.npy", mean1)
        print("  matplotlib not available — saved .npy arrays instead")
        return out_dir_path / f"canonical_means_s{s}_delta{delta:.1f}.npy"


def print_asymmetry_stats(
    mean0: np.ndarray, mean1: np.ndarray, s: int, delta: float
) -> None:
    """Print top-minus-bottom asymmetry for each class mean."""
    h = 256
    top0 = float(mean0[: h // 2].mean())
    bot0 = float(mean0[h // 2 :].mean())
    top1 = float(mean1[: h // 2].mean())
    bot1 = float(mean1[h // 2 :].mean())
    print(f"\n  s={s:+d}, δ={delta:.2f}°")
    print(
        f"    Class 0 (Depth):  top={top0:.2f}  bot={bot0:.2f}  asymmetry={top0-bot0:+.2f}"
    )
    print(
        f"    Class 1 (Rise):   top={top1:.2f}  bot={bot1:.2f}  asymmetry={top1-bot1:+.2f}"
    )
    separable = (top0 - bot0) * (top1 - bot1) < 0
    if separable:
        print("    ✅ SEPARABLE — classes have OPPOSITE top-bottom asymmetry!")
        print(
            "    → This is likely the correct (s, delta). Commit these frozen config values."
        )
    else:
        print("    ❌ NOT separable — classes have SAME-SIGN asymmetry.")
        print("    → Try the other handedness option.")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cache...")
    images, records = load_cache(args.data_dir)
    print(f"Loaded {len(records)} training records.")

    # Determine which (s, delta) pairs to run
    if args.s is not None and args.delta is not None:
        pairs = [(args.s, args.delta)]
    elif args.s is not None:
        # Use default delta for the given s
        pairs = [(args.s, d) for (sv, d) in _PARAM_PAIRS if sv == args.s]
        if not pairs:
            pairs = [(args.s, _PARAM_PAIRS[0][1])]
    elif args.delta is not None:
        pairs = [(-1, args.delta), (+1, args.delta)]
    else:
        pairs = _PARAM_PAIRS

    print(f"\nRunning {len(pairs)} handedness configuration(s)...")
    for s, delta in pairs:
        print(f"\n  Computing canonical means with s={s:+d}, δ={delta:.2f}°...")
        mean0, mean1, n0, n1 = compute_canonical_means_fast(
            images, records, s, delta, max_images=args.max_images
        )
        print(f"  Class 0: {n0} images, Class 1: {n1} images")
        print_asymmetry_stats(mean0, mean1, s, delta)
        out_path = save_side_by_side(mean0, mean1, s, delta, str(out_dir), n0, n1)
        print(f"  Saved → {out_path}")

    print(
        "\n──────────────────────────────────────────────────────────────\n"
        "MANUAL ACTION REQUIRED:\n"
        "Open the saved figures. The correct (s, δ) is the one where\n"
        "Class 0 and Class 1 mean images look VISIBLY DIFFERENT.\n"
        "Expected: Class 0 bright-top (or dark-top) AND Class 1 the opposite.\n"
        "Once confirmed, the frozen config values are validated.\n"
        "If NEITHER separates → escalate per AGENTS.md §11.\n"
        "──────────────────────────────────────────────────────────────"
    )


if __name__ == "__main__":
    main()
