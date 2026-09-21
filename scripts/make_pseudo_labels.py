"""scripts/make_pseudo_labels.py — Generate pseudo-label CSV from test predictions.

Usage:
    python scripts/make_pseudo_labels.py \
        --test-probs artifacts/20260916_ensemble_e4_e5_swin_ba0.7361/test_probs_tta_20260916-223312.npy \
        --lower-thresh 0.41 \
        --upper-thresh 0.68 \
        --output data/pseudo_labels.csv

Keeps only test images where P(Rise) <= lower-thresh (Depth) or P(Rise) >= upper-thresh (Rise).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from test predictions")
    parser.add_argument("--test-probs", required=True, type=str, help="Path to test_probs .npy file")
    parser.add_argument("--lower-thresh", type=float, default=0.30,
                        help="Confidence threshold for Depth: keep p <= t")
    parser.add_argument("--upper-thresh", type=float, default=0.70,
                        help="Confidence threshold for Rise: keep p >= t")
    parser.add_argument("--output", type=str, default="data/pseudo_labels.csv")
    parser.add_argument("--test-meta", type=str, default="data/raw/test_metadata.csv")
    args = parser.parse_args()

    probs = np.load(args.test_probs)
    meta = pd.read_csv(args.test_meta, dtype={"image_id": str})

    assert len(probs) == len(meta), f"Length mismatch: {len(probs)} probs vs {len(meta)} rows"

    depth_mask = probs <= args.lower_thresh
    rise_mask = probs >= args.upper_thresh

    depth_df = meta[depth_mask].copy()
    depth_df["label"] = 0
    depth_df["pseudo_prob"] = probs[depth_mask]

    rise_df = meta[rise_mask].copy()
    rise_df["label"] = 1
    rise_df["pseudo_prob"] = probs[rise_mask]

    out = pd.concat([depth_df, rise_df]).sort_values("pseudo_prob").reset_index(drop=True)

    print(f"Pseudo-label summary (lower={args.lower_thresh}, upper={args.upper_thresh}):")
    print(f"  Depth (p <= {args.lower_thresh}):        {depth_mask.sum():4d} images")
    print(f"  Rise  (p >= {args.upper_thresh}):        {rise_mask.sum():4d} images")
    print(f"  Total:                        {len(out):4d} / {len(meta)} test images ({len(out)/len(meta)*100:.1f}%)")
    print(f"\nAzimuth quadrant breakdown:")
    for lo, hi in [(0, 90), (90, 180), (180, 270), (270, 360)]:
        qmask = (out["sun_azimuth_angle"] >= lo) & (out["sun_azimuth_angle"] < hi)
        n_d = ((out["label"] == 0) & qmask).sum()
        n_r = ((out["label"] == 1) & qmask).sum()
        print(f"  Az {lo:3d}-{hi:3d}: Depth={n_d:3d}  Rise={n_r:3d}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved pseudo_labels.csv -> {args.output}")


if __name__ == "__main__":
    main()
