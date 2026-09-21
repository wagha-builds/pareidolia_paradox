"""
scripts/eval_sun_gate.py — Go/No-Go Gate Evaluator for Option A (Sun Predictor).

Evaluates:
1. Rotation Equivariance Error (mean angular error on held-out random rotations).
2. Physical Separation Metric:
   Rotate images so predicted sun is at top (90 deg).
   Compute top-minus-bottom brightness asymmetry A = mean(I_top) - mean(I_bottom).
   Compute point-biserial correlation r(A, y) with ground truth labels y in {0, 1}.
3. Decision:
   - GATE PASS: r >= 0.35 (canonicalization genuinely reveals relief)
   - GATE FAIL: r < 0.20 (180 deg ambiguity unresolved or signal in noise)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch

from src.sun_predictor import SunDirectionModel

SQRT2 = 2.0**0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Go/No-Go Gate for Sun Predictor.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="experiments/sun_predictor/sun_predictor_best.pt",
        help="Path to trained sun predictor checkpoint.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny.fb_in22k_ft_in1k",
        help="Backbone architecture.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of training samples to evaluate for physical correlation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating Sun Predictor Gate using checkpoint: {args.ckpt}")
    print(f"Device: {device}")

    # 1. Load model
    model = SunDirectionModel(backbone_name=args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 2. Load cached images and labels
    train_imgs = np.load("data/processed/train_images.npy")
    df = pd.read_csv("data/raw/train_metadata.csv")
    y_all = df["label"].values

    n = min(args.num_samples, len(train_imgs))
    np.random.seed(42)
    indices = np.random.choice(len(train_imgs), size=n, replace=False)
    sub_imgs = train_imgs[indices]
    sub_y = y_all[indices]

    norm_mean = 100.86
    norm_std = 54.55

    # 3. Test Rotation Equivariance Error
    print("\n[Metric 1/2] Measuring Rotation Equivariance Error...")
    angular_errors = []
    with torch.no_grad():
        for i in range(min(500, n)):
            img = sub_imgs[i]
            h, w = img.shape
            alpha_deg = float(np.random.uniform(10.0, 350.0))

            M_orig = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 0.0, SQRT2)
            img_orig = cv2.warpAffine(img, M_orig, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

            M_rot = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), alpha_deg, SQRT2)
            img_rot = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

            t_orig = torch.from_numpy((img_orig.astype(np.float32) - norm_mean) / norm_std).unsqueeze(0).unsqueeze(0).to(device)
            t_rot = torch.from_numpy((img_rot.astype(np.float32) - norm_mean) / norm_std).unsqueeze(0).unsqueeze(0).to(device)

            v_orig = model(t_orig).cpu().numpy()[0]
            v_rot = model(t_rot).cpu().numpy()[0]

            theta_orig = np.degrees(np.arctan2(v_orig[1], v_orig[0])) % 360.0
            theta_rot = np.degrees(np.arctan2(v_rot[1], v_rot[0])) % 360.0

            expected_rot = (theta_orig + alpha_deg) % 360.0
            diff = abs(theta_rot - expected_rot)
            diff = min(diff, 360.0 - diff)
            angular_errors.append(diff)

    mean_ang_err = float(np.mean(angular_errors))
    median_ang_err = float(np.median(angular_errors))
    print(f"  Mean Angular Error:   {mean_ang_err:.2f} deg")
    print(f"  Median Angular Error: {median_ang_err:.2f} deg")

    # 4. Test Physical Separation Metric (Top-minus-bottom brightness asymmetry)
    print("\n[Metric 2/2] Measuring Physical Separation Correlation r(A, y)...")
    asymmetries = []
    with torch.no_grad():
        for i in range(n):
            img = sub_imgs[i]
            h, w = img.shape

            M_orig = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 0.0, SQRT2)
            img_orig = cv2.warpAffine(img, M_orig, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            t_orig = torch.from_numpy((img_orig.astype(np.float32) - norm_mean) / norm_std).unsqueeze(0).unsqueeze(0).to(device)

            v_orig = model(t_orig).cpu().numpy()[0]
            theta_sun = np.degrees(np.arctan2(v_orig[1], v_orig[0])) % 360.0

            # Rotate so predicted sun is at top (90 deg)
            rot_angle = (90.0 - theta_sun) % 360.0
            M_canon = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), rot_angle, SQRT2)
            img_canon = cv2.warpAffine(img, M_canon, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

            # Central crop 128x128
            c_img = img_canon[64:192, 64:192].astype(np.float32)
            top_half = c_img[:64, :]
            bottom_half = c_img[64:, :]

            asym = float(top_half.mean() - bottom_half.mean())
            asymmetries.append(asym)

    asymmetries = np.array(asymmetries)
    r_val, p_val = pearsonr(asymmetries, sub_y)

    print("=" * 70)
    print("GO / NO-GO GATE RESULTS")
    print("=" * 70)
    print(f"Equivariance Median Error: {median_ang_err:.2f}° (random chance = 90.0°)")
    print(f"Physical Asymmetry Correlation r(A, y): {r_val:+.4f} (p = {p_val:.2e})")
    print(f"Metadata Baseline Correlation:          +0.0110")
    print("-" * 70)

    if r_val >= 0.35:
        verdict = "GATE PASS"
        action = "Proceed with full training & re-canonicalization. Expected BA: 0.82–0.86+."
    elif r_val >= 0.20:
        verdict = "GATE INCONCLUSIVE / MODERATE"
        action = "Partial illumination signal detected. Can be blended into FiLM."
    else:
        verdict = "GATE FAIL"
        action = "Illumination signal is insufficient or 180° ambiguity unresolved. ABORT to Option C."

    print(f"VERDICT: {verdict}")
    print(f"ACTION:  {action}")
    print("=" * 70 + "\n")

    out_file = Path("experiments/sun_predictor/gate_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "r_asymmetry": float(r_val),
                "p_value": float(p_val),
                "median_angular_error_deg": float(median_ang_err),
                "mean_angular_error_deg": float(mean_ang_err),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
