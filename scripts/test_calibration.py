"""
Diagnostic calibration script: per-class R analysis and physical convention test.
READ-ONLY: Does NOT modify data/folds.csv or configs/config.yaml.
Run from repo root: python scripts/test_calibration.py
"""

import os
import cv2
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.abspath("."))
from src.canonical import calibrate, dark_minus_bright_deg, circ_mean_R


def main():
    print("=" * 60, flush=True)
    print("CALIBRATION & R-VALUE FORENSICS", flush=True)
    print("=" * 60, flush=True)

    # 1. Load data
    train_cache = "data/processed/train_images.npy"
    if os.path.exists(train_cache):
        print(f"Loading cached images from {train_cache}...", flush=True)
        images = np.load(train_cache, mmap_mode="r")
    else:
        print("Processed cache not found. Reading from raw...", flush=True)
        import cv2
        df = pd.read_csv("data/raw/train_metadata.csv")
        images = [
            cv2.imread(os.path.join("data/raw/train_images", f), cv2.IMREAD_GRAYSCALE)
            for f in df["image_id"]
        ]
        images = np.stack(images)

    meta_path = "data/raw/train_metadata.csv"
    train_df = pd.read_csv(meta_path)
    azs = train_df["sun_azimuth_angle"].values
    labels = train_df["label"].values

    print(f"Dataset: {len(images)} images | Prior pi1 (Rise): {labels.mean():.4f}", flush=True)
    print(f"Class 0 (Depth): {(labels == 0).sum()} images | Class 1 (Rise): {(labels == 1).sum()} images", flush=True)

    # 2. Azimuth-only probe
    print("\n--- 1. Azimuth-Only Logistic Probe ---", flush=True)
    az_rad = np.radians(azs)
    X_az = np.stack([np.sin(az_rad), np.cos(az_rad)], axis=1)
    clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(X_az, labels)
    preds = clf.predict(X_az)
    az_ba = balanced_accuracy_score(labels, preds)
    print(f"Azimuth-only Logistic BA (train set): {az_ba:.4f}", flush=True)
    if az_ba > 0.55:
        print(f"  [NOTE] BA={az_ba:.4f} > 0.55 — azimuth distribution alone predicts label.", flush=True)
        print("         This is a known label-leakage concern for FiLM conditioning.", flush=True)

    # 3. Calibration sweep
    print("\n--- 2. Calibration Sweep Over Crop Sizes ---", flush=True)
    results = []
    for crop in [None, 192, 160, 128, 96, 64, 48, 32]:
        calib = calibrate(images, azs, labels, center_crop=crop)
        label = f"{crop}x{crop}" if crop else "256x256"
        results.append({
            "crop": str(crop) if crop else "Full (256)",
            "delta": calib["delta"],
            "s": calib["s"],
            "R": calib["R"],
            "R_other_s": calib["R_other_s"],
        })
        print(
            f"  Crop {label:>7}: R={calib['R']:.4f}  s={calib['s']:+d}  "
            f"delta={calib['delta']:6.2f}°  |  R(other_s)={calib['R_other_s']:.4f}",
            flush=True,
        )

    best = max(results, key=lambda x: x["R"])
    print(
        f"\nBest: crop={best['crop']}  R={best['R']:.4f}  s={best['s']}  delta={best['delta']:.2f}°",
        flush=True,
    )

    # 4. Per-class polarity — full image, no crop, using best s
    print("\n--- 3. Per-Class Polarity (Full 256px, best s) ---", flush=True)
    phi_all = np.array([dark_minus_bright_deg(im) for im in images])
    s_best = best["s"]

    phi_c0 = phi_all[labels == 0]
    az_c0 = azs[labels == 0]
    phi_c1 = phi_all[labels == 1]
    az_c1 = azs[labels == 1]

    residual_c0 = (phi_c0 - s_best * az_c0) % 360.0
    residual_c1 = (phi_c1 - s_best * az_c1) % 360.0

    mode_c0, R_c0 = circ_mean_R(residual_c0)
    mode_c1, R_c1 = circ_mean_R(residual_c1)
    separation = abs((mode_c1 - mode_c0 + 180.0) % 360.0 - 180.0)

    print(f"  Depth (Class 0): mode_delta={mode_c0:6.2f}°  R={R_c0:.4f}  n={len(phi_c0)}", flush=True)
    print(f"  Rise  (Class 1): mode_delta={mode_c1:6.2f}°  R={R_c1:.4f}  n={len(phi_c1)}", flush=True)
    print(f"  Angular separation between class modes: {separation:.2f}° (ideal ~180°)", flush=True)

    # 5. Class-0-only calibration (the friend's AI's suggestion)
    print("\n--- 4. Class-0-Only Calibration (Craters Only) ---", flush=True)
    print("  This tests whether craters alone give R > 0.4 ...", flush=True)
    imgs_c0 = images[labels == 0]
    az_c0_arr = azs[labels == 0]
    labels_c0 = np.zeros(len(imgs_c0), dtype=int)  # all depth, label=0
    calib_c0 = calibrate(imgs_c0, az_c0_arr, labels_c0)
    print(
        f"  Class-0-only: R={calib_c0['R']:.4f}  s={calib_c0['s']}  "
        f"delta={calib_c0['delta']:.2f}°  |  R(other_s)={calib_c0['R_other_s']:.4f}",
        flush=True,
    )

    # 6. Physical convention test: delta=90, s=-1
    print("\n--- 5. Physical Convention Test (delta=90°, s=-1) ---", flush=True)
    print("  If images are North-up orbital tiles, this is the expected convention.", flush=True)
    physical_toward = (phi_all + np.where(labels == 1, 180.0, 0.0)) % 360.0
    physical_residuals = (physical_toward - (-1) * azs) % 360.0
    mode_phys, R_phys = circ_mean_R(physical_residuals)
    print(f"  Physical (delta=90, s=-1): mode={mode_phys:.2f}°  R={R_phys:.4f}", flush=True)
    print(f"  (If this R is high and mode~90°, the physical convention is correct)", flush=True)

    # 7. Summary
    print("\n" + "=" * 60, flush=True)
    print("DIAGNOSTIC SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Combined calibration best R:  {best['R']:.4f}  (threshold: 0.40)", flush=True)
    print(f"  Class-0-only R:               {calib_c0['R']:.4f}", flush=True)
    print(f"  Class-0 per-class R:          {R_c0:.4f}", flush=True)
    print(f"  Class-1 per-class R:          {R_c1:.4f}", flush=True)
    print(f"  Class mode separation:        {separation:.1f}°  (ideal: 180°)", flush=True)
    print(f"  Physical convention R:        {R_phys:.4f}", flush=True)
    print(f"  Azimuth-only BA:              {az_ba:.4f}", flush=True)

    print("\nVERDICT:", flush=True)
    if calib_c0["R"] >= 0.40:
        print(
            f"  [RECOVERABLE] Class-0-only R={calib_c0['R']:.4f} >= 0.40.\n"
            "  Canonicalization may work using parameters from crater-only calibration.\n"
            f"  Use delta={calib_c0['delta']:.2f}°, s={calib_c0['s']} and inspect canonical mean images.",
            flush=True,
        )
    elif R_phys >= 0.40:
        print(
            f"  [PHYSICAL OK] Physical convention R={R_phys:.4f} >= 0.40 even though heuristic fails.\n"
            "  The pixel heuristic is wrong but the physical model (North-up) may still apply.\n"
            "  Try canonicalization with delta=90°, s=-1 and inspect canonical mean images.",
            flush=True,
        )
    else:
        print(
            f"  [DEAD END] Both class-0-only R ({calib_c0['R']:.4f}) and physical R ({R_phys:.4f}) < 0.40.\n"
            "  The heuristic cannot calibrate this dataset at all.\n"
            "  Raw-frame pivot confirmed correct. Do not revisit canonicalization.",
            flush=True,
        )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
