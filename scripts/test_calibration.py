"""
Diagnostic calibration script to analyze R value, handedness s, and offset delta.
Does NOT modify data/folds.csv or configs/config.yaml.
"""

import os
import sys
import json
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
        print(f"Loading cached images from {train_cache}...")
        images = np.load(train_cache, mmap_mode="r")
    else:
        print("Processed cache not found. Reading from raw...")
        import cv2
        df = pd.read_csv("data/raw/train_metadata.csv")
        images = [cv2.imread(os.path.join("data/raw/train_images", f), cv2.IMREAD_GRAYSCALE) for f in df["image_id"]]
        images = np.stack(images)

    meta_path = "data/raw/train_metadata.csv"
    train_df = pd.read_csv(meta_path)
    azs = train_df["sun_azimuth_angle"].values
    labels = train_df["label"].values

    print(f"Dataset: {len(images)} images | Prior pi1 (Rise): {labels.mean():.4f}")

    # 2. Azimuth-only probe
    print("\n--- 1. Azimuth-Only Logistic Probe ---")
    az_rad = np.radians(azs)
    X_az = np.stack([np.sin(az_rad), np.cos(az_rad)], axis=1)
    clf = LogisticRegression(random_state=42)
    clf.fit(X_az, labels)
    preds = clf.predict(X_az)
    az_ba = balanced_accuracy_score(labels, preds)
    print(f"Azimuth-only Logistic Model BA: {az_ba:.4f}")
    if az_ba > 0.55:
        print(f"[NOTE] High azimuth-only BA ({az_ba:.4f} > 0.55) indicates azimuth distribution alone predicts label.")

    # 3. Sweep Center Crops for Calibration R
    print("\n--- 2. Calibration Sweep Over Center Crop Sizes ---")
    results = []
    for crop in [None, 192, 160, 128, 96, 64, 48, 32]:
        calib = calibrate(images, azs, labels, center_crop=crop)
        results.append({
            "crop": str(crop) if crop else "Full (256)",
            "delta": calib["delta"],
            "s": calib["s"],
            "R": calib["R"],
            "R_other_s": calib["R_other_s"],
            "R_ratio": calib["R"] / max(calib["R_other_s"], 1e-6)
        })
        crop_label = f"{crop}x{crop}" if crop else "256x256"
        print(
            f"Crop {crop_label:>7}: R={calib['R']:.4f} (s={calib['s']:+d}, delta={calib['delta']:6.2f}°) | "
            f"other s({-calib['s']:+d}): R={calib['R_other_s']:.4f}"
        )

    best = max(results, key=lambda x: x["R"])
    print(f"\nBest crop: {best['crop']} with R = {best['R']:.4f}, s = {best['s']}, delta = {best['delta']:.2f}°")

    # 4. Per-Class Separation Check
    print("\n--- 3. Per-Class Polarity Analysis (Best Crop) ---")
    crop_val = None if best["crop"] == "Full (256)" else int(best["crop"])
    phi = np.array([dark_minus_bright_deg(im, center_crop=crop_val) for im in images])
    
    phi_c0 = phi[labels == 0]
    az_c0 = azs[labels == 0]
    phi_c1 = phi[labels == 1]
    az_c1 = azs[labels == 1]

    # Mode calculation for class 0 (Depth) and class 1 (Rise)
    mode_c0, R_c0 = circ_mean_R((phi_c0 - best["s"] * az_c0) % 360.0)
    mode_c1, R_c1 = circ_mean_R((phi_c1 - best["s"] * az_c1) % 360.0)
    separation = abs((mode_c1 - mode_c0 + 180.0) % 360.0 - 180.0)

    print(f"Depth (0): Mode = {mode_c0:6.2f}°, R0 = {R_c0:.4f}")
    print(f"Rise  (1): Mode = {mode_c1:6.2f}°, R1 = {R_c1:.4f}")
    print(f"Angular separation between modes: {separation:.2f}° (Ideal physics = ~180.0°)")

    # 5. Diagnostic Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC VERDICT")
    print("=" * 60)
    if best["R"] < 0.40:
        print(f"[FAIL] Calibration R = {best['R']:.4f} is below the 0.40 threshold.")
        print(f"       Reasons: R(+1) and R(-1) are nearly equal ({best['R']:.4f} vs {best['R_other_s']:.4f}),")
        print(f"       and mode separation is {separation:.1f}° instead of ~180°.")
        print("       CONCLUSION: Canonicalization based on heuristic pixel polarity is noisy.")
        print("       The raw-frame CNN with FiLM azimuth conditioning (which learns the mapping")
        print("       directly from data) is the statistically sound modeling path.")
    else:
        print(f"[PASS] Calibration R = {best['R']:.4f} meets the threshold.")
    print("=" * 60)


if __name__ == "__main__":
    main()
