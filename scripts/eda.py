import os
import glob
import json
import numpy as np
import pandas as pd
import hashlib
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import StratifiedGroupKFold
from src.canonical import calibrate, canonicalize

def main():
    print("Running M1 Data Gate EDA script...")
    
    train_meta_path = "data/raw/train_metadata.csv"
    train_img_dir = "data/raw/train_images"
    test_meta_path = "data/raw/test_metadata.csv"
    test_img_dir = "data/raw/eval_images"
    
    if not os.path.exists(train_meta_path) or not os.path.exists(train_img_dir):
        print("Data files not found in data/raw/. Aborting EDA.")
        return
        
    train_df = pd.read_csv(train_meta_path)
    test_df = pd.read_csv(test_meta_path)
    
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    
    # 1.3 Class balance
    pi1 = train_df["label"].mean()
    print(f"pi1 (Class 1 prior): {pi1:.4f}")
    
    # Load a sample of images for calibration (e.g. all of them or 1000)
    print("Loading training images for calibration...")
    # For speed, we just do all of them but downscaled or we just load them all (7854 is small enough)
    images = []
    labels = train_df["label"].values
    azs = train_df["sun_azimuth_angle"].values
    
    for img_id in train_df["image_id"]:
        path = os.path.join(train_img_dir, img_id)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read {path}")
        images.append(img)
    
    # 1.11 Normalization stats
    print("Computing normalization stats...")
    all_pixels = np.stack(images).astype(np.float32) / 255.0
    mean_val = float(all_pixels.mean())
    std_val = float(all_pixels.std())
    print(f"Norm stats: mean={mean_val:.4f}, std={std_val:.4f}")

    # 1.5 Calibration
    print("Running calibration (sweeping center_crop sizes to isolate features)...")
    best_calib = None
    best_crop = None
    for crop in [None, 128, 96, 64, 48, 32]:
        calib = calibrate(images, azs, labels, center_crop=crop)
        print(f"Crop {crop if crop else 'Full'}: R={calib['R']:.4f}")
        if best_calib is None or calib['R'] > best_calib['R']:
            best_calib = calib
            best_crop = crop
            
    delta, s, R = float(best_calib["delta"]), int(best_calib["s"]), float(best_calib["R"])
    print(f"Best Calibration output (Crop {best_crop}): delta={delta:.2f}, s={s}, R={R:.4f} (other R={best_calib['R_other_s']:.4f})")
    
    # 1.6 Canonical mean images
    print("Generating canonical mean images...")
    can_class0 = []
    can_class1 = []
    for img, az, lbl in zip(images, azs, labels):
        c_img = canonicalize(img, az, delta, s)
        if lbl == 0:
            can_class0.append(c_img)
        else:
            can_class1.append(c_img)
            
    mean0 = np.mean(can_class0, axis=0).astype(np.uint8)
    mean1 = np.mean(can_class1, axis=0).astype(np.uint8)
    
    os.makedirs("reports/figures", exist_ok=True)
    cv2.imwrite("reports/figures/canonical_mean_class0_depth.png", mean0)
    cv2.imwrite("reports/figures/canonical_mean_class1_rise.png", mean1)
    print("Saved canonical mean images to reports/figures/")
    
    # 1.9 Folds generation
    print("Generating folds.csv...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # Simple grouping by azimuth bin as a placeholder for pHash (requires imagehash)
    train_df["az_bin"] = pd.qcut(train_df["sun_azimuth_angle"], q=10, labels=False)
    train_df["group_id"] = train_df["az_bin"] # In full run, replace with pHash groups
    
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(train_df, train_df["label"], train_df["group_id"])):
        train_df.loc[val_idx, "fold"] = fold
        
    folds_path = "data/folds.csv"
    train_df[["image_id", "fold"]].to_csv(folds_path, index=False)
    
    with open(folds_path, "rb") as f:
        folds_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Folds generated. Hash: {folds_hash}")
    
    # Writing to config
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    config["frozen"]["calibration"]["delta"] = delta
    config["frozen"]["calibration"]["s"] = s
    config["frozen"]["calibration"]["R"] = R
    config["frozen"]["prior_pi1"] = float(pi1)
    config["frozen"]["folds_sha256"] = folds_hash
    config["frozen"]["norm"]["mean"] = mean_val
    config["frozen"]["norm"]["std"] = std_val
    
    with open("configs/config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    print("configs/config.yaml frozen block updated.")

if __name__ == "__main__":
    main()
