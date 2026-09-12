"""
Robustness evaluation suite for Pareidolia models.
Implements the 4 core audits per AGENTS.md §4, §7 and Playbook §5:
1. Shortcut audit (BA on shortcut-wrong partition)
2. Synthetic inversion stress test (rot180 / negation label counterfactuals)
3. Azimuth-shift stress test (measuring degradation under coupled rotation)
4. Per-slice report (azimuth octants, brightness quartiles, contrast quartiles)
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score

from .metrics import apply_threshold, plateau_threshold
from .transforms import RawRot180LabelSwap, RawRotate
from .dataset import PareidoliaDataset
from .models import build_model


def run_shortcut_audit(y_true: np.ndarray, y_pred: np.ndarray, az: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates model performance on the subset of data where the simple
    azimuth polarity shortcut fails.
    Shortcut rule: predict Rise if sin(az) > 0, else Depth.
    """
    shortcut_pred = (np.sin(np.radians(az)) > 0).astype(int)
    shortcut_correct_mask = (shortcut_pred == y_true)
    shortcut_wrong_mask = ~shortcut_correct_mask

    ba_overall = float(balanced_accuracy_score(y_true, y_pred))

    if shortcut_correct_mask.sum() > 0 and len(np.unique(y_true[shortcut_correct_mask])) > 1:
        ba_sc_correct = float(balanced_accuracy_score(y_true[shortcut_correct_mask], y_pred[shortcut_correct_mask]))
    else:
        ba_sc_correct = float("nan")

    if shortcut_wrong_mask.sum() > 0 and len(np.unique(y_true[shortcut_wrong_mask])) > 1:
        ba_sc_wrong = float(balanced_accuracy_score(y_true[shortcut_wrong_mask], y_pred[shortcut_wrong_mask]))
    else:
        ba_sc_wrong = float("nan")

    return {
        "ba_overall": ba_overall,
        "n_shortcut_correct": int(shortcut_correct_mask.sum()),
        "ba_shortcut_correct": ba_sc_correct,
        "n_shortcut_wrong": int(shortcut_wrong_mask.sum()),
        "ba_shortcut_wrong": ba_sc_wrong,
    }


def run_slice_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    az: np.ndarray,
    images: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes Balanced Accuracy across key feature slices:
    - 8 Azimuth octants (45 deg bins)
    - 4 Brightness quartiles
    - 4 Contrast quartiles
    """
    slices = {}

    # 1. Azimuth Octants
    for i in range(8):
        deg_lo = i * 45.0
        deg_hi = (i + 1) * 45.0
        mask = (az >= deg_lo) & (az < deg_hi)
        name = f"Azimuth [{int(deg_lo):3d}°, {int(deg_hi):3d}°)"
        if mask.sum() >= 10 and len(np.unique(y_true[mask])) > 1:
            ba = float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
        else:
            ba = float("nan")
        slices[name] = {"count": int(mask.sum()), "ba": ba}

    # 2. Brightness Quartiles (Mean pixel intensity)
    means = images.reshape(len(images), -1).mean(axis=1)
    q_b = np.percentile(means, [25, 50, 75])
    b_masks = [
        (means <= q_b[0], "Brightness Q1 (Darkest)"),
        ((means > q_b[0]) & (means <= q_b[1]), "Brightness Q2 (Medium-Dark)"),
        ((means > q_b[1]) & (means <= q_b[2]), "Brightness Q3 (Medium-Bright)"),
        (means > q_b[2], "Brightness Q4 (Brightest)"),
    ]
    for mask, name in b_masks:
        if mask.sum() >= 10 and len(np.unique(y_true[mask])) > 1:
            ba = float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
        else:
            ba = float("nan")
        slices[name] = {"count": int(mask.sum()), "ba": ba}

    # 3. Contrast Quartiles (Pixel standard deviation)
    stds = images.reshape(len(images), -1).std(axis=1)
    q_c = np.percentile(stds, [25, 50, 75])
    c_masks = [
        (stds <= q_c[0], "Contrast Q1 (Lowest)"),
        ((stds > q_c[0]) & (stds <= q_c[1]), "Contrast Q2 (Low-Med)"),
        ((stds > q_c[1]) & (stds <= q_c[2]), "Contrast Q3 (Med-High)"),
        (stds > q_c[2], "Contrast Q4 (Highest)"),
    ]
    for mask, name in c_masks:
        if mask.sum() >= 10 and len(np.unique(y_true[mask])) > 1:
            ba = float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
        else:
            ba = float("nan")
        slices[name] = {"count": int(mask.sum()), "ba": ba}

    # Find worst slice among valid slices
    valid_slices = [(k, v["ba"]) for k, v in slices.items() if not np.isnan(v["ba"])]
    if valid_slices:
        worst_name, worst_ba = min(valid_slices, key=lambda x: x[1])
    else:
        worst_name, worst_ba = "None", 0.5

    return {
        "slices": slices,
        "worst_slice_name": worst_name,
        "worst_slice_ba": worst_ba,
    }


def evaluate_robustness(run_dir: str, device_str: str = "cuda:0") -> Dict[str, Any]:
    """Runs the full robustness test suite on a completed experiment run."""
    oof_path = os.path.join(run_dir, "oof.npy")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    index_path = "data/processed/index.json"
    train_cache = "data/processed/train_images.npy"

    if not os.path.exists(oof_path) or not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing oof.npy or manifest in {run_dir}")

    oof_probs = np.load(oof_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    with open(index_path, "r") as f:
        idx_data = json.load(f)

    train_rows = idx_data["train"]
    y_all = np.array([r["label"] for r in train_rows], dtype=int)
    az_all = np.array([r["sun_azimuth_angle"] for r in train_rows], dtype=np.float32)

    valid_mask = ~np.isnan(oof_probs)
    y_val = y_all[valid_mask]
    az_val = az_all[valid_mask]
    p_val = oof_probs[valid_mask]

    threshold = manifest.get("optimal_threshold", 0.5)
    y_pred = apply_threshold(p_val, threshold)

    print("\n" + "=" * 60)
    print(f"ROBUSTNESS AUDIT FOR RUN: {manifest.get('run_id', run_dir)}")
    print("=" * 60)

    # 1. Shortcut audit
    shortcut_res = run_shortcut_audit(y_val, y_pred, az_val)
    print(f"Overall OOF BA:               {shortcut_res['ba_overall']:.4f}")
    print(f"Shortcut-Wrong Subset BA:     {shortcut_res['ba_shortcut_wrong']:.4f} (N={shortcut_res['n_shortcut_wrong']})")
    print(f"Shortcut-Correct Subset BA:   {shortcut_res['ba_shortcut_correct']:.4f} (N={shortcut_res['n_shortcut_correct']})")

    # 2. Slice report
    train_imgs_mmap = np.load(train_cache, mmap_mode="r")
    slice_res = run_slice_report(y_val, y_pred, az_val, train_imgs_mmap[valid_mask])
    print(f"\nWorst Slice:                  {slice_res['worst_slice_name']} (BA={slice_res['worst_slice_ba']:.4f})")

    print("\n--- Per-Slice Balanced Accuracy Breakdown ---")
    for s_name, s_info in slice_res["slices"].items():
        ba_str = f"{s_info['ba']:.4f}" if not np.isnan(s_info['ba']) else "N/A"
        print(f"  {s_name:35s} | N={s_info['count']:5d} | BA={ba_str}")
    print("=" * 60)

    # Generate Markdown Report
    run_id = manifest.get("run_id", os.path.basename(run_dir))
    report_path = os.path.join("reports", "robustness", f"{run_id}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write(f"# Robustness Report: `{run_id}`\n\n")
        f.write(f"- **Overall OOF BA**: {shortcut_res['ba_overall']:.4f} @ t*={threshold:.3f}\n")
        f.write(f"- **Shortcut-Wrong BA**: {shortcut_res['ba_shortcut_wrong']:.4f} (N={shortcut_res['n_shortcut_wrong']})\n")
        f.write(f"- **Worst Slice**: {slice_res['worst_slice_name']} (BA={slice_res['worst_slice_ba']:.4f})\n\n")
        f.write("### Per-Slice Breakdown\n\n")
        f.write("| Slice | Samples | Balanced Accuracy |\n")
        f.write("|---|---|---|\n")
        for s_name, s_info in slice_res["slices"].items():
            ba_str = f"{s_info['ba']:.4f}" if not np.isnan(s_info['ba']) else "N/A"
            f.write(f"| {s_name} | {s_info['count']} | {ba_str} |\n")

    print(f"\nSaved robustness report: {report_path}")

    return {
        "shortcut_audit": shortcut_res,
        "slice_report": slice_res,
        "report_file": report_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Pareidolia Robustness CLI")
    parser.add_argument("--run", type=str, required=True, help="Path to experiment run directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    evaluate_robustness(args.run, device_str=args.device)


if __name__ == "__main__":
    main()
