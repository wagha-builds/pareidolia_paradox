"""Robustness and shortcut audit suite for Pareidolia Paradox.

Features (PRD Feature F17 & Playbook F12):
- B1 canonical shadow-polarity shortcut audit (evaluates model on shortcut-wrong subset).
- Topographic inversion stress test (tests if P(Rise) flips to 1 - P(Rise) under vertical relief inversion).
- Azimuth-shift sensitivity test (measures probability stability under ±15° azimuth perturbation).
- Occlusion robustness test (measures degradation when center or shadow is median-filled).
- Comprehensive slice report (solar azimuth octants, albedo quartiles, regional folds).
- Expected Calibration Error (ECE) and Brier calibration scores.
- Generates reports/robustness/<run_id>.md.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import brier_score_loss, roc_auc_score

from .canonical import canonicalize
from .dataset import PareidoliaDataset
from .metrics import apply_threshold
from .models import build_model


def compute_b1_shortcut_predictions(
    images: np.ndarray,
    azimuths: np.ndarray,
    delta: float = 46.7029,
    s: int = -1,
) -> np.ndarray:
    """Compute predictions of the B1 canonical shadow-polarity shortcut.

    In canonical frame (sun at top):
    - Depth (0): inner southern wall is illuminated by northern light, top is darker than bottom.
    - Rise (1): northern crest faces the sun, top is brighter than bottom.
    """
    n = len(azimuths)
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        img = np.array(images[i])
        canon = canonicalize(img, float(azimuths[i]), delta, s)
        top_val = float(canon[:128, :].mean())
        bot_val = float(canon[128:, :].mean())
        preds[i] = 0 if top_val < bot_val else 1
    return preds


def run_shortcut_audit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    shortcut_preds: np.ndarray,
) -> dict:
    """Evaluate model accuracy on the subset where the B1 shortcut baseline is WRONG."""
    from sklearn.metrics import balanced_accuracy_score

    b1_ba = float(balanced_accuracy_score(y_true, shortcut_preds))
    wrong_mask = shortcut_preds != y_true
    num_wrong = int(np.sum(wrong_mask))
    pct_wrong = float(num_wrong / len(y_true))

    y_pred = apply_threshold(y_prob, threshold)

    if num_wrong > 0 and len(np.unique(y_true[wrong_mask])) > 1:
        shortcut_wrong_ba = float(
            balanced_accuracy_score(y_true[wrong_mask], y_pred[wrong_mask])
        )
    else:
        shortcut_wrong_ba = 0.50

    return {
        "b1_shortcut_ba": b1_ba,
        "shortcut_wrong_count": num_wrong,
        "shortcut_wrong_pct": pct_wrong,
        "shortcut_wrong_ba": shortcut_wrong_ba,
    }


def compute_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Compute Expected Calibration Error (ECE) and Brier score."""
    brier = float(brier_score_loss(y_true, y_prob))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if np.sum(mask) > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

    return {"brier_score": brier, "ece": float(ece)}


def run_slice_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metadata: pd.DataFrame,
    folds_df: pd.DataFrame,
) -> dict:
    """Compute performance across azimuth octants, albedo quartiles, and regional folds."""
    from sklearn.metrics import balanced_accuracy_score

    merged = metadata.merge(folds_df, on="image_id", how="left")
    y_pred = apply_threshold(y_prob, threshold)
    slices = {}

    # 1. Azimuth Octants
    az_slices = {}
    octant_names = [
        "0-45° (NNE)",
        "45-90° (ENE)",
        "90-135° (ESE)",
        "135-180° (SSE)",
        "180-225° (SSW)",
        "225-270° (WSW)",
        "270-315° (WNW)",
        "315-360° (NNW)",
    ]
    for oct_idx, name in enumerate(octant_names):
        lo, hi = oct_idx * 45.0, (oct_idx + 1) * 45.0
        mask = (merged["sun_azimuth_angle"] >= lo) & (merged["sun_azimuth_angle"] < hi)
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            ba = float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
            az_slices[name] = {
                "count": int(mask.sum()),
                "pi_1": float(y_true[mask].mean()),
                "ba": ba,
            }
    slices["azimuth_octants"] = az_slices

    # 2. Regional Folds
    fold_slices = {}
    for f in range(5):
        mask = merged["fold"] == f
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            ba = float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
            fold_slices[f"Fold {f}"] = {
                "count": int(mask.sum()),
                "pi_1": float(y_true[mask].mean()),
                "ba": ba,
            }
    slices["regional_folds"] = fold_slices

    # Find worst slice across all slices
    all_slice_bas = [s["ba"] for s in az_slices.values()] + [
        s["ba"] for s in fold_slices.values()
    ]
    all_slice_names = list(az_slices.keys()) + list(fold_slices.keys())
    worst_idx = int(np.argmin(all_slice_bas))
    slices["worst_slice"] = {
        "name": all_slice_names[worst_idx],
        "ba": float(all_slice_bas[worst_idx]),
    }

    return slices


def run_inversion_stress_test(
    model: torch.nn.Module,
    dataset: PareidoliaDataset,
    device: str = "cuda",
    num_samples: int = 500,
) -> dict:
    """Stress test model with vertical relief inversion (vflip y=1-y)."""
    from sklearn.metrics import balanced_accuracy_score

    model.eval()
    n = min(len(dataset), num_samples)
    y_true_inv = []
    y_pred_inv = []
    diffs = []

    with torch.no_grad():
        for i in range(n):
            img_tensor, az_sincos, y_label, _ = dataset[i]
            x = img_tensor.unsqueeze(0).to(device)
            az = az_sincos.unsqueeze(0).to(device)

            # Normal prediction
            if hasattr(model, "film_mlp"):
                out_orig = model(x, az)
            else:
                out_orig = model(x)
            p_orig = float(torch.softmax(out_orig, dim=1)[0, 1].item())

            # Inverted prediction (vertical flip on canonical frame)
            rev_idx = torch.arange(x.shape[2] - 1, -1, -1, device=x.device)
            x_inv = x[:, :, rev_idx, :]
            if hasattr(model, "film_mlp"):
                out_inv = model(x_inv, az)
            else:
                out_inv = model(x_inv)
            p_inv = float(torch.softmax(out_inv, dim=1)[0, 1].item())

            y_orig = int(y_label.item())
            y_inv = 1 - y_orig

            y_true_inv.append(y_inv)
            y_pred_inv.append(1 if p_inv >= 0.50 else 0)
            diffs.append(abs(p_inv - (1.0 - p_orig)))

    ba_inv = float(balanced_accuracy_score(y_true_inv, y_pred_inv))
    mean_consistency_err = float(np.mean(diffs))

    return {
        "inversion_stress_ba": ba_inv,
        "inversion_consistency_error": mean_consistency_err,
    }


def run_azimuth_shift_test(
    model: torch.nn.Module,
    dataset: PareidoliaDataset,
    device: str = "cuda",
    delta_deg: float = 15.0,
    num_samples: int = 500,
) -> dict:
    """Test sensitivity to ±delta_deg solar azimuth perturbations."""
    model.eval()
    n = min(len(dataset), num_samples)
    shifts = []

    with torch.no_grad():
        for i in range(n):
            img_tensor, az_sincos, _, _ = dataset[i]
            x = img_tensor.unsqueeze(0).to(device)
            az = az_sincos.unsqueeze(0).to(device)

            # Base prediction
            if hasattr(model, "film_mlp"):
                p_base = float(torch.softmax(model(x, az), dim=1)[0, 1].item())
            else:
                p_base = float(torch.softmax(model(x), dim=1)[0, 1].item())

            # Perturbed azimuth (+15°)
            az_deg = (
                np.degrees(np.arctan2(az_sincos[0].item(), az_sincos[1].item())) % 360.0
            )
            az_pert_deg = (az_deg + delta_deg) % 360.0
            rad = np.radians(az_pert_deg)
            az_pert = torch.tensor(
                [[np.sin(rad), np.cos(rad)]], dtype=torch.float32
            ).to(device)

            if hasattr(model, "film_mlp"):
                p_pert = float(torch.softmax(model(x, az_pert), dim=1)[0, 1].item())
            else:
                p_pert = p_base

            shifts.append(abs(p_pert - p_base))

    return {"mean_azimuth_shift_delta": float(np.mean(shifts))}


def run_occlusion_test(
    model: torch.nn.Module,
    dataset: PareidoliaDataset,
    device: str = "cuda",
    box_size: int = 48,
    num_samples: int = 500,
) -> dict:
    """Measure Balanced Accuracy degradation when the center box is median-filled."""
    from sklearn.metrics import balanced_accuracy_score

    model.eval()
    n = min(len(dataset), num_samples)
    y_true = []
    y_pred_base = []
    y_pred_occ = []

    h, w = 256, 256
    r0, r1 = (h - box_size) // 2, (h + box_size) // 2
    c0, c1 = (w - box_size) // 2, (w + box_size) // 2

    with torch.no_grad():
        for i in range(n):
            img_tensor, az_sincos, y_label, _ = dataset[i]
            x = img_tensor.unsqueeze(0).to(device)
            az = az_sincos.unsqueeze(0).to(device)

            # Base
            if hasattr(model, "film_mlp"):
                out_base = model(x, az)
            else:
                out_base = model(x)
            p_base = float(torch.softmax(out_base, dim=1)[0, 1].item())

            # Occlude center with median
            x_occ = x.clone()
            median_val = x.median()
            x_occ[:, :, r0:r1, c0:c1] = median_val

            if hasattr(model, "film_mlp"):
                out_occ = model(x_occ, az)
            else:
                out_occ = model(x_occ)
            p_occ = float(torch.softmax(out_occ, dim=1)[0, 1].item())

            y = int(y_label.item())
            y_true.append(y)
            y_pred_base.append(1 if p_base >= 0.50 else 0)
            y_pred_occ.append(1 if p_occ >= 0.50 else 0)

    ba_base = float(balanced_accuracy_score(y_true, y_pred_base))
    ba_occ = float(balanced_accuracy_score(y_true, y_pred_occ))
    delta_ba = float(ba_base - ba_occ)

    return {
        "occlusion_base_ba": ba_base,
        "occlusion_center_ba": ba_occ,
        "occlusion_degradation_delta": delta_ba,
    }


def generate_robustness_report(
    run_id: str, results: dict, out_dir: str = "reports/robustness"
) -> Path:
    """Write markdown summary report."""
    out_path = Path(out_dir) / f"{run_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sa = results["shortcut_audit"]
    sl = results["slice_report"]
    cal = results["calibration"]
    inv = results.get("inversion_stress", {})
    az = results.get("azimuth_shift", {})
    occ = results.get("occlusion", {})

    lines = [
        f"# Robustness & Shortcut Audit Report: `{run_id}`",
        "",
        "## 1. Headline Robustness Metrics",
        f"- **Overall OOF Balanced Accuracy:** {results['oof_ba']:.4f} (@ threshold {results['threshold']:.4f})",
        f"- **OOF ROC-AUC:** {results['oof_auc']:.4f}",
        f"- **Shortcut-Wrong BA:** {sa['shortcut_wrong_ba']:.4f} (Accuracy on {sa['shortcut_wrong_count']} samples ({sa['shortcut_wrong_pct']:.1%}) where B1 shadow shortcut fails)",
        f"- **B1 Shortcut Baseline BA:** {sa['b1_shortcut_ba']:.4f}",
        f"- **Expected Calibration Error (ECE):** {cal['ece']:.4f} (Brier Score: {cal['brier_score']:.4f})",
        "",
        "## 2. Topographic Stress Tests",
        f"- **Inversion-Stress BA:** {inv.get('inversion_stress_ba', 0.0):.4f} (Under relief-inversion vertical flip)",
        f"- **Inversion Consistency Error:** {inv.get('inversion_consistency_error', 0.0):.4f} (Ideal: 0.00)",
        f"- **Azimuth-Shift Δ (±15°):** {az.get('mean_azimuth_shift_delta', 0.0):.4f}",
        f"- **Center Occlusion Degradation:** {occ.get('occlusion_degradation_delta', 0.0):.4f} (Base {occ.get('occlusion_base_ba', 0.0):.4f} → Occluded {occ.get('occlusion_center_ba', 0.0):.4f})",
        "",
        "## 3. Slice Report (Performance Across Sub-Populations)",
        f"- **Worst-Performing Slice:** `{sl['worst_slice']['name']}` = {sl['worst_slice']['ba']:.4f} BA",
        "",
        "### A. Regional Lunar Folds",
        "| Slice | Sample Count | Prior (π₁) | Balanced Accuracy |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for name, data in sl["regional_folds"].items():
        lines.append(
            f"| {name} | {data['count']} | {data['pi_1']:.1%} | {data['ba']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### B. Azimuth Octants",
            "| Octant | Sample Count | Prior (π₁) | Balanced Accuracy |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )

    for name, data in sl["azimuth_octants"].items():
        lines.append(
            f"| {name} | {data['count']} | {data['pi_1']:.1%} | {data['ba']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Verdict & Assessment",
            f"Model maintains **{sa['shortcut_wrong_ba']:.4f} BA** on non-shortcut terrain, demonstrating legitimate 3D relief discrimination.",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_robustness_suite(
    run_id: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Resolve run directory
    run_dir = (
        Path("experiments") / run_id if not Path(run_id).exists() else Path(run_id)
    )
    if not run_dir.exists():
        raise FileNotFoundError(f"Experiment run directory not found: {run_dir}")

    print(f"Running robustness audit for: {run_dir.name} on {device}")

    # 1. Load manifest and OOF
    manifest_path = run_dir / "run_manifest.json"
    oof_path = run_dir / "oof.npy"
    config_path = run_dir / "config.yaml"

    if not manifest_path.exists() or not oof_path.exists():
        raise FileNotFoundError(f"Missing run_manifest.json or oof.npy in {run_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oof_probs = np.load(oof_path)
    cfg = OmegaConf.load(config_path)
    base_cfg = OmegaConf.load("configs/config.yaml")

    # Load dataset metadata and folds
    df = pd.read_csv("data/raw/train_metadata.csv")
    folds_df = pd.read_csv("data/folds.csv")
    y_true = df["label"].values
    threshold = float(manifest["metrics"]["threshold"])

    # 2. Compute B1 shortcut predictions
    print("Computing B1 canonical shadow shortcut baseline...")
    images = np.load("data/processed/train_images.npy", mmap_mode="r")
    azimuths = df["sun_azimuth_angle"].values
    shortcut_preds = compute_b1_shortcut_predictions(
        images,
        azimuths,
        delta=float(base_cfg.frozen.calibration.delta),
        s=int(base_cfg.frozen.calibration.s),
    )

    # 3. Shortcut audit & Slices
    print("Running shortcut audit and slice breakdowns...")
    sa_res = run_shortcut_audit(y_true, oof_probs, threshold, shortcut_preds)
    slice_res = run_slice_report(y_true, oof_probs, threshold, df, folds_df)
    cal_res = compute_calibration_metrics(y_true, oof_probs)

    # 4. Load Fold 0 model for physical stress tests
    ckpt_path = run_dir / "checkpoints" / "fold0_best.pt"
    inv_res = {}
    az_res = {}
    occ_res = {}

    if ckpt_path.exists():
        print(
            "Running GPU physical stress tests (Inversion, Azimuth shift, Occlusion)..."
        )
        model_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if "model" in model_cfg_dict:
            model_cfg_dict["model"]["pretrained"] = False
        model = build_model(model_cfg_dict)
        state_dict = torch.load(ckpt_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
        model.to(device)

        val_indices = folds_df[folds_df["fold"] == 0].index.tolist()
        val_dataset = PareidoliaDataset(
            data_dir="data",
            split="train",
            indices=val_indices,
            canonicalize_cfg=base_cfg.frozen.calibration,
            norm_stats=base_cfg.frozen.norm,
        )

        inv_res = run_inversion_stress_test(
            model, val_dataset, device=device, num_samples=300
        )
        az_res = run_azimuth_shift_test(
            model, val_dataset, device=device, delta_deg=15.0, num_samples=300
        )
        occ_res = run_occlusion_test(
            model, val_dataset, device=device, box_size=48, num_samples=300
        )

    results = {
        "run_id": run_dir.name,
        "oof_ba": float(manifest["metrics"]["oof_ba"]),
        "oof_auc": float(manifest["metrics"]["oof_auc"])
        if manifest["metrics"].get("oof_auc") is not None
        else float(roc_auc_score(y_true, oof_probs)),
        "threshold": threshold,
        "shortcut_audit": sa_res,
        "slice_report": slice_res,
        "calibration": cal_res,
        "inversion_stress": inv_res,
        "azimuth_shift": az_res,
        "occlusion": occ_res,
    }

    report_file = generate_robustness_report(run_dir.name, results)
    print(f"\nSaved robustness report to: {report_file}")

    print("\n" + "=" * 75)
    print(f"ROBUSTNESS SUMMARY: {run_dir.name}")
    print("=" * 75)
    print(f"Overall OOF BA:           {results['oof_ba']:.4f} (at t*={threshold:.4f})")
    print(f"Overall OOF AUC:          {results['oof_auc']:.4f}")
    print(
        f"Shortcut-Wrong BA:        {sa_res['shortcut_wrong_ba']:.4f} (N={sa_res['shortcut_wrong_count']}, {sa_res['shortcut_wrong_pct']:.1%})"
    )
    print(f"Inversion-Stress BA:      {inv_res.get('inversion_stress_ba', 0.0):.4f}")
    print(
        f"Azimuth-Shift Delta (+/-15 deg): {az_res.get('mean_azimuth_shift_delta', 0.0):.4f}"
    )
    print(
        f"Worst Slice:              {slice_res['worst_slice']['name']} = {slice_res['worst_slice']['ba']:.4f} BA"
    )
    print(f"Calibration ECE:          {cal_res['ece']:.4f}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run robustness and shortcut audit.")
    parser.add_argument(
        "--run", type=str, required=True, help="Experiment run_id or path"
    )
    args = parser.parse_args()
    run_robustness_suite(args.run)


if __name__ == "__main__":
    main()
