"""
Ensemble builder and out-of-fold evaluator.
Ensembles model predictions (simple mean, logit mean, or weighted blend),
validates matching fold hashes, calculates OOF correlations, searches optimal
plateau threshold, and builds validated competition submission CSVs.
"""

import os
import glob
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from omegaconf import OmegaConf, DictConfig
from scipy.special import logit, expit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss

from .metrics import plateau_threshold, apply_threshold
from .submit import build_submission


def load_run(run_dir: str) -> Tuple[np.ndarray, dict]:
    """Loads OOF predictions and manifest for a given experiment run."""
    oof_path = os.path.join(run_dir, "oof.npy")
    manifest_path = os.path.join(run_dir, "run_manifest.json")

    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"Missing oof.npy in {run_dir}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing run_manifest.json in {run_dir}")

    oof = np.load(oof_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    return oof, manifest


def check_matching_folds(manifests: List[dict]):
    """Verifies all runs share identical folds_sha256."""
    first_hash = manifests[0].get("folds_sha256", "")
    for i, m in enumerate(manifests[1:], start=1):
        h = m.get("folds_sha256", "")
        if h != first_hash:
            raise ValueError(
                f"Fold hash mismatch between run 0 ({first_hash}) and run {i} ({h})! "
                "Ensembles can only blend models evaluated on identical fold splits."
            )


def blend_predictions(
    preds_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    method: str = "prob_mean",
) -> np.ndarray:
    """
    Blends a list of probability arrays using either 'prob_mean' or 'logit_mean'.
    """
    K = len(preds_list)
    if weights is None:
        weights = [1.0 / K] * K
    else:
        weights = [w / sum(weights) for w in weights]

    weights = np.array(weights, dtype=np.float32)

    if method == "prob_mean":
        blended = np.zeros_like(preds_list[0], dtype=np.float32)
        for w, p in zip(weights, preds_list):
            blended += w * p
        return blended

    elif method == "logit_mean":
        eps = 1e-6
        blended_logits = np.zeros_like(preds_list[0], dtype=np.float32)
        for w, p in zip(weights, preds_list):
            p_clipped = np.clip(p, eps, 1.0 - eps)
            blended_logits += w * logit(p_clipped)
        return expit(blended_logits).astype(np.float32)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def evaluate_ensemble(
    run_dirs: List[str],
    weights: Optional[List[float]] = None,
    method: str = "prob_mean",
    labels_file: str = "data/processed/index.json",
) -> dict:
    """Evaluates an ensemble over out-of-fold predictions."""
    oof_list = []
    manifests = []
    for r in run_dirs:
        oof, m = load_run(r)
        oof_list.append(oof)
        manifests.append(m)

    check_matching_folds(manifests)

    # Correlation matrix
    K = len(oof_list)
    corr_matrix = np.corrcoef(oof_list)

    print("=" * 60)
    print("ENSEMBLE OOF CORRELATION MATRIX:")
    for i in range(K):
        name_i = manifests[i].get("run_id", f"run_{i}")
        row_str = " ".join(f"{corr_matrix[i, j]:.4f}" for j in range(K))
        print(f"[{i}] {name_i[:40]:40s} : {row_str}")
    print("=" * 60)

    # Ground truth
    with open(labels_file, "r") as f:
        idx_data = json.load(f)
    y_true = np.array([item["label"] for item in idx_data["train"]], dtype=int)

    # Blend
    blended_oof = blend_predictions(oof_list, weights=weights, method=method)

    # Valid mask
    valid_mask = ~np.isnan(blended_oof)
    y_eval = y_true[valid_mask]
    p_eval = blended_oof[valid_mask]

    t_plat, grid, ba_curve = plateau_threshold(y_eval, p_eval)
    preds_plat = apply_threshold(p_eval, t_plat)
    best_ba = balanced_accuracy_score(y_eval, preds_plat)
    auc = roc_auc_score(y_eval, p_eval)
    brier = brier_score_loss(y_eval, p_eval)

    print(f"\nEnsemble Evaluation ({method}):")
    print(f"  Valid OOF Samples: {valid_mask.sum()}/{len(y_true)}")
    print(f"  Optimal Threshold t*: {t_plat:.4f}")
    print(f"  OOF Balanced Accuracy: {best_ba:.4f}")
    print(f"  OOF ROC-AUC:           {auc:.4f}")
    print(f"  OOF Brier Score:       {brier:.4f}")
    print("=" * 60)

    return {
        "runs": run_dirs,
        "weights": weights if weights is not None else [1.0 / K] * K,
        "method": method,
        "optimal_threshold": t_plat,
        "oof_ba": float(best_ba),
        "oof_auc": float(auc),
        "oof_brier": float(brier),
        "blended_oof": blended_oof,
        "correlation_matrix": corr_matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Pareidolia Ensemble Evaluator & Submitter")
    parser.add_argument("--runs", nargs="+", help="List of run directories to ensemble")
    parser.add_argument("--spec", type=str, help="Path to YAML ensemble specification")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Weights for each run")
    parser.add_argument("--method", choices=["prob_mean", "logit_mean"], default="prob_mean")
    parser.add_argument("--submit", action="store_true", help="Generate submission CSV using the ensemble")
    parser.add_argument("--index", default="data/processed/index.json")
    args = parser.parse_args()

    run_dirs = args.runs
    weights = args.weights
    method = args.method
    ens_name = "ensemble"

    if args.spec and os.path.exists(args.spec):
        cfg = OmegaConf.load(args.spec)
        ens_cfg = cfg.get("ensemble", cfg)
        run_dirs = ens_cfg.get("runs", run_dirs)
        weights = ens_cfg.get("weights", weights)
        method = ens_cfg.get("method", method)
        ens_name = ens_cfg.get("name", ens_name)

    if not run_dirs or len(run_dirs) < 2:
        print("Error: Must provide at least 2 run directories to ensemble.")
        return

    res = evaluate_ensemble(run_dirs, weights=weights, method=method, labels_file=args.index)

    if args.submit:
        # Load test predictions from each run
        test_preds = []
        for r in run_dirs:
            test_files = sorted(glob.glob(os.path.join(r, "test_probs*.npy")))
            if not test_files:
                raise FileNotFoundError(f"No test_probs*.npy found in {r}. Run infer.py for this model first!")
            # Load latest test_probs
            tp = np.load(test_files[-1])
            test_preds.append(tp)

        blended_test = blend_predictions(test_preds, weights=res["weights"], method=method)
        threshold = res["optimal_threshold"]

        # Timestamped submission
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        sub_filename = f"sub_{now_str}_{ens_name}_ba{res['oof_ba']:.4f}.csv"
        sub_path = os.path.join("submissions", sub_filename)

        with open(args.index, "r") as f:
            idx_data = json.load(f)
        test_ids = [row["image_id"] for row in idx_data["test"]]

        build_submission(
            image_ids=test_ids,
            probs=blended_test,
            threshold=threshold,
            output_path=sub_path,
        )
        print(f"\nSUCCESS: Created ensemble submission at {sub_path}")


if __name__ == "__main__":
    main()
