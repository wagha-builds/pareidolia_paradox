"""Multi-backbone ensembling, OOF blending, and artifact packaging.

Features:
- Validates identical folds_sha256 across all ensemble members (AGENTS.md invariant).
- Computes weighted ensemble of OOF probabilities (or log-odds).
- Calculates optimal plateau threshold t* using plateau_threshold from src.metrics.
- Evaluates per-fold Balanced Accuracy and paired bootstrap 95% CI vs parent models.
- Generates self-contained artifact directory under artifacts/<date>_<name>_ba<score>/.
- Updates artifacts/registry.json candidates list.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .metrics import apply_threshold, paired_bootstrap, plateau_threshold
import subprocess


def get_git_sha() -> str:
    """Return current git commit SHA."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate_ensemble_members(
    members: list[dict[str, Any]], expected_folds_sha256: str
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[float]]:
    """Verify fold hashes, check OOF shapes, and extract weights.

    Args:
        members: list of member specs with 'run_id' and 'weight'.
        expected_folds_sha256: hash of data/folds.csv.

    Returns:
        tuple of (oof_arrays, manifests, normalized_weights).
    """
    oof_arrays = []
    manifests = []
    weights = []

    for m in members:
        run_id = m["run_id"]
        run_dir = (
            Path("experiments") / run_id if not Path(run_id).exists() else Path(run_id)
        )
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Ensemble member run directory not found: {run_dir}"
            )

        manifest_path = run_dir / "run_manifest.json"
        oof_path = run_dir / "oof.npy"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run_manifest.json in {run_dir}")
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing oof.npy in {run_dir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        member_hash = manifest.get("folds_sha256")

        if member_hash != expected_folds_sha256:
            raise ValueError(
                f"Fold hash mismatch in {run_id}! "
                f"Expected {expected_folds_sha256[:8]}, got {member_hash[:8] if member_hash else 'None'}."
            )

        oof = np.load(oof_path)
        if oof.ndim != 1 or len(oof) != 7854:
            raise ValueError(
                f"Invalid OOF array shape {oof.shape} in {run_id}. Expected (7854,)."
            )

        oof_arrays.append(oof)
        manifests.append(manifest)
        weights.append(float(m.get("weight", 1.0)))

    # Normalize weights
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("Sum of ensemble weights must be positive.")
    norm_weights = [w / total_w for w in weights]

    return oof_arrays, manifests, norm_weights


def blend_predictions(
    probs_list: list[np.ndarray], weights: list[float], method: str = "weighted_average"
) -> np.ndarray:
    """Blend multiple probability arrays."""
    if method == "weighted_average":
        blended = np.zeros_like(probs_list[0], dtype=np.float64)
        for p, w in zip(probs_list, weights, strict=True):
            blended += w * p
        return blended.astype(np.float32)
    elif method == "weighted_logits":
        eps = 1e-6
        blended_logit = np.zeros_like(probs_list[0], dtype=np.float64)
        for p, w in zip(probs_list, weights, strict=True):
            clipped = np.clip(p, eps, 1.0 - eps)
            logit = np.log(clipped / (1.0 - clipped))
            blended_logit += w * logit
        exp_logit = np.exp(blended_logit)
        return (exp_logit / (1.0 + exp_logit)).astype(np.float32)
    else:
        raise ValueError(f"Unknown ensemble blending method: {method}")


def evaluate_ensemble(
    y_true: np.ndarray,
    p_ens: np.ndarray,
    folds: np.ndarray,
    member_oofs: list[np.ndarray],
    member_names: list[str],
) -> dict[str, Any]:
    """Evaluate ensemble OOF metrics and paired bootstrap vs members."""
    threshold, _, _ = plateau_threshold(y_true, p_ens)
    y_pred = apply_threshold(p_ens, threshold)

    overall_ba = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, p_ens))

    # Per-fold BA
    per_fold_ba = []
    for f in range(5):
        mask = folds == f
        f_ba = float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
        per_fold_ba.append(f_ba)

    # Paired bootstrap vs each member
    deltas = {}
    for oof, name in zip(member_oofs, member_names, strict=True):
        t_m, _, _ = plateau_threshold(y_true, oof)
        pred_m = apply_threshold(oof, t_m)
        mean_diff, lo, hi = paired_bootstrap(y_true, y_pred, pred_m, folds, n_iter=1000)
        deltas[name] = {
            "mean_delta": float(mean_diff),
            "ci_95": [float(lo), float(hi)],
        }

    return {
        "oof_ba": overall_ba,
        "oof_auc": auc,
        "threshold": float(threshold),
        "per_fold_ba": per_fold_ba,
        "bootstrap_deltas": deltas,
    }


def package_artifact(
    spec: dict[str, Any],
    eval_results: dict[str, Any],
    base_cfg: Any,
    date_str: str = "20260916",
) -> Path:
    """Create a self-contained production artifact directory."""
    ba_str = f"{eval_results['oof_ba']:.4f}"
    artifact_name = f"{date_str}_{spec.get('name', 'ensemble')}_ba{ba_str}"
    artifact_dir = Path("artifacts") / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config
    OmegaConf.save(base_cfg, artifact_dir / "config.yaml")

    # 2. Threshold
    threshold_meta = {
        "threshold": eval_results["threshold"],
        "method": "plateau_centre",
        "pi1": float(base_cfg.frozen.prior_pi1),
        "per_fold": eval_results["per_fold_ba"],
    }
    (artifact_dir / "threshold.json").write_text(
        json.dumps(threshold_meta, indent=2), encoding="utf-8"
    )

    # 3. Norm stats
    norm_stats = {
        "mean": float(base_cfg.frozen.norm.mean),
        "std": float(base_cfg.frozen.norm.std),
    }
    (artifact_dir / "norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2), encoding="utf-8"
    )

    # 4. Calibration
    cal_dict = {
        "delta_deg": float(base_cfg.frozen.calibration.delta),
        "handedness": int(base_cfg.frozen.calibration.s),
        "R": float(base_cfg.frozen.calibration.R),
        "convention": "toward_sun_ccw_from_right",
    }
    (artifact_dir / "calibration.json").write_text(
        json.dumps(cal_dict, indent=2), encoding="utf-8"
    )

    # 5. Class map
    class_map = {"0": "depth", "1": "rise"}
    (artifact_dir / "class_map.json").write_text(
        json.dumps(class_map, indent=2), encoding="utf-8"
    )

    # 6. Metrics
    metrics_meta = {
        "oof_ba": eval_results["oof_ba"],
        "oof_auc": eval_results["oof_auc"],
        "per_fold_ba": eval_results["per_fold_ba"],
        "bootstrap_deltas": eval_results["bootstrap_deltas"],
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics_meta, indent=2), encoding="utf-8"
    )

    # 7. Git SHA
    (artifact_dir / "git_sha.txt").write_text(get_git_sha(), encoding="utf-8")

    # 8. Ensemble Members Spec
    (artifact_dir / "members.json").write_text(
        json.dumps(spec.get("members", []), indent=2), encoding="utf-8"
    )

    # 9. Run manifest
    manifest = {
        "run_id": spec.get("name", "ensemble"),
        "git_sha": get_git_sha(),
        "folds_sha256": str(
            getattr(base_cfg.get("frozen", {}), "folds_sha256", None)
            or (
                base_cfg.get("frozen", {}).get("folds_sha256", "unknown")
                if isinstance(base_cfg.get("frozen", {}), dict)
                else "unknown"
            )
        ),
        "metrics": {
            "oof_ba": eval_results["oof_ba"],
            "oof_auc": eval_results["oof_auc"],
            "threshold": eval_results["threshold"],
            "per_fold_ba": eval_results["per_fold_ba"],
        },
        "members": spec.get("members", []),
    }
    (artifact_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # 10. Update artifacts/registry.json candidates
    reg_path = Path("artifacts/registry.json")
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    else:
        reg = {"production": str(artifact_dir).replace("\\", "/"), "candidates": []}

    cand_str = str(artifact_dir).replace("\\", "/")
    if cand_str not in reg.get("candidates", []):
        reg.setdefault("candidates", []).append(cand_str)

    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return artifact_dir


def build_and_evaluate_ensemble(spec_path: str | Path):
    """Main routine to run ensemble specification."""
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Ensemble spec file not found: {spec_path}")

    spec = OmegaConf.to_container(OmegaConf.load(spec_path), resolve=True)
    base_cfg = OmegaConf.load("configs/config.yaml")

    folds_path = Path("data/folds.csv")
    expected_folds_hash = get_file_sha256(folds_path)

    print(f"\nBuilding ensemble: {spec.get('name', 'ensemble')}")
    print(f"Verified folds.csv SHA-256: {expected_folds_hash[:12]}...")

    members = spec.get("members", [])
    oof_arrays, manifests, weights = validate_ensemble_members(
        members, expected_folds_hash
    )

    print(f"Loaded {len(members)} members with normalized weights:")
    for m, w in zip(members, weights, strict=True):
        print(f"  - {m.get('name', m['run_id'])}: weight = {w:.2f}")

    # Load targets and fold assignments
    df = pd.read_csv("data/raw/train_metadata.csv")
    folds_df = pd.read_csv("data/folds.csv")
    y_true = df["label"].values
    folds = folds_df["fold"].values

    method = spec.get("method", "weighted_average")
    p_ens = blend_predictions(oof_arrays, weights, method=method)

    member_names = [m.get("name", m["run_id"]) for m in members]
    eval_res = evaluate_ensemble(y_true, p_ens, folds, oof_arrays, member_names)

    # Save ensemble OOF array and manifest
    out_dir = Path(spec.get("output", {}).get("dir", "experiments/ensemble"))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", p_ens)

    manifest = {
        "run_id": spec.get("name", "ensemble"),
        "config_hash": get_file_sha256(spec_path),
        "git_sha": get_git_sha(),
        "folds_sha256": expected_folds_hash,
        "metrics": {
            "oof_ba": eval_res["oof_ba"],
            "oof_auc": eval_res["oof_auc"],
            "threshold": eval_res["threshold"],
            "per_fold_ba": eval_res["per_fold_ba"],
        },
        "members": members,
        "weights": weights,
        "oof_path": str(out_dir / "oof.npy"),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Package self-contained artifact
    artifact_dir = package_artifact(spec, eval_res, base_cfg)

    print("\n" + "=" * 75)
    print(f"ENSEMBLE RESULTS: {spec.get('name', 'ensemble')}")
    print("=" * 75)
    print(
        f"Overall OOF BA:       {eval_res['oof_ba']:.4f} (@ plateau t*={eval_res['threshold']:.4f})"
    )
    print(f"Overall OOF AUC:      {eval_res['oof_auc']:.4f}")
    print("Per-Fold BA:")
    for f, f_ba in enumerate(eval_res["per_fold_ba"]):
        print(f"  Fold {f}: {f_ba:.4f}")

    print("\nPaired Bootstrap vs Parent Models (95% CI):")
    for name, d in eval_res["bootstrap_deltas"].items():
        print(
            f"  vs {name}: {d['mean_delta']:+.4f} [95% CI: {d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}]"
        )

    print(f"\nSaved ensemble OOF to: {out_dir / 'oof.npy'}")
    print(f"Created production artifact: {artifact_dir}")
    print("Registered in: artifacts/registry.json")
    print("=" * 75 + "\n")


def predict_ensemble(
    spec_path: str | Path, split: str = "test", tta: bool = True
) -> pd.DataFrame:
    """Run test inference across all ensemble members and blend predictions with TTA."""
    from .infer import predict_run

    spec_path = Path(spec_path)
    spec = OmegaConf.to_container(OmegaConf.load(spec_path), resolve=True)
    members = spec.get("members", [])
    weights = [float(m.get("weight", 1.0)) for m in members]
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    member_probs = []
    image_ids = None
    for m in members:
        run_id = m["run_id"]
        run_dir = (
            Path("experiments") / run_id if not Path(run_id).exists() else Path(run_id)
        )
        df_pred = predict_run(run_dir, split=split, tta=tta)
        member_probs.append(df_pred["p_rise"].to_numpy())
        if image_ids is None:
            image_ids = df_pred["image_id"].tolist()

    method = spec.get("method", "weighted_average")
    p_ens = blend_predictions(member_probs, norm_weights, method=method)
    out_df = pd.DataFrame({"image_id": image_ids, "p_rise": p_ens})

    out_dir = Path(spec.get("output", {}).get("dir", "experiments/ensemble"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "tta" if tta else "raw"
    pred_path = (
        out_dir / f"test_probs_{tag}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.npy"
    )
    np.save(pred_path, p_ens)
    out_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"Ensemble test predictions written to {out_dir / 'test_predictions.csv'}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Ensemble builder and evaluator.")
    parser.add_argument(
        "--spec",
        type=str,
        default="configs/ensemble.yaml",
        help="Path to ensemble spec YAML",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run test inference across ensemble members and blend",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable TTA during inference",
    )
    args = parser.parse_args()
    if args.infer:
        predict_ensemble(args.spec, tta=not args.no_tta)
    else:
        build_and_evaluate_ensemble(args.spec)


if __name__ == "__main__":
    main()
