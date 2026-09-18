"""End-to-end reproduction and verification of the final submission artifact."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .ensemble import blend_predictions, validate_ensemble_members
from .metrics import apply_threshold, plateau_threshold


def reproduce_artifact(artifact_dir: str | Path | None = None) -> dict:
    """Verify and reproduce metrics for the production artifact."""
    if artifact_dir is None:
        reg_path = Path("artifacts/registry.json")
        if not reg_path.exists():
            raise FileNotFoundError("Missing artifacts/registry.json")
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        artifact_dir = Path(reg["production"])
    else:
        artifact_dir = Path(artifact_dir)

    print(f"Reproducing artifact: {artifact_dir}")
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact dir {artifact_dir} does not exist")

    # Load artifact metadata
    members = json.loads((artifact_dir / "members.json").read_text(encoding="utf-8"))
    threshold_meta = json.loads(
        (artifact_dir / "threshold.json").read_text(encoding="utf-8")
    )
    expected_thresh = float(threshold_meta["threshold"])
    metrics_meta = json.loads(
        (artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )
    expected_ba = float(metrics_meta["oof_ba"])

    # Load targets
    train_meta = pd.read_csv("data/raw/train_metadata.csv")
    y_true = train_meta["label"].to_numpy()
    folds_df = pd.read_csv("data/folds.csv")

    # Load members and blend
    expected_folds_hash = folds_df.attrs.get("sha256", None)
    if expected_folds_hash is None:
        import hashlib

        expected_folds_hash = hashlib.sha256(
            Path("data/folds.csv").read_bytes()
        ).hexdigest()

    oof_arrays, manifests, weights = validate_ensemble_members(
        members, expected_folds_hash
    )
    p_ens = blend_predictions(oof_arrays, weights, method="weighted_average")

    # Calculate metrics
    thresh, _, _ = plateau_threshold(y_true, p_ens)
    y_pred = apply_threshold(p_ens, expected_thresh)
    reproduced_ba = float(balanced_accuracy_score(y_true, y_pred))
    reproduced_auc = float(roc_auc_score(y_true, p_ens))

    diff = abs(reproduced_ba - expected_ba)
    print("=" * 65)
    print("REPRODUCTION REPORT")
    print("=" * 65)
    print(f"Artifact:          {artifact_dir.name}")
    print(f"Stored OOF BA:     {expected_ba:.4f}")
    print(f"Reproduced OOF BA: {reproduced_ba:.4f} (diff = {diff:.6f})")
    print(f"Stored Threshold:  {expected_thresh:.4f}")
    print(f"Computed Plateau:  {thresh:.4f}")
    print(f"Reproduced AUC:    {reproduced_auc:.4f}")
    print("=" * 65)

    if diff > 0.002:
        raise AssertionError(
            f"Reproduction failed! BA discrepancy {diff:.4f} exceeds 0.002 tolerance."
        )

    print("[PASS] Reproduction verified within 0.2 pt tolerance.")
    return {
        "stored_ba": expected_ba,
        "reproduced_ba": reproduced_ba,
        "diff": diff,
        "auc": reproduced_auc,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce final submission artifact.")
    parser.add_argument(
        "--artifact", type=str, default=None, help="Path to artifact directory"
    )
    args = parser.parse_args()
    reproduce_artifact(args.artifact)


if __name__ == "__main__":
    main()
