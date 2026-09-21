"""
inference.py — Pareidolia Paradox: Lunar Terrain Classification
================================================================
Top-level inference entry point. Runs TTA inference on the test set
and produces a competition-ready submission CSV.

Usage
-----
# Run inference with the best saved model weights:
    python inference.py --run-dir experiments/<run_id>

# With explicit threshold override:
    python inference.py --run-dir experiments/<run_id> --threshold 0.45

# Ensemble inference (multiple run directories):
    python inference.py \\
        --run-dirs experiments/run_a experiments/run_b \\
        --weights 0.6 0.4 \\
        --threshold 0.45

# Skip TTA (faster, slightly lower accuracy):
    python inference.py --run-dir experiments/<run_id> --no-tta

Output
------
submissions/sub_<timestamp>_<run_id>.csv   — validated submission CSV
reports/sanity_<timestamp>.txt             — sanity report (class balance, drift)

Model Weights
-------------
Pre-trained weights for the best ensemble (E17 + E5 multi-seed):
  https://drive.google.com/drive/folders/YOUR_FOLDER_ID
  (see README.md for the exact link)

Prerequisites
-------------
1.  python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
2.  Place test images under data/raw/eval_images/ (or test_images/)
3.  Place data/raw/test_metadata.csv
4.  Download and unzip model weights into experiments/
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareidolia Paradox — inference + submission builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Single-run mode
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a single experiment run directory (contains checkpoints/ and run_manifest.json)",
    )
    # Ensemble mode
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=None,
        help="Two or more run directories to ensemble",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Ensemble weights (one per --run-dirs entry, must sum to 1). Defaults to equal weights.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold p >= t => Rise (1). Reads from run_manifest.json if not given.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation (faster but slightly lower accuracy)",
    )
    parser.add_argument(
        "--skip-inversion-check",
        action="store_true",
        help="Skip label-inversion check (use when checkpoint is for a subset of folds only)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to run inference on",
    )
    args = parser.parse_args()

    use_tta = not args.no_tta

    # ── Single-run mode ───────────────────────────────────────────────────────
    if args.run_dir is not None and args.run_dirs is None:
        from src.submit import build_submission

        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Running inference on: {run_dir}")
        sub_path = build_submission(
            run_dir,
            threshold=args.threshold,
            skip_inversion_check=args.skip_inversion_check,
        )
        print(f"\n✅ Submission written: {sub_path}")
        return

    # ── Ensemble mode ─────────────────────────────────────────────────────────
    if args.run_dirs is not None:
        import json
        import numpy as np
        import pandas as pd
        from datetime import datetime
        from src.infer import predict_run
        from src.metrics import apply_threshold, plateau_threshold
        from src.submit import validate_submission, generate_sanity_report

        run_dirs = [Path(d) for d in args.run_dirs]
        for d in run_dirs:
            if not d.exists():
                print(f"ERROR: run directory not found: {d}", file=sys.stderr)
                sys.exit(1)

        n = len(run_dirs)
        weights = args.weights if args.weights else [1.0 / n] * n
        if len(weights) != n:
            print("ERROR: --weights count must match --run-dirs count", file=sys.stderr)
            sys.exit(1)
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()  # normalise

        print(f"Ensemble of {n} runs:")
        for d, w in zip(run_dirs, weights_arr):
            print(f"  {w:.3f}  {d.name}")

        test_meta = pd.read_csv("data/raw/test_metadata.csv", dtype={"image_id": str})
        blended = np.zeros(len(test_meta), dtype=np.float64)

        for run_dir, w in zip(run_dirs, weights_arr):
            preds = predict_run(run_dir, split=args.split, tta=use_tta)
            blended += w * preds["p_rise"].to_numpy()

        p_rise = blended.astype(np.float32)

        # Threshold
        if args.threshold is not None:
            t = args.threshold
        else:
            # Use first run's manifest threshold as reference
            manifest = json.loads((run_dirs[0] / "run_manifest.json").read_text())
            t = float(manifest["metrics"]["threshold"])
        print(f"Threshold: {t:.4f}")

        labels = apply_threshold(p_rise, t)

        out_dir = Path("submissions")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        sub_path = out_dir / f"sub_{ts}_ensemble_{n}runs.csv"

        test_meta["label"] = labels.astype(int)
        test_meta[["image_id", "label"]].to_csv(
            sub_path, index=False, lineterminator="\n"
        )

        validate_submission(sub_path)
        generate_sanity_report(sub_path, p_rise, t)
        print(f"\n✅ Submission written: {sub_path}")
        return

    parser.error("Pass either --run-dir (single model) or --run-dirs (ensemble)")


if __name__ == "__main__":
    main()
