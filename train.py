"""
train.py — Pareidolia Paradox: Lunar Terrain Classification
============================================================
Top-level training entry point.

Usage
-----
# Full 5-fold cross-validation (default config):
    python train.py --config configs/config.yaml

# Specific experiment config:
    python train.py --config configs/exp/e18_physics_tensor.yaml

# Single fold (fast exploration):
    python train.py --config configs/exp/e18_physics_tensor.yaml --fold 0

# Multiple seeds:
    python train.py --config configs/exp/e18_physics_tensor.yaml --seeds "42 43 44"

# Debug / smoke test (finishes in < 5 min):
    python train.py --config configs/debug.yaml

Prerequisites
-------------
1.  python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
2.  Place raw data under data/raw/ (train_images/, eval_images/, test_images/,
    train_metadata.csv, test_metadata.csv)
3.  python -m src.cache          # build uint8 caches + index.json
4.  python train.py ...

Outputs are written to experiments/<run_id>/:
    oof.npy            — out-of-fold probability array (float32, length 7854)
    run_manifest.json  — git SHA, folds SHA256, per-fold metrics
    checkpoints/       — fold{k}_best.pt, fold{k}_last.pt
"""

from src.train import main

if __name__ == "__main__":
    main()
