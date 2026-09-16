"""Unit tests for src/ensemble.py (Ensemble builder and evaluator)."""

import json
from pathlib import Path
import numpy as np

from src.ensemble import (
    blend_predictions,
    evaluate_ensemble,
    get_file_sha256,
    package_artifact,
)


def test_blend_predictions_weighted_average():
    p1 = np.array([0.2, 0.8], dtype=np.float32)
    p2 = np.array([0.4, 0.6], dtype=np.float32)
    weights = [0.25, 0.75]

    blended = blend_predictions([p1, p2], weights, method="weighted_average")
    expected = 0.25 * p1 + 0.75 * p2
    np.testing.assert_allclose(blended, expected, atol=1e-6)


def test_blend_predictions_weighted_logits():
    p1 = np.array([0.5, 0.5], dtype=np.float32)
    p2 = np.array([0.5, 0.5], dtype=np.float32)
    weights = [0.5, 0.5]

    blended = blend_predictions([p1, p2], weights, method="weighted_logits")
    np.testing.assert_allclose(blended, [0.5, 0.5], atol=1e-5)


def test_get_file_sha256(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"hello world")
    h = get_file_sha256(test_file)
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_evaluate_ensemble():
    n = 100
    y_true = np.array([0, 1] * (n // 2))
    p_ens = np.array([0.1, 0.9] * (n // 2), dtype=np.float32)
    folds = np.array([i % 5 for i in range(n)])

    oof1 = np.array([0.2, 0.8] * (n // 2), dtype=np.float32)
    member_oofs = [oof1]
    member_names = ["m1"]

    res = evaluate_ensemble(y_true, p_ens, folds, member_oofs, member_names)
    assert res["oof_ba"] == 1.0
    assert res["oof_auc"] == 1.0
    assert len(res["per_fold_ba"]) == 5
    assert "m1" in res["bootstrap_deltas"]


def test_package_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf

    base_cfg = OmegaConf.create(
        {
            "frozen": {
                "prior_pi1": 0.6366,
                "norm": {"mean": 0.3835, "std": 0.2496},
                "calibration": {"delta": 46.7029, "s": -1, "R": 0.1758},
            }
        }
    )

    spec = {
        "name": "test_ensemble",
        "members": [{"run_id": "r1", "weight": 0.5}],
    }
    eval_results = {
        "oof_ba": 0.7350,
        "oof_auc": 0.7580,
        "threshold": 0.4400,
        "per_fold_ba": [0.75, 0.76, 0.71, 0.60, 0.77],
        "bootstrap_deltas": {"r1": {"mean_delta": 0.01, "ci_95": [0.005, 0.02]}},
    }

    art_dir = package_artifact(spec, eval_results, base_cfg, date_str="20260916")
    assert art_dir.exists()
    assert (art_dir / "config.yaml").exists()
    assert (art_dir / "threshold.json").exists()
    assert (art_dir / "metrics.json").exists()
    assert (art_dir / "norm_stats.json").exists()
    assert (art_dir / "calibration.json").exists()
    assert (art_dir / "class_map.json").exists()
    assert (art_dir / "git_sha.txt").exists()

    reg_path = Path("artifacts/registry.json")
    assert reg_path.exists()
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert "candidates" in reg
