"""Unit tests for src/robustness.py (Robustness & Shortcut Audit Suite)."""

import numpy as np
import pandas as pd

from src.robustness import (
    compute_b1_shortcut_predictions,
    compute_calibration_metrics,
    generate_robustness_report,
    run_shortcut_audit,
    run_slice_report,
)


def test_b1_shortcut_predictions():
    # Construct 2 synthetic 256x256 images:
    # Image 0: already in canonical frame (azimuth gives theta_sun = 90 deg)
    # Top is bright (200), bottom is dark (50) -> Rise (1)
    img_rise = np.zeros((256, 256), dtype=np.uint8)
    img_rise[:128, :] = 200
    img_rise[128:, :] = 50

    # Image 1: top is dark (50), bottom is bright (200) -> Depth (0)
    img_depth = np.zeros((256, 256), dtype=np.uint8)
    img_depth[:128, :] = 50
    img_depth[128:, :] = 200

    images = np.stack([img_rise, img_depth])
    # With s = -1, delta = 0: theta_sun = (-az) % 360
    # For theta_sun = 90, az = 270 (-270 = -270 % 360 = 90)
    azimuths = np.array([270.0, 270.0])

    preds = compute_b1_shortcut_predictions(images, azimuths, delta=0.0, s=-1)
    assert preds[0] == 1, "Top-bright canonical terrain should be predicted Rise (1)"
    assert preds[1] == 0, "Top-dark canonical terrain should be predicted Depth (0)"


def test_shortcut_audit():
    y_true = np.array([0, 0, 1, 1])
    shortcut_preds = np.array(
        [0, 1, 0, 1]
    )  # Wrong on index 1 (true 0, pred 1) and index 2 (true 1, pred 0)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])  # Model predicts correct classes at t=0.5
    threshold = 0.5

    audit = run_shortcut_audit(y_true, y_prob, threshold, shortcut_preds)

    assert audit["shortcut_wrong_count"] == 2
    assert np.isclose(audit["shortcut_wrong_pct"], 0.50)
    assert np.isclose(audit["b1_shortcut_ba"], 0.50)
    # Model gets both wrong cases right (y_prob[1]=0.2 < 0.5 -> 0, y_prob[2]=0.8 >= 0.5 -> 1)
    assert np.isclose(audit["shortcut_wrong_ba"], 1.0)


def test_calibration_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert np.isclose(metrics["brier_score"], 0.0)
    assert np.isclose(metrics["ece"], 0.0)


def test_calibration_metrics_uncalibrated():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert np.isclose(metrics["brier_score"], 0.25)


def test_slice_report():
    n = 16
    y_true = np.array([0, 1] * 8)
    y_prob = np.array([0.1, 0.9] * 8)
    threshold = 0.5

    metadata = pd.DataFrame(
        {
            "image_id": [f"img_{i}.png" for i in range(n)],
            "sun_azimuth_angle": [float(i * 22.5) for i in range(n)],
            "label": y_true,
        }
    )
    folds_df = pd.DataFrame(
        {
            "image_id": [f"img_{i}.png" for i in range(n)],
            "fold": [i % 5 for i in range(n)],
        }
    )

    slices = run_slice_report(y_true, y_prob, threshold, metadata, folds_df)

    assert "azimuth_octants" in slices
    assert "regional_folds" in slices
    assert "worst_slice" in slices
    assert slices["worst_slice"]["ba"] >= 0.0


def test_generate_robustness_report(tmp_path):
    results = {
        "oof_ba": 0.7250,
        "oof_auc": 0.7500,
        "threshold": 0.4500,
        "shortcut_audit": {
            "b1_shortcut_ba": 0.5800,
            "shortcut_wrong_count": 100,
            "shortcut_wrong_pct": 0.40,
            "shortcut_wrong_ba": 0.6300,
        },
        "slice_report": {
            "worst_slice": {"name": "Fold 2", "ba": 0.6500},
            "regional_folds": {
                "Fold 0": {"count": 50, "pi_1": 0.5, "ba": 0.7500},
                "Fold 2": {"count": 50, "pi_1": 0.5, "ba": 0.6500},
            },
            "azimuth_octants": {
                "0-45° (NNE)": {"count": 25, "pi_1": 0.5, "ba": 0.7200},
            },
        },
        "calibration": {
            "ece": 0.0450,
            "brier_score": 0.1800,
        },
        "inversion_stress": {
            "inversion_stress_ba": 0.7100,
            "inversion_consistency_error": 0.0800,
        },
        "azimuth_shift": {
            "mean_azimuth_shift_delta": 0.0150,
        },
        "occlusion": {
            "occlusion_base_ba": 0.7200,
            "occlusion_center_ba": 0.6800,
            "occlusion_degradation_delta": 0.0400,
        },
    }

    report_path = generate_robustness_report(
        "test_run_123", results, out_dir=str(tmp_path)
    )
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "test_run_123" in content
    assert "Shortcut-Wrong BA" in content
    assert "Inversion-Stress BA" in content
