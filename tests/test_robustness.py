import numpy as np
from src.robustness import run_shortcut_audit, run_slice_report


def test_shortcut_audit():
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_pred = np.array([0, 1, 1, 0], dtype=int)
    # az: sin(az) > 0 for 45 deg (Rise shortcut), sin(az) < 0 for 225 deg (Depth shortcut)
    az = np.array([225.0, 45.0, 45.0, 225.0], dtype=np.float32)

    res = run_shortcut_audit(y_true, y_pred, az)
    assert "ba_overall" in res
    assert "ba_shortcut_wrong" in res
    assert "ba_shortcut_correct" in res
    assert res["n_shortcut_correct"] == 2
    assert res["n_shortcut_wrong"] == 2


def test_slice_report():
    np.random.seed(42)
    N = 100
    y_true = np.random.randint(0, 2, size=N)
    y_pred = np.random.randint(0, 2, size=N)
    az = np.random.uniform(0, 360, size=N).astype(np.float32)
    images = np.random.randint(0, 256, size=(N, 256, 256), dtype=np.uint8)

    res = run_slice_report(y_true, y_pred, az, images)
    assert "slices" in res
    assert "worst_slice_name" in res
    assert "worst_slice_ba" in res
    assert len(res["slices"]) > 0
