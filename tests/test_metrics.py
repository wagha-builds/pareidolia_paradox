import numpy as np
from src.metrics import apply_threshold, plateau_threshold

def test_apply_threshold():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    res = apply_threshold(p, 0.5)
    np.testing.assert_array_equal(res, [0, 0, 1, 1])

def test_plateau_threshold():
    y = np.array([0, 0, 1, 1])
    # Trivial perfect predictor
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t, grid, ba = plateau_threshold(y, p)
    # Any threshold between 0.2 and 0.8 is perfect. The midpoint is ~0.5.
    assert 0.3 < t < 0.7
    assert np.max(ba) == 1.0
