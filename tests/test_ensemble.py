import pytest
import numpy as np
from src.ensemble import blend_predictions, check_matching_folds


def test_blend_predictions_prob_mean():
    p1 = np.array([0.2, 0.6, 0.8], dtype=np.float32)
    p2 = np.array([0.4, 0.4, 0.2], dtype=np.float32)

    # Equal weights
    blended = blend_predictions([p1, p2], method="prob_mean")
    np.testing.assert_allclose(blended, [0.3, 0.5, 0.5], atol=1e-5)

    # Weighted
    blended_w = blend_predictions([p1, p2], weights=[0.25, 0.75], method="prob_mean")
    expected = 0.25 * p1 + 0.75 * p2
    np.testing.assert_allclose(blended_w, expected, atol=1e-5)


def test_blend_predictions_logit_mean():
    p1 = np.array([0.3, 0.7], dtype=np.float32)
    p2 = np.array([0.4, 0.6], dtype=np.float32)

    blended = blend_predictions([p1, p2], method="logit_mean")
    assert blended.shape == (2,)
    assert (blended >= 0.0).all() and (blended <= 1.0).all()


def test_check_matching_folds():
    m1 = {"folds_sha256": "abc123hash"}
    m2 = {"folds_sha256": "abc123hash"}
    m_diff = {"folds_sha256": "different_hash"}

    # Matching folds should pass
    check_matching_folds([m1, m2])

    # Differing folds should raise ValueError
    with pytest.raises(ValueError, match="Fold hash mismatch"):
        check_matching_folds([m1, m_diff])
