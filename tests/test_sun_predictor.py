import numpy as np
import pytest
import torch

from src.sun_predictor import SunDirectionModel, compute_equivariance_loss


def test_model_unit_norm():
    """Verify SunDirectionModel outputs strict unit vectors."""
    # Using small mock input
    model = SunDirectionModel(pretrained=False)
    model.eval()
    x = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        v = model(x)
    assert v.shape == (4, 2)
    norms = torch.norm(v, p=2, dim=1)
    np.testing.assert_allclose(norms.numpy(), np.ones(4), atol=1e-5)


def test_equivariance_loss_perfect():
    """Verify loss is exactly zero when predictions are perfectly equivariant."""
    batch_size = 4
    # v_orig pointing at 0 deg (1, 0)
    v_orig = torch.tensor([[1.0, 0.0]] * batch_size)
    # alpha = 90 deg (pi/2)
    alpha = torch.tensor([np.pi / 2] * batch_size)
    # v_rot should be pointing at 90 deg (0, 1)
    v_rot = torch.tensor([[0.0, 1.0]] * batch_size)
    u_meta = torch.tensor([[1.0, 0.0]] * batch_size)

    total_loss, equiv_loss, anchor_loss = compute_equivariance_loss(
        v_orig, v_rot, alpha, u_meta=u_meta, lambda_anchor=0.2
    )

    assert equiv_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert anchor_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert total_loss.item() == pytest.approx(0.0, abs=1e-5)


def test_equivariance_loss_opposite():
    """Verify loss is maximum (2.0) when rotated prediction is opposite."""
    batch_size = 2
    v_orig = torch.tensor([[1.0, 0.0]] * batch_size)
    alpha = torch.tensor([0.0] * batch_size)
    # rotated prediction is opposite (-1, 0)
    v_rot = torch.tensor([[-1.0, 0.0]] * batch_size)

    total_loss, equiv_loss, _ = compute_equivariance_loss(
        v_orig, v_rot, alpha, u_meta=None, lambda_anchor=0.0
    )
    assert equiv_loss.item() == pytest.approx(2.0, abs=1e-5)
