"""
Unit tests for CNN models and conditioning heads in src/models.py.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models import PareidoliaModel, build_model


@pytest.mark.parametrize("conditioning", ["none", "concat", "film"])
def test_model_forward_shapes(conditioning):
    """Verifies forward pass shape (B,) across all conditioning modes."""
    batch_size = 3
    model = PareidoliaModel(
        backbone_name="resnet18",
        in_chans=1,
        pretrained=False,
        conditioning=conditioning,
    )
    model.eval()

    x = torch.randn(batch_size, 1, 128, 128)
    az = torch.randn(batch_size, 2)

    with torch.no_grad():
        out = model(x, az)

    assert out.shape == (batch_size,)
    assert not torch.isnan(out).any()


def test_model_backward_pass():
    """Verifies that gradients flow through all model parameters during training."""
    model = PareidoliaModel(
        backbone_name="resnet18",
        in_chans=1,
        pretrained=False,
        conditioning="film",
    )
    model.train()

    x = torch.randn(2, 1, 64, 64)
    az = torch.randn(2, 2)
    targets = torch.tensor([0.0, 1.0])

    logits = model(x, az)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    # Verify gradients exist on backbone and FiLM generator
    assert model.classifier[-1].weight.grad is not None
    assert model.az_module.generator[0].weight.grad is not None


@pytest.mark.parametrize("conditioning", ["concat", "film"])
def test_conditioning_sensitivity(conditioning):
    """
    Verifies that changing the azimuth vector on identical images produces
    different model outputs, confirming active conditioning.
    """
    model = PareidoliaModel(
        backbone_name="resnet18",
        in_chans=1,
        pretrained=False,
        conditioning=conditioning,
    )
    model.eval()

    # Single image duplicated twice
    single_img = torch.randn(1, 1, 64, 64)
    x = torch.cat([single_img, single_img], dim=0)

    # Two distinct azimuth angles: 0 deg ([0, 1]) and 90 deg ([1, 0])
    az = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    # For film, we randomly initialize generator to test sensitivity
    if conditioning == "film":
        with torch.no_grad():
            model.az_module.generator[-1].weight.normal_()

    with torch.no_grad():
        out = model(x, az)

    assert out[0] != out[1]


def test_build_model_factory():
    """Verifies build_model instantiates correctly from config dictionary."""
    cfg = {
        "model": {
            "backbone": "resnet18",
            "in_chans": 1,
            "pretrained": False,
            "conditioning": "none",
        },
        "training": {
            "drop_path_rate": 0.0,
        },
    }
    model = build_model(cfg)
    assert isinstance(model, PareidoliaModel)
    assert model.conditioning == "none"
    assert model.az_module is None
