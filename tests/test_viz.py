import numpy as np
import pytest
import torch
import torch.nn as nn

from src.viz import (
    GradCAM,
    compute_shadow_mass_fraction,
    decanonicalize_cam,
    overlay_cam_on_image,
)


class DummyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x_pool = self.pool(x).flatten(1)
        return self.fc(x_pool)


def test_gradcam_shape_and_range():
    model = DummyConvNet()
    cam_extractor = GradCAM(model, target_layer=model.conv1)
    x = torch.randn(1, 1, 64, 64)

    cam, pred_cls, prob_rise = cam_extractor(x, target_class=1)

    assert cam.shape == (64, 64)
    assert cam.dtype == np.float32
    assert 0.0 <= cam.min() <= cam.max() <= 1.0
    assert pred_cls in (0, 1)
    assert 0.0 <= prob_rise <= 1.0

    cam_extractor.remove_hooks()


def test_shadow_mass_fraction_bounds():
    # Construct a simple CAM map and shadow mask
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[2:5, 2:5] = 1.0  # 9 pixels with weight 1.0
    shadow_mask = np.zeros((10, 10), dtype=bool)
    shadow_mask[2:4, 2:4] = True  # 4 pixels overlap

    fraction = compute_shadow_mass_fraction(cam, shadow_mask)
    assert pytest.approx(fraction, 1e-4) == 4.0 / 9.0
    assert 0.0 <= fraction <= 1.0


def test_overlay_cam_on_image():
    base_img = np.full((32, 32), 128, dtype=np.uint8)
    cam = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32).astype(np.float32)

    overlay = overlay_cam_on_image(base_img, cam, alpha=0.5)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
    assert overlay.min() >= 0
    assert overlay.max() <= 255


def test_decanonicalize_cam():
    cam = np.ones((64, 64), dtype=np.float32)
    decanon = decanonicalize_cam(cam, azimuth=90.0, delta=46.7, s=-1)

    assert decanon.shape == (64, 64)
    assert decanon.dtype == np.float32
    assert 0.0 <= decanon.min() <= decanon.max() <= 1.0
