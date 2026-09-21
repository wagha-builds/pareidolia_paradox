"""
src/sun_predictor.py — Self-Supervised Sun Direction Predictor via Rotation Equivariance.

Learns illumination angle theta_sun from 9,854 lunar tiles without ground-truth labels.
Uses ultra-fast GPU affine grid rotation (0.8s per epoch) for rotation equivariance training.
A soft hemisphere anchor constraint keeps the coordinate system physically aligned with metadata.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

SQRT2 = 2.0**0.5
INV_SQRT2 = 1.0 / SQRT2


class SunDirectionModel(nn.Module):
    """Backbone producing a 2D unit vector (cos theta, sin theta) for illumination direction."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        pretrained: bool = True,
        in_chans: int = 1,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,  # pooled features
        )
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns unit vector [B, 2] = (cos theta, sin theta)."""
        feats = self.backbone(x)
        if feats.dim() == 4:
            feats = feats.mean(dim=(-2, -1))
        elif feats.dim() == 3:
            feats = feats[:, 0]

        v = self.head(feats)
        # Normalize to unit circle
        return F.normalize(v, p=2, dim=1, eps=1e-8)


class FastLunarDataset(Dataset):
    """Loads all lunar tiles in uint8 and returns normalized tensors quickly."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        norm_mean: float = 100.86,
        norm_std: float = 54.55,
        include_test: bool = True,
    ):
        data_dir = Path(data_dir)
        processed = data_dir / "processed"

        train_imgs = np.load(processed / "train_images.npy")
        if include_test and (processed / "test_images.npy").exists():
            test_imgs = np.load(processed / "test_images.npy")
            self.images = np.concatenate([train_imgs, test_imgs], axis=0)
        else:
            self.images = train_imgs

        with (processed / "index.json").open("r", encoding="utf-8") as f:
            idx = json.load(f)

        records = list(idx["train"])
        if include_test and "test" in idx:
            records.extend(idx["test"])

        self.azimuths = np.array([r["sun_azimuth_angle"] for r in records], dtype=np.float32)
        assert len(self.images) == len(self.azimuths), "Images and azimuths mismatch"

        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].astype(np.float32)
        t_img = torch.from_numpy((img - self.norm_mean) / self.norm_std).unsqueeze(0)

        az_rad = np.radians(self.azimuths[idx])
        u_meta = torch.tensor([np.cos(az_rad), np.sin(az_rad)], dtype=torch.float32)
        return t_img, u_meta


def gpu_rotate_and_zoom(x: torch.Tensor, alpha_rad: torch.Tensor) -> torch.Tensor:
    """Rotates batch x by alpha_rad with sqrt(2) zoom on GPU using affine_grid and grid_sample.

    Args:
        x: [B, 1, H, W] image tensor on GPU
        alpha_rad: [B] angles in radians

    Returns:
        [B, 1, H, W] rotated and zoomed image tensor
    """
    B = x.shape[0]
    cos_a = torch.cos(alpha_rad)
    sin_a = torch.sin(alpha_rad)

    # Affine grid maps output coords to input coords:
    # [x_in]  = scale * [ cos  sin] [x_out]
    # [y_in]  = scale * [-sin  cos] [y_out]
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = INV_SQRT2 * cos_a
    theta[:, 0, 1] = INV_SQRT2 * sin_a
    theta[:, 1, 0] = -INV_SQRT2 * sin_a
    theta[:, 1, 1] = INV_SQRT2 * cos_a

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def compute_equivariance_loss(
    v_orig: torch.Tensor,
    v_rot: torch.Tensor,
    alpha_rad: torch.Tensor,
    u_meta: Optional[torch.Tensor] = None,
    lambda_anchor: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes rotation equivariance loss + optional soft hemisphere anchor loss.

    Args:
        v_orig: [B, 2] predicted unit vector for original image (cos, sin)
        v_rot:  [B, 2] predicted unit vector for rotated image
        alpha_rad: [B] rotation angle applied to original
        u_meta: [B, 2] approximate unit vector from metadata
        lambda_anchor: weight of hemisphere anchor loss

    Returns:
        total_loss, equiv_loss, anchor_loss
    """
    cos_a = torch.cos(alpha_rad)
    sin_a = torch.sin(alpha_rad)

    # R_alpha @ v_orig:
    # [cos  -sin] [vx] = [cos*vx - sin*vy]
    # [sin   cos] [vy] = [sin*vx + cos*vy]
    v_orig_x = v_orig[:, 0]
    v_orig_y = v_orig[:, 1]

    expected_rot_x = cos_a * v_orig_x - sin_a * v_orig_y
    expected_rot_y = sin_a * v_orig_x + cos_a * v_orig_y
    expected_v_rot = torch.stack([expected_rot_x, expected_rot_y], dim=1)

    # Equivariance cosine similarity loss: 1 - cos(theta_pred_rot - theta_expected)
    cos_sim = torch.sum(expected_v_rot * v_rot, dim=1)
    equiv_loss = torch.mean(1.0 - cos_sim)

    # Anchor loss: penalize if predicted vector points into opposite hemisphere of metadata
    if u_meta is not None and lambda_anchor > 0:
        meta_cos_sim = torch.sum(v_orig * u_meta, dim=1)
        anchor_loss = torch.mean(F.relu(-meta_cos_sim))
    else:
        anchor_loss = torch.tensor(0.0, device=v_orig.device)

    total_loss = equiv_loss + lambda_anchor * anchor_loss
    return total_loss, equiv_loss, anchor_loss
