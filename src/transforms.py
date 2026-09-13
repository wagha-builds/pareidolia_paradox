"""
src/transforms.py — ALL geometry and photometric label-flips live here.
PROTECTED: changes require a PR, tests, and human approval (AGENTS.md §5).

Canonical frame conventions (AGENTS.md §6.1):
  - Arrays are (row, col), origin top-left, uint8 0-255 until normalization.
  - In canonical frame sun_theta = 90° (sun at top).
  - Vertical flip (reverse rows): label 0 (depth) <-> 1 (rise).
    A crater flipped vertically looks like a mound (bright bottom = rise).
  - Horizontal flip (reverse cols): label UNCHANGED.
    Left-right symmetry does not change crater vs mound.
  - Photometric negation (255 - img): label 0 <-> 1.
    Negating a crater (dark bottom) makes it look like a mound (bright bottom).
  - Never use arithmetic mean of azimuths; sin/cos only.
  - Never fill rotated corners with reflect/replicate padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data carrier
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """A single training example in any frame (raw or canonical).

    Attributes:
        image:   uint8 ndarray of shape (H, W) — never normalized here.
        azimuth: sun azimuth in degrees (metadata convention).
        label:   0 = Depth (crater), 1 = Rise (mound). -1 for test images.
    """

    image: np.ndarray  # shape (256, 256), dtype uint8
    azimuth: float
    label: int


# ---------------------------------------------------------------------------
# Canonical augmentation operators
# All operators return a new Sample (do not mutate the input).
# ---------------------------------------------------------------------------


class CanonicalVerticalFlipLabelSwap:
    """Vertical flip in the canonical frame; swaps the label.

    In the canonical frame (sun at top, theta_sun=90°):
      - A crater (depth=0) has bright top, dark bottom.
      - Flipping rows → dark top, bright bottom → looks like a mound (rise=1).
    Therefore label must be toggled.

    Args:
        p: probability of applying the flip. Default 0.20.
    """

    def __init__(self, p: float = 0.20) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        return Sample(
            image=np.flipud(sample.image).astype(np.uint8),
            azimuth=sample.azimuth,
            label=1 - sample.label if sample.label in (0, 1) else sample.label,
        )


class PhotometricNegationLabelSwap:
    """Photometric negation (255 - img); swaps the label.

    Negating pixel values inverts the shading polarity:
      - A crater (depth=0) goes from bright-top/dark-bottom → dark-top/bright-bottom.
      - That reversed shading is identical to a mound (rise=1).
    Therefore label must be toggled.

    Uses 255 - img exactly (AGENTS.md rule: never img.max() - img).

    Args:
        p: probability of applying negation. Default 0.15.
    """

    def __init__(self, p: float = 0.15) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        negated = (255 - sample.image.astype(np.int16)).clip(0, 255).astype(np.uint8)
        return Sample(
            image=negated,
            azimuth=sample.azimuth,
            label=1 - sample.label if sample.label in (0, 1) else sample.label,
        )


class CanonicalHorizontalFlip:
    """Horizontal flip in the canonical frame; label is UNCHANGED.

    Left-right mirroring preserves the top-bottom shading asymmetry that
    distinguishes craters from mounds. Label is class-neutral.

    Args:
        p: probability of applying the flip. Default 0.50.
    """

    def __init__(self, p: float = 0.50) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        return Sample(
            image=np.fliplr(sample.image).astype(np.uint8),
            azimuth=sample.azimuth,
            label=sample.label,
        )


# ---------------------------------------------------------------------------
# Policy builder
# ---------------------------------------------------------------------------

AugOp = Callable[[Sample], Sample]


def build_augmentation_policy(cfg: dict, training: bool) -> list[AugOp]:
    """Build the ordered list of Sample→Sample augmentation operators.

    At inference (training=False) returns an empty list — no augmentation.
    Reads p_vflip, p_neg, p_hflip from cfg['augmentation'].

    Args:
        cfg:      full config dict (OmegaConf-resolved).
        training: True for training, False for validation/inference.

    Returns:
        List of callable augmentation operators to apply in sequence.
    """
    if not training:
        return []

    aug_cfg = cfg.get("augmentation", {})
    p_vflip = float(aug_cfg.get("p_vflip", 0.0))
    p_neg = float(aug_cfg.get("p_neg", 0.0))
    p_hflip = float(aug_cfg.get("p_hflip", 0.0))

    return [
        CanonicalVerticalFlipLabelSwap(p=p_vflip),
        PhotometricNegationLabelSwap(p=p_neg),
        CanonicalHorizontalFlip(p=p_hflip),
    ]


# ---------------------------------------------------------------------------
# Raw-frame preprocessing (no geometry — label unchanged)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawTransform:
    """Raw-frame tensor preprocessing without geometric augmentation.

    Args:
        mean:       lunar normalization mean (not ImageNet).
        std:        lunar normalization std (not ImageNet).
        training:   if True, applies photometric-only jitter.
        brightness: uniform brightness jitter half-range.
        contrast:   uniform contrast jitter half-range.
        noise_std:  Gaussian noise std.
    """

    mean: float
    std: float
    training: bool = False
    brightness: float = 0.0
    contrast: float = 0.0
    noise_std: float = 0.0

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Convert a uint8 HxW image to a normalized 1xHxW float tensor."""
        x = image.astype(np.float32) / 255.0
        if self.training:
            if self.contrast > 0:
                scale = np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
                x = (x - x.mean()) * scale + x.mean()
            if self.brightness > 0:
                x = x + np.random.uniform(-self.brightness, self.brightness)
            if self.noise_std > 0:
                x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(
                    np.float32
                )
            x = np.clip(x, 0.0, 1.0)
        x = (x - float(self.mean)) / max(float(self.std), 1e-6)
        return torch.from_numpy(x).unsqueeze(0).float()


def build_transform(cfg: dict, training: bool) -> RawTransform:
    """Build the raw-frame normalization transform from config.

    Args:
        cfg:      full config dict (OmegaConf-resolved).
        training: if True, enables photometric jitter.

    Returns:
        RawTransform instance.
    """
    frozen = cfg.get("frozen", {})
    norm = frozen.get("norm", {})
    aug = cfg.get("augmentation", {})
    return RawTransform(
        mean=float(norm.get("mean", 0.0)),
        std=float(norm.get("std", 1.0)),
        training=training,
        brightness=float(aug.get("brightness", 0.0)),
        contrast=float(aug.get("contrast", 0.0)),
        noise_std=float(aug.get("noise_std", 0.0)),
    )
