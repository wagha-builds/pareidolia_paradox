"""
Physics-valid transformation and augmentation algebra for lunar relief classification.
Enforces synchronization between image geometry, solar azimuth angle, and terrain labels.

Per AGENTS.md §5 and SKILL.md §4, standard library flips and rotations inject catastrophic
label noise because relief depends jointly on (image, sun_azimuth_angle).
"""

from dataclasses import dataclass
import random
from typing import Callable, List, Optional
import numpy as np
import cv2

from .canonical import az_to_img, img_to_az, SQRT2


@dataclass
class Sample:
    """Container for an image sample, its associated sun azimuth, and target label."""
    image: np.ndarray      # (H, W) array (uint8 or float32)
    azimuth: float         # Sun azimuth angle in degrees [0, 360)
    label: int             # 0 (Depth), 1 (Rise), or -1 (unlabeled test sample)


# ---------------------------------------------------------------------------
# Canonical Frame Transforms (Sun at theta = 90 deg)
# ---------------------------------------------------------------------------

class CanonicalHorizontalFlip:
    """
    Horizontal flip in the canonical frame (mirroring across the vertical sun axis).
    Sun remains at the top (90 deg). Label is unchanged.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample.image = np.ascontiguousarray(np.fliplr(sample.image))
        return sample


class CanonicalVerticalFlipLabelSwap:
    """
    Vertical flip in canonical frame (mirroring across the axis perpendicular to sunlight).
    Top-vs-bottom shading order inverts, turning pits into mounds and vice versa.
    Label is mathematically swapped (0 <-> 1).
    """
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample.image = np.ascontiguousarray(np.flipud(sample.image))
            if sample.label in (0, 1):
                sample.label = 1 - sample.label
        return sample


class PhotometricNegationLabelSwap:
    """
    Photometric negation (255 - img on uint8, or 1.0 - img on float32).
    Under linearized Lambertian shading, inverting intensity inverts surface relief (z -> -z).
    Label is swapped (0 <-> 1).
    """
    def __init__(self, p: float = 0.15):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            if sample.image.dtype == np.uint8:
                sample.image = 255 - sample.image
            else:
                sample.image = 1.0 - sample.image
            if sample.label in (0, 1):
                sample.label = 1 - sample.label
        return sample


class CanonicalRotationJitter:
    """
    Small rotation jitter (within +/- max_deg) to model azimuth measurement noise.
    Label is unchanged.
    """
    def __init__(self, max_deg: float = 10.0, p: float = 0.5):
        self.max_deg = max_deg
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            angle = random.uniform(-self.max_deg, self.max_deg)
            h, w = sample.image.shape[:2]
            M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
            sample.image = cv2.warpAffine(
                sample.image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
            )
        return sample


# ---------------------------------------------------------------------------
# Raw Frame Transforms (Azimuth & Label tracking)
# ---------------------------------------------------------------------------

class RawRotate:
    """
    Rotates image content counter-clockwise by alpha degrees using the sqrt(2) zoom
    warp so scale is invariant and no corner is ever padded.
    Azimuth is updated: az -> (az + s * alpha) % 360. Label is unchanged.
    """
    def __init__(self, delta: float = 0.0, s: int = 1, angle_deg: Optional[float] = None, p: float = 1.0):
        self.delta = delta
        self.s = s
        self.angle_deg = angle_deg
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            alpha = self.angle_deg if self.angle_deg is not None else random.uniform(0.0, 360.0)
            h, w = sample.image.shape[:2]
            M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), alpha, SQRT2)
            sample.image = cv2.warpAffine(
                sample.image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
            )
            sample.azimuth = (sample.azimuth + self.s * alpha) % 360.0
        return sample


class RawHorizontalFlip:
    """
    Horizontal flip (mirror columns) on raw frame.
    theta_sun -> (180 - theta_sun) % 360. Azimuth updated accordingly. Label is unchanged.
    """
    def __init__(self, delta: float = 0.0, s: int = 1, p: float = 0.5):
        self.delta = delta
        self.s = s
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample.image = np.ascontiguousarray(np.fliplr(sample.image))
            theta = az_to_img(sample.azimuth, self.delta, self.s)
            new_theta = (180.0 - theta) % 360.0
            sample.azimuth = img_to_az(new_theta, self.delta, self.s)
        return sample


class RawVerticalFlip:
    """
    Vertical flip (mirror rows) on raw frame.
    theta_sun -> (-theta_sun) % 360. Azimuth updated accordingly. Label is unchanged.
    """
    def __init__(self, delta: float = 0.0, s: int = 1, p: float = 0.5):
        self.delta = delta
        self.s = s
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample.image = np.ascontiguousarray(np.flipud(sample.image))
            theta = az_to_img(sample.azimuth, self.delta, self.s)
            new_theta = (-theta) % 360.0
            sample.azimuth = img_to_az(new_theta, self.delta, self.s)
        return sample


class RawRot180LabelSwap:
    """
    Rotate 180 degrees with azimuth held fixed.
    Since lighting direction is unchanged relative to world coordinates but image is upside-down,
    apparent relief is inverted (pit <-> mound). Label is swapped (0 <-> 1).
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample.image = np.ascontiguousarray(np.rot90(sample.image, 2))
            if sample.label in (0, 1):
                sample.label = 1 - sample.label
        return sample


class RawNegationLabelSwap:
    """
    Photometric negation with azimuth held fixed.
    Inverting pixel intensities inverts apparent relief. Label is swapped (0 <-> 1).
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            if sample.image.dtype == np.uint8:
                sample.image = 255 - sample.image
            else:
                sample.image = 1.0 - sample.image
            if sample.label in (0, 1):
                sample.label = 1 - sample.label
        return sample


# ---------------------------------------------------------------------------
# Transform Composition Pipeline
# ---------------------------------------------------------------------------

class Compose:
    """Chains a sequence of Sample -> Sample transformations."""
    def __init__(self, transforms: List[Callable[[Sample], Sample]]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_canonical_train_transforms(
    p_hflip: float = 0.5,
    p_vflip: float = 0.2,
    p_neg: float = 0.15,
    p_jitter: float = 0.5,
    max_jitter_deg: float = 10.0
) -> Compose:
    """Builds training augmentation pipeline for canonical-frame samples."""
    return Compose([
        CanonicalHorizontalFlip(p=p_hflip),
        CanonicalVerticalFlipLabelSwap(p=p_vflip),
        PhotometricNegationLabelSwap(p=p_neg),
        CanonicalRotationJitter(max_deg=max_jitter_deg, p=p_jitter),
    ])


def get_raw_train_transforms(
    delta: float,
    s: int,
    p_rotate: float = 1.0,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    p_neg: float = 0.15,
    p_rot180_flip: float = 0.0,
) -> Compose:
    """Builds training augmentation pipeline for raw-frame samples."""
    transforms = [
        RawRotate(delta=delta, s=s, p=p_rotate),
        RawHorizontalFlip(delta=delta, s=s, p=p_hflip),
        RawVerticalFlip(delta=delta, s=s, p=p_vflip),
        RawNegationLabelSwap(p=p_neg),
    ]
    if p_rot180_flip > 0.0:
        transforms.append(RawRot180LabelSwap(p=p_rot180_flip))
    return Compose(transforms)
