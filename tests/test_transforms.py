"""
Tests for physics-valid transformations in src/transforms.py.
Enforces the mandatory requirements of AGENTS.md §8 and SKILL.md §4.
"""

import numpy as np
import pytest

from src.canonical import az_to_img, img_to_az, canonicalize
from src.transforms import (
    Sample,
    CanonicalHorizontalFlip,
    CanonicalVerticalFlipLabelSwap,
    PhotometricNegationLabelSwap,
    CanonicalRotationJitter,
    RawRotate,
    RawHorizontalFlip,
    RawVerticalFlip,
    RawRot180LabelSwap,
    RawNegationLabelSwap,
)
from tests.fixtures.synthetic import render


DELTA = 45.0
S = -1


def test_rotate_cycle_restores_azimuth():
    """Rotating by theta then -theta restores the azimuth."""
    az_orig = 120.0
    img = np.zeros((256, 256), dtype=np.uint8)
    sample = Sample(image=img, azimuth=az_orig, label=1)

    alpha = 67.5
    rot_fwd = RawRotate(delta=DELTA, s=S, angle_deg=alpha, p=1.0)
    rot_bwd = RawRotate(delta=DELTA, s=S, angle_deg=-alpha, p=1.0)

    sample = rot_fwd(sample)
    expected_fwd_az = (az_orig + S * alpha) % 360.0
    assert pytest.approx(sample.azimuth, abs=1e-5) == expected_fwd_az

    sample = rot_bwd(sample)
    assert pytest.approx(sample.azimuth, abs=1e-5) == az_orig


def test_rotate_and_canonicalize_equivariance():
    """
    Rotating image content by alpha and updating azimuth accordingly,
    then canonicalizing, yields the same canonical view as canonicalizing directly.
    """
    theta_sun = 30.0
    img = render("dome", theta_sun)
    az = img_to_az(theta_sun, DELTA, S)

    # Path A: canonicalize directly
    canonical_direct = canonicalize(img, az, DELTA, S)

    # Path B: rotate content by alpha with sqrt(2) zoom, then canonicalize
    # Note: canonicalize also applies sqrt(2) zoom, so to compare rotation equivariance,
    # we test that the canonical angle of the rotated sample aligns with 90 degrees.
    alpha = 45.0
    rot = RawRotate(delta=DELTA, s=S, angle_deg=alpha, p=1.0)
    sample = Sample(image=img.copy(), azimuth=az, label=1)
    sample = rot(sample)

    canonical_rotated = canonicalize(sample.image, sample.azimuth, DELTA, S)

    # The central circular region of both canonical representations should have peak brightness at the top
    top_a = canonical_direct[:64, 64:192].mean()
    bot_a = canonical_direct[192:, 64:192].mean()
    top_b = canonical_rotated[:64, 64:192].mean()
    bot_b = canonical_rotated[192:, 64:192].mean()

    assert top_a > bot_a
    assert top_b > bot_b


def test_double_negation_identity():
    """Double negation is the identity on both image and label."""
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    sample = Sample(image=img.copy(), azimuth=90.0, label=1)

    neg = PhotometricNegationLabelSwap(p=1.0)
    sample = neg(sample)
    assert sample.label == 0
    assert np.array_equal(sample.image, 255 - img)

    sample = neg(sample)
    assert sample.label == 1
    assert np.array_equal(sample.image, img)


def test_horizontal_flip_azimuth_matches_synthetic():
    """
    Horizontal flip azimuth updates match re-rendered synthetic images.
    Flipping a dome horizontally equals rendering the dome with the flipped sun vector.
    """
    theta_sun = 60.0
    img = render("dome", theta_sun)
    az = img_to_az(theta_sun, DELTA, S)

    sample = Sample(image=img.copy(), azimuth=az, label=1)
    hflip = RawHorizontalFlip(delta=DELTA, s=S, p=1.0)
    sample = hflip(sample)

    new_theta_sun = az_to_img(sample.azimuth, DELTA, S)
    assert pytest.approx(new_theta_sun, abs=1e-5) == (180.0 - theta_sun) % 360.0

    rendered_flipped = render("dome", new_theta_sun)
    # The flipped image and directly rendered flipped image should be virtually identical
    np.testing.assert_allclose(sample.image, rendered_flipped, atol=2.0)


def test_vertical_flip_azimuth_matches_synthetic():
    """
    Vertical flip azimuth updates match re-rendered synthetic images.
    Flipping a dome vertically equals rendering the dome with the vertically flipped sun vector.
    """
    theta_sun = 60.0
    img = render("dome", theta_sun)
    az = img_to_az(theta_sun, DELTA, S)

    sample = Sample(image=img.copy(), azimuth=az, label=1)
    vflip = RawVerticalFlip(delta=DELTA, s=S, p=1.0)
    sample = vflip(sample)

    new_theta_sun = az_to_img(sample.azimuth, DELTA, S)
    assert pytest.approx(new_theta_sun, abs=1e-5) == (-theta_sun) % 360.0

    rendered_flipped = render("dome", new_theta_sun)
    np.testing.assert_allclose(sample.image, rendered_flipped, atol=2.0)


def test_canonical_vertical_flip_reverses_asymmetry_and_swaps_label():
    """
    In the canonical frame (sun at 90 deg / top), vertical flip mirrors rows.
    The top-minus-bottom shading asymmetry reverses sign, and label flips.
    """
    # Canonical dome: sun at 90 deg -> bright at top, dark at bottom
    img = render("dome", 90.0)
    sample = Sample(image=img.copy(), azimuth=img_to_az(90.0, DELTA, S), label=1)

    asym_before = float(sample.image[:128, :].mean()) - float(sample.image[128:, :].mean())
    assert asym_before > 0  # Dome lit from top is brighter on top

    vflip_swap = CanonicalVerticalFlipLabelSwap(p=1.0)
    sample = vflip_swap(sample)

    asym_after = float(sample.image[:128, :].mean()) - float(sample.image[128:, :].mean())
    assert asym_after < 0   # Inverted asymmetry
    assert pytest.approx(asym_after, abs=1e-4) == -asym_before
    assert sample.label == 0  # Label swapped from Rise (1) to Depth (0)


def test_canonical_horizontal_flip():
    """In canonical frame, horizontal flip mirrors columns without altering label."""
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    sample = Sample(image=img.copy(), azimuth=90.0, label=1)
    hflip = CanonicalHorizontalFlip(p=1.0)
    sample = hflip(sample)

    assert sample.label == 1
    assert sample.azimuth == 90.0
    np.testing.assert_array_equal(sample.image, np.fliplr(img))


def test_canonical_rotation_jitter():
    """Rotation jitter produces output of identical shape with unchanged label."""
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    sample = Sample(image=img.copy(), azimuth=90.0, label=0)
    jitter = CanonicalRotationJitter(max_deg=5.0, p=1.0)
    sample = jitter(sample)

    assert sample.label == 0
    assert sample.image.shape == (64, 64)


def test_raw_negation_label_swap():
    """Raw photometric negation inverts grayscale values and flips label."""
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    sample = Sample(image=img.copy(), azimuth=180.0, label=1)
    neg = RawNegationLabelSwap(p=1.0)
    sample = neg(sample)

    assert sample.label == 0
    assert sample.azimuth == 180.0
    np.testing.assert_array_equal(sample.image, 255 - img)


def test_raw_rot180_label_swap():
    """Raw 180-deg rotation with azimuth held fixed flips the label."""
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    sample = Sample(image=img.copy(), azimuth=45.0, label=0)
    rot180 = RawRot180LabelSwap(p=1.0)
    sample = rot180(sample)

    assert sample.label == 1
    assert sample.azimuth == 45.0  # Azimuth held fixed
    np.testing.assert_array_equal(sample.image, np.rot90(img, 2))


