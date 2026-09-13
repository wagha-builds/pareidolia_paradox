"""
Tests for canonical augmentation operators in src/transforms.py.
All geometry must go through transforms.py — no library flips/rotations elsewhere.

Coverage required by AGENTS.md §8:
- vflip round-trip identity; label toggles correctly
- photometric negation round-trip identity; label toggles correctly
- hflip changes pixels, label unchanged
- canonical vflip reverses top-minus-bottom asymmetry
- Lint rule: no library flip/rotation with non-zero probability outside transforms.py
"""

import re
from pathlib import Path

import numpy as np

from src.transforms import (
    CanonicalHorizontalFlip,
    CanonicalVerticalFlipLabelSwap,
    PhotometricNegationLabelSwap,
    Sample,
    build_augmentation_policy,
)
from tests.fixtures.synthetic import render


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pit_bright_top() -> np.ndarray:
    """A pit with sun at top (theta_sun=90°) in image-plane convention."""
    # theta_sun=90° → sun is straight up → shadow below → pit has bright top
    return render("pit", theta_sun_deg=90)


def _dome_bright_top() -> np.ndarray:
    """A dome with sun at top."""
    return render("dome", theta_sun_deg=90)


def _make_sample(kind: str = "pit", label: int = 0) -> Sample:
    img = render(kind, theta_sun_deg=90)
    return Sample(image=img, azimuth=90.0, label=label)


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


def test_sample_immutable_or_accessible():
    """Sample should be accessible with .image, .azimuth, .label."""
    img = np.zeros((256, 256), dtype=np.uint8)
    s = Sample(image=img, azimuth=45.0, label=1)
    assert s.label == 1
    assert s.azimuth == 45.0
    assert s.image.shape == (256, 256)


# ---------------------------------------------------------------------------
# CanonicalVerticalFlipLabelSwap
# ---------------------------------------------------------------------------


def test_canonical_vflip_round_trip_image():
    """Apply vflip twice → identical image."""
    aug = CanonicalVerticalFlipLabelSwap(p=1.0)
    s = _make_sample("pit", label=0)
    flipped = aug(s)
    restored = aug(flipped)
    np.testing.assert_array_equal(restored.image, s.image)


def test_canonical_vflip_round_trip_label():
    """Apply vflip twice → label returns to original."""
    aug = CanonicalVerticalFlipLabelSwap(p=1.0)
    for original_label in (0, 1):
        s = _make_sample("pit", label=original_label)
        toggled = aug(s)
        restored = aug(toggled)
        assert (
            restored.label == original_label
        ), f"Double vflip should restore label {original_label}, got {restored.label}"


def test_canonical_vflip_toggles_label():
    """Single application always toggles label when p=1."""
    aug = CanonicalVerticalFlipLabelSwap(p=1.0)
    s0 = Sample(image=np.zeros((256, 256), dtype=np.uint8), azimuth=90.0, label=0)
    s1 = Sample(image=np.zeros((256, 256), dtype=np.uint8), azimuth=90.0, label=1)
    assert aug(s0).label == 1
    assert aug(s1).label == 0


def test_canonical_vflip_flips_pixels():
    """Vertical flip reverses the rows of the image."""
    aug = CanonicalVerticalFlipLabelSwap(p=1.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img, azimuth=90.0, label=0)
    result = aug(s)
    np.testing.assert_array_equal(result.image, np.flipud(img))


def test_canonical_vflip_reverses_top_bottom_asymmetry():
    """In canonical frame (sun at top), a pit has bright top, dark bottom.
    After vflip it should have dark top, bright bottom.
    Asymmetry score = mean(top half) - mean(bottom half); sign must flip.
    """
    img = _pit_bright_top()
    h = 256
    top_mean = float(img[: h // 2].mean())
    bot_mean = float(img[h // 2 :].mean())
    original_asymmetry = top_mean - bot_mean

    aug = CanonicalVerticalFlipLabelSwap(p=1.0)
    s = Sample(image=img, azimuth=90.0, label=0)
    flipped_img = aug(s).image
    flip_top = float(flipped_img[: h // 2].mean())
    flip_bot = float(flipped_img[h // 2 :].mean())
    flipped_asymmetry = flip_top - flip_bot

    assert (
        np.sign(original_asymmetry) != np.sign(flipped_asymmetry)
    ), f"Expected sign flip; original={original_asymmetry:.3f}, flipped={flipped_asymmetry:.3f}"


def test_canonical_vflip_noop_at_p0():
    """p=0 → image and label unchanged."""
    aug = CanonicalVerticalFlipLabelSwap(p=0.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img.copy(), azimuth=90.0, label=0)
    result = aug(s)
    np.testing.assert_array_equal(result.image, img)
    assert result.label == 0


# ---------------------------------------------------------------------------
# PhotometricNegationLabelSwap
# ---------------------------------------------------------------------------


def test_negation_round_trip_image():
    """Apply negation twice → identical image (255-x applied twice = x)."""
    aug = PhotometricNegationLabelSwap(p=1.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img.copy(), azimuth=90.0, label=0)
    result = aug(aug(s))
    np.testing.assert_array_equal(result.image, img)


def test_negation_round_trip_label():
    """Apply negation twice → label back to original."""
    aug = PhotometricNegationLabelSwap(p=1.0)
    for original_label in (0, 1):
        img = np.zeros((256, 256), dtype=np.uint8)
        s = Sample(image=img, azimuth=90.0, label=original_label)
        result = aug(aug(s))
        assert result.label == original_label


def test_negation_toggles_label():
    """Single application: label 0→1, 1→0."""
    aug = PhotometricNegationLabelSwap(p=1.0)
    s0 = Sample(image=np.zeros((256, 256), dtype=np.uint8), azimuth=90.0, label=0)
    s1 = Sample(image=np.zeros((256, 256), dtype=np.uint8), azimuth=90.0, label=1)
    assert aug(s0).label == 1
    assert aug(s1).label == 0


def test_negation_pixel_formula():
    """Negation must use 255 - img (not img.max() - img)."""
    aug = PhotometricNegationLabelSwap(p=1.0)
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    s = Sample(image=img.copy(), azimuth=0.0, label=0)
    result = aug(s)
    expected = (255 - img).astype(np.uint8)
    np.testing.assert_array_equal(result.image, expected)


def test_negation_preserves_uint8():
    """Output image must remain uint8."""
    aug = PhotometricNegationLabelSwap(p=1.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img, azimuth=90.0, label=0)
    result = aug(s)
    assert result.image.dtype == np.uint8


def test_negation_noop_at_p0():
    """p=0 → image and label unchanged."""
    aug = PhotometricNegationLabelSwap(p=0.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img.copy(), azimuth=90.0, label=1)
    result = aug(s)
    np.testing.assert_array_equal(result.image, img)
    assert result.label == 1


# ---------------------------------------------------------------------------
# CanonicalHorizontalFlip
# ---------------------------------------------------------------------------


def test_hflip_changes_pixels():
    """Horizontal flip changes the image pixels.
    Uses an asymmetric image (feature offset from centre) so fliplr differs.
    """
    aug = CanonicalHorizontalFlip(p=1.0)
    # Render a pit with sun from the left (theta_sun=180) so shadow is on the right.
    # This image is NOT left-right symmetric.
    img = render("pit", theta_sun_deg=180)
    s = Sample(image=img.copy(), azimuth=180.0, label=0)
    result = aug(s)
    assert not np.array_equal(
        result.image, img
    ), "hflip must change an asymmetric image"


def test_hflip_label_unchanged():
    """Horizontal flip must NOT change the label (physics: hflip is class-neutral)."""
    aug = CanonicalHorizontalFlip(p=1.0)
    for lbl in (0, 1):
        s = Sample(image=np.zeros((256, 256), dtype=np.uint8), azimuth=90.0, label=lbl)
        assert aug(s).label == lbl, f"label must not change after hflip (label={lbl})"


def test_hflip_round_trip():
    """Apply hflip twice → identical image."""
    aug = CanonicalHorizontalFlip(p=1.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img.copy(), azimuth=90.0, label=0)
    result = aug(aug(s))
    np.testing.assert_array_equal(result.image, img)


def test_hflip_flips_columns():
    """Horizontal flip reverses columns."""
    aug = CanonicalHorizontalFlip(p=1.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img, azimuth=90.0, label=0)
    result = aug(s)
    np.testing.assert_array_equal(result.image, np.fliplr(img))


def test_hflip_noop_at_p0():
    """p=0 → image unchanged."""
    aug = CanonicalHorizontalFlip(p=0.0)
    img = render("pit", theta_sun_deg=90)
    s = Sample(image=img.copy(), azimuth=90.0, label=0)
    result = aug(s)
    np.testing.assert_array_equal(result.image, img)


# ---------------------------------------------------------------------------
# build_augmentation_policy
# ---------------------------------------------------------------------------


def test_build_policy_training_returns_callables():
    """Training policy returns a list of callables."""
    cfg = {"augmentation": {"p_vflip": 0.25, "p_neg": 0.15, "p_hflip": 0.50}}
    policy = build_augmentation_policy(cfg, training=True)
    assert isinstance(policy, list)
    assert len(policy) == 3
    for op in policy:
        assert callable(op)


def test_build_policy_inference_is_empty():
    """Inference policy returns empty list (no augmentation at test time)."""
    cfg = {"augmentation": {"p_vflip": 0.25, "p_neg": 0.15, "p_hflip": 0.50}}
    policy = build_augmentation_policy(cfg, training=False)
    assert policy == [], "Inference policy must be empty"


def test_build_policy_applies_to_sample():
    """Policy transforms a Sample correctly."""
    cfg = {"augmentation": {"p_vflip": 0.0, "p_neg": 1.0, "p_hflip": 0.0}}
    policy = build_augmentation_policy(cfg, training=True)
    img = np.ones((256, 256), dtype=np.uint8) * 100
    s = Sample(image=img.copy(), azimuth=90.0, label=0)
    for op in policy:
        s = op(s)
    # Only negation applied (p_neg=1.0): label should be 1, pixels = 255-100 = 155
    assert s.label == 1
    np.testing.assert_array_equal(s.image, (255 - img).astype(np.uint8))


# ---------------------------------------------------------------------------
# Lint rule: no library flip/rotation outside transforms.py
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = [
    r"\.flip\s*\(",
    r"\.transpose\s*\(",
    r"RandomHorizontalFlip",
    r"RandomVerticalFlip",
    r"RandomRotation",
    r"RandAugment",
    r"TrivialAugmentWide",
    r"AugMix",
    r"albumentations",
    r"kornia\.augmentation",
    r"create_transform\s*\(.*is_training\s*=\s*True",
]
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_PATTERNS))

_TRANSFORMS_PATH = Path("src/transforms.py")
_THIS_FILE = Path(__file__).resolve()


def _source_files_except_transforms() -> list[Path]:
    root = Path("src")
    files = list(root.glob("**/*.py"))
    test_files = list(Path("tests").glob("**/*.py"))
    script_files = list(Path("scripts").glob("**/*.py"))
    all_files = files + test_files + script_files
    # Exclude transforms.py (it legitimately uses fliplr/flipud)
    # Exclude this test file (it contains the forbidden strings as string literals in the regex pattern)
    return [
        f
        for f in all_files
        if f.resolve() != _TRANSFORMS_PATH.resolve() and f.resolve() != _THIS_FILE
    ]


def test_no_library_flip_rotation_outside_transforms():
    """No library flip/rotation with non-zero probability may appear outside transforms.py."""
    violations = []
    for path in _source_files_except_transforms():
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _FORBIDDEN_RE.search(line):
                violations.append(f"{path}:{lineno}: {stripped}")
    assert violations == [], (
        "Library flips/rotations found outside src/transforms.py:\n"
        + "\n".join(violations)
    )
