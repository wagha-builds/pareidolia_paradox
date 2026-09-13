import numpy as np
import cv2

# Phase 1: Azimuth Calibration & Canonicalization

SQRT2 = 2**0.5


def az_to_img(az, delta, s):
    return (s * az + delta) % 360.0


def img_to_az(theta, delta, s):
    return (s * (theta - delta)) % 360.0


def dark_minus_bright_deg(img, frac=0.10, center_crop=None):
    """Angle of centroid(darkest frac) - centroid(brightest frac)."""
    x = img.astype(np.float32)
    if center_crop is not None:
        h, w = x.shape
        r0 = (h - center_crop) // 2
        c0 = (w - center_crop) // 2
        x = x[r0 : r0 + center_crop, c0 : c0 + center_crop]

    lo, hi = np.quantile(x, [frac, 1 - frac])
    r, c = np.indices(x.shape)
    dark, bright = x <= lo, x >= hi
    d_r = r[dark].mean() - r[bright].mean()
    d_c = c[dark].mean() - c[bright].mean()
    return np.degrees(np.arctan2(-d_r, d_c)) % 360.0


def circ_mean_R(deg):
    a = np.radians(deg)
    C, S = np.cos(a).mean(), np.sin(a).mean()
    return np.degrees(np.arctan2(S, C)) % 360.0, float(np.hypot(C, S))


def calibrate(images, az, labels, center_crop=None):
    phi = np.array(
        [dark_minus_bright_deg(im, center_crop=center_crop) for im in images]
    )
    toward = (phi + np.where(labels == 1, 180.0, 0.0)) % 360.0
    fits = {s: circ_mean_R(toward - s * az) for s in (+1, -1)}
    s = max(fits, key=lambda k: fits[k][1])
    return {"delta": fits[s][0], "s": s, "R": fits[s][1], "R_other_s": fits[-s][1]}


def canonical_rotation_deg(az, delta, s):
    return (90.0 - az_to_img(az, delta, s)) % 360.0


def canonicalize(img, az, delta, s, jitter_deg=0.0):
    h, w = img.shape
    angle = canonical_rotation_deg(az, delta, s) + jitter_deg
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, SQRT2)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
    )


def decanonicalize(img, az, delta, s, jitter_deg=0.0):
    h, w = img.shape
    angle = canonical_rotation_deg(az, delta, s) + jitter_deg
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, SQRT2)
    # invert affine mapping
    inv_M = cv2.invertAffineTransform(M)
    # to reverse the canonical warp, we use invertAffineTransform
    return cv2.warpAffine(
        img,
        inv_M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
