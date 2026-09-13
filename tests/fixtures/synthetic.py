import numpy as np


def render(
    kind,
    theta_sun_deg,
    elev_deg=25.0,
    sigma=28.0,
    height=18.0,
    size=256,
    cx=None,
    cy=None,
):
    """Gaussian dome ('dome') or pit ('pit') lit from theta_sun (convention §1). Returns uint8."""
    c = (size - 1) / 2.0
    cx, cy = (c if cx is None else cx), (c if cy is None else cy)
    rows, cols = np.indices((size, size), dtype=np.float64)
    x, y = cols - cx, -(rows - cy)  # y up as displayed
    z = height * np.exp(-(x**2 + y**2) / (2 * sigma**2)) * (1 if kind == "dome" else -1)
    n = np.stack(
        [x / sigma**2 * z, y / sigma**2 * z, np.ones_like(z)], -1
    )  # (-dz/dx, -dz/dy, 1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    t, e = np.radians(theta_sun_deg), np.radians(elev_deg)
    light = np.array([np.cos(e) * np.cos(t), np.cos(e) * np.sin(t), np.sin(e)])
    return np.round(np.clip(n @ light, 0, 1) * 255).astype(np.uint8)
