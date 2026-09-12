import numpy as np

from src.transforms import RawTransform


def test_raw_transform_normalizes_without_geometry():
    image = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256) % 256
    image = image.astype(np.uint8)
    out = RawTransform(mean=0.5, std=0.25, training=False)(image)

    assert out.shape == (1, 256, 256)
    assert float(out[0, 0, 0]) == -2.0
    assert float(out[0, 0, 255]) == 2.0
