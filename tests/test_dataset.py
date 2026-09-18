import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import PareidoliaDataset, build_cache


def _write_png(path: Path, value: int):
    arr = np.full((256, 256), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_build_cache_preserves_metadata_order(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "train_images").mkdir(parents=True)
    (raw / "eval_images").mkdir(parents=True)
    pd.DataFrame(
        {
            "image_id": ["train_b.png", "train_a.png"],
            "sun_azimuth_angle": [10.0, 20.0],
            "label": [1, 0],
        }
    ).to_csv(raw / "train_metadata.csv", index=False)
    pd.DataFrame({"image_id": ["eval_b.png"], "sun_azimuth_angle": [30.0]}).to_csv(
        raw / "test_metadata.csv", index=False
    )
    _write_png(raw / "train_images" / "train_b.png", 7)
    _write_png(raw / "train_images" / "train_a.png", 9)
    _write_png(raw / "eval_images" / "eval_b.png", 11)

    build_cache(tmp_path)

    index = json.loads((tmp_path / "processed" / "index.json").read_text())
    assert [r["image_id"] for r in index["train"]] == ["train_b.png", "train_a.png"]
    arr = np.load(tmp_path / "processed" / "train_images.npy")
    assert int(arr[0, 0, 0]) == 7
    assert int(arr[1, 0, 0]) == 9


def test_dataset_returns_image_id_and_label_from_cache(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "train_images").mkdir(parents=True)
    (raw / "eval_images").mkdir(parents=True)
    pd.DataFrame(
        {"image_id": ["train_1.png"], "sun_azimuth_angle": [90.0], "label": [1]}
    ).to_csv(raw / "train_metadata.csv", index=False)
    pd.DataFrame({"image_id": ["eval_1.png"], "sun_azimuth_angle": [0.0]}).to_csv(
        raw / "test_metadata.csv", index=False
    )
    _write_png(raw / "train_images" / "train_1.png", 255)
    _write_png(raw / "eval_images" / "eval_1.png", 0)
    build_cache(tmp_path)

    image, az_sincos, label, image_id = PareidoliaDataset(tmp_path, "train")[0]

    assert image.shape == (1, 256, 256)
    np.testing.assert_allclose(az_sincos.numpy(), [1.0, 0.0], atol=1e-6)
    assert int(label) == 1
    assert image_id == "train_1.png"


def test_dataset_canonicalize_cfg(tmp_path: Path):
    """Dataset with canonicalize_cfg returns rotated image compared to raw."""
    raw = tmp_path / "raw"
    (raw / "train_images").mkdir(parents=True)
    (raw / "eval_images").mkdir(parents=True)
    pd.DataFrame(
        {"image_id": ["train_asym.png"], "sun_azimuth_angle": [180.0], "label": [0]}
    ).to_csv(raw / "train_metadata.csv", index=False)
    pd.DataFrame({"image_id": ["eval_1.png"], "sun_azimuth_angle": [0.0]}).to_csv(
        raw / "test_metadata.csv", index=False
    )
    # Create asymmetric image: top half 100, bottom half 200
    arr = np.zeros((256, 256), dtype=np.uint8)
    arr[:128, :] = 100
    arr[128:, :] = 200
    Image.fromarray(arr).save(raw / "train_images" / "train_asym.png")
    _write_png(raw / "eval_images" / "eval_1.png", 0)
    build_cache(tmp_path)

    raw_img, _, _, _ = PareidoliaDataset(tmp_path, "train", canonicalize_cfg=None)[0]
    canon_img, _, _, _ = PareidoliaDataset(
        tmp_path, "train", canonicalize_cfg={"delta": 46.7, "s": -1}
    )[0]

    # Canonicalizing with azimuth=180 rotates the image, so pixels differ
    assert not np.allclose(raw_img.numpy(), canon_img.numpy())


def test_dataset_augmentation_policy(tmp_path: Path):
    """Dataset applies augmentation policy, which can toggle the label."""
    from src.transforms import PhotometricNegationLabelSwap

    raw = tmp_path / "raw"
    (raw / "train_images").mkdir(parents=True)
    (raw / "eval_images").mkdir(parents=True)
    pd.DataFrame(
        {"image_id": ["train_1.png"], "sun_azimuth_angle": [0.0], "label": [0]}
    ).to_csv(raw / "train_metadata.csv", index=False)
    pd.DataFrame({"image_id": ["eval_1.png"], "sun_azimuth_angle": [0.0]}).to_csv(
        raw / "test_metadata.csv", index=False
    )
    _write_png(raw / "train_images" / "train_1.png", 50)
    _write_png(raw / "eval_images" / "eval_1.png", 0)
    build_cache(tmp_path)

    policy = [PhotometricNegationLabelSwap(p=1.0)]
    img_tensor, _, label, _ = PareidoliaDataset(
        tmp_path, "train", augmentation_policy=policy
    )[0]

    # Label toggles from 0 to 1
    assert int(label) == 1
    # Pixel value was negated: 255 - 50 = 205 (normalized by /255.0 = 205/255)
    np.testing.assert_allclose(float(img_tensor[0, 0, 0]), 205.0 / 255.0, atol=1e-3)


def test_dataset_canonicalize_with_jitter(tmp_path: Path):
    """Dataset with jitter_deg produces valid float tensor with proper dimensions."""
    raw = tmp_path / "raw"
    (raw / "train_images").mkdir(parents=True)
    (raw / "eval_images").mkdir(parents=True)
    pd.DataFrame(
        {"image_id": ["train_1.png"], "sun_azimuth_angle": [45.0], "label": [0]}
    ).to_csv(raw / "train_metadata.csv", index=False)
    pd.DataFrame({"image_id": ["eval_1.png"], "sun_azimuth_angle": [0.0]}).to_csv(
        raw / "test_metadata.csv", index=False
    )
    _write_png(raw / "train_images" / "train_1.png", 100)
    _write_png(raw / "eval_images" / "eval_1.png", 0)
    build_cache(tmp_path)

    img_tensor, az_sincos, label, img_id = PareidoliaDataset(
        tmp_path,
        "train",
        canonicalize_cfg={"delta": 46.7, "s": -1, "jitter_deg": 10.0},
    )[0]
    assert img_tensor.shape == (1, 256, 256)
    assert az_sincos.shape == (2,)
    assert int(label) == 0
    assert img_id == "train_1.png"
