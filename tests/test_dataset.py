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
