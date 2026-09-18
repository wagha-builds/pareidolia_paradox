"""tests/test_pseudo_dataset.py — Tests for src/pseudo_dataset.py."""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _make_minimal_cache(tmpdir: Path, n_test: int = 5) -> tuple[Path, Path]:
    """Create minimal test cache files for PseudoDataset tests."""
    processed = tmpdir / "processed"
    processed.mkdir(parents=True)

    # Fake uint8 test images: (n_test, 256, 256)
    imgs = np.zeros((n_test, 256, 256), dtype=np.uint8)
    for i in range(n_test):
        imgs[i, 100:150, 100:150] = i * 50  # different patches
    np.save(processed / "test_images.npy", imgs)

    # Minimal index.json with test split
    test_rows = [
        {"image_id": f"test_{i:04d}.png", "sun_azimuth_angle": float(i * 60 % 360)}
        for i in range(n_test)
    ]
    index = {"test": test_rows}
    with (processed / "index.json").open("w") as f:
        json.dump(index, f)

    return tmpdir, processed


def _make_pseudo_csv(tmpdir: Path, image_ids: list[str], labels: list[int]) -> Path:
    df = pd.DataFrame({
        "image_id": image_ids,
        "label": labels,
        "sun_azimuth_angle": [45.0] * len(image_ids),
        "pseudo_prob": [0.1] * len(image_ids),
    })
    csv_path = tmpdir / "pseudo_labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestPseudoDataset:
    def test_len_and_getitem_shape(self, tmp_path):
        """PseudoDataset returns correct length and tensor shapes."""
        from src.pseudo_dataset import PseudoDataset

        data_dir, _ = _make_minimal_cache(tmp_path)
        pseudo_csv = _make_pseudo_csv(
            tmp_path,
            image_ids=["test_0000.png", "test_0001.png", "test_0002.png"],
            labels=[0, 1, 0],
        )
        ds = PseudoDataset(str(data_dir), str(pseudo_csv))

        assert len(ds) == 3
        img, az_sincos, label, image_id = ds[0]
        assert img.shape == (1, 256, 256), f"Unexpected shape {img.shape}"
        assert az_sincos.shape == (2,)
        assert label.item() in (0, 1)
        assert image_id == "test_0000.png"

    def test_label_values_correct(self, tmp_path):
        """Labels are exactly as specified in pseudo_label_csv."""
        from src.pseudo_dataset import PseudoDataset

        data_dir, _ = _make_minimal_cache(tmp_path)
        pseudo_csv = _make_pseudo_csv(
            tmp_path,
            image_ids=["test_0000.png", "test_0001.png"],
            labels=[0, 1],
        )
        ds = PseudoDataset(str(data_dir), str(pseudo_csv))

        _, _, label0, _ = ds[0]
        _, _, label1, _ = ds[1]
        assert label0.item() == 0
        assert label1.item() == 1

    def test_az_sincos_range(self, tmp_path):
        """az_sincos values are within [-1, 1]."""
        from src.pseudo_dataset import PseudoDataset

        data_dir, _ = _make_minimal_cache(tmp_path)
        pseudo_csv = _make_pseudo_csv(tmp_path, ["test_0003.png"], [1])
        ds = PseudoDataset(str(data_dir), str(pseudo_csv))

        _, az_sincos, _, _ = ds[0]
        assert az_sincos.abs().max().item() <= 1.0 + 1e-6

    def test_missing_image_id_raises(self, tmp_path):
        """PseudoDataset raises ValueError for image_ids not in test cache."""
        from src.pseudo_dataset import PseudoDataset

        data_dir, _ = _make_minimal_cache(tmp_path)
        pseudo_csv = _make_pseudo_csv(tmp_path, ["nonexistent_9999.png"], [0])

        with pytest.raises(ValueError, match="not in test cache"):
            PseudoDataset(str(data_dir), str(pseudo_csv))

    def test_concat_with_torch_dataset(self, tmp_path):
        """ConcatDataset works correctly with PseudoDataset."""
        from torch.utils.data import ConcatDataset

        from src.pseudo_dataset import PseudoDataset

        data_dir, _ = _make_minimal_cache(tmp_path)
        pseudo_csv = _make_pseudo_csv(
            tmp_path,
            image_ids=["test_0000.png", "test_0001.png"],
            labels=[0, 1],
        )
        ds1 = PseudoDataset(str(data_dir), str(pseudo_csv))
        ds2 = PseudoDataset(str(data_dir), str(pseudo_csv))
        combined = ConcatDataset([ds1, ds2])
        assert len(combined) == 4
        # Make sure DataLoader works
        loader = torch.utils.data.DataLoader(combined, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        imgs, az, labels, ids = batch
        assert imgs.shape[0] == 2
