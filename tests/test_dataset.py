"""
Unit tests for PareidoliaDataset and transform integration in src/dataset.py.
"""

import os
import json
import numpy as np
import pytest
import torch

from src.dataset import PareidoliaDataset
from src.transforms import CanonicalHorizontalFlip


@pytest.fixture
def mock_processed_dir(tmp_path):
    """Creates a minimal mock data/processed directory with synthetic arrays."""
    data_dir = tmp_path / "data"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True)

    # 10 train samples, 4 test samples
    train_imgs = np.random.randint(0, 256, (10, 64, 64), dtype=np.uint8)
    test_imgs = np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8)

    np.save(proc_dir / "train_images.npy", train_imgs)
    np.save(proc_dir / "test_images.npy", test_imgs)

    train_records = [
        {"image_id": f"train_{i}.png", "sun_azimuth_angle": float(i * 36.0), "label": i % 2, "fold": i % 5}
        for i in range(10)
    ]
    test_records = [
        {"image_id": f"test_{i}.png", "sun_azimuth_angle": float(i * 90.0)}
        for i in range(4)
    ]

    index_data = {
        "train": train_records,
        "test": test_records,
        "norm": {"mean": 0.5, "std": 0.25},
    }
    with open(proc_dir / "index.json", "w") as f:
        json.dump(index_data, f)

    return str(data_dir)


def test_dataset_train_val_split(mock_processed_dir):
    """Verifies train/val splitting based on fold parameter."""
    # Fold 0 has 2 samples (indices 0 and 5 out of 10)
    val_ds = PareidoliaDataset(data_dir=mock_processed_dir, split="val", fold=0)
    assert len(val_ds) == 2

    # Train split has remaining 8 samples
    train_ds = PareidoliaDataset(data_dir=mock_processed_dir, split="train", fold=0)
    assert len(train_ds) == 8


def test_dataset_test_split(mock_processed_dir):
    """Verifies test split loading."""
    test_ds = PareidoliaDataset(data_dir=mock_processed_dir, split="test")
    assert len(test_ds) == 4
    img, az, lbl = test_ds[0]
    assert img.shape == (1, 64, 64)
    assert az.shape == (2,)
    assert lbl.item() == -1.0


def test_dataset_getitem_transforms(mock_processed_dir):
    """Verifies augmentation policy integration in __getitem__."""
    policy = CanonicalHorizontalFlip(p=1.0)
    ds = PareidoliaDataset(
        data_dir=mock_processed_dir,
        split="train",
        fold=0,
        augmentation_policy=policy,
        channel_mode="1ch",
    )
    img_tensor, az_sincos, label_tensor = ds[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dtype == torch.float32
    assert img_tensor.shape == (1, 64, 64)
    assert az_sincos.shape == (2,)
    assert label_tensor.item() in (0.0, 1.0)
