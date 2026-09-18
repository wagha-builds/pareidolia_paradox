"""src/pseudo_dataset.py — Dataset for pseudo-labeled test images.

Loads test images from the processed cache and assigns pseudo-labels from a CSV.
Designed to be concatenated with the real PareidoliaDataset via torch.utils.data.ConcatDataset
in train.py, keeping src/dataset.py (PROTECTED) untouched.

Usage in train.py:
    from .pseudo_dataset import PseudoDataset
    pseudo_ds = PseudoDataset(
        "data",
        pseudo_label_csv="data/pseudo_labels.csv",
        transform=build_transform(cfg_dict, training=True),
        canonicalize_cfg=train_canonicalize_cfg,
        augmentation_policy=train_policy,
    )
    train_ds = ConcatDataset([real_train_ds, pseudo_ds])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_CLASS_MAP = {0: "depth", 1: "rise"}


class PseudoDataset(torch.utils.data.Dataset):
    """Pseudo-labeled test images for self-training.

    Args:
        data_dir:           Path to the data directory (same as PareidoliaDataset).
        pseudo_label_csv:   Path to CSV with columns: image_id, label, sun_azimuth_angle.
        transform:          RawTransform instance for tensor normalization.
        canonicalize_cfg:   Optional dict with 'delta', 's', 'jitter_deg' keys.
        augmentation_policy: List of Sample→Sample augmentation operators.
    """

    def __init__(
        self,
        data_dir: str,
        pseudo_label_csv: str,
        transform=None,
        canonicalize_cfg: Optional[dict] = None,
        augmentation_policy: Optional[list] = None,
    ) -> None:
        assert _CLASS_MAP == {0: "depth", 1: "rise"}, "Class mapping must be {0: depth, 1: rise}"

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.canonicalize_cfg = canonicalize_cfg
        self.augmentation_policy = augmentation_policy or []

        # Load pseudo-label CSV
        self.pseudo_df = pd.read_csv(pseudo_label_csv, dtype={"image_id": str})
        assert "image_id" in self.pseudo_df.columns, "pseudo_label_csv must have 'image_id' column"
        assert "label" in self.pseudo_df.columns, "pseudo_label_csv must have 'label' column"
        assert "sun_azimuth_angle" in self.pseudo_df.columns, "pseudo_label_csv must have 'sun_azimuth_angle'"

        # Load test image cache
        cache_img_path = self.data_dir / "processed" / "test_images.npy"
        cache_idx_path = self.data_dir / "processed" / "index.json"
        if not cache_img_path.exists() or not cache_idx_path.exists():
            raise FileNotFoundError("Test cache not found. Run `make cache` first.")

        with cache_idx_path.open("r", encoding="utf-8") as f:
            idx_data = json.load(f)
        self.test_meta = pd.DataFrame(idx_data["test"]).reset_index(drop=True)
        self.test_images = np.load(cache_img_path, mmap_mode="r")

        # Build image_id → cache row index map
        self._id_to_idx = {str(row["image_id"]): i for i, row in self.test_meta.iterrows()}

        # Validate all pseudo-label image_ids exist in the test cache
        missing = [
            iid for iid in self.pseudo_df["image_id"]
            if iid not in self._id_to_idx
        ]
        if missing:
            raise ValueError(f"Pseudo-label CSV has {len(missing)} image_ids not in test cache: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.pseudo_df)

    def __getitem__(self, idx):
        """Return (img_tensor, az_sincos, label_tensor, image_id)."""
        row = self.pseudo_df.iloc[idx]
        image_id = str(row["image_id"])
        cache_idx = self._id_to_idx[image_id]

        img: np.ndarray = np.array(self.test_images[cache_idx])  # copy from memmap
        azimuth: float = float(row["sun_azimuth_angle"])
        label: int = int(row["label"])

        # Step 1: Canonicalize with optional jitter (training only)
        if self.canonicalize_cfg is not None:
            from .canonical import canonicalize

            delta = float(self.canonicalize_cfg["delta"])
            s = int(self.canonicalize_cfg["s"])
            jitter_std = float(self.canonicalize_cfg.get("jitter_deg", 0.0))
            jitter = float(np.random.normal(0.0, jitter_std)) if jitter_std > 0.0 else 0.0
            img = canonicalize(img, azimuth, delta, s, jitter_deg=jitter)

        # Step 2: Augmentation policy
        if self.augmentation_policy:
            from .transforms import Sample

            sample = Sample(image=img, azimuth=azimuth, label=label)
            for op in self.augmentation_policy:
                sample = op(sample)
            img = sample.image
            label = sample.label

        # Step 3: Normalize → tensor
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        az_sincos = torch.tensor(
            [np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))],
            dtype=torch.float32,
        )
        return (
            img_tensor,
            az_sincos,
            torch.tensor(label, dtype=torch.long),
            image_id,
        )
