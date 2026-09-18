"""
src/dataset.py — Data loading, caching, and PareidoliaDataset.
PROTECTED: changes require a PR, tests, and human approval (AGENTS.md §5).

Class mapping is FROZEN: {0: depth (crater), 1: rise (mound)}.
Derived from neither LabelEncoder nor sorted(unique); always asserted at load time.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


# Frozen class mapping (AGENTS.md §5 — never derive from data)
_CLASS_MAP = {0: "depth", 1: "rise"}

RAW_DIRS = {
    "train": ("train_metadata.csv", "train_images"),
    "test": ("test_metadata.csv", "eval_images"),
}


def load_metadata(data_dir: "str | Path", split: str) -> pd.DataFrame:
    """Load metadata with stable string image IDs.

    Args:
        data_dir: root data directory.
        split:    'train' or 'test'.

    Returns:
        DataFrame with image_id (str), sun_azimuth_angle (float), label (int, train only).
    """
    if split not in RAW_DIRS:
        raise ValueError(f"Unknown split {split!r}")
    data_dir = Path(data_dir)
    meta_name, _ = RAW_DIRS[split]
    df = pd.read_csv(data_dir / "raw" / meta_name, dtype={"image_id": str})
    expected = ["image_id", "sun_azimuth_angle"] + (
        ["label"] if split == "train" else []
    )
    if list(df.columns) != expected:
        raise ValueError(
            f"{meta_name} columns must be {expected}, got {list(df.columns)}"
        )
    if df["image_id"].isna().any() or df["sun_azimuth_angle"].isna().any():
        raise ValueError(f"{meta_name} contains null values")
    if df["image_id"].duplicated().any():
        raise ValueError(f"{meta_name} contains duplicate image_id values")
    if split == "train":
        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError("train labels must be integers in {0, 1}")
        # Assert frozen class mapping (AGENTS.md §5)
        assert _CLASS_MAP == {
            0: "depth",
            1: "rise",
        }, "Class mapping drifted — this must always be {0: depth, 1: rise}"
    return df


def build_cache(data_dir: "str | Path" = "data") -> None:
    """Build uint8 image caches and index.json in metadata order.

    Images are stored in exactly the metadata CSV order so that index.json
    positions are stable and reproducible.

    Args:
        data_dir: root data directory (contains raw/ and processed/ subdirs).
    """
    data_dir = Path(data_dir)
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    index: dict[str, list[dict[str, object]]] = {}

    for split, (meta_name, image_dir_name) in RAW_DIRS.items():
        df = load_metadata(data_dir, split)
        image_dir = data_dir / "raw" / image_dir_name
        images = np.empty((len(df), 256, 256), dtype=np.uint8)
        records: list[dict[str, object]] = []
        for i, row in df.iterrows():
            image_id = str(row["image_id"])
            path = image_dir / image_id
            if not path.exists():
                raise FileNotFoundError(path)
            img = Image.open(path).convert("L")
            arr = np.asarray(img)
            if arr.shape != (256, 256):
                raise ValueError(f"{path} has shape {arr.shape}, expected (256, 256)")
            images[i] = arr.astype(np.uint8, copy=False)
            rec: dict[str, object] = {
                "image_id": image_id,
                "sun_azimuth_angle": float(row["sun_azimuth_angle"]),
            }
            if split == "train":
                rec["label"] = int(row["label"])
            records.append(rec)
        np.save(processed / f"{split}_images.npy", images)
        index[split] = records

    with (processed / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


class PareidoliaDataset(Dataset):
    """Pareidolia lunar terrain dataset.

    Loads uint8 images from the preprocessed cache. Optionally:
      1. Canonicalizes the image (rotates so sun is at top).
      2. Applies a list of Sample→Sample augmentation operators.
      3. Normalizes with a RawTransform (or custom transform).

    The class mapping {0: depth, 1: rise} is asserted at __init__ time.
    Labels are never derived from data ordering or LabelEncoder.

    Args:
        data_dir:           root data directory.
        split:              'train' or 'test'.
        indices:            optional subset of row indices (for fold splits).
        transform:          a callable (np.ndarray → torch.Tensor) applied after
                            canonicalization and augmentation.
        canonicalize_cfg:   if not None, a dict with keys 'delta', 's' from the
                            frozen config; images are rotated to canonical frame
                            using src.canonical.canonicalize().
        augmentation_policy: list of Sample→Sample callables applied in order
                             after canonicalization and before normalization.
                             Must be empty ([]) at validation/inference time.
        norm_stats:         optional dict with 'mean' and 'std' for manual
                            normalization (used when transform is None).
        channel_mode:       '1ch' (default).
    """

    def __init__(
        self,
        data_dir,
        split: str = "train",
        indices=None,
        transform=None,
        canonicalize_cfg: Optional[dict] = None,
        augmentation_policy: Optional[list] = None,
        norm_stats: Optional[dict] = None,
        channel_mode: str = "1ch",
    ):
        # Assert frozen class mapping on every dataset instantiation
        assert _CLASS_MAP == {
            0: "depth",
            1: "rise",
        }, "Class mapping must be {0: depth, 1: rise}"

        self.data_dir = Path(data_dir)
        self.split = split
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.canonicalize_cfg = canonicalize_cfg
        self.augmentation_policy = augmentation_policy or []
        self.norm_stats = norm_stats
        self.channel_mode = channel_mode

        # Load from cache
        cache_img_path = self.data_dir / "processed" / f"{split}_images.npy"
        cache_idx_path = self.data_dir / "processed" / "index.json"

        if cache_img_path.exists() and cache_idx_path.exists():
            with cache_idx_path.open("r", encoding="utf-8") as f:
                idx_data = json.load(f)
            self.metadata = pd.DataFrame(idx_data[split]).reset_index(drop=True)
            self.images = np.load(cache_img_path, mmap_mode="r")
        else:
            raise FileNotFoundError("Cache not found. Run `make cache` first.")

    def __len__(self) -> int:
        return len(self.metadata) if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        """Return (img_tensor, az_sincos, label_tensor, image_id).

        Pipeline:
            raw uint8 image
            → [canonicalize] if canonicalize_cfg is set
            → [augmentation_policy] each Sample→Sample op in sequence
            → [transform / manual normalization] → torch.Tensor 1×H×W
        """
        real_idx = int(idx if self.indices is None else self.indices[idx])
        row = self.metadata.iloc[real_idx]
        img: np.ndarray = np.array(self.images[real_idx])  # copy from memmap → uint8
        azimuth: float = float(row["sun_azimuth_angle"])
        label: int = int(row["label"]) if "label" in row else -1

        # Step 1: Canonicalize (rotate so sun is at top)
        if self.canonicalize_cfg is not None:
            from .canonical import canonicalize  # lazy import to avoid circular

            delta = float(self.canonicalize_cfg["delta"])
            s = int(self.canonicalize_cfg["s"])
            jitter_std = float(self.canonicalize_cfg.get("jitter_deg", 0.0))
            jitter = (
                float(np.random.normal(0.0, jitter_std)) if jitter_std > 0.0 else 0.0
            )
            img = canonicalize(
                img, azimuth, delta, s, jitter_deg=jitter
            )  # returns uint8 ndarray

        # Step 2: Apply augmentation policy (Sample objects carry label)
        if self.augmentation_policy:
            from .transforms import Sample  # lazy import to avoid circular

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
            if self.norm_stats:
                img_tensor = (img_tensor - self.norm_stats["mean"]) / self.norm_stats[
                    "std"
                ]

        az_sincos = torch.tensor(
            [np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))],
            dtype=torch.float32,
        )
        return (
            img_tensor,
            az_sincos,
            torch.tensor(label, dtype=torch.long),
            row["image_id"],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["cache", "folds"])
    args = parser.parse_args()

    if args.mode == "cache":
        build_cache("data")
        print("Cache written to data/processed.")
    else:
        print("Fold generation is intentionally handled by scripts/eda.py.")
