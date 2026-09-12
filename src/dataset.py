"""
Dataset loading, memory-mapped caching, and transform integration.
Follows AGENTS.md §10 (zero per-epoch PNG decoding; memmapped uint8 arrays)
and SKILL.md §1 & §4 (lunar normalization stats, physics transforms).
"""

import os
import json
import argparse
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .transforms import Sample
from .canonical import canonicalize


DEFAULT_NORM_MEAN = 0.3835757076740265
DEFAULT_NORM_STD = 0.2496677041053772


class PareidoliaDataset(Dataset):
    """
    PyTorch Dataset for Pareidolia lunar relief classification.
    Reads uint8 grayscale images directly from memory-mapped numpy cache arrays.
    """
    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",             # "train", "val", or "test"
        fold: Optional[int] = None,       # 0..4 for cross-validation
        canonicalize_cfg: Optional[Dict[str, Any]] = None,
        augmentation_policy: Optional[Callable[[Sample], Sample]] = None,
        norm_stats: Optional[Dict[str, float]] = None,
        channel_mode: str = "1ch",        # "1ch" or "3ch"
    ):
        self.data_dir = data_dir
        self.split = split
        self.fold = fold
        self.canonicalize_cfg = canonicalize_cfg
        self.augmentation_policy = augmentation_policy
        self.norm_stats = norm_stats or {"mean": DEFAULT_NORM_MEAN, "std": DEFAULT_NORM_STD}
        self.channel_mode = channel_mode

        processed_dir = os.path.join(data_dir, "processed")
        index_path = os.path.join(processed_dir, "index.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Cache index not found at {index_path}. Run 'python -m src.dataset --mode cache' first."
            )

        with open(index_path, "r") as f:
            index_data = json.load(f)

        if split in ("train", "val"):
            cache_img_path = os.path.join(processed_dir, "train_images.npy")
            if not os.path.exists(cache_img_path):
                raise FileNotFoundError(f"Cache array not found at {cache_img_path}.")
            
            self.images = np.load(cache_img_path, mmap_mode="r")
            df = pd.DataFrame(index_data["train"])

            if fold is not None:
                if "fold" not in df.columns:
                    raise ValueError("Column 'fold' missing from train metadata index.")
                if split == "train":
                    mask = df["fold"] != fold
                else:
                    mask = df["fold"] == fold
                self.indices = np.where(mask)[0]
                self.metadata = df.iloc[self.indices].reset_index(drop=True)
            else:
                self.indices = np.arange(len(df))
                self.metadata = df
        elif split == "test":
            cache_img_path = os.path.join(processed_dir, "test_images.npy")
            if not os.path.exists(cache_img_path):
                raise FileNotFoundError(f"Cache array not found at {cache_img_path}.")
            
            self.images = np.load(cache_img_path, mmap_mode="r")
            df = pd.DataFrame(index_data["test"])
            self.indices = np.arange(len(df))
            self.metadata = df
        else:
            raise ValueError(f"Unknown split '{split}'. Must be 'train', 'val', or 'test'.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        img = np.array(self.images[real_idx], dtype=np.uint8)
        row = self.metadata.iloc[idx]
        azimuth = float(row["sun_azimuth_angle"])
        label = int(row["label"]) if "label" in row and pd.notna(row["label"]) else -1

        # 1. Canonicalization (if specified)
        if self.canonicalize_cfg is not None:
            delta = float(self.canonicalize_cfg.get("delta", 46.70))
            s = int(self.canonicalize_cfg.get("s", -1))
            img = canonicalize(img, azimuth, delta, s)

        # 2. Physics-consistent augmentation
        if self.augmentation_policy is not None:
            sample = Sample(image=img, azimuth=azimuth, label=label)
            sample = self.augmentation_policy(sample)
            img, azimuth, label = sample.image, sample.azimuth, sample.label

        # 3. Normalization (using lunar surface stats)
        img_float = img.astype(np.float32) / 255.0
        mean = float(self.norm_stats["mean"])
        std = float(self.norm_stats["std"])
        img_norm = (img_float - mean) / std

        # 4. Channel formatting
        if self.channel_mode == "1ch":
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
        elif self.channel_mode == "3ch":
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).repeat(3, 1, 1).float()
        else:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        # 5. Continuous [sin(az), cos(az)] representation
        az_rad = np.radians(azimuth)
        az_sincos = torch.tensor([np.sin(az_rad), np.cos(az_rad)], dtype=torch.float32)

        return img_tensor, az_sincos, torch.tensor(label, dtype=torch.float32)


def cache_dataset(data_dir: str = "data") -> None:
    """
    Compiles raw PNG images into memory-mapped uint8 numpy arrays
    and builds index.json with folds info.
    """
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_meta_path = os.path.join(raw_dir, "train_metadata.csv")
    test_meta_path = os.path.join(raw_dir, "test_metadata.csv")
    folds_path = os.path.join(data_dir, "folds.csv")
    train_img_dir = os.path.join(raw_dir, "train_images")
    test_img_dir = os.path.join(raw_dir, "eval_images")

    if not os.path.exists(train_meta_path) or not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"Training data not found in {raw_dir}.")

    print("Loading train metadata...")
    train_df = pd.read_csv(train_meta_path)
    if os.path.exists(folds_path):
        folds_df = pd.read_csv(folds_path)
        train_df = train_df.merge(folds_df, on="image_id", how="left")
        print(f"Merged folds.csv (found {train_df['fold'].nunique()} folds).")

    print(f"Caching {len(train_df)} training images...")
    train_images = []
    for img_id in tqdm(train_df["image_id"], desc="Train images"):
        path = os.path.join(train_img_dir, img_id)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {path}")
        train_images.append(img)

    train_arr = np.stack(train_images).astype(np.uint8)
    train_cache_path = os.path.join(processed_dir, "train_images.npy")
    np.save(train_cache_path, train_arr)
    print(f"Saved {train_cache_path} (shape={train_arr.shape}, dtype={train_arr.dtype}, size={train_arr.nbytes / (1024*1024):.1f} MB)")

    # Compute normalization stats
    train_norm = train_arr.astype(np.float32) / 255.0
    calc_mean = float(train_norm.mean())
    calc_std = float(train_norm.std())
    print(f"Calculated lunar normalization stats: mean={calc_mean:.6f}, std={calc_std:.6f}")

    # Process evaluation/test images
    test_df = pd.read_csv(test_meta_path)
    print(f"Caching {len(test_df)} test images...")
    test_images = []
    for img_id in tqdm(test_df["image_id"], desc="Test images"):
        path = os.path.join(test_img_dir, img_id)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {path}")
        test_images.append(img)

    test_arr = np.stack(test_images).astype(np.uint8)
    test_cache_path = os.path.join(processed_dir, "test_images.npy")
    np.save(test_cache_path, test_arr)
    print(f"Saved {test_cache_path} (shape={test_arr.shape}, dtype={test_arr.dtype}, size={test_arr.nbytes / (1024*1024):.1f} MB)")

    # Write index.json
    index_data = {
        "train": train_df.to_dict(orient="records"),
        "test": test_df.to_dict(orient="records"),
        "norm": {"mean": calc_mean, "std": calc_std},
    }
    index_path = os.path.join(processed_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"Saved {index_path}")
    print("Dataset caching complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pareidolia Dataset CLI")
    parser.add_argument("--mode", type=str, required=True, choices=["cache"])
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    if args.mode == "cache":
        cache_dataset(args.data_dir)
