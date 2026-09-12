import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


RAW_DIRS = {
    "train": ("train_metadata.csv", "train_images"),
    "test": ("test_metadata.csv", "eval_images"),
}


def load_metadata(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Load metadata with stable string image IDs."""
    if split not in RAW_DIRS:
        raise ValueError(f"Unknown split {split!r}")
    data_dir = Path(data_dir)
    meta_name, _ = RAW_DIRS[split]
    df = pd.read_csv(data_dir / "raw" / meta_name, dtype={"image_id": str})
    expected = ["image_id", "sun_azimuth_angle"] + (["label"] if split == "train" else [])
    if list(df.columns) != expected:
        raise ValueError(f"{meta_name} columns must be {expected}, got {list(df.columns)}")
    if df["image_id"].isna().any() or df["sun_azimuth_angle"].isna().any():
        raise ValueError(f"{meta_name} contains null values")
    if df["image_id"].duplicated().any():
        raise ValueError(f"{meta_name} contains duplicate image_id values")
    if split == "train" and not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("train labels must be integers in {0, 1}")
    return df


def build_cache(data_dir: str | Path = "data") -> None:
    """Build uint8 image caches and index.json in metadata order."""
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
            rec = {
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
    def __init__(
        self,
        data_dir,
        split="train",
        indices=None,
        transform=None,
        canonicalize_cfg=None,
        augmentation_policy=None,
        norm_stats=None,
        channel_mode="1ch",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.canonicalize_cfg = canonicalize_cfg
        self.augmentation_policy = augmentation_policy
        self.norm_stats = norm_stats
        self.channel_mode = channel_mode

        # Load from cache if available
        cache_img_path = self.data_dir / "processed" / f"{split}_images.npy"
        cache_idx_path = self.data_dir / "processed" / "index.json"

        if cache_img_path.exists() and cache_idx_path.exists():
            with cache_idx_path.open("r", encoding="utf-8") as f:
                idx_data = json.load(f)
            self.metadata = pd.DataFrame(idx_data[split]).reset_index(drop=True)
            self.images = np.load(cache_img_path, mmap_mode="r")
        else:
            raise FileNotFoundError("Cache not found. Run `make cache` first.")

    def __len__(self):
        return len(self.metadata) if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(idx if self.indices is None else self.indices[idx])
        row = self.metadata.iloc[real_idx]
        img = np.array(self.images[real_idx])  # copy from memmap
        azimuth = float(row["sun_azimuth_angle"])
        label = int(row["label"]) if "label" in row else -1

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            if self.norm_stats:
                img_tensor = (img_tensor - self.norm_stats["mean"]) / self.norm_stats["std"]

        az_sincos = torch.tensor(
            [np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))], dtype=torch.float32
        )
        return img_tensor, az_sincos, torch.tensor(label, dtype=torch.long), row["image_id"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["cache", "folds"])
    args = parser.parse_args()

    if args.mode == "cache":
        build_cache("data")
        print("Cache written to data/processed.")
    else:
        print("Fold generation is intentionally handled by scripts/eda.py and refuses overwrite there.")
