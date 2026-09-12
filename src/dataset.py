import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class PareidoliaDataset(Dataset):
    def __init__(self, data_dir, split="train", canonicalize_cfg=None, 
                 augmentation_policy=None, norm_stats=None, channel_mode="1ch"):
        self.data_dir = data_dir
        self.split = split
        self.canonicalize_cfg = canonicalize_cfg
        self.augmentation_policy = augmentation_policy
        self.norm_stats = norm_stats
        self.channel_mode = channel_mode
        
        # Load from cache if available
        cache_img_path = os.path.join(data_dir, "processed", f"{split}_images.npy")
        cache_idx_path = os.path.join(data_dir, "processed", "index.json")
        
        if os.path.exists(cache_img_path) and os.path.exists(cache_idx_path):
            with open(cache_idx_path, "r") as f:
                idx_data = json.load(f)
            self.metadata = pd.DataFrame(idx_data[split])
            self.images = np.load(cache_img_path, mmap_mode="r")
        else:
            raise FileNotFoundError("Cache not found. Run dataset cache script first.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img = np.array(self.images[idx]) # copy from memmap
        azimuth = float(row["sun_azimuth_angle"])
        label = int(row["label"]) if "label" in row else -1
        
        # We will implement transformations and canonicalization in transforms.py (M3)
        # For now, just return dummy normalized tensors
        img_tensor = torch.from_numpy(img).float() / 255.0
        az_sincos = torch.tensor([np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))], dtype=torch.float32)
        
        return img_tensor.unsqueeze(0), az_sincos, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["cache", "folds"])
    args = parser.parse_args()
    
    print(f"Running mode: {args.mode}. Implementation stubbed.")
