"""
Inference and Test-Time Augmentation (TTA) engine for Pareidolia.
Imports model and preprocessing from src/ without reimplementing it (per AGENTS.md §5).
"""

import os
import glob
import json
import argparse
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from .models import build_model
from .dataset import PareidoliaDataset
from .submit import build_submission


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
) -> np.ndarray:
    """Computes test predictions with optional horizontal flip TTA."""
    model.eval()
    all_probs = []

    for imgs, az, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        az = az.to(device, non_blocking=True)

        logits = model(imgs, az)
        probs = torch.sigmoid(logits)

        if use_tta:
            # Horizontal flip TTA
            imgs_flipped = torch.flip(imgs, dims=[-1])
            logits_flipped = model(imgs_flipped, az)
            probs_flipped = torch.sigmoid(logits_flipped)
            probs = (probs + probs_flipped) / 2.0

        all_probs.append(probs.float().cpu().numpy())

    return np.concatenate(all_probs)


def run_inference(
    run_dir: str,
    output_csv: Optional[str] = None,
    use_tta: bool = False,
    data_dir: str = "data",
) -> str:
    """Runs inference on test images using all fold checkpoints in run_dir."""
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found in {run_dir}.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    cfg = OmegaConf.create(manifest["config"])
    threshold = float(manifest.get("optimal_threshold", 0.5))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device} using checkpoints from {run_dir}")

    # Load test dataset
    canonicalize = bool(cfg.data.get("canonicalize", False))
    if hasattr(cfg, "frozen") and hasattr(cfg.frozen, "calibration"):
        calib_cfg = dict(cfg.frozen.calibration)
    elif os.path.exists("configs/config.yaml"):
        base_cfg = OmegaConf.load("configs/config.yaml")
        calib_cfg = dict(base_cfg.frozen.calibration) if hasattr(base_cfg, "frozen") and hasattr(base_cfg.frozen, "calibration") else {"delta": 46.7, "s": -1}
    else:
        calib_cfg = {"delta": 46.7, "s": -1}

    test_ds = PareidoliaDataset(
        data_dir=data_dir,
        split="test",
        canonicalize_cfg=calib_cfg if canonicalize else None,
        augmentation_policy=None,
        channel_mode="1ch",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.training.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    ckpt_paths = glob.glob(os.path.join(run_dir, "best_model_fold_*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {run_dir}.")

    print(f"Ensembling {len(ckpt_paths)} checkpoint(s): {[os.path.basename(p) for p in ckpt_paths]}")
    fold_probs = []

    for ckpt in ckpt_paths:
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        probs = predict_loader(model, test_loader, device, use_tta=use_tta)
        fold_probs.append(probs)

    avg_probs = np.mean(fold_probs, axis=0)

    # Save raw test probabilities numpy array
    probs_save_path = os.path.join(run_dir, "test_probs.npy")
    np.save(probs_save_path, avg_probs)
    print(f"Saved test probabilities: {probs_save_path} (shape={avg_probs.shape})")

    # Build and validate submission CSV
    if output_csv is None:
        os.makedirs("submissions", exist_ok=True)
        run_name = os.path.basename(run_dir.rstrip("/\\"))
        output_csv = os.path.join("submissions", f"sub_{run_name}.csv")

    test_meta_path = os.path.join(data_dir, "raw", "test_metadata.csv")
    test_ids = [str(x) for x in test_ds.metadata["image_id"]]

    final_csv = build_submission(
        image_ids=test_ids,
        probs=avg_probs,
        threshold=threshold,
        output_path=output_csv,
        test_metadata_path=test_meta_path,
    )

    print(f"[SUCCESS] Submission ready: {final_csv}")
    return final_csv


def main():
    parser = argparse.ArgumentParser(description="Pareidolia Inference CLI")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to experiment run directory")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path")
    parser.add_argument("--tta", action="store_true", help="Enable horizontal flip TTA")
    args = parser.parse_args()

    run_inference(run_dir=args.run_dir, output_csv=args.output, use_tta=args.tta)


if __name__ == "__main__":
    main()
