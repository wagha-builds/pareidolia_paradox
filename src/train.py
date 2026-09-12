"""
5-Fold GPU cross-validation training loop and experiment runner.
Adheres strictly to AGENTS.md §5 (seed, checkpointing, OOF float32 array, run_manifest.json).
"""

import os
import json
import time
import argparse
import subprocess
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf, DictConfig

from .utils import set_seed, setup_logging
from .models import build_model
from .dataset import PareidoliaDataset
from .transforms import get_canonical_train_transforms, get_raw_train_transforms
from .metrics import plateau_threshold


def get_git_sha() -> str:
    """Returns the current git commit SHA."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def get_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hex digest of a file."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype,
) -> float:
    """Runs a single training epoch with automatic mixed precision."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for imgs, az, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        az = az.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
            logits = model(imgs, az)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs evaluation and returns predicted probabilities and ground-truth labels."""
    model.eval()
    all_probs = []
    all_labels = []

    for imgs, az, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        az = az.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(imgs, az)
            probs = torch.sigmoid(logits)

        all_probs.append(probs.float().cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def train_cv(config: DictConfig, fast: bool = False, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Executes the complete cross-validation training pipeline."""
    seed = int(config.experiment.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Experiment run ID and directory
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    backbone_name = config.model.get("backbone", "convnext_tiny")
    exp_name = config.experiment.get("name", "baseline")
    run_id = run_id or f"{now_str}_{backbone_name}_{exp_name}_s{seed}"
    run_dir = os.path.join("experiments", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Verify folds hash if frozen block exists
    folds_path = "data/folds.csv"
    actual_folds_hash = get_file_sha256(folds_path)
    if hasattr(config, "frozen") and hasattr(config.frozen, "folds_sha256"):
        expected_hash = config.frozen.folds_sha256
        if expected_hash and actual_folds_hash != expected_hash:
            print(f"[WARNING] folds.csv hash mismatch: {actual_folds_hash} vs config {expected_hash}")

    # Mixed precision setup
    mp_mode = config.training.get("mixed_precision", "fp16")
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (mp_mode == "bf16" and use_cuda) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # Hyperparameters
    epochs = int(config.training.get("epochs", 30))
    batch_size = int(config.training.get("batch_size", 32))
    lr = float(config.training.get("learning_rate", 1e-3))
    weight_decay = float(config.training.get("weight_decay", 0.05))
    patience = int(config.training.get("early_stopping_patience", 5))
    debug_mode = bool(config.training.get("debug_mode", False))

    # Caching / Canonicalization parameters
    data_cfg = config.data
    canonicalize = bool(data_cfg.get("canonicalize", False))
    if hasattr(config, "frozen") and hasattr(config.frozen, "calibration"):
        calib_cfg = dict(config.frozen.calibration)
    elif os.path.exists("configs/config.yaml"):
        base_cfg = OmegaConf.load("configs/config.yaml")
        calib_cfg = dict(base_cfg.frozen.calibration) if hasattr(base_cfg, "frozen") and hasattr(base_cfg.frozen, "calibration") else {"delta": 46.7, "s": -1}
    else:
        calib_cfg = {"delta": 46.7, "s": -1}

    # Determine folds to train
    n_folds = 1 if (fast or debug_mode) else 5
    folds_to_train = [0] if (fast or debug_mode) else list(range(5))
    print(f"Training {n_folds} fold(s): {folds_to_train}")

    total_train_samples = 7854
    oof_probs = np.full(total_train_samples, np.nan, dtype=np.float32)
    fold_scores = {}

    for fold in folds_to_train:
        print(f"\n{'='*25} FOLD {fold} / 5 {'='*25}")

        # Transforms policy
        if canonicalize:
            p_vflip = float(config.training.get("p_vflip", 0.20))
            p_neg = float(config.training.get("p_neg", 0.15))
            train_transform = get_canonical_train_transforms(p_vflip=p_vflip, p_neg=p_neg)
        else:
            delta = float(calib_cfg.get("delta", 46.7)) if calib_cfg else 0.0
            s = int(calib_cfg.get("s", -1)) if calib_cfg else 1
            p_rot180_flip = float(config.training.get("p_rot180_flip", 0.0))
            p_neg = float(config.training.get("p_neg", 0.15))
            train_transform = get_raw_train_transforms(
                delta=delta, s=s, p_neg=p_neg, p_rot180_flip=p_rot180_flip
            )

        # Datasets
        train_ds = PareidoliaDataset(
            data_dir="data",
            split="train",
            fold=fold,
            canonicalize_cfg=calib_cfg if canonicalize else None,
            augmentation_policy=train_transform,
            channel_mode="1ch",
        )
        val_ds = PareidoliaDataset(
            data_dir="data",
            split="val",
            fold=fold,
            canonicalize_cfg=calib_cfg if canonicalize else None,
            augmentation_policy=None,
            channel_mode="1ch",
        )

        if debug_mode:
            train_ds = Subset(train_ds, range(min(128, len(train_ds))))
            val_ds = Subset(val_ds, range(min(64, len(val_ds))))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )

        # Model & Optimization
        model = build_model(config).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_ba = -1.0
        best_t = 0.5
        best_val_probs = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp_dtype=amp_dtype,
            )
            scheduler.step()

            val_probs, val_labels = evaluate(model, val_loader, device, amp_dtype)
            t_opt, _, ba_grid = plateau_threshold(val_labels, val_probs)
            val_ba = float(np.max(ba_grid))
            elapsed = time.time() - t0

            is_best = val_ba > best_val_ba
            marker = "*" if is_best else ""
            print(
                f"Epoch {epoch:2d}/{epochs} [{elapsed:4.1f}s] - "
                f"Loss: {train_loss:.4f} - val_BA: {val_ba:.4f} (t*={t_opt:.3f}) {marker}",
                flush=True,
            )

            if is_best:
                best_val_ba = val_ba
                best_t = t_opt
                best_val_probs = val_probs
                patience_counter = 0
                ckpt_path = os.path.join(run_dir, f"best_model_fold_{fold}.pt")
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        fold_scores[f"fold_{fold}"] = {"best_val_ba": best_val_ba, "threshold": best_t}
        print(f"Fold {fold} finished. Best val_BA: {best_val_ba:.4f} @ t*={best_t:.3f}")

        # Store OOF predictions
        if not debug_mode:
            target_indices = val_ds.indices
            oof_probs[target_indices] = best_val_probs
        else:
            oof_probs[:len(best_val_probs)] = best_val_probs

    # Save OOF probabilities array
    oof_path = os.path.join(run_dir, "oof.npy")
    np.save(oof_path, oof_probs)
    print(f"\nSaved OOF predictions: {oof_path} (shape={oof_probs.shape})")

    # Overall metrics computation
    valid_oof_mask = ~np.isnan(oof_probs)
    if valid_oof_mask.sum() > 0:
        index_path = "data/processed/index.json"
        with open(index_path, "r") as f:
            idx_data = json.load(f)
        all_train_labels = np.array([row["label"] for row in idx_data["train"]], dtype=int)
        
        eval_y = all_train_labels[valid_oof_mask]
        eval_p = oof_probs[valid_oof_mask]
        overall_t, _, overall_ba_grid = plateau_threshold(eval_y, eval_p)
        overall_ba = float(np.max(overall_ba_grid))
    else:
        overall_t, overall_ba = 0.5, 0.5

    print(f"Overall OOF Balanced Accuracy: {overall_ba:.4f} @ threshold {overall_t:.3f}")

    # Generate run_manifest.json per AGENTS.md §5
    manifest = {
        "run_id": run_id,
        "timestamp": now_str,
        "git_sha": get_git_sha(),
        "folds_sha256": actual_folds_hash,
        "fast_mode": fast,
        "debug_mode": debug_mode,
        "device": str(device),
        "overall_oof_ba": overall_ba,
        "optimal_threshold": overall_t,
        "fold_scores": fold_scores,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Pareidolia Training CLI")
    parser.add_argument("--config", type=str, default="configs/debug.yaml", help="Path to YAML config")
    parser.add_argument("--fast", action="store_true", help="Run 1 fold only")
    parser.add_argument("--seeds", type=str, default="", help="Optional space-separated seed list")
    args = parser.parse_args()

    setup_logging()
    cfg = OmegaConf.load(args.config)

    seeds = [int(s) for s in args.seeds.split()] if args.seeds.strip() else [int(cfg.experiment.get("seed", 42))]
    for s in seeds:
        cfg.experiment.seed = s
        train_cv(cfg, fast=args.fast)


if __name__ == "__main__":
    main()
