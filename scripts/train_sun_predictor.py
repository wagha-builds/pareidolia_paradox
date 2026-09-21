"""
scripts/train_sun_predictor.py — Self-supervised training of SunDirectionModel.

Usage:
    python scripts/train_sun_predictor.py --epochs 5 --batch-size 32
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.sun_predictor import (
    FastLunarDataset,
    SunDirectionModel,
    compute_equivariance_loss,
    gpu_rotate_and_zoom,
)
from src.utils import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sun Direction Model via Rotation Equivariance.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--backbone", type=str, default="convnext_tiny.fb_in22k_ft_in1k", help="Backbone.")
    parser.add_argument("--lambda-anchor", type=float, default=0.2, help="Hemisphere anchor weight.")
    parser.add_argument("--save-dir", type=str, default="experiments/sun_predictor", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_logging()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training sun predictor on device={device}, backbone={args.backbone}, epochs={args.epochs}, bs={args.batch_size}", flush=True)

    # 1. Dataset & DataLoader (all 9,854 images)
    dataset = FastLunarDataset(include_test=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Loaded {len(dataset)} images for self-supervised training ({len(loader)} batches/epoch).", flush=True)

    # 2. Model & Optimizer
    model = SunDirectionModel(backbone_name=args.backbone, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16

    best_loss = float("inf")
    history = []

    # Zero angle reference for zoom parity
    zeros = torch.zeros(args.batch_size, device=device)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_losses = []
        epoch_equiv = []
        epoch_anchor = []

        for b_idx, (x_raw, u_meta) in enumerate(loader):
            x_raw = x_raw.to(device, non_blocking=True)
            u_meta = u_meta.to(device, non_blocking=True)

            # Sample random rotation angle on GPU
            alpha_rad = torch.rand(args.batch_size, device=device) * (2.0 * np.pi)

            # Rotate on GPU with sqrt(2) zoom
            x_rot = gpu_rotate_and_zoom(x_raw, alpha_rad)
            x_orig = gpu_rotate_and_zoom(x_raw, zeros)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                v_orig = model(x_orig)
                v_rot = model(x_rot)
                loss, equiv_loss, anchor_loss = compute_equivariance_loss(
                    v_orig, v_rot, alpha_rad, u_meta=u_meta, lambda_anchor=args.lambda_anchor
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            epoch_equiv.append(float(equiv_loss.item()))
            epoch_anchor.append(float(anchor_loss.item()))

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(loader):
                print(
                    f"  [Epoch {epoch:02d}/{args.epochs:02d} | Batch {b_idx+1:03d}/{len(loader)}] "
                    f"loss={loss.item():.4f} (equiv={equiv_loss.item():.4f}, anchor={anchor_loss.item():.4f})",
                    flush=True,
                )

        scheduler.step()
        dt = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))
        mean_equiv = float(np.mean(epoch_equiv))
        mean_anchor = float(np.mean(epoch_anchor))

        msg = (
            f">> EPOCH {epoch:02d}/{args.epochs:02d} COMPLETE in {dt:.1f}s | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"total_loss={mean_loss:.4f} | equiv_loss={mean_equiv:.4f} | anchor_loss={mean_anchor:.4f}"
        )
        print(msg + "\n", flush=True)

        record = {
            "epoch": epoch,
            "time_sec": dt,
            "total_loss": mean_loss,
            "equiv_loss": mean_equiv,
            "anchor_loss": mean_anchor,
        }
        history.append(record)

        # Save latest checkpoint
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "loss": mean_loss},
            save_dir / "sun_predictor_last.pt",
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "loss": mean_loss},
                save_dir / "sun_predictor_best.pt",
            )

    (save_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best loss: {best_loss:.4f}. Saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
