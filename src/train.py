import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import PareidoliaDataset
from .losses import build_loss
from .metrics import apply_threshold, plateau_threshold
from .models import build_model, predict_proba
from .transforms import build_transform
from .utils import set_seed, setup_logging


def _to_container(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def _load_config(path: str):
    base_path = Path("configs/config.yaml")
    if Path(path).resolve() == base_path.resolve():
        return OmegaConf.load(path)
    return OmegaConf.merge(OmegaConf.load(base_path), OmegaConf.load(path))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_id(cfg, seed: int) -> str:
    now = datetime.now().strftime("%Y%m%d-%H%M")
    backbone = str(cfg.model.backbone).replace("/", "_").replace(".", "_")
    name = str(cfg.experiment.name)
    return f"{now}_{backbone}_{name}_s{seed}"


def _indices_for_fold(folds: pd.DataFrame, fold: int, debug_mode: bool):
    train_idx = folds.index[folds["fold"] != fold].to_numpy()
    val_idx = folds.index[folds["fold"] == fold].to_numpy()
    if debug_mode:
        train_idx = train_idx[: min(96, len(train_idx))]
        val_idx = val_idx[: min(64, len(val_idx))]
    return train_idx, val_idx


def _train_one_fold(cfg, fold: int, train_idx, val_idx, labels, run_dir: Path, device):
    cfg_dict = _to_container(cfg)
    train_ds = PareidoliaDataset(
        "data", "train", indices=train_idx, transform=build_transform(cfg_dict, training=True)
    )
    val_ds = PareidoliaDataset(
        "data", "train", indices=val_idx, transform=build_transform(cfg_dict, training=False)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg_dict).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = build_loss(labels[train_idx], float(cfg.training.label_smoothing)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(cfg.training.epochs), 1)
    )
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(cfg.training.epochs) - warmup_epochs, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    best_ba = -1.0
    best_epoch = 0
    best_probs = np.zeros(len(val_idx), dtype=np.float32)
    patience = int(cfg.training.early_stopping_patience)
    effective_warmup = int(cfg.training.get("warmup_epochs", 0))
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = device.type == "cuda" and str(cfg.training.mixed_precision).lower() in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if str(cfg.training.mixed_precision).lower() == "bf16" else torch.float16

    for epoch in range(1, int(cfg.training.epochs) + 1):
        model.train()
        losses = []
        for images, _, y, _ in train_loader:
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        probs = []
        y_true = []
        with torch.no_grad():
            for images, _, y, _ in val_loader:
                images = images.to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                probs.append(predict_proba(model, images).cpu().numpy())
                y_true.append(y.numpy())
        probs_np = np.concatenate(probs)
        y_np = np.concatenate(y_true)
        val_ba = balanced_accuracy_score(y_np, apply_threshold(probs_np, 0.5))
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else float(cfg.training.learning_rate)
        logging.info(
            "fold=%s epoch=%s lr=%.2e loss=%.4f val_ba@0.5=%.4f",
            fold,
            epoch,
            current_lr,
            float(np.mean(losses)),
            val_ba,
        )

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_ba": val_ba},
            ckpt_dir / f"fold{fold}_last.pt",
        )
        if val_ba > best_ba:
            best_ba = val_ba
            best_epoch = epoch
            best_probs = probs_np.astype(np.float32)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_ba": val_ba},
                ckpt_dir / f"fold{fold}_best.pt",
            )
        elif epoch > effective_warmup and epoch - best_epoch >= patience:
            logging.info("fold=%s early stopping at epoch %s (patience=%s, warmup=%s)", fold, epoch, patience, effective_warmup)
            break

    return best_probs, {"fold": fold, "best_ba_at_0_5": best_ba, "best_epoch": best_epoch}


def train_cv(config_path: str, fast: bool = False, seed_override: int | None = None) -> Path:
    cfg = _load_config(config_path)
    seed = int(seed_override if seed_override is not None else cfg.experiment.seed)
    set_seed(seed)
    setup_logging()

    if bool(cfg.data.get("canonicalize", False)):
        raise RuntimeError("M2 raw baseline requires data.canonicalize=false")

    folds_path = Path("data/folds.csv")
    if _sha256(folds_path) != str(cfg.frozen.folds_sha256):
        raise RuntimeError("data/folds.csv SHA does not match configs/config.yaml frozen.folds_sha256")
    folds = pd.read_csv(folds_path, dtype={"image_id": str})
    train_meta = pd.read_csv("data/raw/train_metadata.csv", dtype={"image_id": str})
    merged = train_meta.merge(folds, on="image_id", how="left", validate="one_to_one")
    if merged["fold"].isna().any():
        raise RuntimeError("folds.csv is missing training rows")
    labels = merged["label"].to_numpy(dtype=np.int64)
    debug_mode = bool(cfg.training.get("debug_mode", False))

    run_id = _run_id(cfg, seed)
    run_dir = Path("experiments") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    n_folds = 1 if (fast or debug_mode) else int(cfg.training.get("n_folds", 5))
    fold_ids = sorted(merged["fold"].unique().astype(int).tolist())[:n_folds]
    oof = np.full(len(merged), np.nan, dtype=np.float32)
    fold_metrics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("run_id=%s device=%s folds=%s", run_id, device, fold_ids)

    for fold in fold_ids:
        train_idx, val_idx = _indices_for_fold(merged, fold, debug_mode)
        probs, metrics = _train_one_fold(cfg, fold, train_idx, val_idx, labels, run_dir, device)
        oof[val_idx] = probs
        fold_metrics.append(metrics)

    valid = ~np.isnan(oof)
    threshold, _, _ = plateau_threshold(labels[valid], oof[valid])
    oof_ba = balanced_accuracy_score(labels[valid], apply_threshold(oof[valid], threshold))
    auc = roc_auc_score(labels[valid], oof[valid]) if len(np.unique(labels[valid])) == 2 else None
    np.save(run_dir / "oof.npy", oof)

    manifest = {
        "run_id": run_id,
        "config": config_path,
        "seed": seed,
        "folds_sha256": str(cfg.frozen.folds_sha256),
        "folds_have_group_id": "group_id" in folds.columns,
        "metrics": {
            "oof_ba": float(oof_ba),
            "oof_auc": None if auc is None else float(auc),
            "threshold": float(threshold),
        },
        "fold_metrics": fold_metrics,
        "oof_path": str(run_dir / "oof.npy"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("OOF BA @ plateau t=%.4f: %.4f", threshold, oof_ba)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split() if s.strip()] or [None]
    for seed in seeds:
        train_cv(args.config, fast=args.fast, seed_override=seed)


if __name__ == "__main__":
    main()
