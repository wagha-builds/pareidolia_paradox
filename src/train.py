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
from torch.utils.data import ConcatDataset, DataLoader

from .dataset import PareidoliaDataset
from .losses import build_loss
from .metrics import apply_threshold, plateau_threshold
from .models import build_model, predict_proba
from .transforms import build_augmentation_policy, build_transform
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


def _train_one_fold(
    cfg,
    fold: int,
    train_idx,
    val_idx,
    labels,
    run_dir: Path,
    device,
    fast: bool = False,
):
    cfg_dict = _to_container(cfg)

    # Build canonicalize_cfg from frozen config if canonicalize=true
    # frozen.calibration.{delta,s} as laid out in configs/config.yaml
    canonicalize_cfg = None
    train_canonicalize_cfg = None
    if bool(cfg.data.get("canonicalize", False)):
        frozen = cfg_dict.get("frozen", {})
        cal = frozen.get("calibration", frozen)  # fallback to frozen itself if flat
        canonicalize_cfg = {
            "delta": float(cal["delta"]),
            "s": int(cal["s"]),
            "jitter_deg": 0.0,
        }
        jitter_std = float(
            cfg_dict.get("augmentation", {}).get("canonicalize_jitter_deg", 0.0)
        )
        train_canonicalize_cfg = {
            "delta": float(cal["delta"]),
            "s": int(cal["s"]),
            "jitter_deg": jitter_std,
        }

    train_policy = build_augmentation_policy(cfg_dict, training=True)
    val_policy: list = []  # no augmentation at validation time

    train_ds = PareidoliaDataset(
        "data",
        "train",
        indices=train_idx,
        transform=build_transform(cfg_dict, training=True),
        canonicalize_cfg=train_canonicalize_cfg,
        augmentation_policy=train_policy,
    )
    val_ds = PareidoliaDataset(
        "data",
        "train",
        indices=val_idx,
        transform=build_transform(cfg_dict, training=False),
        canonicalize_cfg=canonicalize_cfg,
        augmentation_policy=val_policy,
    )

    # Optional: append pseudo-labeled test images to training set
    pseudo_csv = str(cfg_dict.get("training", {}).get("pseudo_label_csv", "") or "")
    if pseudo_csv:
        from .pseudo_dataset import PseudoDataset

        pseudo_ds = PseudoDataset(
            "data",
            pseudo_label_csv=pseudo_csv,
            transform=build_transform(cfg_dict, training=True),
            canonicalize_cfg=train_canonicalize_cfg,
            augmentation_policy=train_policy,
        )
        logging.info(
            "fold=%s pseudo-labels: %d images added from %s", fold, len(pseudo_ds), pseudo_csv
        )
        train_ds = ConcatDataset([train_ds, pseudo_ds])
    sampler = None
    if bool(cfg.training.get("azimuth_balanced_sampler", False)):
        # Azimuth-balanced sampler only works on plain PareidoliaDataset (not ConcatDataset)
        if hasattr(train_ds, "metadata"):
            az_angles = train_ds.metadata.iloc[train_idx]["sun_azimuth_angle"].values
            train_labels = labels[train_idx]
            az_bins = np.digitize(az_angles, bins=[0.0, 90.0, 180.0, 270.0, 360.0]) - 1
            az_bins = np.clip(az_bins, 0, 3)
            cell_counts = {}
            for b, y in zip(az_bins, train_labels):
                cell_counts[(b, y)] = cell_counts.get((b, y), 0) + 1
            sample_weights = [
                1.0 / cell_counts[(b, y)] for b, y in zip(az_bins, train_labels)
            ]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
        else:
            logging.warning("azimuth_balanced_sampler disabled when pseudo-labels active (ConcatDataset)")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=int(cfg.training.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.training.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg_dict).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = build_loss(labels[train_idx], float(cfg.training.label_smoothing)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    total_epochs = 15 if fast else int(cfg.training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs, 1)
    )
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1)
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

    amp_enabled = device.type == "cuda" and str(
        cfg.training.mixed_precision
    ).lower() in {"bf16", "fp16"}
    amp_dtype = (
        torch.bfloat16
        if str(cfg.training.mixed_precision).lower() == "bf16"
        else torch.float16
    )

    for epoch in range(1, total_epochs + 1):
        model.train()
        losses = []
        for images, az_sincos, y, _ in train_loader:
            images = images.to(device, non_blocking=True)
            az_sincos = az_sincos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                from .models import AdversarialBackbone, FiLMBackbone

                if isinstance(model, AdversarialBackbone):
                    logits, az_logits = model(images, az_sincos, return_az_logits=True)
                    loss = criterion(logits, y)
                    sin_a, cos_a = az_sincos[:, 0], az_sincos[:, 1]
                    az_deg = (
                        torch.atan2(sin_a, cos_a) * (180.0 / 3.141592653589793)
                    ) % 360.0
                    quadrant = (az_deg // 90.0).long().clamp(0, 3)
                    az_loss = torch.nn.functional.cross_entropy(az_logits, quadrant)
                    adv_weight = float(cfg.training.get("adv_weight", 0.2))
                    loss = loss + adv_weight * az_loss
                elif isinstance(model, FiLMBackbone):
                    logits = model(images, az_sincos)
                    loss = criterion(logits, y)
                else:
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
            for images, az_sincos, y, _ in val_loader:
                images = images.to(device, non_blocking=True)
                az_sincos = az_sincos.to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                probs.append(predict_proba(model, images, az_sincos).cpu().numpy())
                y_true.append(y.numpy())
        probs_np = np.concatenate(probs)
        y_np = np.concatenate(y_true)
        val_ba = balanced_accuracy_score(y_np, apply_threshold(probs_np, 0.5))
        current_lr = (
            scheduler.get_last_lr()[0]
            if hasattr(scheduler, "get_last_lr")
            else float(cfg.training.learning_rate)
        )
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
            logging.info(
                "fold=%s early stopping at epoch %s (patience=%s, warmup=%s)",
                fold,
                epoch,
                patience,
                effective_warmup,
            )
            break

    return best_probs, {
        "fold": fold,
        "best_ba_at_0_5": best_ba,
        "best_epoch": best_epoch,
    }


def train_cv(
    config_path: str,
    fast: bool = False,
    seed_override: int | None = None,
    fold_override: int | None = None,
    folds_to_run: list[int] | None = None,
) -> Path:
    cfg = _load_config(config_path)
    seed = int(seed_override if seed_override is not None else cfg.experiment.seed)
    set_seed(seed)
    setup_logging()

    # canonicalize guard removed — dataset.py handles it correctly when data.canonicalize=true

    folds_path = Path("data/folds.csv")
    if _sha256(folds_path) != str(cfg.frozen.folds_sha256):
        raise RuntimeError(
            "data/folds.csv SHA does not match configs/config.yaml frozen.folds_sha256"
        )
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

    if folds_to_run is not None:
        fold_ids = [int(f) for f in folds_to_run]
    elif fold_override is not None:
        fold_ids = [int(fold_override)]
    elif "fold" in cfg.training:
        fold_ids = [int(cfg.training.fold)]
    else:
        n_folds = 1 if (fast or debug_mode) else int(cfg.training.get("n_folds", 5))
        fold_ids = sorted(merged["fold"].unique().astype(int).tolist())[:n_folds]

    oof = np.full(len(merged), np.nan, dtype=np.float32)
    fold_metrics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("run_id=%s device=%s folds=%s", run_id, device, fold_ids)

    for fold in fold_ids:
        train_idx, val_idx = _indices_for_fold(merged, fold, debug_mode)
        probs, metrics = _train_one_fold(
            cfg, fold, train_idx, val_idx, labels, run_dir, device, fast=fast
        )
        oof[val_idx] = probs
        fold_metrics.append(metrics)

    valid = ~np.isnan(oof)
    threshold, _, _ = plateau_threshold(labels[valid], oof[valid])
    oof_ba = balanced_accuracy_score(
        labels[valid], apply_threshold(oof[valid], threshold)
    )
    auc = (
        roc_auc_score(labels[valid], oof[valid])
        if len(np.unique(labels[valid])) == 2
        else None
    )
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
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logging.info("OOF BA @ plateau t=%.4f: %.4f", threshold, oof_ba)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument(
        "--fold", type=int, default=None, help="Train specific fold only"
    )
    parser.add_argument(
        "--folds", type=str, default="",
        help="Space-separated list of fold IDs to run (e.g. '2 3 4')"
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split() if s.strip()] or [None]
    folds_to_run = [int(f) for f in args.folds.split() if f.strip()] if args.folds else None
    for seed in seeds:
        train_cv(
            args.config,
            fast=args.fast,
            seed_override=seed,
            fold_override=args.fold,
            folds_to_run=folds_to_run,
        )


if __name__ == "__main__":
    main()
