import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .dataset import PareidoliaDataset
from .models import build_model, predict_proba
from .transforms import build_transform


def predict_run(
    run_dir: str | Path,
    split: str = "test",
    tta: bool = True,
    indices: list[int] | None = None,
) -> pd.DataFrame:
    """Predict P(Rise) for a run or an ensemble artifact with optional TTA and subsetting."""
    run_dir = Path(run_dir)

    # If run_dir is an ensemble artifact (has members.json), blend member runs
    members_path = run_dir / "members.json"
    if members_path.exists():
        members = json.loads(members_path.read_text(encoding="utf-8"))
        weights = [float(m.get("weight", 1.0)) for m in members]
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        member_probs = []
        ids = None
        for m in members:
            m_run = (
                Path("experiments") / m["run_id"]
                if not Path(m["run_id"]).exists()
                else Path(m["run_id"])
            )
            df_m = predict_run(m_run, split=split, tta=tta, indices=indices)
            member_probs.append(df_m["p_rise"].to_numpy())
            if ids is None:
                ids = df_m["image_id"].tolist()

        p_rise = np.zeros_like(member_probs[0], dtype=np.float64)
        for p, w in zip(member_probs, norm_weights, strict=True):
            p_rise += w * p
        p_rise = p_rise.astype(np.float32)

        out = pd.DataFrame({"image_id": ids, "p_rise": p_rise})
        if indices is None and split == "test":
            tag = "tta" if tta else "raw"
            pred_path = (
                run_dir
                / f"test_probs_{tag}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.npy"
            )
            np.save(pred_path, p_rise)
        return out

    # Standard single training run with checkpoints
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {run_dir}")
    cfg = OmegaConf.load(cfg_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Build canonicalize_cfg if data.canonicalize is True
    canonicalize_cfg = None
    if bool(cfg.data.get("canonicalize", False)):
        frozen = cfg_dict.get("frozen", {})
        cal = frozen.get("calibration", frozen)
        if isinstance(cal, dict) and "delta" in cal and "s" in cal:
            canonicalize_cfg = {
                "delta": float(cal["delta"]),
                "s": int(cal["s"]),
            }
        else:
            base_cfg = OmegaConf.load("configs/config.yaml")
            canonicalize_cfg = {
                "delta": float(base_cfg.frozen.calibration.delta),
                "s": int(base_cfg.frozen.calibration.s),
            }

    ds = PareidoliaDataset(
        "data",
        split=split,
        indices=indices,
        transform=build_transform(cfg_dict, training=False),
        canonicalize_cfg=canonicalize_cfg,
    )
    batch_size = int(cfg.training.get("batch_size", 32))
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted((run_dir / "checkpoints").glob("fold*_best.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No fold*_best.pt checkpoints found in {run_dir / 'checkpoints'}"
        )

    if "model" in cfg_dict:
        cfg_dict["model"]["pretrained"] = False

    all_probs = []
    ids = None
    for ckpt_path in ckpts:
        model = build_model(cfg_dict).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        probs = []
        ckpt_ids = []
        with torch.no_grad():
            for images, az_sincos, _, image_ids in loader:
                x = images.to(device)
                az = az_sincos.to(device)
                p0 = predict_proba(model, x, az)
                if tta:
                    col_rev = torch.arange(x.shape[-1] - 1, -1, -1, device=x.device)
                    xh = x[:, :, :, col_rev]
                    ph = predict_proba(model, xh, az)
                    p = 0.5 * (p0 + ph)
                else:
                    p = p0
                probs.append(p.cpu().numpy())
                if ids is None:
                    ckpt_ids.extend(list(image_ids))
        all_probs.append(np.concatenate(probs))
        if ids is None:
            ids = ckpt_ids

    p_rise = np.mean(np.stack(all_probs, axis=0), axis=0).astype(np.float32)
    out = pd.DataFrame({"image_id": ids, "p_rise": p_rise})
    if indices is None and split == "test":
        tag = "tta" if tta else "raw"
        pred_path = (
            run_dir
            / f"test_probs_{tag}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.npy"
        )
        np.save(pred_path, p_rise)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=str, required=True)
    parser.add_argument(
        "--no-tta", action="store_true", help="Disable test-time augmentation"
    )
    args = parser.parse_args()
    preds = predict_run(args.artifact, tta=not args.no_tta)
    out_path = Path(args.artifact) / "test_predictions.csv"
    preds.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} ({len(preds)} rows, mean P(Rise) = {preds['p_rise'].mean():.4f})"
    )


if __name__ == "__main__":
    main()
