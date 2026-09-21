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
                if tta:
                    # Legal azimuth-consistent jitter TTA:
                    # Rotate image by ±delta degrees AND update az_sincos by the same
                    # delta so FiLM conditioning stays physically consistent.
                    # Uses affine_grid (no library augmentation policy).
                    import torch.nn.functional as _F
                    _INV_SQRT2 = 1.0 / (2.0 ** 0.5)
                    jitter_deg = [-20.0, -10.0, 0.0, 10.0, 20.0]
                    all_p = []
                    for d_deg in jitter_deg:
                        d_rad = torch.tensor(d_deg * 3.141592653589793 / 180.0,
                                             dtype=x.dtype, device=x.device)
                        cos_d = torch.cos(d_rad)
                        sin_d = torch.sin(d_rad)
                        B = x.shape[0]
                        # Build affine matrix: rotate + sqrt(2) zoom (no border artifacts)
                        theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
                        theta[:, 0, 0] = _INV_SQRT2 * cos_d
                        theta[:, 0, 1] = _INV_SQRT2 * sin_d
                        theta[:, 1, 0] = -_INV_SQRT2 * sin_d
                        theta[:, 1, 1] = _INV_SQRT2 * cos_d
                        grid = _F.affine_grid(theta, x.size(), align_corners=False)
                        x_jit = _F.grid_sample(x, grid, mode="bilinear",
                                               padding_mode="reflection",
                                               align_corners=False)
                        # Rotate az_sincos vector by same delta:
                        # (sin_az, cos_az) → (sin_az*cos_d + cos_az*sin_d,
                        #                     cos_az*cos_d - sin_az*sin_d)
                        sin_az = az[:, 0]
                        cos_az = az[:, 1]
                        az_jit = torch.stack([
                            sin_az * cos_d + cos_az * sin_d,
                            cos_az * cos_d - sin_az * sin_d,
                        ], dim=1)
                        all_p.append(predict_proba(model, x_jit, az_jit))
                    p = torch.stack(all_p, dim=0).mean(dim=0)
                else:
                    p = predict_proba(model, x, az)
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
