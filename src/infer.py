import argparse
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
    run_dir: str | Path, split: str = "test", tta: bool = True
) -> pd.DataFrame:
    """Predict P(Rise) for one training run by averaging its best fold checkpoints with TTA."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ds = PareidoliaDataset(
        "data", split, transform=build_transform(cfg_dict, training=False)
    )
    loader = DataLoader(
        ds, batch_size=int(cfg.training.batch_size), shuffle=False, num_workers=0
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
    for ckpt_path in ckpts:
        model = build_model(cfg_dict).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        probs = []
        ids = []
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
                ids.extend(list(image_ids))
        all_probs.append(np.concatenate(probs))

    p_rise = np.mean(np.stack(all_probs, axis=0), axis=0).astype(np.float32)
    out = pd.DataFrame({"image_id": ids, "p_rise": p_rise})
    tag = "tta" if tta else "raw"
    pred_path = (
        run_dir / f"test_probs_{tag}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.npy"
    )
    np.save(pred_path, p_rise)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=str, required=True)
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    args = parser.parse_args()
    preds = predict_run(args.artifact, tta=not args.no_tta)
    out_path = Path(args.artifact) / "test_predictions.csv"
    preds.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
