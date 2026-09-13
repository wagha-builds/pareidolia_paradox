"""
Overfit sanity check + training collapse diagnosis for ConvNeXt.
Trains on a tiny 64-image subset for 60 epochs, no early stopping.
If loss can't drop below 0.4, something fundamental is broken.
Run: python scripts/overfit_test.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import PareidoliaDataset
from src.losses import build_loss
from src.models import build_model, predict_proba
from src.transforms import build_transform
from src.utils import set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load base config merged with exp config
cfg = OmegaConf.merge(
    OmegaConf.load("configs/config.yaml"),
    OmegaConf.load("configs/exp/raw_convnext_baseline.yaml"),
)
cfg_dict = OmegaConf.to_container(cfg, resolve=True)

# Build dataset — take first 32 Depth + 32 Rise for a balanced micro-set
with open("data/processed/index.json") as f:
    idx_data = json.load(f)
meta = pd.DataFrame(idx_data["train"])
c0 = meta[meta["label"] == 0].index[:32].tolist()
c1 = meta[meta["label"] == 1].index[:32].tolist()
micro_idx = c0 + c1
micro_labels = [0] * 32 + [1] * 32

train_ds = PareidoliaDataset(
    "data",
    "train",
    indices=micro_idx,
    transform=build_transform(cfg_dict, training=True),
)
loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = build_model(cfg_dict).to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

criterion = build_loss(micro_labels, label_smoothing=0.0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)

print(
    f"\nOverfit test: {len(micro_idx)} images | 60 epochs | LR=5e-5 | no label smoothing"
)
print(f"Class weights: {criterion.weight.tolist()}")
print("Expected: loss should drop well below 0.4 by epoch 30\n")

for epoch in range(1, 61):
    model.train()
    losses, preds_all, labels_all = [], [], []
    for images, _, y, _ in loader:
        images = images.to(device)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            probs = predict_proba(model, images).cpu().numpy()
        preds_all.extend((probs >= 0.5).astype(int).tolist())
        labels_all.extend(y.cpu().numpy().tolist())

    from sklearn.metrics import balanced_accuracy_score

    ba = balanced_accuracy_score(labels_all, preds_all)
    mean_loss = float(np.mean(losses))

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}: loss={mean_loss:.4f}  BA@0.5={ba:.4f}")
        if mean_loss < 0.3 and ba > 0.75:
            print("  [PASS] Model can overfit. Training pipeline is healthy.")
            break

print("\nFinal diagnosis:")
if mean_loss > 0.55:
    print(
        "  [FAIL] BROKEN: Loss is stuck near log(2)=0.693. Model cannot overfit 64 images."
    )
    print(
        "     Likely causes: normalization wrong, data pipeline bug, or model init issue."
    )
elif mean_loss > 0.35:
    print("  ⚠️  SLUGGISH: Loss improving but slow. LR or weight decay may need tuning.")
else:
    print(
        "  ✅ HEALTHY: Pipeline is fine. Training collapse is a hyperparameter/config issue."
    )
    print(
        "     Fix: add LR warmup, reduce LR for full run, or check early stopping patience."
    )
