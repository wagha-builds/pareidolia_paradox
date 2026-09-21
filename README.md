# Pareidolia Paradox — Lunar Terrain Depth vs. Rise Classification

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Classifies 256×256 grayscale lunar surface tiles as **Depth (crater = 0)** or **Rise (mound = 1)** using a physics-informed CNN ensemble. Scored by **Balanced Accuracy** on 2,000 hidden-label evaluation images.

---

## Core Idea: Sun Azimuth Canonicalization

The central challenge is that **the same terrain patch looks completely different depending on sun direction**. A crater lit from the left casts shadows to the right; lit from the right it casts them to the left — and can visually resemble a mound.

### How We Handle `sun_azimuth_angle`

Every image comes with a `sun_azimuth_angle` (degrees, compass convention). We use this to **rotate each image into a canonical frame where the sun always illuminates from the top** before any learning or inference.

```
θ_sun = (s · az + δ) mod 360          # calibrated angle in image plane
α     = 90° − θ_sun                   # rotation needed to put sun at top
```

**Calibration** (`make calibrate`) fits the two parameters (scale `s ∈ {+1, −1}` and offset `δ`) from training data by maximising the correlation between sun direction and image brightness asymmetry.

**Rotation** uses a √2-zoom warp (expand, rotate, crop) to avoid black corner artifacts — reflected/replicated padding would produce mirror-inverted terrain which reverses the depth/rise label.

```python
# src/canonical.py  (PROTECTED — do not modify without a PR)
def canonicalize(img: np.ndarray, az: float, cfg: dict) -> np.ndarray:
    theta_sun = (cfg["s"] * az + cfg["delta"]) % 360
    alpha = 90.0 - theta_sun          # degrees CCW to rotate
    zoom = 1.0 / math.cos(math.radians(45))   # ≈ √2, eliminates corners
    M = cv2.getRotationMatrix2D(center, alpha, zoom)
    return cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
```

After canonicalization every training and test image has the sun at 12 o'clock, making craters consistently darker at the top and brighter at the bottom.

### Physics-Informed 3-Channel Input (E18)

Beyond rotation we also compute **Sobel gradients on GPU** and stack them as extra channels:

| Channel | Signal | Physical meaning |
|---|---|---|
| Ch 0 | Grayscale I | Raw luminance |
| Ch 1 | dI/dy (Sobel-Y) | Shadow slope — craters dark above, bright below |
| Ch 2 | dI/dx (Sobel-X) | Rim curvature |

This means the network receives the shadow-polarity signal explicitly rather than learning it from scratch.

---

## Repository Structure

```
pareidolia/
├── train.py              ← entry point: python train.py --config ...
├── inference.py          ← entry point: python inference.py --run-dir ...
├── requirements.txt      ← pinned dependencies
├── Makefile              ← make setup | cache | calibrate | train | submit
├── configs/
│   ├── config.yaml       ← base config (frozen block is immutable)
│   └── exp/              ← one YAML per experiment (E4–E18)
├── src/
│   ├── canonical.py      ← calibration + canonicalize/decanonicalize
│   ├── transforms.py     ← ALL geometry + label-flip augmentations
│   ├── dataset.py        ← data loading and caching
│   ├── models.py         ← timm backbones, FiLM, PhysicsTensorWrapper
│   ├── train.py          ← training loop + CV
│   ├── infer.py          ← TTA inference
│   ├── ensemble.py       ← SLSQP weight optimisation
│   ├── metrics.py        ← BA, plateau_threshold, apply_threshold
│   └── submit.py         ← builder + validator + sanity report
├── data/
│   ├── folds.csv         ← 5-fold stratified groups (PROTECTED)
│   └── processed/        ← uint8 caches, index.json (gitignored)
├── experiments/          ← run directories (gitignored)
└── submissions/          ← competition CSVs (gitignored)
```

---

## Quickstart

### 1. Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\pip install -r requirements.txt
# Linux/Mac:
source .venv/bin/activate && pip install -r requirements.txt
```

### 2. Data

Place competition data files:
```
data/raw/
    train_images/        ← 7,854 PNG files
    eval_images/         ← 2,000 PNG files (test set)
    train_metadata.csv   ← image_id, label, sun_azimuth_angle
    test_metadata.csv    ← image_id, sun_azimuth_angle
```

### 3. Build caches

```bash
python -m src.cache
# Verifies integrity, builds uint8 numpy caches, writes index.json
```

### 4. Train

```bash
# Full 5-fold CV with best config:
python train.py --config configs/exp/e17_pseudo_no_canon.yaml

# Quick smoke test (< 5 min):
python train.py --config configs/debug.yaml
```

### 5. Inference + Submission

```bash
# Single model:
python inference.py --run-dir experiments/<run_id>

# Ensemble (E17 + E5 multi-seed):
python inference.py \
    --run-dirs experiments/run_e17 experiments/run_e5_s42 experiments/run_e5_s43 \
    --weights 0.40 0.30 0.30

# Or use make:
make submit ARTIFACT=experiments/<run_id>
```

---

## Model Weights

Pre-trained checkpoints for the best ensemble:

| Model | Config | OOF BA | Weights |
|---|---|---|---|
| E17 — ConvNeXt-Tiny, pseudo-label, no-canon | `configs/exp/e17_pseudo_no_canon.yaml` | 0.7515 | [Download ↗](https://drive.google.com/drive/folders/PLACEHOLDER_E17) |
| E5 s42 — ConvNeXt-Tiny, FiLM | `configs/exp/e5_canonical_film.yaml` | 0.7271 | [Download ↗](https://drive.google.com/drive/folders/PLACEHOLDER_E5) |
| E18 — ConvNeXt-Tiny, 3-ch Physics Tensor | `configs/exp/e18_physics_tensor.yaml` | 0.79+ | [Download ↗](https://drive.google.com/drive/folders/PLACEHOLDER_E18) |

> **Share setting:** Anyone with the link can view.

After downloading, unzip into `experiments/` so the structure is:
```
experiments/
    20260921-0910_.../
        checkpoints/fold0_best.pt ... fold4_best.pt
        run_manifest.json
```

---

## Experiments Summary

| Exp | Description | OOF BA | Key change |
|---|---|---|---|
| E4 | ConvNeXt-Tiny, canonical | 0.7161 | Baseline CNN |
| E5 | + FiLM azimuth conditioning | 0.7271 | Sun angle as FiLM input |
| E6b | Swin-Tiny, canonical | 0.7038 | Architecture diversity |
| E13 | Strong jitter (±15°) | 0.7196 | Rotation robustness |
| E15 | No canonicalization + FiLM | **0.7530** | Best single model |
| E16 | ConvNeXt-Small, no-canon | 0.7355 | Larger backbone |
| E17 | + Pseudo-labels (808 imgs) | 0.7515 | Semi-supervised |
| E18 | + 3-ch physics tensor | **0.79+** | Physics channels |

---

## Methodology: Sun Azimuth Handling (Full Detail)

### Why azimuth matters

Depth (crater) vs Rise (mound) is determined purely by the **shadow direction**. In a canonical frame (sun at top), craters are dark at the top, bright at the bottom. Without rotation, the same crater lit from 8 different azimuths looks like 8 different scenes and the model must learn all 8 patterns.

### Calibration

The metadata `sun_azimuth_angle` is in compass convention (clockwise from North). The image convention is counter-clockwise from East. The mapping involves an unknown offset δ and handedness s:

```
θ_sun_image = (s · az_metadata + δ) mod 360
```

We fit (s, δ) by:
1. Computing the per-image **top-minus-bottom brightness ratio** as a proxy for sun direction
2. Fitting a circular regression against `az_metadata` for both s = +1 and s = −1
3. Selecting the handedness that gives higher circular correlation R

The fitted values are frozen in `configs/config.yaml` and checked at every training run.

### No-Canonicalization Models (E15, E17, E18)

In our best models we **do not rotate the image**. Instead, `(sin(θ_sun), cos(θ_sun))` is fed into a **FiLM (Feature-wise Linear Modulation)** layer that modulates CNN feature maps per-channel. The model learns to condition on sun direction directly rather than relying on correct rotation.

This approach avoids corner artifacts and trains faster, at the cost of requiring the azimuth at inference time (always available from metadata).

### Augmentations (Physics-Valid Only)

Per `AGENTS.md §5`, all geometry goes through `src/transforms.py` which updates azimuth alongside the image:

| Transform | Azimuth update | Label flip? |
|---|---|---|
| Canonical rotation by α | az → az − s·α | No |
| Horizontal flip | θ → 180° − θ | No |
| Vertical flip (canonical only) | θ → −θ | **Yes** (top↔bottom = depth↔rise) |
| Image negation | — | **Yes** (255−img reverses shadow polarity) |

Standard library flips (torchvision, albumentations) are **never used** as they don't update azimuth.

---

## Requirements

See [`requirements.txt`](requirements.txt) for pinned versions. Key dependencies:

```
torch >= 2.3
timm == 1.0.9
omegaconf == 2.3.0
opencv-python-headless
scipy
scikit-learn
numpy
pandas
```

---

## Citation / Competition

**Pareidolia Paradox** — Lunar terrain classification challenge.  
Metric: **Balanced Accuracy** (average of per-class recall).  
Deadline: Mon 21 Sep 2026, 23:59 IST.
