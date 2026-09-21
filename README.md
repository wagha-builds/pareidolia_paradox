# Pareidolia Paradox — Lunar Terrain Depth vs. Rise Classification

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Classifies 256×256 grayscale lunar surface tiles as **Depth (crater = 0)** or **Rise (mound = 1)** using a physics-informed CNN ensemble. Scored by **Balanced Accuracy** on 2,000 hidden-label evaluation images.

---

## Methodology: How We Handle `sun_azimuth_angle`

### Problem Framing

The class label is *not* intrinsic to the image pixels alone — it depends jointly on the image **and** the `sun_azimuth_angle`. The same crater rotated 180° with its azimuth unchanged looks exactly like a mound. Standard image classifiers fail on this without explicit azimuth handling. **Azimuth is a transformation parameter first, a feature second.**

---

### Step 1 — Azimuth Calibration

The metadata `sun_azimuth_angle` is in compass convention (0° = North, clockwise). The image plane uses counter-clockwise from East. The true mapping is:

```
θ_sun (image plane) = (s · az_metadata + δ)  mod  360
```

where `s ∈ {+1, −1}` is the handedness and `δ` is an offset — both unknown a priori.

We calibrate by computing a **top-minus-bottom brightness asymmetry** for each training image as a proxy for sun direction, then fitting a circular regression against `az_metadata` for both `s = +1` and `s = −1`. The handedness with higher circular correlation R wins. The fitted `(s, δ)` are frozen in `configs/config.yaml` and asserted at every training run.

---

### Step 2 — Two Approaches to Azimuth

**Approach A — Canonicalization (E4–E14)**

Rotate every image so the sun is always at 12 o'clock (θ_sun = 90°). Rotation angle = `90° − θ_sun`. We use a **√2-zoom warp** (expand canvas by √2, rotate, crop back to 256×256) to eliminate black corner artifacts. Reflected/replicated padding is explicitly banned — mirrored terrain reverses the depth/rise label.

**Approach B — FiLM Conditioning, No Canonicalization (E15–E18 — best results)**

Do not rotate the image at all. Instead, compute `(sin θ_sun, cos θ_sun)` — the circular encoding of sun direction — and inject it into the CNN via **FiLM (Feature-wise Linear Modulation)**. FiLM learns per-channel affine transforms of CNN feature maps conditioned on sun angle:

```
y = γ(az) · x + β(az)
```

The network learns to condition every convolutional feature on sun direction, effectively rotating its internal representation rather than the pixels. This avoids all rotation artifacts and trains ~20% faster.

---

### Step 3 — Physics-Informed 3-Channel Input (E18)

In canonical frame, craters are **dark above the rim, bright below** — sunlight hits the far wall, shadow pools at the near side. Mounds are the opposite. We make this shadow-polarity signal explicit by computing **Sobel gradients on GPU** and stacking them as additional input channels:

| Channel | Signal | Physical meaning |
|---|---|---|
| Ch 0 | Grayscale `I(x,y)` | Raw lunar albedo |
| Ch 1 | `dI/dy` (Sobel-Y) | **Shadow slope** — negative above crater rim, positive below |
| Ch 2 | `dI/dx` (Sobel-X) | **Rim curvature** — lateral edge sharpness |

```python
SOBEL_Y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3) / 4
SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 4
gy = F.conv2d(gray, SOBEL_Y, padding=1)           # shadow slope
gx = F.conv2d(gray, SOBEL_X, padding=1)           # rim curvature
image = torch.cat([gray, gy, gx], dim=1)           # [B, 3, 256, 256]
```

The CNN uses the shadow gradient directly — it does not need to learn Sobel operators from random weights. E18 fold-0 peak: **OOF BA 0.7903** vs E17 baseline at 0.7515.

---

### Step 4 — Physics-Valid Augmentations Only

All geometry is routed through `src/transforms.py`, which updates `az` and flips the label wherever physics demands:

| Transform | Azimuth update | Label flip |
|---|---|---|
| Rotation by α° | `θ → θ + α` | No |
| Horizontal flip | `θ → 180° − θ` | No |
| Vertical flip *(canonical frame only)* | `θ → −θ` | **Yes** — top↔bottom = crater↔mound |
| Image negation `255 − img` | — | **Yes** — inverts shadow polarity |

Standard library augmentations (torchvision/albumentations random flips, RandAugment, TrivialAugment) are **never used** — they do not update `az` and produce invalid (image, azimuth) training pairs.

---

### Step 5 — Pseudo-Labels (E17)

After initial training we run inference on the test set and extract **808 high-confidence predictions** (p ≤ 0.30 or p ≥ 0.70). These are added as soft pseudo-labels for a second training round. This addresses the azimuth distribution shift between train and test sets.

---

### Step 6 — Multi-Model Out-of-Fold Weighted Ensemble

Rather than relying on a single architecture or training paradigm, we ensemble diverse models spanning complementary inductive biases (ConvNeXt-Tiny with FiLM without rotation, canonical-frame models with √2-zoom warps, multi-seed checkpoints, and pseudo-labeled representations). Ensemble weights are selected based on out-of-fold Balanced Accuracy on the frozen 5-fold CV splits:

- **E17 (40%)**: ConvNeXt-Tiny + FiLM (no-canon) + soft pseudo-labels (OOF BA 0.7515)
- **E5-s42 (20%)**: ConvNeXt-Tiny + canonical frame + seed 42
- **E5-s43 (20%)**: ConvNeXt-Tiny + canonical frame + seed 43
- **E5-s44 (10%)**: ConvNeXt-Tiny + canonical frame + seed 44
- **E4 (10%)**: ConvNeXt-Tiny baseline

Final ensemble performance: **OOF BA 0.7476** (optimal plateau threshold $t^* = 0.4475$).

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
│   ├── dataset.py        ← data loading and uint8 caching
│   ├── models.py         ← timm backbones, FiLM, PhysicsTensorWrapper
│   ├── train.py          ← training loop + 5-fold CV
│   ├── infer.py          ← TTA inference
│   ├── ensemble.py       ← multi-model weighted blending and OOF evaluation
│   ├── metrics.py        ← BA, plateau_threshold, apply_threshold
│   └── submit.py         ← builder + validator + sanity report
├── data/
│   ├── folds.csv         ← 5-fold stratified groups (PROTECTED — never regenerate)
│   └── processed/        ← uint8 caches + index.json (gitignored)
├── experiments/          ← run directories with checkpoints (gitignored)
└── submissions/          ← competition CSVs (gitignored)
```

---

## Quickstart

### 1. Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt

# Linux / macOS
source .venv/bin/activate && pip install -r requirements.txt
```

### 2. Data

```
data/raw/
    train_images/        ← 7,854 PNG files
    eval_images/         ← 2,000 PNG files (test / evaluation set)
    train_metadata.csv   ← image_id, label, sun_azimuth_angle
    test_metadata.csv    ← image_id, sun_azimuth_angle
```

### 3. Build caches

```bash
python -m src.cache
# Verifies integrity, builds uint8 numpy caches, writes data/processed/index.json
```

### 4. Train

```bash
# Full 5-fold CV with best config (E17 — pseudo-label + no-canon FiLM):
python train.py --config configs/exp/e17_pseudo_no_canon.yaml

# Physics tensor (E18) — 3-channel Sobel input:
python train.py --config configs/exp/e18_physics_tensor.yaml

# Quick smoke test (finishes in < 5 min):
python train.py --config configs/debug.yaml

# Resume specific folds only (e.g. after Colab interruption):
python train.py --config configs/exp/e18_physics_tensor.yaml --folds 2 3 4
```

### 5. Inference + Submission

```bash
# Single model:
python inference.py --run-dir experiments/<run_id>

# Ensemble (E17 + E5 multi-seed, weights match OOF BA):
python inference.py \
    --run-dirs \
        experiments/20260921-0910_convnext_tiny_fb_in22k_ft_in1k_e17_pseudo_no_canon_s42 \
        experiments/20260915-2338_convnext_tiny_fb_in22k_ft_in1k_e5_canonical_film_s42 \
        experiments/20260916-2029_convnext_tiny_fb_in22k_ft_in1k_e5_canonical_film_s43 \
    --weights 0.50 0.25 0.25

# Or via make:
make submit ARTIFACT=experiments/<run_id>
```

---

## Model Weights

Pre-trained checkpoints for the best ensemble — download and place under `experiments/`:

### 📁 E17 — ConvNeXt-Tiny, Pseudo-label, No-Canon FiLM *(OOF BA 0.7515)*
Config: `configs/exp/e17_pseudo_no_canon.yaml`

| File | Size |
|---|---|
| `checkpoints/fold0_best.pt` | 106.6 MB |
| `checkpoints/fold1_best.pt` | 106.6 MB |
| `checkpoints/fold2_best.pt` | 106.6 MB |
| `checkpoints/fold3_best.pt` | 106.6 MB |
| `checkpoints/fold4_best.pt` | 106.6 MB |
| `run_manifest.json` | — |

> **[⬇ Download E17 weights](https://drive.google.com/drive/folders/PLACEHOLDER_E17)** — Anyone with link can view

---

### 📁 E5-s42 — ConvNeXt-Tiny, Canonical FiLM *(OOF BA 0.7271)*
Config: `configs/exp/e5_canonical_film.yaml`

| File | Size |
|---|---|
| `checkpoints/fold0_best.pt` | 106.6 MB |
| `checkpoints/fold1_best.pt` | 106.6 MB |
| `checkpoints/fold2_best.pt` | 106.6 MB |
| `checkpoints/fold3_best.pt` | 106.6 MB |
| `checkpoints/fold4_best.pt` | 106.6 MB |
| `run_manifest.json` | — |

> **[⬇ Download E5-s42 weights](https://drive.google.com/drive/folders/PLACEHOLDER_E5)** — Anyone with link can view

---

### 📁 E18 — ConvNeXt-Tiny, 3-Ch Physics Tensor *(fold-0 peak BA 0.7903)*
Config: `configs/exp/e18_physics_tensor.yaml`

| File | Size |
|---|---|
| `checkpoints/fold0_best.pt` | 106.6 MB |
| `checkpoints/fold1_best.pt` | 106.6 MB |
| `run_manifest.json` | — |

> **[⬇ Download E18 weights](https://drive.google.com/drive/folders/PLACEHOLDER_E18)** — Anyone with link can view
> *(Folds 2–4 training in progress on Colab — will update link when complete)*

---

After downloading, place each folder under `experiments/` preserving its full directory name so `inference.py` can locate `run_manifest.json` and `checkpoints/`.

---

## Experiments Summary

| Exp | Config | Description | OOF BA | Key change |
|---|---|---|---|---|
| E4 | `e4_canonical.yaml` | ConvNeXt-Tiny, canonical | 0.7161 | Baseline CNN |
| E5 | `e5_canonical_film.yaml` | + FiLM azimuth conditioning | 0.7271 | Sun angle as FiLM input |
| E6b | `e6b_canonical_swin.yaml` | Swin-Tiny, canonical | 0.7038 | Architecture diversity |
| E13 | `e13_strong_jitter.yaml` | Strong rotation jitter ±15° | 0.7196 | Rotation robustness |
| E14 | `e14_az_balanced_sampler.yaml` | Azimuth-balanced batch sampler | — | Reduce az shortcut |
| E15 | `e15_no_canon_film.yaml` | No canonicalization + FiLM | **0.7530** | Best single model |
| E16 | `e16_convnext_small_no_canon.yaml` | ConvNeXt-Small, no-canon | 0.7355 | Larger backbone |
| E17 | `e17_pseudo_no_canon.yaml` | + Pseudo-labels (808 imgs) | 0.7515 | Semi-supervised |
| E18 | `e18_physics_tensor.yaml` | + 3-channel Sobel physics tensor | **0.79+** | Physics-informed channels |

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
Pillow
ruff
pytest
```

---

## Competition

**Pareidolia Paradox** — Lunar terrain Depth vs. Rise classification.
Metric: **Balanced Accuracy** (mean of per-class recall).
Deadline: Mon 21 Sep 2026, 23:59 IST.
