# Pareidolia Paradox — Lunar Terrain Depth vs. Rise Classification

[![Tests](https://img.shields.io/badge/tests-40%2F40%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![Ruff](https://img.shields.io/badge/code%20style-ruff-black)]()
[![Current Best OOF BA](https://img.shields.io/badge/OOF%20Balanced%20Accuracy-0.7161-success)]()

Classifying $256 \times 256$ grayscale lunar terrain patches into **Depth (0)** (craters, pits) or **Rise (1)** (mounds, domes) for the **Pareidolia Paradox** competition, scored by **Balanced Accuracy** across 2,000 hidden-label test images.

---

## 🌓 The Core Physical Principle (Why Standard Classifiers Fail)

In lunar satellite imagery under oblique solar illumination, relief perception (craters vs mounds) is governed by **shape-from-shading**:
- A crater illuminated with the sun from the top has an illuminated interior on the top rim and a dark shadow on the bottom floor.
- If an image is rotated $180^\circ$ without modifying the azimuth parameter, the shading pattern flips: the crater now visually mimics an elevated mound (shadow on top, bright on bottom).
- **The label depends on the image pixels and `sun_azimuth_angle` together.**

### The Azimuth Shortcut
Standard CNNs trained directly on raw images quickly discover that `sun_azimuth_angle` alone correlates with label distribution in the dataset, leading to an **azimuth shortcut**. When evaluated on held-out regional terrain groups, naive raw-frame models collapse to **`0.5000` Balanced Accuracy**.

---

## 🔭 The Canonical Pipeline Solution

We eliminate the azimuth confound through rigorous coordinate transformation and physics-preserving augmentation:

1. **Coordinate Calibration:**
   The image-plane solar direction is modeled as:
   $$\theta_{\text{sun}} = (s \cdot \text{azimuth} + \delta) \pmod{360^\circ}$$
   Forensics on crater shading confirmed **$s = -1, \delta = 46.702^\circ$** with strong class separation (top-bottom mean asymmetry of $-18.30$ for Depth vs $+0.26$ for Rise).

2. **Canonical Frame Rotation:**
   Every input image is rotated via an affine $\sqrt{2}$-zoom warp so that the sun direction is always pointing directly from the top ($\theta_{\text{sun}} = 90^\circ$). In this canonical frame:
   - Craters (0) always have **bright tops and dark bottoms**.
   - Mounds (1) always have **dark tops and bright bottoms**.

3. **Physics-Preserving Augmentation Algebra:**
   Standard random flips and rotations corrupt the physical relationship between illumination and relief. Instead, all geometry passes through `src/transforms.py`:
   - **Canonical Vertical Flip ($p=0.25$):** Flipping rows reverses top/bottom shading $\implies$ **toggles the label ($0 \leftrightarrow 1$)**.
   - **Photometric Negation ($p=0.15$):** Applying $255 - \text{img}$ inverts shading polarity $\implies$ **toggles the label ($0 \leftrightarrow 1$)**.
   - **Canonical Horizontal Flip ($p=0.50$):** Left-right mirroring preserves top/bottom asymmetry $\implies$ **label is unchanged**.
   - **Strict Lint Rule:** No arbitrary library flips or rotations outside `src/transforms.py`.

---

## 📊 Benchmark Results

| Model / Experiment | Configuration | 5-Fold OOF Balanced Accuracy | ROC-AUC | Status |
|---|---|:---:|:---:|---|
| **Constant / Random** | Baseline rule | 0.5000 | 0.5000 | Reference |
| **Raw ConvNeXt-T (B3)** | Raw frame, no canonicalization | 0.5000 | ~0.5000 | ❌ Collapsed (Azimuth shortcut) |
| **E4 Canonical ConvNeXt-T** | Canonical frame + physics augmentations | **0.7161** (Plateau $t^*=0.4850$) | **0.7376** | ✅ **M3 Canonical Reference (5 Folds)** |
| **E4a Ablation ($p_{\text{vflip}}=0$)** | No vertical flip label swap | 0.7815 (Fold 0 fast screen) | 0.7859 | Threshold drifts to $t^*=0.6025$ |
| **E4b Ablation ($p_{\text{neg}}=0$)** | No photometric negation | **0.7824** (Fold 0 fast screen) | **0.7966** | Highest AUC, clean threshold $t^*=0.5175$ |

*Per-fold scores for E4 (Seed 42): Fold 0: `0.7673`, Fold 1: `0.7735`, Fold 2: `0.7079`, Fold 3: `0.5799`, Fold 4: `0.7730`.*

---

## 📁 Repository Structure

```
pareidolia_paradox/
├── README.md               # This document
├── AGENTS.md               # Operating manual and hard constraints for agents & engineers
├── task.md                 # Live phase roadmap & task execution tracker
├── PROGRESS.md             # Shared memory, experiment reports, and metrics log
├── Makefile                # Standardized developer CLI interface
├── configs/
│   ├── config.yaml         # Base config (contains PROTECTED frozen calibration)
│   ├── debug.yaml          # Fast smoke test configuration
│   └── exp/
│       ├── e4_canonical_convnext.yaml # Current best canonical model
│       ├── e4a_no_vflip.yaml          # M3 ablation: p_vflip=0
│       ├── e4b_no_neg.yaml            # M3 ablation: p_neg=0
│       └── e5_convnext_film.yaml      # Canonical + FiLM azimuth conditioning
├── src/
│   ├── canonical.py        # [PROTECTED] Calibration & canonical rotation routines
│   ├── dataset.py          # [PROTECTED] uint8 cache loader & PareidoliaDataset class
│   ├── transforms.py       # [PROTECTED] Canonical augmentation operators & label-swap algebra
│   ├── models.py           # timm backbones & FiLM conditioning wrapper
│   ├── metrics.py          # [PROTECTED] Balanced Accuracy, threshold sweep & bootstrap
│   ├── train.py            # Cross-validation training loop & checkpointing
│   ├── submit.py           # [PROTECTED] Submission builder, validator & sanity check
│   └── utils.py            # Reproducibility seeds, paths, logging
├── scripts/
│   ├── compare_runs.py        # Standup experiment leaderboard & comparison table
│   ├── viz_canonical_means.py # Visual handedness & class separation diagnostic
│   ├── viz_augment.py         # 16-variant augmentation grid visualizer
│   ├── test_calibration.py    # Per-class calibration forensics
│   └── eda.py                 # Exploratory data analysis
├── tests/                  # 40 pytest unit tests enforcing physics & contracts
└── reports/figures/        # Visual verification figures & canonical mean diagnostics
```

---

## 🚀 Quickstart for New Contributors

### 1. Environment Setup
Python 3.11 is recommended.
```bash
# Clone the repository
git clone https://github.com/wagha-builds/pareidolia_paradox.git
cd pareidolia_paradox

# Create virtual environment and install pinned dependencies
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Prepare Data Cache
Place raw competition data inside `data/raw/`:
```
data/raw/
├── train_images/          # train_00000.png ...
├── eval_images/           # eval_00001.png ...
├── train_metadata.csv     # image_id, sun_azimuth_angle, label
└── test_metadata.csv      # image_id, sun_azimuth_angle
```
Then build the memory-mapped `.npy` caches:
```bash
make cache
```

### 3. Run Test Suite & Linting
All tests must be green before proposing any changes:
```bash
make test    # Runs 40/40 pytest tests
make lint    # Runs ruff check & ruff format --check
```

### 4. Visual Diagnostics
Verify canonical separation and physical augmentations:
```bash
# Generate per-class canonical mean images (verifies s=-1, delta=46.7°)
python scripts/viz_canonical_means.py

# Generate 16-variant augmentation grid
python scripts/viz_augment.py --policy canonical
```
The resulting figures are saved to `reports/figures/`.

### 5. Training Models

#### A. Fast Smoke Test (< 15 seconds)
Verify end-to-end training and checkpoint pipeline:
```bash
python -m src.train --config configs/debug.yaml
```

#### B. Exploration Run (1 Fold, 15 Epochs)
```bash
make fast CONFIG=configs/exp/e4_canonical_convnext.yaml
```

#### C. Full 5-Fold Cross-Validation Run
```bash
make train CONFIG=configs/exp/e4_canonical_convnext.yaml SEEDS="42"
```

---

## 🛡️ Protected Rules & Invariants (from AGENTS.md)

1. **No Library Flips/Rotations:** Never use `torchvision.transforms.RandomHorizontalFlip`, `albumentations`, or auto-augment directly. All geometry must route through `src/transforms.py`.
2. **Never Touch Folds:** `data/folds.csv` is frozen and committed.
3. **Never Strip `.png`:** Image IDs must remain identical to metadata.
4. **Submissions via Validator Only:** All final CSV submissions must pass `src/submit.py` sanity and label-inversion checks.
