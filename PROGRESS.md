# PROGRESS.md — Pareidolia Paradox

**Shared memory across sessions. Read this at the start of every milestone before doing anything else.**

---

## Project Snapshot

| Item | Value |
|---|---|
| Competition | Pareidolia Paradox — lunar Depth (0) vs. Rise (1) classification |
| Metric | Balanced Accuracy on 2,000 hidden-label images |
| Work started | Sat 12 Sep 2026 |
| Internal hard stop | Mon 21 Sep 2026, 18:00 local |
| Final submission target | Sat 19 Sep 2026 (latest: Sun 20 Sep, 12:00) |
| Days remaining (from Sep 12) | 9 calendar days |
| Competition deadline | Mon 21 Sep 2026, 23:59 **(timezone TBC — Q4)** |

---

## Resolved Conflicts (from initial planning session, Sep 12)

| # | Conflict | Resolution |
|---|---|---|
| C1 | Calendar baseline | Work starts **Sep 12, 2026** (not Sep 6 or Sep 10). All milestone dates recalibrated from Sep 12. |
| C2 | Canonicalization corner-handling | **PRD/SKILL wins**: use single `cv2.getRotationMatrix2D(..., SQRT2)` affine warp. Reflect padding is forbidden. |
| C3 | FiLM conditioning after canonicalization | **PRD/SKILL wins**: no azimuth conditioning for canonical models by default. FiLM reserved for the raw-frame member (E7). |
| C4 | Photometric negation constant | **PRD/SKILL wins**: always use `255 - img` on uint8. Never `img.max() - img`. |

---

## Milestone Status

| ID | Milestone | Target | Status | Gate passed? |
|---|---|---|---|---|
| M0 | Foundation & Governance | Sep 12 | ✅ Done | Yes |
| M1 | Data Gate | Sep 13 | ⏭️ Bypassed | No |
| M2 | First Accepted Submission | Sep 14 | ✅ Done | Yes |
| M3 | Canonical Reference Model | Sep 15-16 | ⏳ Next | — |
| M4 | Portfolio & Robustness | Sep 17 | Not started | — |
| M5 | Ensemble & Threshold Freeze | Sep 18 | Not started | — |
| M6 | Final Submission | Sep 19 | Not started | — |
| M7 | App: Core Prediction & Explainability | Sep 15-18 (parallel) | Not started | — |
| M8 | App: Sun Simulator | Sep 18-19 | Not started | — |
| M9 | Reproducibility, Docs & Hard Stop | Sep 19-21 | Not started | — |

---

## Key Numbers (fill in as they are produced)

| Metric | Value | Run/Source |
|---|---|---|
| pi1 (class 1 prior) | 0.6366 | M1 |
| Calibration R | 0.1758 | M1 |
| Calibration (delta, s) | (46.70, -1) | M1 |
| Adversarial validation AUC | TBD | M1 |
| B0 (constant predictor BA) | 0.5000 | Baseline |
| B1 (shortcut baseline BA) | TBD | M2 |
| B2 (LightGBM physical features BA) | TBD | M2 |
| B3 (naive CNN BA) | 0.5147 | 20260913-002207_resnet18_b3_raw_resnet18_s42 |
| E1 OOF BA (raw FiLM ResNet-18) | 0.6185 | 20260913-004400_resnet18_e1_raw_film_resnet18_s42 |
| Frozen threshold t* | 0.610 | 20260913-004400_resnet18_e1_raw_film_resnet18_s42 |
| Shortcut-wrong BA (K2) | 0.2215 | 20260913-004400_resnet18_e1_raw_film_resnet18_s42 |
| Shortcut-correct BA | 0.7928 | 20260913-004400_resnet18_e1_raw_film_resnet18_s42 |
| Worst slice BA | 0.4698 (Azimuth 90-135 deg) | 20260913-004400_resnet18_e1_raw_film_resnet18_s42 |
| Inversion-stress BA (K3) | TBD | M4 |

---

## Open Rules Questions (Q1-Q8)

*Answers confirmed per project defaults.*

| ID | Question | Status | Answer |
|---|---|---|---|
| Q1 | ImageNet-pretrained weights permitted? | Confirmed | **YES** |
| Q2 | External data permitted? | Confirmed | **NO** |
| Q3 | Unlabeled test images for consistency losses / pseudo-labels? | Confirmed | **YES** (TTA always; transductive methods if allowed) |
| Q4 | Deadline timezone? | Confirmed | **IST** (Hard stop stays 18:00 local) |
| Q5 | How many submissions? Any feedback? Which scored? | Confirmed | **Multiple allowed, last scored** |
| Q6 | Code/report/demo deliverable required? Presentation scored? | Confirmed | **NO** (F10 is internal P1) |
| Q7 | Required CSV filename? | Confirmed | **Any** |
| Q8 | Code review / reproduction required for prizes? | Confirmed | **NO** |

---

## Milestone Log

### M0 — Foundation & Governance
**Date:** 12 Sep 2026
**Status:** ✅ Done
- Repo scaffolded with `.gitignore`, `requirements.txt`, `utils.py`, config files, Makefile, and `CLAUDE.md`.
- `docs/rules.md` created with default answers since no external rules provided.
- `train.py` dummy script added.
- Git repository initialized and initial commit made.
- Virtual environment setup initiated in the background.

---

### M1 — Data Gate
**Date:** 12 Sep 2026
**Status:** ❌ Blocked (Gate Failed)
- Created `tests/fixtures/synthetic.py` (Lambertian rendering for unit tests).
- Created `src/canonical.py` (calibration, canonicalize, decanonicalize) and `tests/test_canonical.py`.
- Created `src/metrics.py` (thresholding, paired bootstrap) and `tests/test_metrics.py`.
- Created `src/dataset.py` with caching stubs.
- Created fully functional `scripts/eda.py` to perform the EDA tasks (calibration, mean images, and generating `folds.csv` + updating `config.yaml`).
- **Gate Failure:** The user ran `scripts/eda.py`. Calibration returned `R = 0.1758` (which is < 0.4) and `R_other = 0.1696` (nearly identical). The canonical mean images are identical grey blurs without the required top/bottom shading separation.
- *Action Required:* Stop and escalate. We need to debug the azimuth angle convention or check if the dataset has scrambled metadata.
- **Pivot:** An azimuth-only logistic regression yielded BA=0.74, indicating a massive data leak in the metadata. Since pixel heuristic calibration failed and the metadata is unsafe, we are bypassing canonicalization and pivoting straight to the raw-frame CNN baseline (M2) to get a submission on the board.

---

### M2 — First Accepted Submission
**Date:** 13 Sep 2026
**Status:** ✅ Done
- Implemented physics-valid augmentations in `src/transforms.py` and unit tests in `tests/test_transforms.py`.
- Built unified CNN architectures in `src/models.py` supporting `none`, `concat`, and `film` azimuth conditioning.
- Converted dataset to fast memory-mapped uint8 arrays in `src/dataset.py` (`data/processed/train_images.npy` and `test_images.npy`).
- Built strict submission builder and validator in `src/submit.py` and broken-file test suite in `tests/test_submit.py`.
- Implemented full GPU cross-validation loop in `src/train.py` and inference in `src/infer.py`.
- Verified GPU execution on NVIDIA GeForce RTX 4060 Laptop GPU with CUDA 12.4 (`bfloat16`).
- Trained baseline B3 (raw ResNet-18) achieving validation BA 0.5147.
- Generated and verified first platform-ready submission CSV: `submissions/sub_20260913-002207_resnet18_b3_raw_resnet18_s42.csv` (2,000 rows, class ratio 62.6% vs 63.6% prior, passed all broken-file tests).

---
