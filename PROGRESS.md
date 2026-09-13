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
| Days remaining (from Sep 13) | 8 calendar days |
| Competition deadline | Mon 21 Sep 2026, 23:59 **IST (Q4 confirmed)** |

---

## Resolved Conflicts

| # | Conflict | Resolution |
|---|---|---|
| C1 | Calendar baseline | Work starts **Sep 12, 2026**. All milestone dates recalibrated. |
| C2 | Canonicalization corner-handling | **PRD/SKILL wins**: use single `cv2.getRotationMatrix2D(..., SQRT2)` affine warp. Reflect padding is forbidden. |
| C3 | FiLM conditioning after canonicalization | **PRD/SKILL wins**: no azimuth conditioning for canonical models by default. FiLM reserved for E5 (raw-frame member). |
| C4 | Photometric negation constant | **PRD/SKILL wins**: always use `255 - img` on uint8. Never `img.max() - img`. |
| C5 | Platform upload policy | **Single final upload only.** No baseline or dry-run upload. The platform slot is reserved for the single final go/no-go CSV at M6. |
| C6 | Calibration recovery | **Pivot 2 (Sep 13):** Combined R=0.1758 failed but Class-0-only R=0.4736 ≥ 0.40. Canonical pipeline is viable; M1 gate was bypassed but NOT permanently abandoned. |
| C7 | Training collapse | **Pivot 3 (Sep 13):** Root cause = LR too high (1e-3/2e-4) for ConvNeXt fine-tuning. LR fixed to 5e-5 + 3-epoch warmup. Secondary cause = azimuth shortcut learning on raw frames. Fix = canonical training. |

---

## Milestone Status

| ID | Milestone | Target | Status | Gate passed? |
|---|---|---|---|---|
| M0 | Foundation & Governance | Sep 12 | ✅ Done | Yes |
| M1 | Data Gate | Sep 13 | ⏭️ Bypassed | No (combined R=0.1758) |
| M2 | Raw-Frame Baseline & Submission Pipeline | Sep 13-14 | ⏳ In Progress | — |
| M2.5 | Canonical Pipeline Implementation | Sep 13-14 | 🟡 Approved, not started | — |
| M3 | Canonical Reference Model | Sep 14-15 | Not started | — |
| M4 | Portfolio & Robustness | Sep 16-17 | Not started | — |
| M5 | Ensemble & Threshold Freeze | Sep 18 | Not started | — |
| M6 | Final Submission | Sep 19 | Not started | — |
| M7 | App: Core Prediction & Explainability | Sep 15-18 (parallel) | Not started | — |
| M8 | App: Sun Simulator | Sep 18-19 | Not started | — |
| M9 | Reproducibility, Docs & Hard Stop | Sep 19-21 | Not started | — |

---

## Key Numbers

| Metric | Value | Run/Source |
|---|---|---|
| π₁ (class 1 prior / Rise fraction) | **0.6366** | M1 — `scripts/eda.py` |
| Calibration R (combined, all images) | **0.1758** | M1 — `scripts/eda.py` |
| Calibration R (Class 0 / Craters only) | **0.4736** | M1b — `scripts/test_calibration.py` |
| Calibration R (Class 1 / Mounds only) | **0.0094** | M1b — `scripts/test_calibration.py` |
| Calibration δ (frozen config, s=−1) | **46.702°** | M1 — class-0-only fit |
| Calibration s (frozen config) | **−1** | M1 — class-0-only fit |
| Azimuth-only logistic regression BA | **~0.741** | M1 — label leakage diagnostic |
| Crater mode residual matches δ? | **Yes — 46.15° vs 46.70° (±0.55°)** | M1b — `scripts/test_calibration.py` |
| Normalization mean | **0.3836** | M1 — lunar stats (not ImageNet) |
| Normalization std | **0.2497** | M1 — lunar stats (not ImageNet) |
| Adversarial validation AUC | TBD | M1 — still needed |
| B0 (constant predictor BA) | **0.500** (by definition) | M2 |
| B1 (shortcut baseline BA) | INVALID — canonicalization gate failed | M2 |
| B2 (LightGBM physical features BA) | TBD | M2 |
| B3 (naive CNN BA) — ConvNeXt | **0.5000** (collapsed) ❌ | M2 — raw-frame, LR too high |
| B3 (naive CNN BA) — ResNet18 | **0.5082** at t=0.47 (azimuth shortcut) ❌ | M2 — raw-frame, shortcut |
| Debug run OOF BA | **0.5738** at t=0.53 (CPU, 3 epochs, SmallCNN) | M2 — debug.yaml |
| Overfit test (64 images, LR=5e-5) | **BA=0.906 at epoch 15** ✅ pipeline healthy | M2 — `scripts/overfit_test.py` |
| E4 OOF BA (canonical ConvNeXt) | TBD | M2.5/M3 |
| Final ensemble OOF BA | TBD | M5 |
| Frozen threshold t* | TBD | M5 |
| Shortcut-wrong BA (K2) | TBD | M4 |
| Inversion-stress BA (K3) | TBD | M4 |

---

## Open Rules Questions (Q1-Q8)

| ID | Question | Status | Answer |
|---|---|---|---|
| Q1 | ImageNet-pretrained weights permitted? | Confirmed | **YES** |
| Q2 | External data permitted? | Confirmed | **NO** |
| Q3 | Unlabeled test images for consistency losses / pseudo-labels? | Confirmed | **YES** (TTA always; transductive methods if allowed) |
| Q4 | Deadline timezone? | Confirmed | **IST** (Hard stop stays 18:00 local) |
| Q5 | How many submissions? Any feedback? Which scored? | Confirmed | **Multiple allowed, last scored** |
| Q6 | Code/report/demo deliverable required? | Confirmed | **NO** |
| Q7 | Required CSV filename? | Confirmed | **Any** |
| Q8 | Code review / reproduction required for prizes? | Confirmed | **NO** |

---

## Milestone Log

### M0 — Foundation & Governance
**Date:** 12 Sep 2026 | **Status:** ✅ Done
- Repo scaffolded: `.gitignore`, `requirements.txt`, `utils.py`, configs, Makefile, `CLAUDE.md`.
- `docs/rules.md` created with Q1-Q8 answers.
- Git initialized and initial commit made.
- Virtual environment set up with all requirements.

---

### M1 — Data Gate
**Date:** 12–13 Sep 2026 | **Status:** ⏭️ Bypassed (gate failed; canonical path conditionally recoverable)

**Session 1 (Sep 12):**
- Created `tests/fixtures/synthetic.py`, `src/canonical.py`, `src/metrics.py`, `src/dataset.py`.
- Created `scripts/eda.py` for full EDA pipeline.
- **Gate failure:** User ran `scripts/eda.py`. R=0.1758 < 0.4; canonical means = featureless grey.
- Azimuth-only logistic probe: BA=0.741 — massive label leak.
- **Pivot 1 approved:** Bypass canonicalization; train raw-frame models.

**Session 2 (Sep 13):**
- Created `scripts/test_calibration.py` for per-class calibration forensics.
- **Key discovery:** Class 0 (Craters) alone gives R=0.4736 ≥ 0.40. Class 1 (Mounds) R=0.0094.
- Combined R=0.1758 was caused by mound noise overwhelming crater signal.
- Calibration IS recoverable; canonical pipeline approved (M2.5).
- Physical convention (North-up) does not apply — confirmed.
- Frozen config values (δ=46.70°, s=−1) match crater-only mode residual to 0.55°.
- **Handedness ambiguity remains:** s=+1 (R=0.4736, δ=135.18°) vs s=−1 (R=0.4678, δ=46.15°). ΔR=0.0058. Visual canonical mean inspection required.

---

### M2 — Raw-Frame Baseline & Submission Pipeline
**Date:** 12–13 Sep 2026 | **Status:** ⏳ In Progress

**Session 1 (Sep 12):**
- Implemented `src/models.py`, `src/transforms.py`, `src/train.py`, `src/losses.py`.
- Implemented `src/infer.py`, `src/submit.py` (validator + builder).
- Created `configs/exp/raw_convnext_baseline.yaml`.
- All 17 pytest tests passing.

**Session 2 (Sep 13) — manual runs by user:**
- `python -m src.dataset --mode cache` → data/processed ✅
- `python -m pytest -q` → 17 passed ✅
- Debug run (CPU): OOF BA 0.5738 at t=0.53 ✅
- ConvNeXt fast run (LR=2e-4): OOF BA 0.5000 — COLLAPSED ❌
- ConvNeXt 3-seed full run: all folds collapsed; stopped manually ❌

**Session 2 — engineering diagnosis and fixes:**
- Root cause 1: LR=1e-3 (base config), LR=2e-4 (exp config) — too high for ConvNeXt fine-tuning.
  Fix: LR→5e-5 in all configs; 3-epoch linear warmup added to `src/train.py`.
- Root cause 2: Early stopping patience=5 fired during warmup (epoch 7, best was epoch 2).
  Fix: early stopping counter now only starts after `warmup_epochs` have elapsed.
- Root cause 3 (architectural): Raw-frame models learn azimuth→label correlation (BA=0.776 from azimuth alone). Grouped CV punishes this — val BA fell from 0.498→0.441 as training progressed.
  Fix: Canonical training (M2.5). This is not a tuning problem.
- Overfit test (64 images, LR=5e-5): BA=0.906 in 15 epochs → pipeline is fundamentally healthy.
- Added `scripts/overfit_test.py`, `scripts/inspect_oof.py`, `scripts/b3_raw_resnet18.yaml`.
- Windows-specific fix: `num_workers` must be 0 (mmap'd numpy arrays + Windows spawn = OSError).

**Files modified this session:**
- `src/train.py` — LR logging; warmup-aware early stopping; SequentialLR warmup scheduler
- `configs/config.yaml` — LR 1e-3→5e-5; label_smoothing 0.1→0.05; warmup_epochs=3; canonicalize=false
- `configs/exp/raw_convnext_baseline.yaml` — LR 2e-4→5e-5; warmup_epochs=3; num_workers=0
- `configs/exp/b3_raw_resnet18.yaml` — NEW file (ResNet18 baseline)
- `scripts/test_calibration.py` — major update (per-class forensics, crop sweep, physical test)
- `scripts/overfit_test.py` — NEW file
- `scripts/inspect_oof.py` — NEW file

---

### M2.5 — Canonical Pipeline Implementation
**Date:** Approved 13 Sep 2026 | **Status:** 🟡 Not yet started

**Approved scope:**
1. Protected-file changes to `src/transforms.py` and `src/dataset.py` (TDD: tests first)
2. Visual handedness check (generate canonical mean images for both s=+1 and s=−1)
3. FiLM conditioning implementation in `src/models.py` (E5 experiment)
4. E4 canonical ConvNeXt experiment

**Expected outcome:** E4 val_ba@0.5 > 0.55 by epoch 5 of the fast run.
**Reference:** sanjog branch achieved E4 OOF BA ~0.65-0.67.

---

### M3 — Canonical Reference Model
**Date:** Target Sep 14-15 | **Status:** Not started

**Re-scoped:** E4 from M2.5 IS the M3 canonical reference. M3 adds ablations (p_vflip=0 vs E4, p_neg=0 vs E4) and backbone diversity.

---
