# task.md — Pareidolia Paradox

Task tracker. Updated throughout execution. Status symbols: `[ ]` not started, `[/]` in progress, `[x]` done, `[!]` blocked/caveat.

**Active milestone:** M2.5 — Canonical Pipeline Implementation (protected-file changes approved by human, 13 Sep 2026).

**Platform upload policy (IMMUTABLE):** There is no baseline or dry-run upload. The competition platform slot is reserved for the **single final go/no-go CSV** produced by `make submit` at M6 only.

---

## Pivot History

### Pivot 1 — M1 Gate Failed → Raw-Frame Bypass (Sep 12-13)
- Calibration returned combined R = 0.1758 (threshold: 0.4).
- Central-crop sweep: 128→0.0628, 96→0.0320, 64→0.0174 — made R worse.
- Canonical per-class mean images: featureless grey boxes.
- Azimuth-only logistic regression: BA ≈ 0.741 — severe metadata label leak.
- **Resolution:** Bypass canonicalization. Train raw-frame models. Use offline OOF as only signal.

### Pivot 2 — Per-Class Forensics → Calibration Recoverable (Sep 13)
**OVERTURNS PIVOT 1 for the canonical path.**
- `scripts/test_calibration.py` ran class-stratified analysis on the full dataset.
- **Class 0 (Depth/Craters) only:** R = 0.4736, δ = 135.18°, s = +1 (marginally wins).
  Alternative: R = 0.4678, δ = 46.15°, s = −1 (within measurement noise).
- **Class 1 (Rise/Mounds) only:** R = 0.0094 — nearly zero signal (mounds have no shadow pools).
- **Combined R = 0.1758 was polluted by 5,000 noisy mound images overwhelming 2,854 crater images.**
- **Calibration IS recoverable** using crater-only parameters. The camera convention fits δ≈46.7°, s=−1 (frozen config values).
- Handedness ambiguity: both s=+1 and s=−1 fit craters almost equally well (ΔR = 0.0058).
  **Visual confirmation of canonical mean images required before E4 full run.**
- Physical convention (North-up, δ=90°, s=−1): R = 0.1758 — confirmed inapplicable.
- Mode of crater residuals = 46.15°, matches frozen δ=46.70° to within 0.55° ✅

### Pivot 3 — Training Collapse → Root-Cause Found and Fixed (Sep 13)
- Overfit test (64 images, LR=5e-5): loss 0.77→0.23 in 15 epochs. Pipeline is healthy.
- Root cause 1: LR=1e-3 (base config) / LR=2e-4 (exp config) → destroyed pretrained ConvNeXt features in epoch 1.
- Root cause 2: Early stopping patience=5 counted from warmup epoch 2, fired at epoch 7 — during warmup.
- B3 ResNet18 at LR=3e-4: loss decreases but val_ba falls from 0.498→0.441.
- Root cause 3 (architectural): Raw-frame models learn the azimuth-label correlation (BA=0.776 from azimuth alone).
  Grouped CV punishes this — val fold has different azimuth distribution → model scores BELOW chance.
- **Resolution:** Canonical training removes the azimuth confound entirely. Canonical pipeline approved and in progress.

---

## M0 — Foundation & Governance ✅

### OPS tasks
- [x] Create git repo with full directory structure from AGENTS.md §3
- [x] Create `.gitignore`
- [x] Create and pin `requirements.txt`
- [x] `src/utils.py::set_seed(seed)`
- [x] `configs/config.yaml` with `frozen:` block
- [x] `configs/debug.yaml`
- [x] Initialize experiment tracker
- [x] `docs/rules.md` with Q1-Q8 answers
- [x] `Makefile` with all targets from AGENTS.md §4
- [x] `CLAUDE.md`
- [x] Verify `python -m src.train --config configs/debug.yaml` runs (loss decreases)
- [x] Update `PROGRESS.md` M0 entry

### Manual actions (DONE)
- [x] MANUAL: Read competition rules → answered Q1-Q8 in `docs/rules.md`
- [x] MANUAL: Confirm GPU access and platform account
- [x] MANUAL: Place data files in `data/raw/`

---

## M1 — Data Gate ⏭️ Bypassed (gate failed, but calibration partially recoverable)

**Outcome:** Combined calibration gate failed (R=0.1758 < 0.4). However, per-class forensics (Pivot 2)
showed R₀=0.4736 for craters alone — above the gate threshold. Calibration IS recoverable using crater-only
parameters. The gate is formally bypassed but the canonical path is NOT permanently closed.

### DATA tasks
- [x] `tests/fixtures/synthetic.py` — Lambertian dome/pit renderer
- [x] `src/canonical.py` — calibrate(), canonicalize(), decanonicalize() with unit tests
  - [x] Synthetic tests pass (delta, R, handedness recovery)
  - [x] Rotate-then-update-azimuth == canonical
  - [x] Reflect-padding ghost test
- [x] Step 1.1: load_metadata(), verify_images()
- [x] Step 1.2: dtype, value range, unique-value count
- [x] Step 1.3: class balance → frozen.prior_pi1 = 0.6366 (~63.7% Rise)
- [x] Step 1.4: azimuth distribution analysis — spans 0-360°; azimuth-only probe BA=0.741
- [x] Step 1.5: CALIBRATION full run — R=0.1758, gate failed
- [x] Step 1.5b: PER-CLASS CALIBRATION (`scripts/test_calibration.py`)
  - [x] Crop sweep: R worsens with cropping; full 256px is best
  - [x] Per-class polarity: Class 0 R=0.4736, Class 1 R=0.0094
  - [x] Class-0-only calibration: R=0.4736 ≥ 0.40 ✅ calibration recoverable
  - [x] Physical convention test: δ=90°, s=−1 → R=0.1758 (fails — not North-up)
  - [!] Handedness ambiguity: s=+1 (R=0.4736, δ=135.18°) vs s=−1 (R=0.4678, δ=46.15°). ΔR=0.0058.
        Frozen config uses δ=46.70°, s=−1. **Visual check of canonical means required before E4 full run.**
- [x] Step 1.6: canonical per-class mean images — featureless grey (combined) — gate failed
- [/] Step 1.7: visual audit
  - [x] User: far-away lunar shots; shadows not obviously isolated
  - [ ] Generate canonical mean images with BOTH (s=+1,δ=135°) and (s=−1,δ=46.7°) — required before E4
- [ ] Step 1.8: near-duplicate detection (pHash + embeddings + FAISS)
- [x] Step 1.9: generate folds.csv (StratifiedGroupKFold, n=5)
  - [!] folds.csv has `image_id,fold` only — lacks `group_id`. Do not regenerate without approval.

### OPS tasks
- [x] `src/metrics.py` — balanced_accuracy, apply_threshold, sweep, plateau_threshold, slice_report, bootstrap
- [ ] Step 1.10: adversarial validation (still needed as train/test shift audit)
- [x] Step 1.11: normalization stats → frozen.norm.mean=0.3836, frozen.norm.std=0.2497
- [x] Step 1.12: build_cache() → data/processed ✅
- [x] Fill frozen config block (delta=46.702°, s=-1, R=0.1758, pi1=0.6366, norm, folds_sha256)
- [x] Update `PROGRESS.md` M1 entry

### Manual actions
- [x] MANUAL: Look at calibration — R<0.4, STOP triggered, pivot approved
- [x] MANUAL: Look at canonical per-class mean images — featureless grey boxes
- [x] MANUAL: Confirm images have clear shadows — far-away lunar shots, not obvious
- [ ] MANUAL: **Inspect canonical mean images with (s=−1,δ=46.7°) AND (s=+1,δ=135.18°)**
      Choose the pair where Class 0 and Class 1 means visibly differ (bright-top vs dark-top or vice versa).
- [ ] MANUAL: Adversarial validation AUC (must be < 0.65)

---

## M2 — Raw-Frame Baseline & Submission Pipeline ⏳ In Progress

**Status (13 Sep 2026):** Training pipeline is healthy (overfit test passed). Raw-frame models cannot
produce good grouped-CV results due to azimuth shortcut learning. Canonical pipeline (M2.5) is the fix.

### Baselines
- [ ] B0: constant predictor → assert BA == 0.500; record in PROGRESS.md
- [ ] B1: shortcut baseline — mark as invalid (canonicalization gate failed)
- [ ] `src/features.py` — C5 feature set for LightGBM
- [ ] B2: LightGBM on physical/image features, 5-fold CV, no raw azimuth as feature
- [/] B3: raw-frame CNN
  - [x] `configs/exp/raw_convnext_baseline.yaml` — ConvNeXt-T; LR fixed to 5e-5; warmup=3
  - [x] `configs/exp/b3_raw_resnet18.yaml` — ResNet18, LR=3e-4, warmup=0
  - [x] Dataset cache built ✅
  - [x] 17 pytest tests pass ✅
  - [x] Debug run (CPU): OOF BA 0.5738 at t=0.53 ✅
  - [x] ConvNeXt fast run (original LR=2e-4): collapsed, OOF BA 0.5000 ❌
  - [x] ConvNeXt 3-seed full run: collapsed across all seeds, stopped manually ❌
  - [x] Overfit test (64 images, LR=5e-5): BA=0.906 in 15 epochs ✅ Pipeline healthy
  - [x] ConvNeXt fast run (LR=5e-5, warmup=3): early stopping fired during warmup ❌
  - [x] B3 ResNet18 fast run (LR=3e-4): val_ba 0.498→0.441 — azimuth shortcut punished ❌
  - [!] Raw-frame CNN without canonicalization CANNOT produce good grouped-CV val BA.
        **Do not launch further raw-frame full runs. Canonical pipeline (M2.5) is the fix.**

### OPS tasks
- [/] `src/infer.py` — parity test still needed
- [/] `src/submit.py` — label-inversion check and sanity report still needed
- [x] `make submit` target in Makefile
- [/] Update `PROGRESS.md` M2 entry

### Files created/modified this session (13 Sep 2026)
- [x] `scripts/test_calibration.py` — per-class calibration forensics (major update)
- [x] `scripts/overfit_test.py` — 64-image overfit sanity check
- [x] `scripts/inspect_oof.py` — OOF probability distribution inspector
- [x] `src/train.py` — LR logging added; warmup-aware early stopping; SequentialLR warmup scheduler
- [x] `configs/config.yaml` — LR 1e-3→5e-5; label_smoothing 0.1→0.05; warmup_epochs=3; canonicalize=false
- [x] `configs/exp/raw_convnext_baseline.yaml` — LR 2e-4→5e-5; warmup_epochs=3; num_workers=0
- [x] `configs/exp/b3_raw_resnet18.yaml` — NEW: ResNet18 baseline, LR=3e-4

### Manual actions (DONE)
- [x] MANUAL: `python -m src.dataset --mode cache` → data/processed ✅
- [x] MANUAL: `python -m pytest -q` → 17 passed ✅
- [x] MANUAL: Debug run → OOF BA 0.5738 ✅
- [x] MANUAL: ConvNeXt fast run → 0.5000 COLLAPSED ❌
- [x] MANUAL: 3-seed full run → collapsed across all seeds ❌
- [ ] MANUAL: Upload the **single final** CSV to platform (reserved for M6 only)

---

## M2.5 — Canonical Pipeline Implementation 🟢 IN PROGRESS (Fast Run Underway)

**Human approved 13 Sep 2026:** Protected-file changes to `src/transforms.py` and `src/dataset.py`,
train-loop wiring, FiLM conditioning (item 3), and E4 canonical ConvNeXt experiment.

**Why this is needed:** Raw-frame models learn azimuth→label shortcut. Canonical frame removes the
confound entirely — in canonical frame, crater always has bright-top, mound always has dark-top.

### Phase 1 — TDD: Write tests FIRST
- [x] `tests/test_transforms.py` — add:
  - [x] `test_canonical_vflip_label_swap` — flip+unflip = identity; double label toggle = original
  - [x] `test_photometric_negation_label_swap` — double negation = identity; double toggle = original
  - [x] `test_canonical_hflip_no_label_change` — pixels change, label unchanged
  - [x] `test_vflip_top_minus_bottom_asymmetry` — canonical vflip reverses top-minus-bottom asymmetry
  - [x] Lint rule test: no library flip/rotation outside transforms.py
- [x] `tests/test_dataset.py` — add:
  - [x] `test_dataset_canonicalize_cfg` — dataset with canonicalize_cfg returns different images than without
  - [x] `test_dataset_augmentation_policy` — policy called, label can change

### Phase 2 — Protected File Changes
- [x] `src/transforms.py` (PROTECTED — approved):
  - [x] Add `Sample` dataclass (image: ndarray uint8, azimuth: float, label: int)
  - [x] Add `CanonicalVerticalFlipLabelSwap(p=0.20)` — vflip, label 0↔1
  - [x] Add `PhotometricNegationLabelSwap(p=0.15)` — `255 - img`, label 0↔1
  - [x] Add `CanonicalHorizontalFlip(p=0.50)` — hflip, label unchanged
  - [x] Add `build_augmentation_policy(cfg, training)` — reads p_vflip, p_neg, p_hflip
- [x] `src/dataset.py` (PROTECTED — approved):
  - [x] Add `canonicalize_cfg` parameter to `PareidoliaDataset.__init__`
  - [x] Add `augmentation_policy` parameter
  - [x] In `__getitem__`: call `canonicalize()` if `canonicalize_cfg` is set
  - [x] Apply augmentation policy after canonicalization
  - [x] Assert class mapping: {0: depth, 1: rise}

### Phase 3 — Train Loop Wiring
- [x] `src/train.py` — remove the `canonicalize=true` RuntimeError guard
- [x] `src/train.py` — wire `canonicalize_cfg` from frozen config into `PareidoliaDataset`
- [x] `src/train.py` — wire `augmentation_policy` into `PareidoliaDataset`

### Phase 4 — Visual Handedness Check (MANUAL — REQUIRED before E4 full run)
- [x] Create `scripts/viz_canonical_means.py`
- [x] Run with (s=−1, δ=46.7°) → saved to `reports/figures/canonical_means_sm1_delta46.7.png`
- [x] Run with (s=+1, δ=135.18°) → saved to `reports/figures/canonical_means_sp1_delta135.2.png`
- [x] MANUAL: Inspected both. Confirmed `s=-1, delta=46.70°` is separable with opposite top-bottom asymmetry (-18.30 vs +0.26).
  Committed parameters as confirmed frozen calibration.

### Phase 5 — E4 Canonical ConvNeXt Experiment
- [x] Create `configs/exp/e4_canonical_convnext.yaml`
      (canonicalize=true; p_vflip=0.25; p_neg=0.15; p_hflip=0.50; LR=5e-5; warmup=3; epochs=30)
- [x] Smoke test: `python -m src.train --config configs/debug.yaml` (Passed in 12s)
- [x] Fast run: `python -m src.train --config configs/exp/e4_canonical_convnext.yaml --fast`
      Gate: val_ba@0.5 > 0.55 by epoch 5 — **PASSED (0.7434 @ epoch 5, best 0.7656 @ epoch 6, OOF BA @ plateau 0.7737, AUC 0.7920)**
- [x] Check augmentation grid: `python scripts/viz_augment.py --policy canonical --image-id <id>`
- [x] MANUAL: Inspected augmentation grid (`reports/figures/augment_canonical_train_00005.png`) — confirmed label flips match physics
- [x] Full 5-fold CV run for E4 (Seed 42) — **OOF BA @ plateau t=0.4850: 0.7161, AUC: 0.7376**
      (Fold 0: 0.7673, Fold 1: 0.7735, Fold 2: 0.7079, Fold 3: 0.5799, Fold 4: 0.7730)
- [ ] Seeds 1 and 2 (for 3-seed variance)
- [ ] `make robustness RUN=<run_id>` (scheduled for M4)
- [x] Record EXP-E4 report in PROGRESS.md

### Phase 6 — FiLM Conditioning (Item 3, approved)
- [x] `src/models.py` — FiLM wrapper (feeds sin/cos azimuth into feature maps)
- [x] `configs/exp/e5_convnext_film.yaml` — canonical + FiLM azimuth conditioning
- [x] Run E5 fast screen (Fold 0 BA: 0.7899 @ t=0.4650, AUC: 0.7976)
- [x] Run E5 full 5-fold CV run — **OOF BA @ plateau t=0.4450: 0.7271, AUC: 0.7528** (New Best!)
      (Fold 0: 0.7885, Fold 1: 0.7747, Fold 2: 0.6233, Fold 3: 0.5977, Fold 4: 0.7778)
- [x] E4 + E5 50/50 Ensemble Blend test: **OOF BA @ plateau t=0.4525: 0.7312, AUC: 0.7559**

### Manual actions required
- [x] MANUAL: Inspect canonical mean images — break handedness ambiguity (Confirmed s=-1, delta=46.70°)
- [x] MANUAL: Review augmentation grid before full E4 run (Confirmed top-bottom asymmetry and label-flips match physics)
- [x] MANUAL: Approve E4 full-run launch after fast-run gate (Approved and Completed)

---

## M3 — Canonical Reference Model & Ablations

**Re-scoped (13 Sep):** E4 from M2.5 IS the M3 canonical reference. M3 adds ablations and backbone diversity.

### MODEL tasks
- [x] E4a ablation: p_vflip=0 — Fold 0 BA: 0.7791 @ 0.5, 0.7815 @ t=0.6025 (Threshold shifted from 0.5375 to 0.6025 due to lost relief-inversion label balance)
- [x] E4b ablation: p_neg=0 — Fold 0 BA: 0.7776 @ 0.5, 0.7824 @ t=0.5175 (Highest AUC: 0.7966, best calibrated threshold near 0.50)
- [x] E5 FiLM Full 5-Fold — **OOF BA: 0.7271 @ t=0.4450, AUC: 0.7528** (Rank #1 across all 5-fold models)
- [x] `scripts/compare_runs.py` — sorted table with seed std devs
- [ ] Grad-CAM utility in src/viz.py

### OPS tasks
- [ ] Artifact contract: `artifacts/<date>_<name>_ba<x.xxxx>/`
- [ ] `artifacts/registry.json`
- [ ] `src/train.py` kill-and-resume test
- [ ] Update `PROGRESS.md` M3 entry

### Manual actions
- [ ] MANUAL: Review Grad-CAM overlays (30 correct + 30 wrong)
- [ ] MANUAL: M3 gate — E4 OOF BA must beat B0 and B3 by >1 pt

---

## M4 — Portfolio & Robustness

### MODEL tasks
- [ ] E3 ablation grid (vflip p in {0, 0.2} × negation p in {0, 0.15}) — 4 fast screens
- [ ] E5 backbone diversity: EfficientNetV2-S + Swin-T
- [ ] E6 negation-consistency loss (lambda in {0, 0.1, 0.5})
- [ ] E7: raw-frame FiLM member (azimuth conditioning without canonicalization)

### DATA tasks
- [ ] `src/robustness.py` — shortcut audit, inversion stress, azimuth-shift, slice report
- [ ] `make robustness` target
- [ ] Top-200 highest-loss OOF image grid

### OPS tasks
- [ ] `src/ensemble.py` skeleton
- [ ] Update `PROGRESS.md` M4 entry

### Manual actions
- [ ] MANUAL: Top-200 error grid review; sub-type breakdown
- [ ] MANUAL: M4 gate — select 3-4 diverse members; approve selection

---

## M5 — Ensemble & Threshold Freeze

### MODEL tasks
- [ ] `src/ensemble.py` complete — fold-hash check, mean-of-logits, weighted average
- [ ] TTA policy in `src/infer.py`
- [ ] F7 plateau threshold sweep on final ensemble OOF
- [ ] Fill frozen.threshold in config

### OPS tasks
- [ ] Selection paragraph committed
- [ ] Commit tagged `ensemble-frozen`
- [ ] `make reproduce` from fresh clone
- [ ] Update `PROGRESS.md` M5 entry

### Manual actions
- [ ] MANUAL: Approve final ensemble spec, TTA policy, threshold
- [ ] MANUAL: Model-code freeze (no src/ changes after Fri 18 Sep ~17:00)

---

## M6 — Final Submission

### OPS tasks
- [ ] Confirm `make reproduce` OOF BA within 0.2 pt
- [ ] Final inference with TTA → `test_probs_<timestamp>.npy`
- [ ] `make submit` → validate → inversion check → sanity report
- [ ] Spot-check grid (20 predicted-Depth + 20 predicted-Rise)
- [ ] Update `PROGRESS.md` M6 entry

### Manual actions
- [ ] MANUAL: Review spot-check grid
- [ ] MANUAL: Final go/no-go decision
- [ ] MANUAL: Upload final CSV — **ONLY planned upload. No dry-runs.**
- [ ] MANUAL: Back up rollback file and `test_probs_*.npy`
- [ ] MANUAL: Tag commit `sub-v1`

---

## M7 — App: Core Prediction & Explainability

### APP tasks
- [ ] `app/main.py` — FastAPI with lifespan model loader
- [ ] GET /health, GET /model-info, POST /predict, POST /explain
- [ ] `src/viz.py` — HiResCAM, shadow-mass fraction
- [ ] Streamlit app
- [ ] `app/Dockerfile` + `docker-compose.yml`
- [ ] Parity check: app == `make infer` to 1e-6 on 5 images

### Manual actions
- [ ] MANUAL: Stranger test on clean machine

---

## M8 — App: Sun Simulator

### APP tasks
- [ ] POST /simulate — 72 azimuths, batched forward pass
- [ ] Streamlit Mode A (azimuth swept) and Mode B (image rotated + azimuth updated)
- [ ] 3-4 one-click presets
- [ ] Docker tested on clean machine

### Manual actions
- [ ] MANUAL: Record 3-minute demo video
- [ ] MANUAL: Docker stranger test

---

## M9 — Reproducibility, Docs & Hard Stop

### OPS tasks
- [ ] `src/reproduce.py` / `make reproduce`
- [ ] `README.md` — one-command reproduction
- [ ] `docs/pareidolia_playbook.md` — final metrics, ablation table, robustness table
- [ ] PROGRESS.md final entry
- [ ] Confirm main is green; protected files intact

### Manual actions
- [ ] MANUAL: Approve README and model card
- [ ] MANUAL: Hard stop — 18:00 local Mon 21 Sep 2026
