# task.md — Pareidolia Paradox

Task tracker. Updated throughout execution. Status symbols: [ ] not started, [/] in progress, [x] done.

**Active milestone:** M2 — Post-pivot raw-frame baseline and offline submission pipeline.

**Current pivot note (13 Sep 2026):** M1 Data Gate did not pass. Calibration failed
(`R=0.1758`, `R_other=0.1696`) and canonical class means did not visually separate. A later
azimuth-only probe scored about `0.741` Balanced Accuracy, so azimuth is also unsafe as a direct
feature because it leaks the training labels. Canonicalization is therefore bypassed for the first
working model path. Until we have stronger evidence, train raw-frame pixel models with no azimuth
conditioning, no geometric augmentation, and offline validation only. Do not upload any baseline or
dry-run CSV; the platform upload slot should be treated as final/go-no-go only.

---

## M0 — Foundation & Governance

### OPS tasks
- [x] Create git repo with full directory structure from AGENTS.md §3 / Playbook 0.2
- [x] Create `.gitignore` (data/, experiments/, artifacts/ except registry.json, __pycache__, .venv)
- [x] Create and pin `requirements.txt` (Python 3.11, all deps with ==)
- [x] Write `src/utils.py::set_seed(seed)` seeding random, numpy, torch, torch.cuda, PYTHONHASHSEED, cuDNN flags
- [x] Create skeleton `configs/config.yaml` with `frozen:` block (all values null)
- [x] Create `configs/debug.yaml` (tiny config: 1 fold, 5 epochs, 32 batch, no CV)
- [x] Initialize experiment tracker (W&B project or equivalent)
- [x] Create `docs/rules.md` with Q1-Q8 template
- [x] Create `Makefile` with all targets from AGENTS.md §4
- [x] Create `CLAUDE.md` (one line: @AGENTS.md)
- [x] Verify `python -m src.train --config configs/debug.yaml` runs (loss decreases)
- [x] Update `PROGRESS.md` M0 entry

### Manual actions (YOU must do these)
- [x] MANUAL: Read the official competition rules and answer Q1-Q8 in `docs/rules.md`
- [x] MANUAL: Send Q1-Q8 to organizers if any are unanswerable from the rules
- [x] MANUAL: Confirm GPU access and check platform account + submission limits
- [x] MANUAL: Place data files in `data/raw/` (train/*.png, test/*.png, train_metadata.csv, test_metadata.csv)

---

## M1 — Data Gate

**Outcome:** Bypassed after failed gate. The checked items below mean the code/script was created or
the diagnostic was run; they do **not** mean the data gate passed.

### DATA tasks
- [x] `tests/fixtures/synthetic.py` — Lambertian dome/pit renderer from SKILL §12
- [x] `src/canonical.py` — calibrate(), canonicalize(), decanonicalize() with all unit tests
  - [x] Synthetic tests: recover known (delta, s) within 3deg, R > 0.95, other-handedness R < 0.3
  - [x] Rotate-then-update-azimuth == canonical (MAE < 1 grey level on central disc)
  - [x] Reflect-padding ghost test (documents why it's forbidden)
- [x] Step 1.1: load_metadata(), verify_images() — assert row counts, no nulls, no dup IDs, 256x256, single channel
- [x] Step 1.2: record dtype (uint8 vs uint16), value range, unique-value count
- [x] Step 1.3: class balance -> fill frozen.prior_pi1 in config
- [x] Step 1.4: azimuth distribution analysis started; azimuth range looked healthy, but azimuth-only logistic probe showed severe label leak (BA ≈ 0.741)
- [x] Step 1.5: CALIBRATION — run calibrate(), produce diagnostics -> reports/figures/
  - [x] Gate failed: full-tile best `R=0.1758`, `R_other=0.1696`; central crop sweep worsened R (`128 -> 0.0628`, `96 -> 0.0320`, `64 -> 0.0174`, `48 -> 0.0122`, `32 -> 0.0086`)
- [x] Step 1.6: canonicalize(), produce per-class mean images -> reports/figures/
  - [x] Gate failed: canonical means were featureless grey boxes rather than opposite top/bottom shading
- [/] Step 1.7: visual audit grid / direct image inspection
  - [x] User inspection: images look like far-away lunar shots; target relief is not obviously isolated
  - [ ] Generate a formal visual audit grid later for error analysis and documentation
- [ ] Step 1.8: near-duplicate detection (pHash + embeddings + FAISS), group_ids, cross-set duplicates
- [x] Step 1.9: generate folds.csv (StratifiedGroupKFold, n=5), compute SHA-256
  - [!] Caveat: existing `data/folds.csv` has `image_id,fold` only; it lacks `group_id`. Do not regenerate without explicit approval, but record this limitation in every manifest.

### OPS tasks
- [x] `src/metrics.py` — balanced_accuracy, apply_threshold, sweep, plateau_threshold, per_fold_optima, slice_report, paired_bootstrap; all with unit tests
- [ ] Step 1.10: adversarial validation — train/test AUC
- [x] Step 1.11: normalization stats -> fill frozen.norm in config
- [x] Step 1.12: build_cache() -> data/processed/train_images.npy, test_images.npy, index.json
- [x] Fill frozen config block (delta, s, R, pi1, norm, folds_sha256) via config update
- [x] Update `PROGRESS.md` M1 entry

### Manual actions (YOU must do these)
- [x] MANUAL: Look at the calibration output — result failed (`R < 0.4`, handedness ambiguous); STOP condition triggered and resolved by explicit pivot approval
- [x] MANUAL: Look at the canonical per-class mean images — result failed; means did not separate
- [x] MANUAL: Confirm whether images have clear central crater/dome shadows — user could not reliably identify them; images are far-away lunar shots
- [ ] MANUAL: Formal visual audit remains useful later, but it no longer blocks M2
- [ ] MANUAL: Check adversarial validation AUC (must be < 0.65); still needed as a train/test shift audit

---

## M2 — Offline Baseline and Final-Ready Submission Pipeline

**Re-scoped:** Because there is no safe/available option to upload a baseline CSV, M2 no longer
means "first accepted platform submission." It means: build a local training + inference + validator
pipeline that can produce a final-ready CSV later, and get at least one nontrivial offline baseline
on the frozen folds.

- [ ] B0: constant predictor (predict all 0), assert BA == 0.500 exactly
- [ ] B1: shortcut baseline — re-scope for failed canonicalization
  - [ ] Option A: skip canonical B1 and record as invalid because canonicalization gate failed
  - [ ] Option B: raw-frame central asymmetry diagnostic only, clearly marked "not selection metric"
- [ ] `src/features.py` — C5 feature set (asymmetry, shadow containment, shadow elongation, radial profile, circularity, sun-elevation proxy, sin/cos az)
- [ ] B2: LightGBM on physical/image features, 5-fold CV, feature importances saved
  - [!] Do not include raw azimuth as a feature unless clearly labelled as a leak diagnostic.
- [/] B3: raw-frame CNN, no canonicalization, no azimuth conditioning, no geometric augmentation
  - [x] `configs/exp/raw_convnext_baseline.yaml` created
  - [x] Fast run completed on GPU, but result is not healthy: OOF BA @ plateau `0.5000`, val BA @ 0.5 stuck at `0.5000`
  - [ ] Diagnose why ConvNeXt fast run is collapsed before launching 3-seed full run

### OPS tasks
- [/] `src/infer.py` — run-based test prediction averaging
  - [x] First implementation added
  - [ ] Parity test still needed (100 training images, val path vs infer path, max |delta_p| < 1e-6)
- [/] `src/submit.py` — CSV builder (join on image_id, never position), validator
  - [x] Strict validator added
  - [x] CSV builder added
  - [ ] Label-inversion check still needed before final submission use
  - [ ] Sanity report still needed before final submission use
- [x] `src/submit.py` validator — A7 checks implemented (BOM, CR, line count, header, .png, labels as int, no dups, order, no index col)
- [x] Broken-file test suite added for submission validator
- [x] `make submit` target exists in Makefile
- [ ] Run offline `src.submit` only after a usable baseline exists; do not upload the produced CSV
- [/] Update `PROGRESS.md` M2 entry
  - [x] Implementation state recorded
  - [ ] Manual run results from 13 Sep should be appended

### Manual actions (YOU must do these)
- [x] MANUAL: Run `python -m src.dataset --mode cache`
  - Result: cache written to `data/processed`
- [x] MANUAL: Run `python -m pytest -q`
  - Result: `17 passed in 10.24s`
- [x] MANUAL: Run `python -m src.train --config configs/debug.yaml`
  - Result: CPU debug run completed; OOF BA @ plateau `0.5738`, but val BA @ 0.5 stayed `0.5000`
- [x] MANUAL: Run `python -m src.train --config configs/exp/raw_convnext_baseline.yaml --fast`
  - Result: GPU fast run completed; OOF BA @ plateau `0.5000`; val BA @ 0.5 stayed `0.5000`
- [ ] MANUAL: Do **not** upload any baseline CSV. Platform upload is reserved for the final go/no-go file.
- [ ] MANUAL: Do not launch `--seeds "0 1 2"` until the collapsed fast run is diagnosed.

### Immediate next engineering tasks
- [ ] Inspect the fast-run manifest/checkpoints and prediction distribution for collapse (all probabilities near constant? wrong class weighting? threshold artifact?)
- [ ] Add a tiny overfit test: train on a very small subset and confirm the model can drive training loss down and predict both classes
- [ ] Add B0 constant predictor script/table entry
- [ ] Implement a fast non-deep baseline (logistic/ridge or LightGBM on raw intensity summary features without azimuth) to check whether pixels carry any signal
- [ ] Fix/extend `src.train` as needed before full 3-seed raw ConvNeXt run

---

## M3 — Post-Pivot Strong Baselines and Diagnostics

**Re-scoped:** The original "Canonical Reference Model" is blocked by M1. M3 should not launch a
canonical E1 run unless a new calibration method passes the original gate. Current priority is to
make the raw-frame path learn, measure leakage/shift honestly, and build robustness diagnostics.

### MODEL tasks
- [/] `src/transforms.py` — raw-frame-safe operators first; canonical-frame operators deferred
  - [ ] Unit tests: rotation round-trips, double-negation identity, flip az updates vs synthetic re-renders, canonical vflip asymmetry reversal
  - [ ] Visual debugger: scripts/viz_augment.py (16-variant grid annotated with az + label); only after geometry operators return
  - [ ] Lint test: build fails if any library flip/rotation appears outside transforms.py
- [/] `src/dataset.py` — PareidoliaDataset (cache, normalization, raw-frame path); canonical policy support deferred
- [/] `src/models.py` — timm backbone (in_chans=1, drop_path_rate, conditioning none); FiLM/conditioning deferred
- [/] `src/losses.py` — class-weighted CE with label smoothing; consistency losses deferred
- [/] `src/train.py` — bf16 AMP, channels_last, cosine schedule, grad clip 1.0, early stopping, per-epoch checkpoint + best, writes oof.npy + run_manifest.json
  - [ ] Add warmup, EMA, and auto-resume
- [ ] configs/exp/e1_convnext_t_canonical_safe.yaml — deferred until calibration is repaired
- [ ] configs/exp/e2_convnext_t_nocanon.yaml — replace with stronger raw-frame baseline configs after fast-run diagnosis
- [ ] Run E1 canonical: blocked unless a new calibration gate passes
- [ ] Run stronger raw-frame CV candidate after M2 fast-run collapse is fixed
- [ ] scripts/compare_runs.py — sorted table with seed std devs
- [ ] Grad-CAM utility in src/viz.py (basic, for sanity check)

### OPS tasks
- [ ] Artifact contract: artifacts/<date>_<name>_ba<x.xxxx>/ structure
- [ ] artifacts/registry.json created
- [ ] `src/train.py` kill-and-resume test (interrupt mid-epoch, restart, confirm it picks up)
- [ ] Update `PROGRESS.md` M3 entry

### Manual actions (YOU must do these)
- [ ] MANUAL: Review augmentation debugger grid BEFORE launching any geometry/label-flip training
- [ ] MANUAL: Review Grad-CAM overlays on 30 correct + 30 wrong validation images — are maps on the feature or corner artifacts?
- [ ] MANUAL: M3 gate review — confirm the selected post-pivot candidate beats B0 and the raw baselines by meaningful margin; canonical E1 comparison is deferred

---

## M4 — Portfolio & Robustness

### MODEL tasks
- [ ] configs/exp/e3_*.yaml x4 — label-flip ablation grid (vflip p in {0, 0.2} x negation p in {0, 0.15})
- [ ] Run E3 fast screens (1 fold, 15 epochs) x4; run winner full CV x3 seeds
- [ ] configs/exp/e4_physics_stack.yaml — [I, dI/ds, dI/ds_perp] input
- [ ] Run E4: 1-channel vs physics-stack (2 full runs)
- [ ] configs/exp/e5_effnetv2s.yaml, e5_swin_t.yaml — backbone diversity
- [ ] Run E5: 2 backbones (EfficientNetV2-S + Swin-T or MaxViT-T)
- [ ] configs/exp/e6_negcon_*.yaml — negation-consistency lambda in {0, 0.1, 0.5}
- [ ] `src/losses.py` — negation-consistency loss implemented
- [ ] Run E6: 3 fast screens; winner full CV
- [ ] configs/exp/e7_rawframe_film.yaml — raw-frame FiLM member
- [ ] `src/models.py` — FiLM wrapper for raw-frame conditioning
- [ ] Run E7: 1-2 full runs

### DATA tasks
- [ ] `src/robustness.py` — shortcut audit, inversion stress, azimuth-shift test, shadow occlusion, centre occlusion, reliability diagram, slice report, shadow-mass fraction (CAM)
- [ ] `make robustness` target
- [ ] Run make robustness on every finished candidate run
- [ ] Top-200 highest-loss OOF image grid generated -> reports/
- [ ] Per-sub-type accuracy table written

### OPS tasks
- [ ] `src/ensemble.py` skeleton — OOF matrix, fold-hash matching, mean-of-logits
- [ ] Update `PROGRESS.md` M4 entry

### Manual actions (YOU must do these)
- [ ] MANUAL: Look at the top-200 error grid and write the sub-type breakdown
- [ ] MANUAL: M4 gate review — SELECT the 3-4 diverse members; write one sentence justifying each; approve the selection
- [ ] MANUAL: Review Grad-CAM overlays (K6 shadow-mass fraction) for each candidate
- [ ] MANUAL: Decide on any inconclusive experiments (CI includes zero) — adopt or reject

---

## M5 — Ensemble & Threshold Freeze

### MODEL tasks
- [ ] `src/ensemble.py` complete — OOF matrix with fold-hash check, mean of logits, optional weighted average
- [ ] Evaluate each TTA view on OOF BA before adopting
- [ ] `src/infer.py` — TTA policy implemented (hflip, ±5-10deg jitter, negation contributing 1-p)
- [ ] F7 plateau threshold sweep on final ensemble OOF
- [ ] Check per-fold spread <= 0.10 and |t* - pi1| <= 0.10; temperature-scale if needed
- [ ] Fill frozen.threshold in config only after final model/ensemble is selected; do not freeze from collapsed/debug runs

### OPS tasks
- [ ] Written selection paragraph committed to git
- [ ] Commit tagged (e.g. ensemble-frozen)
- [ ] `make reproduce` kicked off from a fresh clone (overnight)
- [ ] Update `PROGRESS.md` M5 entry

### Manual actions (YOU must do these)
- [ ] MANUAL: Selection meeting — approve the final ensemble spec, TTA policy, and threshold
- [ ] MANUAL: Declare model-code freeze (no more changes to src/ model code after Fri 18 Sep ~17:00)

---

## M6 — Final Submission

### OPS tasks
- [ ] Confirm make reproduce result: OOF BA within 0.2 pt of recorded value (K11)
- [ ] Run final inference with TTA; save test_probs_<timestamp>.npy
- [ ] Run make submit: inference -> validate -> label-inversion check -> sanity report
- [ ] Sanity report review: class balance vs pi1, shortcut agreement, histogram bimodal
- [ ] Generate spot-check grid: 20 predicted-Depth + 20 predicted-Rise test images with azimuths
- [ ] Update `PROGRESS.md` M6 entry (final entry)

### Manual actions (YOU must do these)
- [ ] MANUAL: Review the spot-check image grid (40 images)
- [ ] MANUAL: Final go/no-go decision — yours, always explicit
- [ ] MANUAL: Upload the final CSV to the competition platform; screenshot the acceptance
  - [!] This is the only planned platform upload. There is no baseline/dry-run upload.
- [ ] MANUAL: Back up rollback file and test_probs_*.npy
- [ ] MANUAL: Tag the producing commit sub-v1 (or sub-vN)

---

## M7 — App: Core Prediction & Explainability

### APP tasks
- [ ] app/main.py — FastAPI app with lifespan model loader
- [ ] GET /health endpoint
- [ ] GET /model-info endpoint
- [ ] POST /predict endpoint (multipart: image PNG + azimuth float)
- [ ] POST /explain endpoint (Grad-CAM overlay mapped back via decanonicalize)
- [ ] src/viz.py — Grad-CAM (HiResCAM for ConvNeXt), overlay at ~40% alpha, shadow-mass fraction
- [ ] Streamlit app (app/streamlit_app.py): upload PNG, set azimuth slider, show prediction + canonical view + CAM overlay
- [ ] F13 artifact registry: app resolves production pointer from artifacts/registry.json
- [ ] app/Dockerfile + docker-compose.yml (CPU-only torch)
- [ ] Parity check: app predictions agree with make infer to 1e-6 on 5 images
- [ ] Update `PROGRESS.md` M7 entry

### Manual actions (YOU must do these)
- [ ] MANUAL: Stranger test — run the app on a clean machine with no project environment; confirm it works end-to-end

---

## M8 — App: Sun Simulator

### APP tasks
- [ ] POST /simulate endpoint: 72 azimuths (5deg steps), one batched forward pass, returns {azimuths[72], p_rise[72], std, canonical_thumbs_b64[]}
- [ ] Streamlit Mode A: fixed image, azimuth swept, slider, polar plot, prediction transitions shown
- [ ] Streamlit Mode B: image rotated + azimuth updated in lockstep, std displayed, panels labelled
- [ ] 3-4 one-click presets (fresh crater, boulder, degraded ambiguous case, mound)
- [ ] Preset loads in <= 5s on CPU
- [ ] Mode B std <= 0.02 on canonical-model presets verified
- [ ] docker compose up tested on a clean machine
- [ ] Update `PROGRESS.md` M8 entry

### Manual actions (YOU must do these)
- [ ] MANUAL: Record 3-minute demo video showing the illusion (Mode A flip) and invariance (Mode B)
- [ ] MANUAL: Stranger test on Docker

---

## M9 — Reproducibility, Docs & Hard Stop

### OPS tasks
- [ ] src/reproduce.py / make reproduce target: full end-to-end from fresh clone
- [ ] README.md: one-command reproduction, project overview, setup instructions
- [ ] docs/pareidolia_playbook.md (model card): update with final metrics, ablation table, robustness table
- [ ] Technical report (if Q6 requires it)
- [ ] PROGRESS.md final entry
- [ ] Confirm main branch is green, no uncommitted changes, protected files intact

### Manual actions (YOU must do these)
- [ ] MANUAL: Read and approve README / model card before hard stop
- [ ] MANUAL: Hard stop — 18:00 local Mon 21 Sep. No new submissions after M6 unless a verified bug was found.
