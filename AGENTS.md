# AGENTS.md — Pareidolia

Operating manual for AI coding agents (Claude Code, Codex, Cursor, Copilot, and others) and for humans working in this repository. Read the whole file before your first change; it is short on purpose.

Where things live:

- Product scope and acceptance criteria → `docs/PRD.md`
- Current phase, milestones, deadlines, risk register → `docs/Roadmap.md`
- Domain procedures (calibration, canonicalization, augmentation algebra, evaluation, thresholds, submission) → `.claude/skills/pareidolia-lunar-terrain/SKILL.md` — **read it before touching anything under `src/`**
- Full rationale and derivations → `docs/pareidolia_playbook.md`

---

## 1. The project in one paragraph

We classify 256×256 grayscale lunar tiles as **Depth (0)** or **Rise (1)** for the Pareidolia Paradox competition, scored by Balanced Accuracy on 2,000 hidden-label images, with a hard internal stop of Mon 21 Sep 2026, 23:59 IST (deadline extended). We also ship a small demo app. The pipeline: calibrate the azimuth convention → rotate every image so the sun is at the top → train a CNN ensemble with physics-valid augmentations on frozen group-aware folds → pick the BA threshold on out-of-fold predictions → build a validated CSV.

## 2. The fact that overrides your defaults

**The label depends on the image and `sun_azimuth_angle` together.** A crater rotated 180° with its azimuth unchanged looks exactly like a mound. Therefore:

- Standard flips and rotations are **label noise** unless they also update the azimuth (or flip the label, where the physics says so).
- Azimuth is a **transformation parameter** first and a feature second.
- A shadow-polarity shortcut scores well on easy data and may be punished on the hidden set, so robustness metrics matter as much as the headline score.
- Cross-validation is the **only** signal (test labels are hidden), so the folds and the metric are sacred.

If you are about to write code that looks like ordinary image classification, stop and check the SKILL first.

## 3. Repository map

```
pareidolia/
  AGENTS.md                 this file
  CLAUDE.md                 one line: @AGENTS.md   (for Claude Code)
  Makefile                  the command interface (see §4)
  requirements.txt          pinned with ==
  configs/
    config.yaml             base config; its `frozen:` block is PROTECTED
    debug.yaml              tiny config for smoke tests
    exp/                    one YAML per experiment
  data/
    raw/                    gitignored
    processed/              gitignored cached .npy arrays + index.json
    folds.csv               PROTECTED — generated once, committed, never regenerated
  src/
    utils.py                set_seed, paths, logging
    dataset.py              PROTECTED — loading, cache, Dataset class
    canonical.py            PROTECTED — calibration, canonicalize, decanonicalize
    transforms.py           PROTECTED — the only place geometry and label flips happen
    features.py             physical features for the LightGBM baseline
    models.py               timm backbones, FiLM wrapper, heads
    losses.py               weighted CE, consistency losses
    metrics.py              PROTECTED — BA, apply_threshold, sweeps, slices, bootstrap
    train.py                training loop, train_cv, resume
    robustness.py           shortcut audit, stress tests, slice report
    ensemble.py             OOF matrix, averaging, weights
    infer.py                inference + TTA (imports preprocessing, never reimplements it)
    submit.py               PROTECTED — builder, validator, inversion check
    reproduce.py            end-to-end reproduction
    viz.py                  standard figures
  scripts/                  compare_runs.py, viz_augment.py, one-off utilities
  tests/                    pytest; fixtures/synthetic.py renders known domes and pits
  notebooks/                EDA only — never imported by production code
  reports/                  figures, robustness reports (tracked, small files only)
  experiments/              gitignored — runs, checkpoints, OOF arrays, manifests
  artifacts/                gitignored except registry.json (PROTECTED)
  submissions/              gitignored — synced to shared storage
  app/                      FastAPI backend + Streamlit frontend, Dockerfile, compose
  docs/                     PRD.md, Roadmap.md, pareidolia_playbook.md, rules.md
  .claude/skills/pareidolia-lunar-terrain/SKILL.md
```

## 4. Setup and commands

These `make` targets are the interface. If a target doesn't exist yet, the Roadmap says which day it's built; create it to this spec rather than inventing a different command.

| Command | What it does |
|---|---|
| `make setup` | Create `.venv` (Python 3.11), install pinned requirements |
| `make cache` | Verify data integrity, build uint8 caches and `index.json`, compute normalization stats |
| `make calibrate` | Fit (δ, s, R), write the diagnostics and figures; proposes the config values (a human commits them) |
| `make folds` | One-time fold generation; refuses to run if `data/folds.csv` exists |
| `make test` | `pytest -q` — must pass before any PR |
| `make lint` | `ruff check` + `ruff format --check` |
| `python -m src.train --config configs/debug.yaml` | Smoke test; must finish in under 5 minutes |
| `make fast CONFIG=configs/exp/x.yaml` | Exploration run: 1 fold, 15 epochs |
| `make train CONFIG=configs/exp/x.yaml SEEDS="0 1 2"` | Full 5-fold CV run(s) with manifest and OOF |
| `make robustness RUN=<run_id>` | Shortcut audit, stress tests, slice report → `reports/robustness/<run_id>.md` |
| `python scripts/compare_runs.py` | Sorted table of runs with seed standard deviations (morning standup artifact) |
| `python scripts/viz_augment.py --policy <name> --image-id <id>` | 16-variant augmentation grid annotated with azimuth and label |
| `make ensemble SPEC=configs/ensemble.yaml` | Build and evaluate an ensemble from OOF arrays with matching fold hashes |
| `make infer ARTIFACT=<dir>` | Test probabilities with TTA → `test_probs_<timestamp>.npy` |
| `make submit ARTIFACT=<dir>` | Build CSV → validator → inversion check → sanity report |
| `make validate FILE=<csv>` | Validator only |
| `make app` | `docker compose up` for API + UI |
| `make reproduce` | Clean end-to-end reproduction of the final submission |

## 5. Hard rules

### Never

1. **Never use a library flip, rotation, transpose, or auto-augment policy on images.** That includes torchvision, albumentations, kornia, RandAugment/TrivialAugment, and `timm.data.create_transform(is_training=True)` (which flips by default). All geometry goes through `src/transforms.py`, which updates azimuth and label together.
2. Never feed raw azimuth degrees to a model or take an arithmetic mean of angles. Use (sin, cos) and circular statistics.
3. Never regenerate, edit, filter, or reorder `data/folds.csv`, and never evaluate on any other split.
4. Never compute or report a score outside `src/metrics.py`. Never select models on accuracy, F1, or AUC; AUC is a diagnostic only.
5. Never tune anything on test-set predictions. Before the final run, test images may be used only for the label-free audits in SKILL §7 and, if rules question Q3 allows, the transductive methods in SKILL §6 and §9 and pseudo-labelling (Roadmap E9).
6. Never recompute the threshold at submission time; read it from the frozen config.
7. Never reimplement preprocessing in `infer.py`, the app, or a notebook; import it from `src/`.
8. Never write a submission file except through `src/submit.py`, which runs the validator and the label-inversion check internally.
9. Never join predictions to IDs by position; join on `image_id` read as `str`. Never strip `.png`.
10. Never derive the class mapping from data (`LabelEncoder`, `sorted(unique)`). It is `{0: depth, 1: rise}` in the frozen config, asserted at load time.
11. Never use ImageNet mean/std; use the lunar statistics from config.
12. Never fill rotated corners with reflect or replicate padding, in canonicalization or in raw-frame rotation. Mirrored terrain is relief-inverted (SKILL §3). Use the √2-zoom warp, which needs no padding.
13. Never auto-delete high-loss or "mislabelled-looking" training images.
14. Never claim an improvement under 1 BA point from a single seed.
15. Never hand-label test images or hand-edit test predictions.
16. Never upload to the competition platform, email the organizers, change the registry `production` pointer, or force-push `main`. Those are human actions.
17. Never commit data, checkpoints, artifacts (other than `registry.json`), submission files, or secrets.

### Always

1. Call `set_seed(seed)` at the top of every entry point.
2. Give every experiment its own config in `configs/exp/`, a tracker run, and a `run_manifest.json` with git SHA and `folds_sha256`.
3. Change exactly one thing relative to a named parent run.
4. Save OOF and test probabilities as timestamped `.npy` files before thresholding; never overwrite them.
5. Checkpoint every epoch to persistent storage; training must resume automatically.
6. Look at pictures: the augmentation grid before training with a new policy, the worst-error grid after.
7. Report worst-slice BA and the robustness metrics next to OOF BA.
8. Threshold only through `metrics.apply_threshold` (predict 1 iff p ≥ t) so every path uses the same comparison.
9. Run `make test` before opening a PR, and add a test whenever you touch a protected file or fix a bug.

### Protected files

`data/folds.csv`, `src/dataset.py`, `src/canonical.py`, `src/transforms.py`, `src/metrics.py`, `src/submit.py`, the `frozen:` block of `configs/config.yaml`, and `artifacts/registry.json`.

Changing any of them requires a PR, a test covering the change, and explicit human approval. If a change alters behaviour, the PR must list which existing runs it invalidates.

## 6. Conventions

### 6.1 Angles and axes — the sign-error firewall

Most silent bugs in this project will be sign errors. The canonical definitions live in the `src/canonical.py` module docstring and are enforced by tests; this is a summary.

- Arrays are `(row, col)`, origin top-left, uint8 0–255 until normalization.
- Image-plane angles θ are degrees **counter-clockwise from +col (pointing right), with up = 90°**, as displayed. From a displacement: `θ = atan2(-Δrow, Δcol)`.
- `θ_sun` is the direction **toward** the sun. Light travels toward `θ_sun + 180°`.
- Calibration: `θ_sun = (s·az + δ) mod 360`, with `s ∈ {+1, −1}`. Inverse: `az = s·(θ_sun − δ) mod 360`.
- Canonical frame: sun at the top (`θ_sun = 90°`).
- `cv2.getRotationMatrix2D(center, α, scale)` with α > 0 rotates content counter-clockwise, so θ → θ + α and az → az + s·α.
- Horizontal flip (reverse columns): θ → 180° − θ. Vertical flip (reverse rows): θ → −θ.
- Negation: `255 − img` on uint8.
- `p` always means P(Rise).

### 6.2 Code style

Python 3.11; type hints on public functions; `ruff` for lint and format (line length 100); docstrings state array shapes and dtypes; `np.float32` throughout; configuration via OmegaConf/Hydra; no global state except the model loaded in the API lifespan handler; notebooks never contain production logic.

### 6.3 Naming

- Run IDs: `<YYYYMMDD-HHMM>_<backbone>_<tag>_s<seed>`, e.g. `20260914-2210_convnext_t_labelflip_s1`.
- OOF: `experiments/<run_id>/oof.npy`, float32, length 7,854, in `index.json` train order.
- Artifacts: `artifacts/<YYYYMMDD>_<name>_ba<0.xxxx>/`.
- Submissions: `submissions/sub_<YYYYMMDD-HHMM>_<artifact_id>.csv`; the producing commit is tagged `sub-vN`.

## 7. How to work a task

### 7.1 Before you start

1. Check `docs/Roadmap.md` for the current phase and the last passed milestone. Don't build on work whose gate hasn't passed; for example, no backbone sweeps before M3.
2. Read the SKILL section for the area you're touching.
3. State your plan in two or three lines, then make the smallest diff that achieves it.

### 7.2 Definition of done by task type

| Task type | Done when |
|---|---|
| Change to a protected module | Tests added and passing; visual check where relevant (augmentation grid, canonical mean images); PR approved by a human; invalidated runs listed |
| New experiment | Config committed; manifest and tracker run exist; ≥ 3 seeds if claiming a win; report filled in (template below); explicit verdict |
| Model or ensemble candidate | OOF saved; `make robustness` run; slice report attached |
| Submission | Produced by `make submit`; validator and inversion check pass; sanity report reviewed; human uploads and tags |
| App feature | Resolves the artifact via the registry; no duplicated preprocessing; works inside Docker; screenshot attached |
| Bug fix | Regression test that fails before the fix and passes after |

### 7.3 Experiment report template

```
### EXP-<id>: <one-line hypothesis>
Parent: <run_id>        Change: <exactly one thing>
Config: configs/exp/<file>.yaml · git <sha> · folds <sha256[:8]> · seeds 0,1,2
OOF BA @ plateau t:  <mean ± std>   (parent <mean ± std>)
Δ vs parent: <value>  [group-level paired bootstrap 95% CI: <lo>, <hi>]
AUC: <v>   t*: <v>  (π₁ = <v>, per-fold spread <v>)
Shortcut-wrong BA: <v>   Inversion-stress BA: <v>   Az-shift Δ: <v>   CAM shadow-mass: <v>
Worst slice: <slice name> = <v>
Verdict: ADOPT / REJECT / INCONCLUSIVE (CI includes 0)
```

## 8. Testing

These tests are mandatory, because a silent error in any of them costs the competition:

- **Transforms:** rotate by θ then −θ restores the azimuth; rotate-and-update then canonicalize equals canonicalize; double negation is the identity; horizontal and vertical flip azimuth updates match re-rendered synthetic images; a canonical vertical flip reverses the top-minus-bottom asymmetry.
- **Calibration:** recovers a known (δ, s) from synthetic renders; the rejected handedness has low R.
- **Metrics:** BA and plateau threshold against hand-computed examples; constant predictor gives exactly 0.5.
- **Submission:** the broken-file suite (index column, float and boolean labels, stripped `.png`, CRLF, BOM, missing row, duplicate, shuffled order, extra trailing newline) is rejected, and a correct file passes.
- **Parity:** validation and inference paths agree to 1e-6 on 100 images; ONNX or TorchScript exports agree with PyTorch to 1e-4.
- **Lint rule:** no library flip or rotation with non-zero probability outside `src/transforms.py`.

`tests/fixtures/synthetic.py` renders Lambertian domes and pits under a known sun direction; SKILL §12 has the reference implementation. Use it instead of real data wherever ground truth is needed.

## 9. Git and pull requests

- Branches: `feat/…`, `fix/…`, `exp/…`, `app/…`, `docs/…`.
- Commits: conventional style, e.g. `feat(transforms): add CanonicalVFlipLabelSwap`, `exp(convnext): label-flip p=0.2, 3 seeds`.
- PRs touching protected files need a human reviewer and must include test output and, for transforms or canonicalization, the relevant figure.
- Tag every submitted commit `sub-vN`. Keep `main` green.

## 10. Compute notes

- Load the uint8 caches into RAM once; never decode PNGs per epoch. Epochs over ~3 minutes on a GPU mean an I/O bug, not a hardware problem.
- Use bf16 autocast on Ampere or newer, `channels_last` for convnets, and `torch.compile` once the code is stable.
- `cudnn.benchmark=True` is fine for exploration. For `make reproduce`, use deterministic mode: `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Hosted notebooks time out; checkpoint every epoch to persistent storage and auto-resume.
- Use `make fast` for screening and reserve `make train` with three seeds for candidates that have already shown promise.

## 11. Stop and ask a human when…

- Calibration R < 0.4, the class modes are not roughly 180° apart, both handedness values give similar R, or δ drifts across azimuth octants.
- The canonical per-class mean images don't visibly separate.
- An azimuth-only logistic model scores above 0.55 BA (azimuth–label leakage).
- Grouped and random CV differ by more than 2 points, or any test image is a near-duplicate of a training image.
- Adversarial validation AUC exceeds 0.65.
- A change would touch a protected file or the frozen config block.
- Per-fold threshold spread exceeds 0.10, or the empirical threshold differs from π₁ by more than 0.10.
- A new submission disagrees with the previous best on more than about 15% of rows, or its predicted class balance looks implausible.
- An export's parity check fails.
- Anything would involve uploading, emailing organizers, external data, or training on test images (rules questions Q2 and Q3).
- It is after Mon 21 Sep 18:00 (model-code freeze) and the change touches model code.

## 12. Tool-specific notes

- **Claude Code** loads `CLAUDE.md`, not `AGENTS.md`. Keep a one-line `CLAUDE.md` at the repo root containing `@AGENTS.md` (or symlink it). The project skill at `.claude/skills/pareidolia-lunar-terrain/SKILL.md` loads automatically when relevant. See https://code.claude.com/docs/en/skills.
- **Other agents** generally read `AGENTS.md` directly. If your tool looks for skills elsewhere, symlink the skill folder rather than copying it, so there is one source of truth.
