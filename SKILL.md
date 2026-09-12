---
name: pareidolia-lunar-terrain
description: "Physics-aware procedures and guardrails for the Pareidolia Paradox lunar Depth-vs-Rise classifier (256x256 grayscale tiles plus sun_azimuth_angle, scored by Balanced Accuracy). Use this skill for ANY task in this repo that touches images, azimuth, calibration or canonicalization, augmentation or TTA (flips, rotations, negation), datasets or folds, losses, training, metrics, thresholds, ensembling, robustness tests, Grad-CAM, the submission CSV, or the Sun Simulator app, even when the request sounds like routine computer-vision work such as 'add augmentations', 'try a new backbone', 'speed up training' or 'make a submission'. Standard CV defaults are wrong for this problem, so consult this before writing or reviewing code."
---

# Pareidolia: illumination-aware lunar relief classification

This skill holds the procedures where ordinary computer-vision instincts produce silent, score-destroying bugs. Find the section for your task, follow it, and run the listed checks. §1 (conventions) applies to every line of code. Repo rules and commands are in `AGENTS.md`; rationale and derivations are in `docs/pareidolia_playbook.md` (section refs like [A3]).

## 0. The model of the problem

Orbital cameras look straight down; the sun is low and to one side. Relief shows up only as shading and shadow, and those depend on where the sun is.

| Class | Sun-facing side | Anti-sun side | Dark region |
|---|---|---|---|
| Rise (1) | bright | dark | cast outward, extends beyond the feature |
| Depth (0) | dark (near wall, rim shadow) | bright (far wall) | contained inside the rim |

Two operations invert apparent relief: **a 180° rotation with the azimuth held fixed**, and **photometric negation** (under linearized Lambertian shading, I ≈ a − b·(s·∇z), so the mirror surface −z renders as 2a − I). So the label is a function of (image, azimuth), never of the image alone. Every rule below follows from that.

## 1. Conventions (all code; tests enforce them)

- Arrays are `(row, col)`, origin top-left; images stay uint8 0–255 until normalization.
- Image-plane angle θ: degrees **counter-clockwise from +col (right), up = 90°**, as displayed. From a displacement: `θ = atan2(-Δrow, Δcol)`.
- `θ_sun` = direction **toward** the sun. Light travels toward `θ_sun + 180`.
- Calibration maps metadata to image angles: `θ_sun = (s·az + δ) mod 360`, `s ∈ {+1, −1}`.
- Canonical frame: sun at the top (`θ_sun = 90°`). There, Rise is bright-top/dark-bottom and Depth is dark-top/bright-bottom.
- `cv2.getRotationMatrix2D(center, α, scale)` with α > 0 rotates content CCW: θ → θ + α, so az → az + s·α.
- Horizontal flip (reverse columns): θ → 180 − θ. Vertical flip (reverse rows): θ → −θ. Rot180: θ → θ + 180.
- Negation: `255 − img` on uint8 (or `−x` after normalization). Not `img.max() − img`, which adds a per-image offset.
- Classes `{0: depth, 1: rise}` from the frozen config; `p` = P(Rise); predict 1 iff `p ≥ t`, via `metrics.apply_threshold` only.

```python
def az_to_img(az, delta, s):     return (s * az + delta) % 360.0      # metadata -> theta_sun
def img_to_az(theta, delta, s):  return (s * (theta - delta)) % 360.0 # theta_sun -> metadata
```

## 2. Calibrate the azimuth convention (Phase 1.5) — the most important experiment

Documentation can't tell you whether azimuth points toward the sun or along the light, whether images are north-up, or how world-clockwise maps to array coordinates. Fit it from pixels.

```python
import numpy as np

def dark_minus_bright_deg(img, frac=0.10):
    """Angle (convention §1) of centroid(darkest frac) - centroid(brightest frac)."""
    x = img.astype(np.float32)
    lo, hi = np.quantile(x, [frac, 1 - frac])
    r, c = np.indices(x.shape)
    dark, bright = x <= lo, x >= hi
    d_r = r[dark].mean() - r[bright].mean()
    d_c = c[dark].mean() - c[bright].mean()
    return np.degrees(np.arctan2(-d_r, d_c)) % 360.0

def circ_mean_R(deg):
    a = np.radians(deg)
    C, S = np.cos(a).mean(), np.sin(a).mean()
    return np.degrees(np.arctan2(S, C)) % 360.0, float(np.hypot(C, S))

def calibrate(images, az, labels):
    phi = np.array([dark_minus_bright_deg(im) for im in images])
    # Depth: dark side faces the sun -> phi ~ theta_sun.  Rise: dark side faces away -> phi ~ theta_sun + 180.
    toward = (phi + np.where(labels == 1, 180.0, 0.0)) % 360.0
    fits = {s: circ_mean_R(toward - s * az) for s in (+1, -1)}
    s = max(fits, key=lambda k: fits[k][1])
    return {"delta": fits[s][0], "s": s, "R": fits[s][1], "R_other_s": fits[-s][1]}
```

Then:

1. Plot the circular histogram of `(phi − s·az) mod 360` split by class. Expect two tight modes about 180° apart. This is the single most important figure in the project; save it to `reports/figures/`.
2. Report δ, s, R, `R_other_s`, per-class R, and the angle between class modes.
3. Refit per azimuth octant; δ must be stable. Drift means the convention varies within the dataset.
4. Interpret R: near 1 means shading polarity is almost deterministic given class and azimuth (the shortcut is strong, and the competition is about the residual). Near 0.4 means noisy polarity, and the deep model carries more of the load.
5. Write the proposed values to the frozen config via a PR.

Escalate to a human if R < 0.4, modes aren't about 180° apart, `R_other_s` is close to R (the azimuth range is too narrow to identify handedness), or δ drifts across octants. If tiles contain background gradients or several features, restrict the centroids to a central window. Fitting two parameters on labels leaks negligibly into CV.

## 3. Canonicalize (Phase 1.6, F2)

Default: **one affine warp combining rotation and a √2 zoom.** Every output pixel then samples the input's inscribed circle, so no pixel is synthesized at any angle, scale is identical for every image (feature size can't encode azimuth), and there is only one interpolation.

```python
import cv2
SQRT2 = 2 ** 0.5

def canonical_rotation_deg(az, delta, s):
    return (90.0 - az_to_img(az, delta, s)) % 360.0

def canonicalize(img, az, delta, s, jitter_deg=0.0):
    h, w = img.shape
    angle = canonical_rotation_deg(az, delta, s) + jitter_deg
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, SQRT2)
    # At sqrt(2) zoom no output pixel maps outside the tile; borderMode only affects the outermost bicubic taps.
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
```

`decanonicalize` uses `cv2.invertAffineTransform(M)`; overlays are valid only over the central region the warp sampled.

**Trade-off.** The view keeps the central ~181×181 px (about 50% of the tile) upsampled 1.41×. In the visual audit (Phase 1.7), check how often the dominant feature extends beyond ~90 px from the centre. If often, the ablation mode is native scale with corners filled by the image median; note that the corner pattern then encodes θ mod 90°, which can leak azimuth if azimuth correlates with the label.

**Never reflect-pad.** Mirroring across a tile edge that is perpendicular to the sun reverses the shading order along the sun axis, so the corners fill with relief-inverted ghosts: a mirrored dome reads as a pit. This is exactly the label-flip transform, applied to random corners without flipping the label.

**Gate (do not proceed until it passes).** Compute per-class mean images before and after canonicalization. Before: similar blurs. After: Depth mean dark-top/bright-bottom, Rise mean the reverse, with opposite-signed top-minus-bottom asymmetry. If they don't separate, calibration or conventions are wrong.

Cache canonical arrays for fixed-policy training. Rotation jitter (±10°, modelling azimuth error) should re-canonicalize from the raw tile with `jitter_deg`, which costs under a millisecond per image.

## 4. Add or review an augmentation (F3)

### 4.1 Canonical frame (default pipeline)

| Transform | Label | Use |
|---|---|---|
| Horizontal flip (mirror across the sun axis) | keep | p = 0.5 |
| Vertical flip (mirror across the perpendicular axis) | **flip** | p 0.15–0.25; ablate over 3 seeds |
| Photometric negation | **flip** | p 0.10–0.20; ablate over 3 seeds |
| Rot180 | **flip** | equals vflip ∘ hflip; redundant with the two above |
| Rotation jitter ≤ ±10° via re-canonicalization | keep | models azimuth measurement error |
| Rotation > 10° | — | **forbidden**: breaks the frame |
| Shift, scale 0.9–1.1, mild elastic, Gaussian noise, mild blur, CoarseDropout | keep | safe |
| Brightness, contrast, gamma (monotone increasing only) | keep | safe; never invert contrast implicitly |
| Shadow masking (fill darkest connected component with local median) | keep | anti-shortcut; ablate |
| Mixup / CutMix | ? | test only; blends across the class boundary |

### 4.2 Raw frame (the un-canonicalized FiLM member)

| Operation | Azimuth | Label |
|---|---|---|
| Rotate content CCW by α | az + s·α | keep |
| Horizontal flip | `img_to_az(180 − θ_sun)` | keep |
| Vertical flip | `img_to_az(−θ_sun)` | keep |
| Rot180 with azimuth held fixed | unchanged | **flip** |
| Negation with azimuth held fixed | unchanged | **flip** |
| Negation with az + 180 | az + 180 | keep (approx.) |
| Any other flip or rotation without the azimuth update | — | **invalid**: pure label noise |

The playbook's [A3] table marks "mirror across the perpendicular axis, azimuth unchanged" as invalid; that is true if the label is kept, and it becomes the label-flipping vflip above if the label is flipped.

### 4.3 Implementation rules

- Operators take and return `Sample(image, azimuth, label)`. Flip hard labels **before** label smoothing.
- Raw-frame rotations use the same √2-zoom warp as §3, applied to every raw-frame input (α = 0 included) so scale stays constant and no corner is ever padded.
- Albumentations only for photometric and mild spatial ops, with `rotate_limit=0` and all flip probabilities at 0. Never use `timm.data.create_transform(is_training=True)`: it flips by default.
- Every new operator needs a unit test against the synthetic renderer (§12) and a look at `scripts/viz_augment.py` output before any training run.
- Label-flip transforms are approximate: negation doesn't reproduce cast-shadow geometry (a crater's contained shadow becomes an implausible contained bright blob), and regolith isn't Lambertian (Hapke / Lunar-Lambert). Keep probabilities moderate and adopt only if grouped CV improves across 3 seeds.

## 5. Model inputs and conditioning (F4)

- `timm.create_model(name, pretrained=True, in_chans=1, num_classes=2, drop_path_rate=0.1)` folds the RGB stem correctly. Prefer IN-22k-pretrained variants.
- **Physics stack (3 channels), test it:** `[I, ∂I/∂s, ∂I/∂s⊥]`. In the canonical frame the sun axis is vertical, so ∂I/∂s is the vertical Scharr derivative (positive along the light, i.e. +row) and ∂I/∂s⊥ is the horizontal one. Standardize each channel with its own training statistics.
- Azimuth conditioning after canonicalization adds little; if used, FiLM (per-stage γ, β from an MLP over sin/cos). Late concatenation is a baseline only. For the raw-frame member, FiLM is the recommended mode.
- Optional auxiliary inputs: sun-elevation proxies (fraction of pixels below a low threshold, global std), which reconstruct the missing elevation column.
- **Leak check:** if a logistic regression on (sin, cos) alone scores > 0.55 BA, azimuth correlates with the label. Drop azimuth conditioning from canonical models; they don't need it.

Starting recipe (not tuned; values without a playbook source are marked †):

| Setting | Value |
|---|---|
| Backbone | `convnext_tiny.fb_in22k_ft_in1k`, 256 px |
| Optimizer | AdamW, lr 2e-4 †, weight decay 0.05, batch 64 † |
| Schedule | cosine, 2–3 warm-up epochs, 30 epochs (fast screen: 1 fold, 15 epochs) |
| Regularization | drop_path 0.1, label smoothing 0.05, grad clip 1.0, EMA decay 0.999 † |
| Early stopping | validation BA, patience 5 |
| Loss | CE with class weights `N / (2·N_c)` |
| Precision | bf16 autocast, `channels_last` |

## 6. Losses (C4)

- Base: class-weighted CE with label smoothing (or a balanced sampler).
- **Negation consistency** (the substantive one): `L_neg = mean((p(neg x) − (1 − p(x)))²)` or a symmetric KL on logits; λ 0.1–0.5, linear warm-up over 3 epochs. It may also run on unlabeled test images only if rules question Q3 allows it.
- **Rotation equivariance** `f(R_α x, az + s·α) = f(x, az)`: in the canonical pipeline both views canonicalize to the same pixels, so the loss is near zero by construction. Use it for the raw-frame FiLM member.
- Ablate detaching the clean view versus symmetric gradients. Consistency losses and label-flip augmentation overlap, so ablate them together, not only separately.

## 7. Evaluate honestly (F5, F16, F17)

- Use only `data/folds.csv` (5-fold `StratifiedGroupKFold`, groups = near-duplicate components, strata = label × azimuth bin). Assert its SHA-256 matches `frozen.folds_sha256` before training or ensembling.
- Select on OOF BA at the plateau threshold; log AUC to separate ranking problems from threshold problems.
- **Noise floor:** seed-to-seed σ is 0.3–0.8 pt. Adopt a change only with ≥ 3 seeds and a **group-level paired bootstrap** on OOF (resample `group_id`s, recompute both models' BA, report Δ with a 95% CI). A CI containing 0 means inconclusive.
- **Baselines in every comparison table:** B0 constant (exactly 0.5); B1 shortcut = canonicalize, `asym = mean(top half) − mean(bottom half)` of the central window (rows and cols 64–192), predict Rise if `asym > 0`; B2 LightGBM on the [C5] physical features; B3 naïve CNN on raw tiles with no azimuth and standard flips.
- **Robustness suite** (`make robustness`), for every candidate:
  - *Shortcut audit:* BA on OOF images where B1 is wrong. Near 0.5 means a polarity detector with extra steps.
  - *Inversion stress:* validation images under rot180-with-fixed-azimuth and negation, labels flipped. Near 0 means the model ignores azimuth. Canonical models trained with label-flip augmentation pass almost by construction, so this mostly catches bugs and azimuth-blind members.
  - *Azimuth shift:* rotate a validation fold by random α with az updated; ΔBA must be ≤ 1 pt.
  - *Occlusion:* mask the shadow component, then the feature centre; record ΔBA.
  - *Calibration:* reliability diagram; temperature-scale on OOF if poor.
  - *Slices:* azimuth octant, brightness quartile, contrast quartile, morphological sub-type, B1-correct vs B1-wrong. Select on the worst slice, not just the mean.
- **Label-free test audits (with the Q3-gated methods in §6 and §9, the only uses of test data before the final run):** train-vs-test azimuth histograms with a Kuiper test, adversarial-validation AUC, simple intensity statistics, cross-set near-duplicates, and the shortcut-agreement rate on test.
- Look at the top 200 highest-loss OOF images yourself, categorize them by sub-type, and never auto-delete them.

## 8. Threshold (F7)

On calibrated posteriors the BA-optimal threshold is **π₁**, the training prior of Rise [A6]. In practice, sweep on the final ensemble's OOF, take the centre of the longest plateau, and compare.

```python
from sklearn.metrics import balanced_accuracy_score

def apply_threshold(p, t):
    return (np.asarray(p) >= t).astype(int)

def plateau_threshold(y, p, grid=np.linspace(0.01, 0.99, 197), tol=0.001):
    ba = np.array([balanced_accuracy_score(y, apply_threshold(p, t)) for t in grid])
    ok = np.append(ba >= ba.max() - tol, False)
    best, start = (0, 0), None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return float((grid[best[0]] + grid[best[1] - 1]) / 2), grid, ba
```

Also compute per-fold optima. If their spread exceeds 0.10 or the plateau centre differs from π₁ by more than 0.10, temperature-scale and redo; if still unstable, fall back to π₁. Freeze the value in config. Never recompute it at submission time.

## 9. Ensemble, TTA, inference (F6)

- Ensemble only runs whose `folds_sha256` match. Prefer members with OOF correlation below ~0.95; a decorrelated LightGBM can add more than a fifth CNN.
- Start with a plain mean of logits. Use optimized weights only if a nested split shows > 0.5 pt gain; stacking only if nested CV beats averaging clearly.
- Canonical TTA set: identity, horizontal flip, ±5–10° re-canonicalization jitter, and negation contributing `1 − p` (in logit space, `−logit`). Adopt each view only if OOF BA doesn't drop.
- Test-time BN adaptation affects BatchNorm backbones only (ResNet, EfficientNet); ConvNeXt and Swin use LayerNorm. Simulate on a validation fold before adopting, and only if Q3 permits.
- The final model is the fold-ensemble. Full-data refits are extra averaged members at most.
- Save `test_probs_<timestamp>.npy` before thresholding. Inference imports the training preprocessing; a parity test on 100 training images must agree to 1e-6.

## 10. Build a submission (F8)

`make submit ARTIFACT=…` runs this sequence; don't reorder it:

1. Resolve the artifact through `artifacts/registry.json`; load calibration, normalization, threshold, and class map from it.
2. Run inference with the frozen TTA policy and save the probabilities.
3. Apply the frozen threshold, join on `image_id` (str) in `test_metadata.csv` order, and write with `index=False`, `lineterminator="\n"`, int labels.
4. Validate (code below).
5. Label-inversion check: 20 known training images through the exact production path must come back correct.
6. Sanity report: predicted class balance vs π₁, shortcut-agreement rate, agreement with the previous submission (investigate if it's below about 85%), and a probability histogram that should be bimodal rather than piled at the threshold.
7. A human views 20 predicted-Depth and 20 predicted-Rise test images with their azimuths, uploads, screenshots the acceptance, and tags the commit.

```python
import pandas as pd

def validate_submission(path, test_meta_path, n_expected=2000):
    raw = open(path, "rb").read()
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM present"
    assert b"\r" not in raw, "CR characters found (use LF line endings)"
    text = raw.decode("utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n"), "must end with exactly one newline"
    lines = text[:-1].split("\n")
    assert lines[0] == "image_id,label", f"bad header {lines[0]!r}"
    assert len(lines) == n_expected + 1, f"expected {n_expected + 1} lines, got {len(lines)}"
    rows = [ln.split(",") for ln in lines[1:]]
    assert all(len(r) == 2 for r in rows), "every row needs exactly 2 fields (index column?)"
    ids, labels = [r[0] for r in rows], [r[1] for r in rows]
    assert all(v in ("0", "1") for v in labels), "labels must be literal 0 or 1"
    assert all(i.endswith(".png") and i == i.strip() for i in ids), "image_id must keep .png, no spaces"
    assert len(set(ids)) == len(ids), "duplicate image_id"
    expected = pd.read_csv(test_meta_path, dtype={"image_id": str})["image_id"].tolist()
    assert set(ids) == set(expected), "image_id set differs from test_metadata.csv"
    assert ids == expected, "row order differs from test_metadata.csv"
    return path
```

## 11. Explainability and the Sun Simulator (F9, F10)

- Grad-CAM (or HiResCAM) on the canonical input; for ConvNeXt target `model.stages[-1]`; Swin needs a reshape transform. Map to the original orientation with `decanonicalize`; overlay at ~40% alpha with a perceptually uniform colormap.
- **Shadow-mass fraction:** CAM mass inside the Otsu shadow mask of the canonical image, averaged over validation. At equal BA, prefer the model with the lower value.
- **Mode A (the illusion):** fixed image, assumed azimuth swept over 72 values in one batched forward pass. Expected: two plateaus with transitions near the true azimuth ± 90°.
- **Mode B (equivariance check):** rotate the image by α and set az ← az + s·α. Expected: constant prediction; display the std. For canonical members this holds by construction, so std > ~0.02 points to a convention or corner bug. For the raw-frame member it tests learned equivariance. Label the panels accordingly.

## 12. Synthetic fixtures for tests

Real data has no ground-truth geometry; these renders do. Lambertian shading without cast shadows is enough to pin every sign convention.

```python
def render(kind, theta_sun_deg, elev_deg=25.0, sigma=28.0, height=18.0, size=256, cx=None, cy=None):
    """Gaussian dome ('dome') or pit ('pit') lit from theta_sun (convention §1). Returns uint8."""
    c = (size - 1) / 2.0
    cx, cy = (c if cx is None else cx), (c if cy is None else cy)
    rows, cols = np.indices((size, size), dtype=np.float64)
    x, y = cols - cx, -(rows - cy)                      # y up as displayed
    z = height * np.exp(-(x**2 + y**2) / (2 * sigma**2)) * (1 if kind == "dome" else -1)
    n = np.stack([x / sigma**2 * z, y / sigma**2 * z, np.ones_like(z)], -1)   # (-dz/dx, -dz/dy, 1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    t, e = np.radians(theta_sun_deg), np.radians(elev_deg)
    light = np.array([np.cos(e) * np.cos(t), np.cos(e) * np.sin(t), np.sin(e)])
    return np.round(np.clip(n @ light, 0, 1) * 255).astype(np.uint8)
```

Tests these fixtures must support (all verified against the code in this file):

1. `render("dome", 90)` has top-minus-bottom > 0; `render("pit", 90)` < 0.
2. `calibrate` recovers a known (δ, s) within 3° with R > 0.95 on 300 random renders; the other handedness gives R < 0.3.
3. After `canonicalize`, B1 classifies synthetic renders with > 98% accuracy.
4. Rotate by α with az updated, then canonicalize: MAE < 1 grey level on the central disc versus canonicalizing the original.
5. Horizontal and vertical flips with azimuth updates equal a fresh render under the updated azimuth (max diff ≤ 1).
6. A canonical vertical flip reverses the sign of the top-minus-bottom asymmetry.
7. `255 − render("dome")` correlates > 0.98 with `render("pit")` at gentle slopes (height 8, elevation 40°).
8. Reflect-padding a dome near the top edge under a top sun produces a ghost whose asymmetry sign is reversed (documents why §3 forbids it).

## 13. Red flags

| Observation | Likely cause | Action |
|---|---|---|
| Canonical class means look alike | Wrong δ or s, angle-convention mismatch, azimuth semantics differ | Stop; rerun §2 with the synthetic tests |
| Calibration R low or modes not ~180° apart | Weak shortcut, noisy or scrambled azimuth, images not north-up | Refit per octant; escalate |
| Azimuth-only model > 0.55 BA | Azimuth–label leakage | Remove azimuth conditioning; rely on canonicalization |
| Grouped CV well below random CV | Near-duplicate leakage | Trust grouped only; revisit earlier decisions |
| Adversarial AUC > 0.65 | Covariate shift | Inspect test-like training images; weight robustness in selection |
| Canonical on/off ablation gains little | Calibration error | Back to §2 |
| Shortcut-wrong BA ≈ 0.5 | Polarity detector | Label-flip aug, negation consistency, shadow masking |
| Azimuth-shift ΔBA > 1 pt | Canonicalization bug or leak (corners, scale) | Check §3 defaults |
| Per-fold threshold spread > 0.10 | Miscalibration or noise | Temperature scaling; fall back to π₁ |
| Submission far from expected class balance | Threshold bug or class inversion | Inversion check; inspect probabilities |
| Epoch > 3 min on GPU | I/O bound | Preload uint8 arrays |
| Train − validation accuracy > 8 pts | Overfitting | More regularization, earlier stopping |

## 14. Where to go deeper

Physics [A3]; azimuth semantics [A4]; organizer warning and strategy [A5]; BA maths [A6]; submission forensics [A7]; failure modes [B1–B14]; canonicalization and augmentation algebra [C1–C2]; conditioning [C3]; consistency losses [C4]; physical features [C5]; phase steps [Part D]; features F1–F15 [Part E]; stack [Part F]; checklists [G4], [G6].
