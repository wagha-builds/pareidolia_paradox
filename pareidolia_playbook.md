# The Pareidolia Paradox — Complete Team Playbook

**A start-to-finish engineering and research document for classifying lunar Depth vs. Rise terrain under variable solar illumination.**

Prepared: 6 September 2026 · Competition window: 1–21 September 2026 · **Days remaining: 15**

---

## Table of Contents

**PART A — Understanding the Problem**
- A1. Executive summary
- A2. Line-by-line deconstruction of the problem statement
- A3. The physics of topographic inversion (the actual core of the challenge)
- A4. What `sun_azimuth_angle` really is and why they gave it to you
- A5. Decoding the organizers' hidden warning
- A6. Balanced Accuracy: the mathematics and what it forces you to do
- A7. Submission format forensics

**PART B — The Pain Point Register**
- B1. Fourteen concrete failure modes, each with a diagnosis and a fix

**PART C — The Central Scientific Strategy**
- C1. Illumination canonicalization
- C2. The augmentation algebra (which transforms preserve, update, or flip the label)
- C3. Azimuth conditioning architectures
- C4. Equivariance and anti-equivariance losses
- C5. Physically-grounded feature engineering

**PART D — Phase-by-Phase Execution Plan**
- Phase 0 through Phase 9, each with numbered steps, owners, deliverables and exit criteria

**PART E — Core Application Features**
- Fifteen features, each with rationale and step-by-step implementation

**PART F — Complete Tech Stack**
- Core, recommended, and optional, each with what/why/how

**PART G — Resources, Risk Register, and Final Checklists**

---
---

# PART A — UNDERSTANDING THE PROBLEM

## A1. Executive Summary

Strip away the framing and this is a **binary image classification problem with 7,854 labelled training samples, 2,000 hidden-label evaluation samples, 256×256 single-channel inputs, one scalar covariate, and Balanced Accuracy as the metric.** On paper that is a two-hour task for anyone who has touched a CNN before.

It is not a two-hour task, and the reason is contained in exactly one sentence of the brief: *"A crater illuminated from one direction can appear like a depression, while the same formation under different lighting can resemble a raised surface."*

What this sentence means in machine learning terms is that **the mapping from pixels to labels is not a function.** The same 256×256 pixel array can legitimately correspond to label 0 or label 1 depending on the value of `sun_azimuth_angle`. There exist pairs of images in this problem space that are pixel-for-pixel nearly identical but carry opposite labels. Any model that consumes only pixels has a hard ceiling on its achievable accuracy, and that ceiling is determined by how much the dataset happens to conflate appearance with illumination geometry. The label is a function of **(image, azimuth)** jointly, and never of the image alone.

This single fact reorganizes the entire project. It means:

1. The azimuth is not an optional bonus feature to tack on for a marginal gain. It is a **required input to disambiguate a genuinely ambiguous signal.** Treating it as optional is the single most common way teams lose this competition.
2. Standard augmentation practice is actively dangerous here. `RandomRotation`, `RandomHorizontalFlip`, and `RandomVerticalFlip` — the three most reflexively applied augmentations in all of computer vision — each silently corrupt the physical relationship between the image and its recorded azimuth. Applying them naïvely teaches your model that illumination geometry is noise, which is precisely the opposite of what you need.
3. There is a **shortcut** available (shadow polarity) that will score well on a naïve validation split and may or may not survive contact with the hidden evaluation set. The organizers have explicitly and deliberately warned you about it. Deciding how much to trust the shortcut is the single biggest strategic decision your team will make.
4. Because evaluation labels are hidden and there is (per the brief) a single submission of predictions, **your local validation protocol is the only signal you will ever have.** Building a trustworthy cross-validation scheme is not housekeeping — it is the highest-leverage engineering work in the project, and it must happen before you train anything serious.

The winning recipe, stated in one paragraph: **calibrate the azimuth convention empirically against the pixel data; rotate every image into a canonical frame where the sun always arrives from the same direction; train an ensemble of pretrained CNN backbones on the canonicalized images with an azimuth-consistent augmentation policy and an equivariance-consistency loss; add a photometric-negation label-flip augmentation to double your effective dataset with physically valid counterfactuals; validate with group-aware stratified k-fold; tune the decision threshold on out-of-fold predictions to maximize Balanced Accuracy; apply azimuth-consistent test-time augmentation; and wrap all of it in an application whose flagship feature is an interactive sun-position simulator that visibly demonstrates your model's invariance to the very illusion the competition is named after.**

Everything below is the elaboration of that paragraph.

---

## A2. Line-by-Line Deconstruction of the Problem Statement

I am going to walk through every clause of the brief you supplied and extract the operational consequence of each. Nothing here is decoration; each item changes something you will actually do.

### "an online machine learning challenge focused on image classification and computer vision"

This tells you the deliverable is judged on a metric, not on novelty, elegance, or presentation. It also tells you this is **not** a detection or segmentation problem. You are not localizing craters within a scene, you are not drawing boxes, you are not producing masks. Each 256×256 image is presumed to contain **one dominant feature** whose nature you must name. This has a direct architectural consequence: you want a classification head over globally pooled features, and you should be suspicious of any impulse to reach for YOLO, U-Net, or Mask R-CNN. Much of the published lunar-crater literature is about detection, and if your team reads that literature uncritically they will waste three days building the wrong kind of model. Read the literature for its **domain insight** about illumination and morphology, not for its architectures.

### "classify 256 × 256 grayscale lunar surface images"

Four separate operational facts are packed in here.

**256×256** is a fixed, known, uniform input size. This is a gift — no aspect-ratio handling, no variable-size batching, no resize-vs-crop debate. It also sits conveniently between the 224×224 that most ImageNet backbones were pretrained at and the 384×384 that many fine-tune well at. Your default should be to train at native 256 (nearly all modern convolutional backbones are fully convolutional and handle 256 without modification), and to run one controlled experiment upscaling to 320 or 384, because subtle rim and shadow-edge structure sometimes benefits from the extra effective receptive-field resolution. Do not downscale to 224 to "match the pretrained model" — you would be discarding 23% of your pixels to satisfy a constraint that does not exist.

**Grayscale** means single-channel input. Every ImageNet-pretrained backbone expects three channels. You have three options and they are not equivalent. The naïve option is to replicate the single channel three times — this works, it is what most people do, and it wastes two-thirds of your first-layer computation on redundant copies. The clean option is to use `timm.create_model(..., in_chans=1)`, which automatically sums the pretrained RGB first-layer kernels into a single-channel kernel, preserving the pretrained filter responses exactly while using one-third the input bandwidth. **The clever option, and the one I recommend you test, is to build a purpose-designed 3-channel input where channel 0 is the raw normalized image, channel 1 is the directional derivative of the image along the estimated sun vector, and channel 2 is the directional derivative perpendicular to it.** This hands the network the physically meaningful quantity on a plate. I expand on this in Part C5.

**Lunar surface** tells you the domain: no atmosphere, therefore no atmospheric scattering, therefore **shadows are extremely dark and extremely sharp**. On Earth, a shadow is filled in by diffuse skylight and has a soft penumbra. On the Moon, a shadowed region receives light only from secondary reflection off nearby illuminated slopes, so shadowed pixels are near-black with a hard edge. This is enormously helpful — it means shadow regions are cleanly thresholdable, and simple morphological analysis of the shadow blob (its position relative to the feature, whether it is contained or cast outward, its aspect ratio) is genuinely informative. It also means there is **no colour or texture cue from vegetation, water, or built structures** — the only signal in the image is shading, shadow, albedo variation, and texture. Your model has less to latch onto than in natural-image classification, which is why transfer learning from ImageNet helps less than you might hope and why domain-specific feature engineering helps more than usual.

**Images**, plural and independent, with no stated scene grouping. This is a trap I return to in B10: even though the brief does not mention it, tiles cut from a small number of source orbital images are near-certain to contain near-duplicates and overlapping content. If two crops of the same crater land in different cross-validation folds, your CV score will be optimistically biased and you will make bad model-selection decisions all week. You must actively test for this.

### "into one of two categories: Class 0: Depth — Craters, holes, and surface depressions. Class 1: Rise — Mounds, hills, rocks, and boulders"

Note carefully that these are **not two classes, they are two superclasses**, and the sub-populations inside each are morphologically very different from one another.

Inside Class 0 (Depth): a fresh simple crater is a near-perfect circular bowl with a raised rim, a sharp interior shadow crescent, and possibly a bright ejecta blanket. A degraded crater is a shallow circular depression with soft edges and no rim. A "hole" or collapse pit (e.g. a lava tube skylight) is a steep-walled, often irregular, near-vertical shaft with an almost entirely black interior. A general surface depression may have no closed boundary at all.

Inside Class 1 (Rise): a boulder is a small, high-contrast object of a few dozen pixels with a long, thin, sharply-defined cast shadow extending away from it. A mound or dome is a large, smooth, low-relief swell with gentle shading and possibly no cast shadow at all. A hill or massif is a large steep-sided positive relief feature with a big triangular cast shadow. A rocky outcrop is a cluster of many small shadow-casting objects.

**The operational consequences of this are significant and most teams miss them.**

First, **the intra-class variance is much larger than a two-class problem suggests.** A boulder and a mound share the label "Rise" but look nothing alike; a boulder and a small fresh crater at the same scale may look more similar to each other than either does to the other members of its own class. This argues for a model with enough capacity to learn multi-modal class-conditional distributions, and it argues strongly against tiny models or classical feature-plus-SVM approaches as your primary method. It also means that if you have any capacity for it, **clustering your training images into morphological sub-types and checking per-sub-type accuracy is one of the most informative diagnostics you can run** — it will immediately show you that, say, your model is at 96% on fresh craters and 71% on degraded ones, which tells you exactly where to spend your remaining time.

Second, the two classes are **not symmetric in their shadow geometry**, and this is the single most useful piece of domain physics in the whole problem. For a depression, the shadow is **cast by the near rim into the interior of the feature**, so the shadow is *bounded by and contained within* the feature's own outline. For a rise, the shadow is **cast by the feature outward onto the surrounding flat terrain**, so the shadow *extends beyond* the feature's outline. This "containment" property is invariant to illumination direction in a way that raw shadow polarity is not, and I will show you how to turn it into an explicit feature in C5. If the organizers have adversarially perturbed shadow polarity to punish shortcut learners, containment geometry is likely to survive.

Third, **scale matters and you have not been told the ground sample distance.** A 256×256 tile might span 50 metres or 5 kilometres. If the dataset mixes scales, then the apparent size of a feature is not directly interpretable, and you should treat scale augmentation as legitimate. If the dataset is single-scale, apparent size becomes a real feature (boulders are small, craters span a wide range) and aggressive scale augmentation would destroy useful signal. **Determine this empirically in Phase 1** by examining whether feature sizes cluster into discrete bands.

### "The appearance of lunar terrain changes depending on the direction of sunlight... This phenomenon is known as topographic inversion."

This is the thesis statement of the competition and I treat it fully in A3 below. Note the vocabulary: <cite index="16-1">in astronomical and Earth imaging this is also called the crater illusion or dome illusion, an optical illusion which makes impact craters and other depressions appear raised as domes or mountains, believed to be caused by human accustomation to seeing light from overhead</cite>. <cite index="15-1">The literature also calls it the terrain reversal effect, and notes that both a 180° rotation of an image and a photometric negative of an image can remove (or induce) the illusion.</cite> Those two facts — rotation and negation both flip apparent relief — are not trivia. **They are the two augmentation operators at the heart of the solution I am proposing.** Search the literature under all four names: topographic inversion, relief inversion, terrain reversal effect, crater/dome illusion.

### "To account for this, each image is accompanied by a sun_azimuth_angle in the metadata. Participants are expected to use this information appropriately while developing their models."

Read the last five words again: **"use this information appropriately."** This is not filler. The organizers are telling you two things simultaneously: that using azimuth is expected (so a pixels-only model is a knowingly incomplete solution), and that there is an *appropriate* versus an *inappropriate* way to use it. The inappropriate way is to concatenate the raw degree value to your feature vector and hope the network figures it out — which fails for reasons I detail in A4 and C3. The appropriate way is to use it as a **geometric transformation parameter** that puts every image into a common illumination frame, and/or as a **conditioning signal** injected through sin/cos encoding and FiLM modulation. This clause is the organizers pointing directly at the intended solution.

### "7,854 training images / 2,000 evaluation images"

7,854 is a **small dataset by deep learning standards** and this constrains almost every downstream choice. With a 5-fold split you train on roughly 6,283 images per fold. At that scale:

- Training a large model from scratch is out of the question; you will overfit within a handful of epochs. **Transfer learning from ImageNet-pretrained weights is mandatory**, not optional.
- Model capacity should be modest. ConvNeXt-Tiny, EfficientNetV2-S, ResNet-50, Swin-Tiny — this is the right weight class. A ConvNeXt-Large will memorize your training set and teach you nothing.
- **Regularization must be aggressive**: heavy augmentation, weight decay in the 0.01–0.05 range for AdamW, stochastic depth (drop_path 0.1–0.2), label smoothing 0.05–0.1, and early stopping on a proper validation metric.
- **Cross-validation, not a single holdout.** A single 80/20 split gives you a validation set of ~1,570 images; the standard error on an accuracy estimate at that size is around ±1.2 percentage points, which is larger than the differences you will be trying to detect between candidate models. 5-fold CV gives you out-of-fold predictions for all 7,854 images and a far more stable estimate. Use it. **Repeated k-fold or multi-seed averaging is worth it if compute allows.**
- The whole training set fits comfortably in RAM (7,854 × 256 × 256 × 1 byte ≈ 515 MB as uint8). **Load the entire dataset into memory once as a single NumPy array.** Do not build a DataLoader that reads PNGs from disk every epoch; you will be I/O bound and your epochs will take five times longer than they need to. This single decision may buy you an extra 3–5× more experiments over the fortnight, which matters enormously at this deadline.

2,000 evaluation images is a **small test set**, which has a consequence people rarely think about: the **standard error on your final Balanced Accuracy is roughly ±1 percentage point.** With a balanced test set of 1,000 per class, the standard error on each class's recall is about 1.4% at 80% accuracy, giving a standard error on the average of about 1.0%. **This means differences of under about 1.5 points between two submissions are statistical noise.** Do not burn three days chasing a 0.4-point CV improvement. Spend that time on robustness to distribution shift, which has a much larger expected payoff. It also means that on the final leaderboard, luck plays a real role, and the correct strategy is to maximize *expected* score with a robust method rather than to squeeze the last drop out of a fragile one.

Note also that 2,000 unlabeled evaluation images is a substantial **unlabeled corpus, roughly 25% the size of your labelled set.** This is a resource. Legitimate uses include test-time augmentation, test-time batch-norm statistic adaptation, unsupervised consistency regularization on the test images (no labels needed — you enforce that predictions are stable under azimuth-consistent transforms), and confidence-thresholded pseudo-labelling. Check the competition rules for any prohibition on using test data, but in the absence of one, transductive methods are standard practice and can be worth a point or more.

### "train_metadata.csv containing image_id, sun_azimuth_angle, and label / test_metadata.csv containing image_id and sun_azimuth_angle"

Three columns and two columns respectively. The important structural observation is that **`sun_azimuth_angle` appears in both files.** It is therefore available at inference time, which is what makes conditioning on it legal and makes canonicalization possible for the test set. If it had been train-only you would have needed to estimate it from pixels. (You should build that estimator anyway as a cross-check — see Phase 1 Step 6.)

The equally important observation is what is **absent**. There is no `sun_elevation_angle` (or incidence angle). This is a real limitation. <cite index="10-1">Solar azimuth and incidence angles both critically affect image-based topographic interpretation; the solar incidence angle has a direct impact on image intensity via the interaction between incoming light and surface reflectance, and variations alter the observed brightness and reduce the reliability of topography cues.</cite> Because elevation is unknown, **shadow length is an uncontrolled nuisance variable.** A high sun gives short shadows and low contrast; a low sun gives long shadows and dramatic contrast. Your model must be robust to both. Two practical responses: include brightness/contrast jitter in your augmentation policy so the model does not key on absolute contrast, and consider **estimating a proxy for solar elevation from each image** (e.g. the fraction of pixels below a shadow threshold, or the global image contrast) and feeding it as an auxiliary conditioning feature. This is a genuine edge — most teams will not think to reconstruct the missing metadata column.

Also absent: any scene/source identifier, any geographic coordinate, any timestamp, any feature diameter. If a source-scene identifier existed it would solve your grouping problem for free. Since it does not, you must **recover the grouping structure yourself** via near-duplicate detection (Phase 1 Step 8).

### "The evaluation labels will remain hidden and will be used for final scoring."

No feedback loop. No public leaderboard to probe. **Your cross-validation score is your entire information channel about model quality.** Everything in Part D Phase 2 exists to make that channel trustworthy. If you get the validation protocol wrong, every subsequent decision you make for two weeks is made on corrupted evidence, and you will not find out until the results are published.

Corollary: **do not tune anything on the test set, and do not let anyone on the team develop the habit of "just checking" a submission's predicted class balance and adjusting until it looks right.** That is leaderboard-probing without a leaderboard, and it is how teams talk themselves into overfitting to a set they cannot see. The one exception, which is legitimate and which you should do, is a **distribution-shift audit**: comparing the *unlabeled* statistical properties of train and test (azimuth histograms, brightness distributions, adversarial-validation AUC) to detect whether the test set was drawn differently. That uses no labels and is sound methodology.

### "Participants must train a classification model on the provided training dataset"

Note "on the provided training dataset." Read your competition rules carefully for whether external data and pretrained weights are permitted. ImageNet-pretrained backbones are almost universally allowed and I am assuming they are here, but **if pretrained weights are disallowed your entire plan changes** — you would need heavier augmentation, self-supervised pretraining on the combined train+test images (SimCLR, MAE, or DINO on 9,854 unlabeled lunar images is very feasible and would be a strong differentiator), and smaller architectures. **Confirm this on Day 1.** It is the single highest-impact ambiguity in the brief.

### "generate predictions for all 2,000 evaluation images"

**All 2,000.** Not the ones you are confident about. There is no abstention option. Every row must carry a hard 0 or 1. This interacts with Balanced Accuracy in a specific way covered in A6: because you must guess on every image and because the metric averages per-class recall, your handling of low-confidence cases directly determines your score, and the optimal threshold is generally *not* 0.5.

### "The submission must be a single CSV file containing exactly: image_id,label"

Covered exhaustively in A7. The word "exactly" is doing work. Assume a strict automated parser with no fuzzy matching.

### "Submissions will be evaluated using Balanced Accuracy."

Covered in A6.

### "Participants are encouraged to focus on the underlying terrain rather than relying solely on visual shadow patterns."

Covered in A5. This is the most consequential sentence in the entire document after the topographic-inversion sentence, and it deserves its own section.

### "Competition Opens: 1 September 2026 ... Closes: 21 September 2026, 11:59 PM"

Today is 6 September. **You have 15 days, or approximately 15 working evenings if this is being done alongside other commitments.** The phase plan in Part D is calendarized against that. The most important scheduling principle: **produce a valid, submittable CSV by end of Day 2, not end of Day 15.** A mediocre submission in hand on Day 2 that you improve twelve times is infinitely better than an excellent model on Day 21 that fails format validation at 11:47 PM. Build the pipeline end-to-end first, optimize second.

Also note: the deadline is 11:59 PM on the 21st, and the timezone is not specified in the text you have. **Confirm the timezone on Day 1** and set your internal deadline to at least six hours earlier than the stated one. Submission portals go down; internet connections fail; files get corrupted on upload. Treat the real deadline as 21 September, 6:00 PM local.

---

## A3. The Physics of Topographic Inversion — In Full

Everything strategic in this document descends from this section, so I am going to derive it carefully rather than assert it.

### The imaging geometry

Orbital imagery of the Moon is captured **nadir-looking**: the camera points straight down at the surface. The Sun, meanwhile, is typically **low on the horizon** in the images that get used for terrain analysis, because low sun angles produce long shadows and strong shading that reveal topography. <cite index="17-1">Satellite photos of craters are taken from overhead, and typically a shadow is only cast inside the cavity when the sun's rays are nearly parallel to the surface; when light comes in from the horizon rather than from overhead, our perception of the scene is altered.</cite>

So: **camera from above, light from the side.** The `sun_azimuth_angle` tells you which side.

### Shading and shadow for a rise

Take a mound, hill, or boulder — a positive relief feature. Let sunlight arrive travelling in direction **s** (a horizontal unit vector in the image plane, pointing *from* the sun *toward* the scene).

The flank of the mound that faces back toward the sun has a surface normal tilted toward the light, so it receives a higher cosine of incidence and appears **bright**. The far flank is tilted away, receives grazing or no light, and appears **dark**. Beyond the far flank, the mound's own bulk blocks the light from reaching the ground, so a **cast shadow extends outward onto the surrounding terrain, away from the sun.**

**Signature of a rise: bright on the sun-facing side, dark on the anti-sun side, and the dark region extends *outside* the feature's footprint.**

### Shading and shadow for a depth

Now take a crater — a negative relief feature, a bowl.

The interior wall on the *sun-facing side of the rim* is the near wall. Its surface faces *away* from the sun (it is the inside of the bowl on the sun side, so its normal points away from the light and down into the bowl). Furthermore, the raised rim on the sun side physically occludes the light, throwing a shadow *down into the bowl*. So the near-side interior is **dark**.

The far interior wall — the one on the opposite side from the sun — faces back toward the sun and catches the light almost square-on. It appears **bright**. <cite index="13-1">As the CosmoQuest lunar explanation puts it, the light shines over the rim of the crater, illuminating the inside on the far side and casting shadows on the near side.</cite>

Crucially, the shadow is **bounded by the crater rim**. It cannot escape the bowl; it is a crescent contained within the feature's own outline.

**Signature of a depth: dark on the sun-facing side, bright on the anti-sun side, and the dark region is *contained within* the feature's footprint.**

### The inversion, stated precisely

Compare the two signatures. Along the axis of illumination, **the bright/dark ordering is exactly reversed between the two classes.** A rise is bright-then-dark going from sun-side to anti-sun-side; a depth is dark-then-bright.

Now consider what happens if you rotate a crater image by 180° without telling anyone. The dark region, previously on the sun-facing side, is now on the anti-sun side of the frame. The image now carries the *exact* signature of a rise. **A 180° rotation of a depth image is pixel-wise indistinguishable from a rise image — unless you also know that the azimuth has changed by 180°.**

This is the Pareidolia Paradox. <cite index="12-1">The illusion arises because our brains are used to perceiving images as lit from above, and one quick way to make an image "pop" into its correct relief is to rotate it until the light source appears to come from above.</cite> <cite index="11-1">In lunar-mapping training materials, the same pair of images illuminated from top-left versus bottom-right will read as craters to most people in one orientation and as hills and pimples in the other — the hills being an optical illusion created by the brain's trained preference for a specific illumination direction.</cite>

A convolutional network is subject to precisely the same illusion, for precisely the same reason: it learns a fixed spatial arrangement of light and dark as diagnostic, and that arrangement is only diagnostic conditional on the illumination direction.

### The second inversion operator: photometric negation

There is a second way to flip apparent relief, and it is less well known but equally important to you. <cite index="15-1">A negative of an image also removes (or induces) the terrain reversal illusion, without any rotation.</cite>

The reason is straightforward under a Lambertian shading model. Image intensity is approximately I ≈ ρ·(**n**·**l**) where **n** is the surface normal and **l** points toward the light. For a surface height field z(x,y) with small slopes, the normal is approximately (−∂z/∂x, −∂z/∂y, 1) normalized, so I ≈ a − b·(**s**·∇z) for constants a, b. Now consider the **mirror-relief surface** −z(x,y) — every crater becomes a mound of identical shape, every mound becomes a crater. Its intensity is I' ≈ a + b·(**s**·∇z) = 2a − I. That is exactly an intensity negation about the mean.

**Therefore: negating an image's intensity, while holding the azimuth fixed, produces the image of the topographically inverted surface — and thus flips the correct label.**

This gives you a way to synthesize label-flipped counterfactual training examples, which is extraordinarily valuable for teaching a model that appearance alone does not determine class. It is approximate — real cast shadows are not symmetric under negation (a crater's contained shadow becomes a "mound" with an implausibly contained bright region), surface reflectance is not perfectly Lambertian (the lunar regolith exhibits strong backscatter, better modelled by Hapke or Lunar-Lambert functions), and albedo variation is not inverted correctly. Use it with moderate probability, and validate empirically that it helps rather than assuming it. But it is physically grounded, it is cheap, and it directly attacks the exact failure mode the competition is built around.

### The three-way relationship you must internalize

| Operation on image | Operation on azimuth | Effect on label |
|---|---|---|
| Rotate by θ | Update azimuth by θ (correct sign) | **Unchanged** |
| Rotate by θ | Leave azimuth unchanged | **Flipped if θ = 180°**; invalid otherwise |
| Rotate by 180° | Leave azimuth unchanged | **Flipped** |
| Negate intensity | Leave azimuth unchanged | **Flipped** |
| Negate intensity | Update azimuth by 180° | **Unchanged** (approximately) |
| Mirror across the illumination axis | Unchanged | **Unchanged** |
| Mirror across the perpendicular axis | Reflect azimuth | **Unchanged** |
| Mirror across the perpendicular axis | Leave azimuth unchanged | **Invalid** — do not do this |

Print this table. Put it on the wall. Every augmentation decision your team makes for the next two weeks must be checked against it. The vast majority of teams entering this competition will apply `RandomHorizontalFlip(p=0.5)` without updating the azimuth, which is row seven, which is *physically incoherent*, and they will be injecting pure label noise into their training signal while believing they are regularizing.

---

## A4. What `sun_azimuth_angle` Really Is, and Why They Gave It To You

### Definition and convention

Solar azimuth is conventionally measured **in degrees clockwise from north**, so 0° = north, 90° = east, 180° = south, 270° = west. There is, however, a genuine ambiguity in every dataset of this kind that you must resolve before you can use the value:

1. **Does the angle point *toward* the sun, or is it the direction the light *travels*?** These differ by 180°.
2. **Is the image north-up?** Map-projected products usually are; raw camera-frame products often are not.
3. **Does "clockwise from north" in world coordinates map to clockwise or counter-clockwise in array coordinates?** Because image row indices increase *downward*, a rotation that is clockwise in the world can be counter-clockwise in the array depending on how the projection was written.

**You cannot resolve these from documentation. You must resolve them empirically from the data.** This is Phase 1, Step 5, and it is the most important single experiment in the entire project. The procedure is:

For each training image, compute a **shadow-direction vector**: take the centroid of the darkest 10% of pixels, subtract the centroid of the brightest 10% of pixels, and normalize. Call its angle in image coordinates φ_img. Now compute, separately for Class 0 and Class 1, the circular mean of (φ_img − azimuth) mod 360.

If the physics holds and the convention is consistent, you will observe **two tight clusters roughly 180° apart** — one for each class. That single plot simultaneously (a) confirms the azimuth column is meaningful and correctly aligned to the pixels, (b) tells you the exact offset and handedness of the convention, and (c) tells you how strong the shadow-polarity shortcut is. If instead you see uniform noise, something is wrong — either the azimuth is scrambled, the images are not north-up, or the organizers have deliberately decorrelated it, and you need to know that on Day 1 rather than Day 12.

### Why raw degrees must never be fed to a network

The value is **circular**. 359° and 1° are two degrees apart, but as real numbers they are 358 apart. A network fed the raw scalar will learn a discontinuity at the wrap point that does not exist in the physics. **Always encode as (sin a, cos a).** This is a two-dimensional embedding on the unit circle in which the metric matches the geometry, and it is non-negotiable. Every single time your pipeline touches an azimuth — for conditioning, for grouping, for stratification, for logging — think about whether the circularity is being handled.

Similarly, when you compute means and standard deviations of azimuths (for EDA, for augmentation range checks), use **circular statistics** (`scipy.stats.circmean`, `circstd`), not arithmetic ones.

### Why they gave it to you

Three reasons, in ascending order of importance.

The trivial reason is that it is a useful feature and features help. The real reason is that **without it the problem is ill-posed**: the label is genuinely not a function of the pixels alone, so any model consuming pixels alone has an irreducible Bayes error determined by how often the dataset contains inversion-ambiguous pairs. The deepest reason is that **it is the key to the intended solution**: it is not primarily a *feature* to be consumed, it is a *transformation parameter* to be applied. It tells you how to rotate the image into a canonical frame. Teams that treat it as a feature get a modest gain. Teams that treat it as a transformation parameter get a decisive one.

---

## A5. Decoding the Organizers' Warning

> *"Participants are encouraged to focus on the underlying terrain rather than relying solely on visual shadow patterns."*

Take this seriously and read it as a threat, not as encouragement.

### What the shortcut is

From A3, the shortcut is: rotate the image so the sun is at the top; check whether the top half of the central feature is darker or brighter than the bottom half; if darker, predict Depth, else predict Rise. This is roughly ten lines of NumPy, requires no training, and on clean synthetic-ish data could plausibly score in the high 80s or low 90s.

### Why the organizers warned you about it

There are only two possibilities, and you should hedge against both.

**Possibility one: it is a genuine hint about the intended difficulty.** The dataset contains a substantial fraction of images where shadow polarity is weak, absent, or misleading — high-sun images with almost no shadow, heavily degraded features with soft shading, overlapping features where multiple shadows interfere, or features so small that the shadow is a handful of pixels. On these, the shortcut fails and a model with genuine morphological understanding wins.

**Possibility two, and the one I would bet on: the evaluation set has been constructed to punish the shortcut.** The most natural way for an organizer to build this competition is to take a pool of features, and then deliberately construct the evaluation set so that its azimuth distribution differs from training, or so that it contains a disproportionate number of inversion-ambiguous cases, or — most aggressively — so that some fraction of evaluation images are *rotated versions of training-like images with correspondingly updated azimuths*, specifically so that a model relying on raw appearance gets them backwards. A competition literally named "The Pareidolia Paradox" that hands you the azimuth and then warns you not to lean on shadows is telegraphing this.

### What "underlying terrain" means concretely

Signals that are *not* raw shadow polarity, and which you should deliberately build capacity for:

- **Shadow containment versus casting** (A3): the crater's shadow is bounded by its own rim; the mound's shadow escapes onto the surroundings. This is a topological property, robust to which direction the light comes from.
- **Rim structure.** Fresh craters have a raised rim producing a characteristic bright annulus on the exterior sun-facing slope and a subtle dark annulus on the exterior anti-sun slope — a *ring* signature absent from mounds.
- **Boundary sharpness and closure.** Impact craters are near-circular with closed boundaries; boulders are angular; mounds are irregular and often have no closed boundary at all.
- **Ejecta and surroundings.** Fresh craters may have bright radial ejecta or a halo of secondary boulders; mounds do not.
- **Interior texture.** Crater floors are often smooth and flat (infill); mound tops are often continuous with the surrounding regolith texture.
- **Scale distribution.** Boulders occupy a narrow small-size band; craters span a wide range.
- **Local relief consistency.** In a scene with several features, they are all lit by the same sun; internal consistency across features in the tile is a strong constraint.

You do not need to hand-engineer all of these. A CNN trained on canonicalized images with strong consistency regularization will learn many of them. But you should **explicitly test for whether your model is using them**, via the shortcut-audit described in Feature F12 and Phase 5.

### The strategic decision

**Do not choose between the shortcut and the deep model. Use both, and use the disagreement as information.**

Build the shortcut as an explicit, fast, interpretable baseline (Phase 2). Build the deep model. Then measure the **agreement rate** on the test set. If they agree on 96% of test images, the shortcut is essentially the whole story and you should be confident. If they agree on 78%, then a fifth of the evaluation set is doing something the shortcut cannot handle, which is direct evidence that Possibility Two is in play, and you should weight your submission heavily toward the deep model and invest your remaining time in robustness. **This agreement statistic uses no labels, is completely legitimate, and gives you the only real intelligence you will get about the hidden set.** It is worth building the baseline for that reason alone.

---

## A6. Balanced Accuracy — The Mathematics and Its Consequences

### The definition

Balanced Accuracy is the arithmetic mean of the per-class recalls:

```
BA = ½ · ( TP/(TP+FN)  +  TN/(TN+FP) )
   = ½ · ( recall_class1 + recall_class0 )
   = ½ · ( sensitivity + specificity )
```

Equivalently it is accuracy computed on a hypothetical test distribution in which both classes are equally frequent, regardless of their actual frequency.

### Consequence 1: class imbalance in training does not automatically hurt, but ignoring it does

If the training set is imbalanced — say 65% Depth, 35% Rise — a model trained with plain cross-entropy will learn a prior that favours Depth, and at a 0.5 threshold it will over-predict Depth. Its plain accuracy might look fine while its Balanced Accuracy suffers badly, because the minority class recall collapses.

Three complementary remedies, and you should use at least two:

1. **Class-weighted loss.** Weight each class inversely to its frequency in the cross-entropy. `torch.nn.CrossEntropyLoss(weight=w)` with `w_c = N/(2·N_c)`. Simple, effective, no data plumbing.
2. **Balanced sampling.** Use a `WeightedRandomSampler` so each epoch draws classes equally. Slightly better for batch-norm statistics than loss weighting, at the cost of some redundancy in the minority class.
3. **Threshold adjustment at inference.** Covered below. This is the cheapest and often the single most effective one.

**Measure the actual class balance on Day 1** — it takes one line of pandas — and choose accordingly. If it turns out to be 50/50, all of this collapses to standard practice, but do not assume it.

### Consequence 2: the optimal decision threshold is the class prior, not 0.5

This is a precise, derivable result and it is worth having in hand.

Maximizing BA is equivalent to maximizing plain accuracy under a re-weighted distribution where each class carries weight ½. If p(y=1|x) is your calibrated posterior under the *training* prior, and π₁ is the training prior of class 1, then the re-weighted posterior is

```
p'(1|x) = [p(1|x)/π₁] / ( p(1|x)/π₁ + p(0|x)/π₀ )
```

Setting p'(1|x) > ½ and simplifying:

```
p(1|x)/π₁ > p(0|x)/π₀
⟺ p(1|x)·π₀ > (1 − p(1|x))·π₁
⟺ p(1|x)·(π₀ + π₁) > π₁
⟺ p(1|x) > π₁
```

**The Balanced-Accuracy-optimal threshold on a well-calibrated posterior is t = π₁, the training prior of the positive class.** If your training set is 65/35 Depth/Rise, threshold at 0.35, not 0.5. If it is balanced, threshold at 0.5 and the correction vanishes.

In practice your posteriors will not be perfectly calibrated, so **do both**: use t = π₁ as your theoretically-motivated default, and also sweep t over a fine grid on your out-of-fold predictions to find the empirical argmax. If the two agree closely, you are calibrated and confident. If the empirical optimum is wildly different from π₁, your model is miscalibrated and you should apply temperature scaling or Platt scaling on out-of-fold predictions before thresholding.

**Guard against threshold overfitting.** The empirical optimum on OOF data is itself an estimate with variance. Two protections: (a) inspect the BA-versus-threshold curve and pick a point in the middle of a *flat plateau* rather than a sharp spike — sharp spikes are noise; (b) compute the optimal threshold independently on each fold and check the spread. If the per-fold optima range from 0.31 to 0.64, do not trust any of them; fall back to π₁.

### Consequence 3: every error costs the same, weighted by class size

There is no asymmetry to exploit — a false Rise and a false Depth cost the same *within their respective class denominators*. But because the denominators are per-class, **an error on a minority-class image costs more than an error on a majority-class image.** If Rise is 35% of the data, each Rise misclassification costs 1/(2·700) of your score while each Depth misclassification costs 1/(2·1300). Errors on the minority class are worth ~1.86× as much. This is exactly what the threshold correction compensates for, and it is why you should spend proportionally more of your error-analysis attention on the minority class.

### Consequence 4: implement the metric yourself, once, correctly

Use `sklearn.metrics.balanced_accuracy_score`. Do not hand-roll it. Do not accidentally optimize for F1, ROC-AUC, or plain accuracy in your model-selection code because that is what your training template printed by default. **Every early-stopping decision, every checkpoint-selection criterion, every hyperparameter comparison, and every logged number in your experiment tracker must be Balanced Accuracy on out-of-fold predictions at the tuned threshold.** Optimizing one metric while being scored on another is the most banal and most common way teams lose competitions.

A subtlety worth knowing: ROC-AUC is still a useful *secondary* metric because it is threshold-independent and therefore separates "my model ranks well but my threshold is off" from "my model ranks badly." Log both. Diagnose with AUC; select and report on BA.

---

## A7. Submission Format Forensics

The brief specifies:

```
image_id,label

eval_00001.png,0
eval_00002.png,1
```

Assume a strict automated parser. Here is the complete checklist your submission validator must enforce, and every single item on it has cost some team somewhere a competition:

1. **Exactly 2,001 lines**: one header plus 2,000 data rows. Count them.
2. **Header is exactly `image_id,label`** — lowercase, no spaces around the comma, no BOM, no quotes.
3. **`image_id` includes the `.png` extension.** The example makes this explicit. A pipeline that strips extensions internally for convenience and forgets to re-append them produces 2,000 unmatched rows and a score of zero. This is the most common single failure.
4. **The `image_id` values match `test_metadata.csv` exactly**, as a set. Compute `set(submission.image_id) == set(test_metadata.image_id)` and assert it. Also assert no duplicates.
5. **`label` values are integers 0 or 1.** Not `0.0`. Not `"0"` with quotes. Not `False`/`True`. Not probabilities. A pandas `float64` column writes as `0.0` and may fail an integer parser — cast with `.astype(int)` before writing.
6. **No index column.** `df.to_csv(path, index=False)`. Forgetting `index=False` prepends an unnamed integer column and breaks the schema.
7. **No trailing whitespace, no trailing blank line beyond a single terminating newline, UTF-8 encoding, LF (not CRLF) line endings** unless the platform specifies otherwise. On Windows, pandas may write CRLF; pass `lineterminator='\n'` if you are unsure.
8. **Ordering**: safest to match `test_metadata.csv` order exactly. Most parsers key on `image_id` and ignore order, but matching costs nothing and eliminates a class of risk.
9. **No missing values.** Assert `df.label.isna().sum() == 0`.
10. **A single CSV file**, not a zip, not a folder, unless the platform says otherwise.

Beyond mechanical validity, run these **sanity checks** on every submission before you upload it:

- **Predicted class balance.** If your training set is roughly balanced and your submission is 94% one class, something is broken — most likely a threshold bug or a label-mapping inversion.
- **The label-inversion check.** This is the nightmare scenario: a `LabelEncoder` or a `sorted(unique)` somewhere maps your classes backwards, and you submit a perfectly good model with every prediction flipped, scoring roughly (1 − BA). Protect against it by asserting the mapping explicitly in code (`assert CLASS_NAMES == {0: 'depth', 1: 'rise'}`), and by running your final model over a **handful of training images with known labels through the exact production inference path** and confirming it gets them right. Do this as the literal last step before upload.
- **Agreement with your previous best submission.** If your new model agrees with your previous submission on only 60% of rows, either you have made a breakthrough or you have introduced a bug. It is almost always the latter. Investigate before uploading.

**Build this validator as a standalone script on Day 2, before you have a good model.** Run it automatically as the final step of every inference run so that an invalid file cannot physically be produced.

---
---

# PART B — THE PAIN POINT REGISTER

Every genuine difficulty embedded in this problem statement, stated as a symptom, diagnosed, and solved. Assign each of these to a named person on your team so that none of them is nobody's job.

## B1. Topographic inversion makes the label a non-function of the pixels

**Symptom.** Your model plateaus somewhere in the low 90s and the confusion matrix shows a stubborn, symmetric residual: a set of Depth images confidently predicted as Rise and a set of Rise images confidently predicted as Depth, with no obvious quality problem in the images themselves.

**Diagnosis.** The model is a function of pixels only. For inversion-ambiguous images, no pixels-only function can be correct on both members of an ambiguous pair. You have hit the Bayes error of the restricted hypothesis class, not of the problem.

**Fix.** The azimuth must enter the model. In descending order of effectiveness: (1) **canonicalize** by rotating every image into a common illumination frame (Part C1); (2) **condition** the network on (sin a, cos a) via FiLM layers (C3); (3) concatenate (sin a, cos a) to the pooled feature vector before the classification head. Do (1). Optionally also do (2) as a residual correction for the canonicalization's imperfections. Do not rely on (3) alone — late concatenation gives the azimuth almost no ability to modulate the spatial feature extraction, which is where it is needed.

**How you will know it worked.** The confidently-wrong symmetric residual shrinks substantially, and your accuracy on the subset of images you have flagged as "low shadow contrast" improves.

## B2. Shortcut learning on shadow polarity

**Symptom.** Excellent CV, and a suspicion you cannot shake. Or: your Grad-CAM maps consistently highlight the shadow boundary and nothing else.

**Diagnosis.** The network has found the cheapest sufficient statistic. Gradient descent is lazy; if shadow polarity separates the training data, no pressure exists to learn rim structure, containment topology, or texture.

**Fix.** Four measures, applied together.

1. **Shadow-region masking augmentation.** With some probability, mask out (fill with local median) the darkest connected component in the image and force the model to classify without it. This is a direct attack on the shortcut and forces the network to find secondary evidence.
2. **Photometric-negation label-flip augmentation** (A3). This creates pairs of images with near-identical structure and opposite labels, so polarity remains diagnostic but the model must also track the physics rather than memorizing an appearance prior.
3. **Contrast and gamma jitter** so the model cannot key on absolute shadow darkness.
4. **The shortcut audit** (F12): explicitly measure your model's accuracy on the subset where the shortcut baseline is *wrong*. If your deep model scores near chance on exactly the images the shortcut gets wrong, your deep model has learned nothing the shortcut does not already know, and you have no robustness margin at all. This number is the single best summary of whether you have solved the actual competition or just the easy version of it.

## B3. Distribution shift between train and evaluation azimuths

**Symptom.** Unknowable directly — that is the whole problem. Detectable only indirectly.

**Diagnosis.** The evaluation set may have been drawn with a different azimuth distribution, a different sun-elevation distribution, a different feature-type mix, or from different source scenes.

**Fix.** Detect it and defend against it.

*Detect*: (a) Plot the azimuth histograms of `train_metadata.csv` and `test_metadata.csv` on the same axes. A visible difference is immediate, decisive intelligence and takes ninety seconds to obtain. Quantify with a two-sample Kuiper test (the circular analogue of Kolmogorov–Smirnov). (b) Run **adversarial validation**: label train images 0 and test images 1, train a small classifier to distinguish them, and measure AUC via cross-validation. AUC ≈ 0.5 means the image distributions are indistinguishable and you can relax. AUC > 0.65 means there is real covariate shift and you should find out what drives it (inspect the most confidently "test-like" training images). (c) Compare simple statistics: mean/std of pixel intensity, fraction of near-black pixels (a proxy for sun elevation), high-frequency energy (a proxy for resolution or compression).

*Defend*: Canonicalization is itself the strongest defence, because after canonicalization the azimuth distribution is irrelevant by construction — every image looks like it was lit from the same direction. This is a major, underappreciated argument for the canonicalization approach: **it makes azimuth distribution shift a non-issue.** Beyond that, uniform-azimuth augmentation (train on all rotations with correctly updated azimuths) ensures coverage of the full circle regardless of what the training marginal looks like.

## B4. Class imbalance interacting with Balanced Accuracy

**Symptom.** High plain accuracy, disappointing Balanced Accuracy, low recall on one class.

**Diagnosis and fix.** Fully covered in A6. Measure the balance on Day 1; apply class-weighted loss or balanced sampling; threshold at the prior; verify with an OOF threshold sweep.

## B5. Small dataset and overfitting

**Symptom.** Training loss drops smoothly toward zero while validation BA peaks early and then degrades. Gap between train and validation accuracy exceeding ~8 points.

**Diagnosis.** 7,854 images is not many for a model with tens of millions of parameters.

**Fix.** ImageNet pretraining (mandatory). Modest architectures (Tiny/Small tier). AdamW with weight decay 0.01–0.05. Stochastic depth `drop_path_rate=0.1`–`0.2`. Label smoothing 0.05–0.1. Cosine LR schedule with 2–3 warmup epochs. Heavy but *physically valid* augmentation. Early stopping on validation BA with patience of about 5 epochs. Mixup and CutMix are worth testing but be aware they blend labels across the class boundary in a way that may be less meaningful here than in natural images — test, do not assume. Exponential moving average of weights (EMA) is a cheap, reliable half-point.

## B6. The augmentation trap — standard augmentations silently corrupt the physics

**Symptom.** You add the usual `RandomHorizontalFlip` / `RandomVerticalFlip` / `RandomRotation(180)` and your validation BA gets *worse*, or gets better but generalizes badly.

**Diagnosis.** You are training the model to be invariant to transformations under which the label is *not* invariant. Every flip without an azimuth update injects a mislabelled example.

**Fix.** Implement a **custom azimuth-aware transform pipeline** where every geometric operation returns both the transformed image and the transformed azimuth. Never use a library augmentation that touches orientation without wrapping it. The algebra is in the table at the end of A3 and the implementation is in C2. If you canonicalize first, the rules become beautifully simple: after canonicalization the sun is at the top, so **horizontal flips are free** (they mirror across the illumination axis, preserving both label and lighting geometry), **vertical flips flip the label**, and **rotations are forbidden** unless you re-canonicalize.

## B7. Grayscale inputs versus RGB-pretrained backbones

**Symptom.** Minor, but it wastes compute and occasionally causes normalization bugs.

**Fix.** Use `timm.create_model(name, pretrained=True, in_chans=1)`, which correctly folds the RGB stem weights. Compute your **own** normalization statistics from the training set rather than using ImageNet's `mean=[0.485,...], std=[0.229,...]` — lunar grayscale imagery has a very different intensity distribution from natural photographs. Compute the dataset mean and std once, hard-code them in a config file, and use the identical values at inference. A train/inference normalization mismatch is a silent, catastrophic, and surprisingly common bug.

## B8. Ambiguous, degraded, and mislabelled images

**Symptom.** A residual of ~2–5% of training images that every model in your ensemble gets wrong with high confidence.

**Diagnosis.** Some are genuinely ambiguous (a shallow degraded feature, a high-sun image with no shadow). Some are probably mislabelled — hand-labelled planetary datasets routinely carry 1–3% label noise.

**Fix.** Extract the top-200 highest-loss out-of-fold training examples and **look at them with your own eyes.** This is the highest-information-per-minute activity available to you and most teams skip it. You will learn more in forty minutes of looking at your model's mistakes than in a day of hyperparameter tuning. Then: if they are mislabelled, consider removing them or applying label smoothing; if they are genuinely hard, they define the frontier of the problem and tell you what capability to build next. Do **not** blindly delete high-loss examples — that is how you delete the hard-but-correct cases that the evaluation set is full of.

## B9. Normalization and contrast pitfalls

**Symptom.** Model performs differently on bright and dark images; sensitive to global contrast.

**Diagnosis.** Absolute brightness depends on solar elevation and local albedo, neither of which is class-relevant. But **relative** brightness structure is the entire signal, so you must be careful what you normalize away.

**Fix.** Use global dataset-level normalization (subtract dataset mean, divide by dataset std) as your default, since it preserves relative structure across images. **Test per-image standardization as an ablation** — it removes brightness nuisance but may also remove useful sun-elevation information. Test **CLAHE** (contrast-limited adaptive histogram equalization) as a preprocessing option; it often helps enormously on low-contrast planetary imagery but can also amplify noise in shadowed regions. Never use an augmentation that *inverts* contrast unless you are deliberately invoking the label-flip rule.

## B10. Validation leakage from near-duplicate tiles

**Symptom.** CV score is 0.97; you have a bad feeling; there is no way to check.

**Diagnosis.** Tiles cropped from the same source orbital image overlap or contain the same feature at different offsets. Random k-fold puts near-duplicates in different folds, so the model has effectively seen the validation data.

**Fix.** Detect and group. Compute a perceptual hash (`imagehash.phash`) or, better, an embedding from a pretrained backbone for every training image; build a nearest-neighbour graph; connect pairs above a similarity threshold; take connected components as groups. Then use `StratifiedGroupKFold` so entire groups stay within a fold. **Compare the grouped-CV score against the random-CV score.** If grouped CV is 4 points lower, you have just discovered that your entire model-selection process was running on inflated numbers, and you have saved your competition. If they are within a point, you have cheaply bought certainty. Either outcome justifies the two hours.

## B11. Submission format errors

**Symptom.** Score of 0.0 or 0.5 on a model you know is good.

**Fix.** The validator in A7, wired into the pipeline as a mandatory final step. Plus the label-inversion check. Plus a dry-run submission uploaded early in the window, if the platform permits, to confirm the file is accepted.

## B12. Reproducibility and seed variance

**Symptom.** You rerun a configuration and get a different score; you cannot tell whether a change helped.

**Diagnosis.** Run-to-run standard deviation on a dataset this size is typically 0.3–0.8 BA points. Many "improvements" you observe will be noise.

**Fix.** Seed everything (`random`, `numpy`, `torch`, `torch.cuda`, and set `torch.backends.cudnn.deterministic=True`, `PYTHONHASHSEED`). Then, critically: **do not trust any single-run comparison of under ~1 point.** For decisions that matter, run 3 seeds and compare means. Log the standard deviation alongside the mean in your experiment tracker so that everyone on the team internalizes the noise floor and stops arguing about 0.2-point differences.

## B13. Compute and time constraints

**Symptom.** Day 12, and you have run four experiments.

**Fix.** Decide your compute story on Day 0. A single mid-range GPU (Colab T4/L4, Kaggle P100, or a local RTX-class card) is sufficient for this dataset if you engineer for throughput: preload the whole dataset to RAM as uint8, use mixed precision (`torch.amp.autocast` with bf16 or fp16), `channels_last` memory format, `num_workers` tuned to your CPU count, and `torch.compile()` if on PyTorch 2.x. A ConvNeXt-Tiny epoch on 6,300 images at 256×256 should take well under a minute on any modern GPU. If your epochs take five minutes, you have an I/O or preprocessing bug, not a compute shortage — **profile before you buy hardware.** Also: run a **fast configuration** (small model, 15 epochs, single fold) for exploratory experiments and reserve full 5-fold multi-seed runs for candidates that have already proven themselves.

## B14. Team coordination

**Symptom.** Three people have three notebooks, two of them have diverged preprocessing, and nobody can reproduce the best score.

**Fix.** Day 0, non-negotiable: a git repository, a single shared `config.yaml`, a single `dataset.py` that everyone imports, a single `metrics.py`, and a shared experiment tracker (Weights & Biases free tier, or a shared Google Sheet if you must). **The rule is: preprocessing and validation splits are shared infrastructure and are changed only by consensus; models and augmentations are individual workstreams.** Precompute the fold assignments **once**, save `folds.csv` to the repo, and have everyone read it. If two team members use different splits, their scores are not comparable and every conversation you have about which model is better is meaningless.

---
---

# PART C — THE CENTRAL SCIENTIFIC STRATEGY

This is the technical heart of the document. Everything here is specific to this problem; none of it is generic ML advice.

## C1. Illumination Canonicalization

### The idea

If the label depends on the image *and* the illumination direction jointly, then rather than teaching a network to handle all illumination directions, **remove the variable**: rotate every image so that the sun always arrives from the same direction. After this transformation, the illumination direction is a constant, the label becomes a function of the pixels alone, and the problem collapses to ordinary image classification.

This is the same trick a planetary scientist uses by hand. (cite index="12-1">One quick way to make a two-dimensional image "pop" into its correct relief is to rotate it until the light source comes from above.</cite> You are automating exactly that, uniformly, for every image in the dataset.

### Why it is the strongest available move

- It **eliminates the ambiguity** entirely rather than asking the network to resolve it.
- It makes **train/test azimuth distribution shift irrelevant**, because after canonicalization both distributions are degenerate at the same point (B3).
- It **reduces the effective complexity** of the function to be learned, which matters enormously with only 7,854 samples. The network no longer has to allocate capacity to representing an illumination-dependent decision rule.
- It **simplifies the augmentation algebra** to two clean rules (C2).
- It makes the model **interpretable and demonstrable** — your app can show the raw image, the canonicalized image, and the prediction side by side, which is compelling.

### The procedure

1. **Calibrate the convention.** Determine the mapping from `sun_azimuth_angle` to a direction in image array coordinates. Do this empirically as described in A4 and Phase 1 Step 5. You are solving for two unknowns: an offset δ and a handedness s ∈ {+1, −1}, such that the sun's direction in image coordinates is θ_img = s·azimuth + δ.
2. **Rotate.** For each image, rotate by −θ_img so that the sun ends up pointing from the top of the frame downward (or whatever canonical direction you choose — the choice is arbitrary, only consistency matters).
3. **Handle the corners.** A rotation of a square image by a non-multiple of 90° leaves undefined corner regions. Three options: (a) **center-crop** to the inscribed circle's bounding square of side 256/√2 ≈ 181 px, then resize back to 256 — clean, loses ~50% of area; (b) **reflect-pad** before rotating so the corners are filled with plausible terrain — preserves area, introduces mirror artifacts; (c) **rotate on an upscaled canvas** (upscale to 362 px, rotate, center-crop 256) — preserves the full original field of view with no invalid pixels, at the cost of interpolation. **Option (c) is usually best**; test (a) as an ablation since the central feature is what matters and cropping to it may actually reduce distraction.
4. **Interpolate well.** Use bilinear or bicubic interpolation, not nearest-neighbour, or you will introduce aliasing artifacts that the network may latch onto. Be careful that the same interpolation is used at train and inference time.
5. **Verify.** After canonicalization, compute the mean image for each class separately. **You should see a striking, clearly interpretable difference**: the mean Depth image dark at the top and bright at the bottom, the mean Rise image the reverse (or vice versa depending on your canonical direction). If the two mean images look identical, your calibration is wrong and you must fix it before proceeding. **This visualization is your proof that the whole approach is working** and it belongs in your final presentation.

### Belt and braces

Canonicalization is not perfect: the azimuth may have measurement error, the convention may vary across the dataset, and the rotation introduces interpolation artifacts. **Therefore still feed (sin a, cos a) to the network as a conditioning signal even after canonicalizing.** It costs almost nothing and lets the model learn residual corrections. And **keep an un-canonicalized model in your ensemble** as a hedge against the possibility that your calibration is subtly wrong — ensemble diversity is cheap insurance.

## C2. The Augmentation Algebra

### In the raw (un-canonicalized) frame

Implement a transform class where every geometric operation updates the azimuth. The core rule: **rotating image content by θ rotates the apparent sun direction by θ.**

```python
def rotate_with_azimuth(img, az_deg, theta_deg, handedness):
    """Rotate image and update azimuth consistently. Label unchanged."""
    img_r = rotate(img, theta_deg, resample=BILINEAR)   # your chosen impl
    az_r  = (az_deg + handedness * theta_deg) % 360
    return img_r, az_r

def hflip_with_azimuth(img, az_deg, delta, handedness):
    """Mirror left-right. Reflects the sun direction. Label unchanged."""
    img_f = img[:, ::-1]
    theta_img = handedness * az_deg + delta          # sun dir in image coords
    theta_ref = (180.0 - theta_img) % 360            # reflection about vertical
    az_f = ((theta_ref - delta) / handedness) % 360
    return img_f, az_f

def negate_flip_label(img, az_deg, label):
    """Photometric inversion => topographically mirrored surface. Label FLIPS."""
    return img.max() - img, az_deg, 1 - label

def rot180_fixed_azimuth_flip_label(img, az_deg, label):
    """180 deg rotation WITHOUT updating azimuth => mirrored relief. Label FLIPS."""
    return img[::-1, ::-1], az_deg, 1 - label
```

The `handedness` and `delta` values come from your Phase 1 calibration. Wrap all of this in one module that everyone imports; do not let each team member reimplement it.

### In the canonical frame (much simpler — prefer this)

After canonicalization the sun is fixed at the top of every image. The rules reduce to:

- **Horizontal flip (left–right): SAFE.** This mirrors across the illumination axis. Both label and lighting geometry are preserved exactly. Use with p = 0.5, free regularization.
- **Vertical flip (up–down): FLIPS THE LABEL.** It converts a dark-top/bright-bottom crater into a bright-top/dark-bottom mound signature under unchanged lighting. Physically valid, approximately. Use with modest probability (p ≈ 0.15–0.25) and validate empirically.
- **Photometric negation: FLIPS THE LABEL.** Same reasoning. Use with p ≈ 0.10–0.20.
- **Arbitrary rotation: FORBIDDEN**, because it breaks canonicalization. Small rotations (±10°) are acceptable as jitter — they represent azimuth measurement uncertainty — but anything larger destroys the frame.
- **Translation, scale, mild elastic, Gaussian noise, mild blur, brightness/contrast/gamma jitter (monotone only), CoarseDropout: SAFE.** Standard regularizers, no physical implications.

The two label-flipping augmentations are your secret weapon. They effectively **double your dataset with perfectly balanced, physically motivated counterfactuals**, and they directly teach the network the inversion relationship the competition is named after. Ablate them carefully — measure with and without, on grouped CV, across three seeds — but I expect them to be worth real points.

## C3. Azimuth Conditioning Architectures

If you condition rather than (or in addition to) canonicalizing, do it properly. Four approaches in ascending order of effectiveness:

**Level 1 — Late concatenation.** Append (sin a, cos a) to the globally pooled feature vector before the final linear layer. Trivial to implement, and largely ineffective: by the time features are pooled, the spatial arrangement information the azimuth needs to modulate has already been destroyed. Use only as a baseline.

**Level 2 — Constant input planes.** Add two extra input channels filled with the constant values sin a and cos a. Now the azimuth is visible from the first layer, and convolutions can in principle combine it with local structure. Cheap; works surprisingly decently.

**Level 3 — FiLM conditioning (recommended if conditioning).** Feature-wise Linear Modulation: pass (sin a, cos a) through a small MLP to produce per-channel scale γ and shift β for each stage of the backbone, and apply `h ← γ ⊙ h + β`. This lets the illumination direction modulate the network's feature computation at every level of abstraction. Roughly 30 lines of code on top of a timm backbone via forward hooks. This is how conditional generation and conditional classification are done properly.

**Level 4 — Physically structured input channels (best, and combines with canonicalization).** Rather than telling the network the azimuth as an abstract number, give it the physically meaningful derived quantity. Construct a 3-channel input:
- Channel 0: the normalized image I.
- Channel 1: the **directional derivative along the sun vector**, ∂I/∂s = s_x·(∂I/∂x) + s_y·(∂I/∂y), computed with Sobel or Scharr filters. Under Lambertian shading this is approximately proportional to the second derivative of the height field along the illumination direction — that is, **it is approximately a curvature map along the sun axis**, which is *exactly* the quantity that distinguishes a bowl from a dome.
- Channel 2: the derivative perpendicular to the sun vector, ∂I/∂s⊥, which carries the (weaker) cross-illumination information. (cite index="10-1">This asymmetry is well established: brightness gradients provide strong normal constraints along the illumination direction while offering only limited information in the orthogonal direction.</cite>

This construction uses your ImageNet 3-channel stem naturally, hands the physics to the network explicitly, and is a genuinely differentiating idea. **Test it.**

## C4. Equivariance and Anti-Equivariance Consistency Losses

This is the most sophisticated component and the one most likely to separate you from the field. It requires no extra labels and can be applied to the unlabeled test images as well.

**The equivariance constraint.** The physics says that rotating an image and correspondingly updating its azimuth must not change the label. So for any image x, azimuth a, and rotation θ:

```
f( R_θ(x), a + θ )  =  f( x, a )
```

Add a loss term penalizing violation of this: draw a random θ each step, compute both predictions, and penalize their KL divergence or squared difference. Weight it with a coefficient λ_eq around 0.1–1.0 (tune it).

**The anti-equivariance constraint.** The physics says that negating an image while holding the azimuth fixed must flip the label:

```
f( negate(x), a )  =  1 − f( x, a )
```

Penalize violation of this too, with coefficient λ_neg.

**Why this is powerful.** These constraints encode the physics as a *differentiable regularizer* rather than as a data transformation. They tell the network not just "here is another example" but "here is a *relationship* your function must satisfy." They provide gradient signal on every image, including images the model already classifies correctly. And critically, **they can be applied to the 2,000 unlabeled evaluation images**, which is a free 25% increase in your effective training data and directly adapts the model to the evaluation distribution without ever touching a label.

Implement it as a separate loss term added to your cross-entropy, warmed up over the first few epochs so it does not destabilize early training. Ablate it. If it works — and I expect it to — it is the centrepiece of your writeup.

## C5. Physically Grounded Feature Engineering

Even if your primary model is a CNN, build these as (a) the shortcut baseline, (b) interpretable features for an auxiliary gradient-boosted model that adds ensemble diversity, and (c) diagnostics for error analysis.

**Sun-axis brightness asymmetry.** In the canonical frame, split the central region into a top half and a bottom half and compute the difference of means. This *is* the shortcut, as a single number. Signed, so its sign is the prediction and its magnitude is the confidence.

**Shadow containment ratio.** Threshold the image to obtain the shadow mask (Otsu, or a fixed low percentile). Separately segment the central "feature" region (e.g. by thresholding the smoothed image, or by a simple blob detector). Compute what fraction of shadow pixels fall *inside* the feature's convex hull. Near 1.0 suggests a depression (shadow contained by the rim); substantially below 1.0 suggests a rise (shadow cast outward). **This is the single most valuable hand-engineered feature in this problem** because it encodes a topological property that survives illumination changes.

**Shadow elongation and offset.** Fit an ellipse to the shadow blob. A boulder's cast shadow is long and thin with a high aspect ratio, offset well away from the object centroid. A crater's interior shadow is a crescent with its centroid near the feature centre. Compute aspect ratio, and the distance from shadow centroid to feature centroid normalized by feature radius.

**Rim annulus signature.** In the canonical frame, compute the mean radial intensity profile around the feature centroid. A fresh crater shows a characteristic bright-rim/dark-interior/bright-far-wall structure; a mound shows a monotone falloff. Even a coarse 16-bin radial profile is a useful feature vector.

**Sun-elevation proxy.** The fraction of pixels below a fixed low threshold, and the global standard deviation of intensity. These recover the missing metadata column (A2) and are worth feeding to your model as auxiliary conditioning.

**Circularity and boundary closure.** Segment the feature, compute 4π·area/perimeter². Impact craters are strikingly circular; mounds and rock clusters are not.

Feed all of these into a LightGBM model alongside the azimuth features. It will not beat your CNN, but it will be **decorrelated** from it, which is exactly what you want in an ensemble, and it takes an afternoon.

---
---

# PART D — PHASE-BY-PHASE EXECUTION PLAN

Fifteen days remain (6–21 September). The plan below is calendarized, with each phase carrying numbered steps, a named deliverable, and an **exit criterion** — a concrete, checkable condition that must be true before the team moves on. Do not advance past an exit criterion on the grounds that you are behind schedule; a phase left half-done poisons everything downstream.

Assume a team of three to five. Role labels below are suggestions: **DATA** (EDA, preprocessing, validation splits), **MODEL** (training, architecture, ensembling), **APP** (application, API, UI), **OPS** (infrastructure, reproducibility, submission).

---

## PHASE 0 — Foundation and Governance
**Day 1 (6 September), first half. Owner: OPS, with the whole team present.**

Nothing in this phase produces a score. All of it prevents disasters.

**Step 0.1 — Resolve the rule ambiguities.** Read the official competition rules end to end and answer, in writing, in a shared document: (a) Are ImageNet-pretrained weights permitted? (b) Is external data permitted? (c) Is use of the unlabeled test images (pseudo-labelling, TTA, transductive learning) permitted? (d) What is the exact deadline timezone? (e) How many submissions are allowed, and is there any intermediate feedback? (f) Is there a required code/report deliverable alongside the CSV? Each of these materially changes the plan. If any is unanswerable from the rules, **email the organizers on Day 1** — you will not get an answer on Day 14.

**Step 0.2 — Stand up the repository.** Create a git repo with this structure:

```
pareidolia/
  configs/          config.yaml, model_*.yaml
  data/             raw/ (gitignored), processed/, folds.csv
  src/
    dataset.py      loading, caching, the Dataset class
    transforms.py   THE azimuth-aware augmentation module
    canonical.py    calibration + canonicalization
    models.py       backbone construction, FiLM, heads
    losses.py       weighted CE, consistency losses
    metrics.py      balanced accuracy, threshold sweep
    train.py        training loop
    infer.py        inference + TTA
    submit.py       submission builder + validator
    features.py     hand-engineered physical features
  notebooks/        EDA only, never production code
  app/              FastAPI backend + frontend
  experiments/      logs, checkpoints (gitignored)
  README.md
```

Add a `.gitignore` for data, checkpoints, and `__pycache__`. Pin your environment in `requirements.txt` with exact versions.

**Step 0.3 — Establish shared infrastructure rules.** Write these into the README as team law: preprocessing, fold assignment, and the metric function are shared and changed only by consensus; `folds.csv` is generated once and committed; every experiment gets a config file and a tracker run; nobody reports a score that was not computed by `src/metrics.py` on the folds in `folds.csv`.

**Step 0.4 — Set up experiment tracking.** Weights & Biases free tier is ideal (`wandb.init(project="pareidolia")`). Log for every run: config hash, git commit SHA, per-fold BA, OOF BA, optimal threshold, ROC-AUC, training curves, and a sample grid of misclassified images. If W&B is not an option, MLflow locally or a shared spreadsheet with a rigid column schema.

**Step 0.5 — Verify compute.** Confirm each team member can access a GPU and run a one-epoch dummy training. Establish where checkpoints live and how they are shared (Google Drive, S3, or W&B artifacts). Do not discover on Day 9 that only one person has a GPU.

**Step 0.6 — Seed everything.** Write `src/utils.py::set_seed(seed)` that seeds `random`, `numpy`, `torch`, `torch.cuda`, sets `PYTHONHASHSEED`, and sets cuDNN deterministic mode. Call it at the top of every script.

> **Deliverable:** working repo, tracked environment, answered rules questions.
> **Exit criterion:** every team member can clone the repo, run `python -m src.train --config configs/debug.yaml`, and see a loss number go down.

---

## PHASE 1 — Data Forensics
**Day 1 (second half) through Day 2. Owner: DATA. This is the highest-value phase in the project.**

Do not skip any step here. The findings determine everything after.

**Step 1.1 — Load and integrity-check.** Read both metadata CSVs. Assert: 7,854 rows in train, 2,000 in test; no duplicate `image_id`; every `image_id` in the CSVs corresponds to a file on disk and vice versa; no nulls. Report any mismatch immediately — a missing file discovered on Day 14 is a crisis; on Day 1 it is an email.

**Step 1.2 — Verify image properties.** For a sample of 200 images (then, in a background job, for all): confirm dimensions are exactly 256×256, confirm single-channel, record the dtype and value range (0–255 uint8 vs 0–65535 uint16 — this matters for normalization), and check for corrupt files. Record the number of unique intensity values; if it is low, the images may be quantized or compressed in a way worth knowing about.

**Step 1.3 — Class balance.** `train_df.label.value_counts(normalize=True)`. Record π₀ and π₁. This single number determines your loss weighting and your decision threshold (A6). Write it into `configs/config.yaml` as a constant.

**Step 1.4 — Azimuth distribution analysis.** Plot the train azimuth histogram and the test azimuth histogram **on the same axes**. Then plot the train azimuth histogram **split by class**. Three questions to answer:
   - *Is the azimuth uniform over 0–360, or clustered?* Clustering suggests a small number of source scenes.
   - *Do train and test differ?* Quantify with a two-sample Kuiper test. **This is your covariate-shift early warning.**
   - ***Is azimuth correlated with label in the training set?*** This is critical. If, say, Depth images skew toward azimuths of 40–120° and Rise toward 220–300°, then **azimuth alone is a leaky feature**, a model will exploit it, and it will very likely not generalize. If you find this, you must actively *decorrelate*: either train with azimuth-randomizing rotation augmentation, or drop azimuth as a direct feature and use it only as a canonicalization parameter (which is another strong argument for canonicalization). Measure it with mutual information between the label and binned azimuth, and by fitting a logistic regression on (sin a, cos a) alone and reporting its cross-validated BA. **If that azimuth-only model scores meaningfully above 0.5, you have found a trap.**

**Step 1.5 — CALIBRATE THE AZIMUTH CONVENTION. (The single most important experiment.)**
   1. For every training image, compute the shadow-direction vector: centroid of the darkest 10% of pixels minus centroid of the brightest 10%, normalized. Take its angle φ_img in image coordinates (using `atan2` with a documented, consistent axis convention).
   2. Compute Δ = (φ_img − azimuth) mod 360 for every image.
   3. Plot the circular histogram of Δ, **split by class**.
   4. **Expected result:** two concentrated modes approximately 180° apart. Read off the modal Δ for each class. This gives you the offset δ; the fact that the classes separate confirms the handedness. If the separation is not clean, retry with handedness s = −1 (i.e. Δ = (φ_img + azimuth) mod 360) and compare which gives tighter clusters.
   5. **Record the concentration** (circular variance / the resultant vector length R) of each mode. R near 1 means shadow polarity is a near-deterministic function of class and azimuth — the shortcut is very strong. R near 0.4 means it is noisy and the deep model has real work to do. **This number tells you how much of the problem the shortcut solves and therefore how to allocate your remaining fortnight.**
   6. Write the resulting (δ, s) into `configs/config.yaml`. Everything downstream depends on it.

**Step 1.6 — Build and validate the canonicalization function.** Implement `canonical.py::canonicalize(img, azimuth)` using the calibrated (δ, s). Then validate visually: compute and display the **per-class mean image before and after canonicalization**. Before: both means should look like featureless blur. After: **they should look strikingly different and physically interpretable** (dark-top/bright-bottom versus the reverse). Save this figure — it is your proof of correctness and belongs in the final presentation. If the means do not separate, stop and debug; do not proceed.

**Step 1.7 — Visual audit.** Build a grid viewer and look at, at minimum: 50 random Class 0, 50 random Class 1, the 20 brightest images, the 20 darkest, the 20 lowest-contrast, and 30 images sampled across the azimuth range. **Actually look at them.** Write down, in prose, in the shared doc, what the sub-populations are (fresh craters, degraded craters, pits, boulders, mounds, rock fields), roughly how common each is, and which pairs look confusable. This qualitative map guides every error analysis you will do later.

**Step 1.8 — Near-duplicate detection and grouping.** Compute a perceptual hash (`imagehash.phash`, hash size 16) for all 7,854 training images. Also compute embeddings from a pretrained ResNet-50 and their cosine similarities via FAISS or sklearn's NearestNeighbors. Build a graph connecting pairs above a similarity threshold; extract connected components as `group_id`. Report: number of groups, size distribution, largest group. **Then also compute cross-set duplicates: are any test images near-duplicates of training images?** If so, that is important intelligence about how the split was made (and, if the duplicates are exact, a partially trivial subset of the test set).

**Step 1.9 — Generate and freeze the folds.** Produce `folds.csv` with columns `image_id, fold`, using `StratifiedGroupKFold(n_splits=5)` stratified on label and grouped on `group_id`. Additionally stratify on binned azimuth if the azimuth distribution is non-uniform — do this by constructing a composite stratification key of `label × azimuth_bin`. **Commit this file. Never regenerate it.** Every model everyone trains uses these exact folds forever.

**Step 1.10 — Adversarial validation.** Train a quick classifier (a small CNN, 3 epochs, or LightGBM on simple image statistics) to distinguish train images from test images. Report cross-validated AUC. Near 0.5 is reassuring. Above 0.65, investigate what is driving the separation by inspecting the most confidently-test-like training images.

**Step 1.11 — Compute normalization statistics.** Dataset-wide mean and standard deviation of pixel intensity over the training set. Hard-code into config.

**Step 1.12 — Cache the data.** Save the entire training set as a single `uint8` NumPy memmap or `.npy` array (7854, 256, 256), plus the test set. ~515 MB and ~131 MB. Every subsequent training run loads this in seconds instead of decoding 7,854 PNGs per epoch.

> **Deliverable:** an EDA report (a notebook plus a one-page written summary of findings), `folds.csv`, calibrated (δ, s) in config, cached arrays, per-class mean-image figure.
> **Exit criterion:** the per-class canonicalized mean images visibly and interpretably differ; `folds.csv` exists and is committed; the team can state in one sentence how strong the shadow shortcut is and whether train/test distributions differ.

---

## PHASE 2 — Baselines and the End-to-End Pipeline
**Days 2–3. Owner: MODEL + OPS.**

The purpose of this phase is not to score well. It is to make a valid submission possible and to establish reference points against which all later work is measured.

**Step 2.1 — Baseline 0: the constant predictor.** Predict all 0. Compute BA. It will be exactly 0.5 by construction. This is your floor and it confirms your metric code is correct.

**Step 2.2 — Baseline 1: the shortcut.** Implement the sun-axis brightness asymmetry rule from C5 with zero training: canonicalize, compare mean intensity of the top versus bottom half of the central region, threshold at zero. Evaluate on the full training set. **Record this number prominently.** It is the score to beat, and the gap between it and your final model is the honest measure of what your machine learning contributed. If this baseline scores 0.93, you are in Possibility One territory (A5) and the competition is about the last 7%. If it scores 0.71, the problem is much richer than pure polarity and your deep model has real room.

**Step 2.3 — Baseline 2: classical features + LightGBM.** Compute the C5 feature set (asymmetry, containment ratio, shadow elongation, radial profile, circularity, sun-elevation proxy, sin/cos azimuth), train LightGBM with 5-fold grouped CV, report OOF BA. Fast, interpretable, and a genuine ensemble member later. Inspect the feature importances — they tell you what actually matters.

**Step 2.4 — Baseline 3: the naïve CNN, deliberately wrong.** Train a ResNet-18 or ConvNeXt-Tiny on **raw, un-canonicalized images with no azimuth input**, using standard augmentations. Report OOF BA. **This is the number every other team will get on their first day, and its gap to Baseline 1 and to your later models quantifies exactly what the physics buys you.** Keep it for your writeup.

**Step 2.5 — Build the full inference and submission path.** Write `infer.py` and `submit.py`. Run your best baseline over the 2,000 test images and produce `submission_baseline.csv`.

**Step 2.6 — Build the submission validator** implementing every check in A7, and wire it as a mandatory final call inside `submit.py` so that an invalid file cannot be written.

**Step 2.7 — SUBMIT.** Upload the baseline submission to the platform (if intermediate submissions are permitted). **The goal is to confirm that the platform accepts your file format.** If the platform accepts it, you have eliminated the single largest catastrophic risk in the project on Day 3 instead of at 11:47 PM on Day 21.

> **Deliverable:** four baseline scores in the tracker, a validated `submission_baseline.csv`, a confirmed-accepted submission.
> **Exit criterion:** a valid CSV has been produced by an automated script and accepted by the platform. **Do not proceed until this is true.**

---

## PHASE 3 — The Canonical Model
**Days 3–5. Owner: MODEL.**

**Step 3.1 — Build the azimuth-aware transform module** (`transforms.py`) implementing C2 exactly. Write **unit tests** for it: assert that rotating by θ and then by −θ recovers the original azimuth; assert that a canonicalized image rotated by θ with azimuth updated, then re-canonicalized, returns to the same canonical image; assert that double negation is the identity. Transform bugs are silent and devastating; test them.

**Step 3.2 — Build the Dataset class.** It should accept a config specifying: canonicalize on/off, corner-handling mode, the augmentation policy, the normalization stats, and the input-channel construction (1-channel vs 3-channel derivative stack from C3 Level 4). Return `(image_tensor, azimuth_sincos_tensor, label)`.

**Step 3.3 — Build the model class.** `timm.create_model(backbone, pretrained=True, in_chans=C, num_classes=2)` with an optional FiLM conditioning wrapper and an optional late-concatenation head. Make the conditioning mode a config flag so ablations are one-line changes.

**Step 3.4 — Build the training loop.** AdamW, cosine schedule with warmup, mixed precision, class-weighted cross-entropy with label smoothing, gradient clipping at 1.0, EMA of weights, early stopping on validation BA with patience 5, checkpoint on best validation BA. Log everything to the tracker.

**Step 3.5 — Train the first real model.** ConvNeXt-Tiny, canonicalized inputs, safe augmentations only (hflip, translate, scale, brightness/contrast jitter, coarse dropout), no label-flip augmentation yet, 30 epochs, 5-fold grouped CV. **This is your reference model.** Record OOF BA and the per-fold spread.

**Step 3.6 — The critical ablation: canonicalization on versus off.** Train the identical configuration with canonicalization disabled. The difference between these two numbers is the empirical value of the central idea in this document. Expect it to be large. If it is not, something is wrong with your calibration and you should return to Step 1.5.

**Step 3.7 — Sanity-check with Grad-CAM.** Generate class activation maps for 30 correctly-classified and 30 misclassified validation images. Are the maps on the feature, or on a corner artifact from rotation? Are they exclusively on the shadow edge, or do they cover the rim and surroundings? This is your first read on shortcut reliance.

> **Deliverable:** a trained, cross-validated, canonicalized CNN with OOF predictions saved to disk as `oof_convnext_tiny.npy`.
> **Exit criterion:** OOF BA meaningfully exceeds both the shortcut baseline and the naïve CNN baseline, with the canonicalization ablation quantified.

---

## PHASE 4 — Model Development and Architecture Search
**Days 5–9. Owner: MODEL, parallelized across team members.**

Run these as parallel workstreams, one per person, all writing to the same tracker and using the same folds. **Change one thing at a time.** Remember the noise floor from B12: treat differences under 1 point as unproven without multi-seed confirmation.

**Step 4.1 — Backbone sweep.** ConvNeXt-Tiny, ConvNeXt-Small, EfficientNetV2-S, ResNet-50, Swin-Tiny, MaxViT-Tiny. Identical training recipe, identical folds. Record OOF BA for each. **Do not pick just the winner** — pick the top three or four *diverse* ones (at least one convnet and one transformer) for your ensemble, because diversity matters more than individual strength when averaging.

**Step 4.2 — Input representation sweep.** (a) 1-channel canonicalized. (b) 3-channel replicated. (c) 3-channel derivative stack (C3 Level 4). (d) CLAHE preprocessing on/off. (e) Resolution 256 versus 320 versus 384.

**Step 4.3 — Conditioning sweep.** No conditioning (canonicalization only), late concatenation, constant planes, FiLM. On canonicalized inputs the marginal value of conditioning should be small but non-zero; on raw inputs it should be large.

**Step 4.4 — Label-flip augmentation ablation.** Vertical flip with label flip at p ∈ {0, 0.15, 0.25}; photometric negation with label flip at p ∈ {0, 0.1, 0.2}. **Run three seeds each**, because this is an important claim and you need to be sure. Report means and standard deviations.

**Step 4.5 — Consistency loss ablation.** Equivariance loss (C4) at λ_eq ∈ {0, 0.1, 0.5, 1.0}; anti-equivariance loss at λ_neg ∈ {0, 0.1, 0.5}. Then the version applying consistency loss to the **unlabeled test images** as well. Three seeds for the winning configuration.

**Step 4.6 — Regularization tuning.** Weight decay, drop_path, label smoothing, mixup/cutmix on-off. Use Optuna if you want to automate it, but honestly, at this dataset size and this deadline, a hand-designed grid of 12 configurations run by one person is a better use of a day than setting up a hyperparameter search framework.

**Step 4.7 — Error analysis, mid-phase and mandatory.** Take the current best model's OOF predictions. Sort by loss descending. **Look at the top 200 errors with your own eyes.** Categorize them into the morphological sub-types you mapped in Step 1.7. Produce a per-sub-type accuracy table. This will tell you, concretely, where the remaining error lives. Then aim the rest of the phase at that. Also break accuracy down by: azimuth bin, image brightness quartile, image contrast quartile, and shortcut-baseline correctness. **The last one is the shortcut audit and it is the most important cell in the whole table.**

**Step 4.8 — Pseudo-labelling (if rules permit and time allows).** Take your best ensemble's test predictions, keep those above a high confidence threshold (e.g. p > 0.95 or p < 0.05), add them to training as soft or hard labels, retrain, and re-evaluate on the *original* grouped OOF. Only keep it if OOF improves. Pseudo-labelling can help meaningfully on small datasets but it can also amplify your existing biases, so treat it as an experiment, not a given.

> **Deliverable:** a ranked table of at least 20 tracked experiments; 4–6 selected diverse configurations; a written error analysis.
> **Exit criterion:** you can name your top four models and state, from evidence, why each is in the ensemble.

---

## PHASE 5 — Robustness and De-Shortcutting
**Days 7–11, overlapping Phase 4. Owner: MODEL + DATA.**

This phase exists because of A5. It is what you do to insure against the possibility that the evaluation set is adversarial.

**Step 5.1 — The shortcut audit.** Partition your OOF predictions by whether the shortcut baseline is right or wrong. Compute your model's BA on each partition. **Target: substantially above 0.5 on the subset where the shortcut is wrong.** If you are at chance there, your model has learned nothing beyond polarity and you have no protection against an adversarial test set. Make improving this number the explicit goal of the phase.

**Step 5.2 — Synthetic inversion stress test.** Construct a synthetic evaluation set from your validation images by applying `rot180_fixed_azimuth_flip_label` and `negate_flip_label` — images whose correct labels are the *opposite* of what they superficially resemble. Measure BA on it. **A model that has genuinely learned the physics scores well here; a model that memorized appearance scores near zero.** This is the sharpest single diagnostic you have, and it is essentially a simulation of the worst-case adversarial evaluation set.

**Step 5.3 — Azimuth-shift stress test.** Hold out a validation fold, rotate every image in it by a random angle with the azimuth correctly updated, and re-evaluate. A correctly canonicalizing, correctly conditioned model should show **near-zero degradation.** Any significant drop indicates a leak or a bug.

**Step 5.4 — Occlusion and masking tests.** Mask the shadow region and re-evaluate. Mask the feature centre and re-evaluate. The degradation profile tells you what the model depends on.

**Step 5.5 — Calibration.** Plot a reliability diagram of your OOF probabilities. If badly miscalibrated, apply temperature scaling fitted on OOF predictions. This matters because your threshold logic (A6) assumes calibrated posteriors.

**Step 5.6 — Per-slice reporting.** Build a standing report that breaks OOF BA down by azimuth octant, brightness quartile, contrast quartile, morphological sub-type, and shortcut-correctness. Regenerate it for every candidate model. **Select your final ensemble on the worst-slice performance, not only on the mean.** In the presence of possible distribution shift, worst-case slice performance is a better predictor of test-set behaviour than the average.

> **Deliverable:** the robustness report.
> **Exit criterion:** the synthetic inversion stress test (5.2) is passed convincingly, and the shortcut-wrong subset accuracy (5.1) is well above chance.

---

## PHASE 6 — Ensembling, TTA, Calibration, Thresholding
**Days 11–13. Owner: MODEL.**

**Step 6.1 — Assemble OOF matrices.** For each of your N candidate models you should already have an OOF probability vector of length 7,854, all computed on the identical folds. Stack them into a 7854×N matrix. **If any model used different folds, discard it** — you cannot ensemble incomparable predictions.

**Step 6.2 — Check pairwise correlations.** Compute the correlation matrix of the OOF probabilities. Prefer members with correlations below ~0.95. A LightGBM feature model at correlation 0.80 with your CNNs may add more than a fifth CNN at 0.98.

**Step 6.3 — Simple averaging first.** Mean of probabilities, and mean of logits (the latter is usually slightly better since it averages in an unbounded space). Evaluate OOF BA. This is your ensemble baseline.

**Step 6.4 — Weighted averaging.** Optimize weights on OOF predictions with `scipy.optimize.minimize` or a simple Dirichlet random search, maximizing OOF BA. **Constrain weights to be non-negative and sum to one, and be alert to overfitting** — with N=6 models and 7,854 samples, weight optimization has real overfitting risk. Compare weighted against simple averaging using a nested split; if the gain is under half a point, **use simple averaging**, which is more robust.

**Step 6.5 — Stacking (optional).** Train a logistic regression on the OOF probability matrix plus (sin a, cos a). Use nested cross-validation to evaluate honestly. Only adopt if the nested estimate beats simple averaging by a clear margin.

**Step 6.6 — Test-time augmentation.** Design the TTA set using the C2 algebra. In the canonical frame, the safe TTA transforms are: identity, horizontal flip, and small rotations of ±5–10° (with re-canonicalization). You may also include **the physics-based TTA pair**: predict on the negated image and take `1 − p` as a second estimate, then average. That is an unusually strong TTA because it probes a different part of the model's function. Evaluate every TTA scheme on OOF before adopting.

**Step 6.7 — Test-time BN adaptation (optional).** Run a forward pass over the test set in train mode to update batch-norm running statistics, then predict. Cheap; occasionally a meaningful gain under covariate shift; evaluate by proxy since you cannot measure it directly on test — simulate by adapting on one validation fold's statistics.

**Step 6.8 — Threshold optimization.** On the final ensemble's OOF probabilities, sweep the threshold over `np.linspace(0.01, 0.99, 197)` and plot BA versus threshold. Identify the plateau, not the spike. Compute the per-fold optima and their spread. Compare against the theoretical t = π₁. **Choose a threshold in the centre of a stable plateau.** Record the chosen value in the config.

**Step 6.9 — Final model selection meeting.** As a team, look at: the OOF BA of each candidate ensemble, the worst-slice performance, the synthetic-inversion stress test result, and the shortcut-agreement rate on test. Choose. Write down why, in one paragraph, in the shared doc.

> **Deliverable:** the final ensemble specification and threshold, frozen in config.
> **Exit criterion:** the final ensemble is chosen and its OOF BA, per-slice breakdown, and stress-test results are all documented.

---

## PHASE 7 — The Application
**Days 8–15, running in parallel throughout. Owner: APP.**

The application track should start on Day 8 and run alongside modelling, consuming model checkpoints as they become available. Design it so it loads whatever model artifact is current, so the app is never blocked on the model being final. Full feature specifications are in Part E; the build sequence is:

**Step 7.1 —** Define the model artifact contract: a directory containing weights, the config, the normalization stats, the calibration constants (δ, s), and the threshold. The app loads this directory and nothing else. Version it.

**Step 7.2 —** Export the final models. ONNX or TorchScript for portability and CPU inference speed; plain PyTorch checkpoints are acceptable if you are deploying on the same environment.

**Step 7.3 —** Build the FastAPI backend with endpoints: `POST /predict` (single image + azimuth), `POST /predict/batch` (zip or CSV+images), `GET /health`, `GET /model-info`, `POST /explain` (returns a Grad-CAM overlay).

**Step 7.4 —** Build the frontend. Streamlit or Gradio if you want it done in a day; React + Vite + Tailwind if you want it to look like a product and you have someone who can build it. Given a 15-day window with modelling as the priority, **Streamlit is the right choice unless the competition explicitly scores presentation.**

**Step 7.5 —** Implement the feature set in the order given in Part E, prioritizing F1, F2, F6, F7, F8 (the pipeline-critical ones) before F9–F12 (the demonstration ones).

**Step 7.6 —** Build the **Sun Simulator** (Feature F10). This is the flagship. Do not skip it.

**Step 7.7 —** Containerize with Docker. Write a one-command startup. Test on a clean machine.

**Step 7.8 —** Record a 3-minute demo video as insurance against live-demo failure.

> **Deliverable:** a running, containerized application.
> **Exit criterion:** a person who has never seen the project can open the app, upload an image, set an azimuth, and get an explained prediction.

---

## PHASE 8 — Final Submission Engineering
**Days 14–15. Owner: OPS. Treat this phase with more paranoia than it seems to deserve.**

**Step 8.1 — Full clean retrain.** From a fresh clone of the repo at a tagged commit, retrain the final ensemble configuration end to end with fixed seeds. Confirm the OOF BA reproduces the number you selected on. If it does not, you have an unpinned source of randomness and you need to find it before submitting.

**Step 8.2 — Full-data refit (decide deliberately).** You have a choice: submit the average of the 5 fold-models (each trained on 80% of the data), or retrain a single model on 100% of the training data. **The fold-ensemble is the safer choice** — it is itself an ensemble, it is what your OOF score actually measured, and it carries no risk of a full-data model behaving differently from what you validated. The full-data refit sees 25% more data per model but is unvalidated. **Recommended: use the fold-ensemble.** If you want the extra data, do both and average all of them.

**Step 8.3 — Generate test predictions** with the full TTA scheme, saving raw probabilities to disk as `test_probs.npy` before thresholding. Never overwrite this file.

**Step 8.4 — Apply the frozen threshold** and generate the CSV.

**Step 8.5 — Run the validator.** Every check in A7.

**Step 8.6 — Run the label-inversion check.** Push 20 training images with known labels through the *exact production inference path* — the same script, the same artifact directory, the same preprocessing — and assert the predictions match the known labels. This catches the class-mapping inversion that would otherwise cost you everything.

**Step 8.7 — Sanity statistics.** Predicted class balance versus training prior. Agreement rate with the shortcut baseline (A5 — record this, it is your best intelligence about the hidden set). Agreement with your previous submission. Distribution of predicted probabilities (should be bimodal, not piled at 0.5).

**Step 8.8 — Visual spot check.** Display 20 test images predicted Depth and 20 predicted Rise, with their azimuths. Look at them. Do they look right to a human who now understands the physics? This takes ten minutes and has caught more bugs than any automated check.

**Step 8.9 — Submit, early.** Target **21 September, 6:00 PM at the latest**, and ideally Day 14. Confirm the platform's acceptance message. Screenshot it.

**Step 8.10 — Keep a rollback.** Retain your previous known-good submission file so that if something goes wrong you can resubmit a known-valid alternative.

> **Exit criterion:** an accepted final submission, with the confirmation captured, well before the deadline.

---

## PHASE 9 — Documentation and Contingency
**Days 14–15, in parallel.**

**Step 9.1 —** Write the README: setup, data placement, one-command reproduction of the final submission, environment versions.

**Step 9.2 —** Write a model card: architecture, training data, validation protocol, OOF BA, per-slice performance, known limitations, and the intended-use statement.

**Step 9.3 —** Write the technical report if one is required: the physics of topographic inversion, your canonicalization approach, the augmentation algebra, the consistency losses, the ablation table, and the robustness results. **Your ablation table is your differentiator** — most teams will report a score; you can report a causal account of why the score is what it is.

**Step 9.4 —** Contingency planning. Write down, in advance: what you do if the GPU dies on Day 19 (answer: you already have a submitted valid CSV from Day 3 and checkpoints in cloud storage); what you do if the platform rejects your file (answer: you tested it on Day 3); what you do if a critical bug is found on Day 20 (answer: roll back to the last tagged commit and its saved `test_probs.npy`).

---

## Timeline at a Glance

| Days | Phase | Primary outcome |
|---|---|---|
| 6 Sep (D1) | 0 + 1 start | Repo, rules answered, EDA begun |
| 6–7 Sep | 1 | Calibration, folds, cached data |
| 7–8 Sep | 2 | Baselines + **first valid submission** |
| 8–10 Sep | 3 | Canonical CNN, canonicalization ablation |
| 10–14 Sep | 4 | Backbone/aug/loss sweeps, error analysis |
| 12–16 Sep | 5 | Robustness, stress tests, de-shortcutting |
| 13–17 Sep | 7 | Application (parallel track from D8) |
| 16–18 Sep | 6 | Ensemble, TTA, threshold |
| 19–20 Sep | 8 | Final retrain, validation, **submit** |
| 20–21 Sep | 9 | Docs, buffer, contingency |

Note the deliberate two-day buffer at the end. **Use it as buffer, not as extra modelling time.** Every competition schedule that plans to finish on the deadline finishes after it.

---
---

# PART E — CORE APPLICATION FEATURES

Fifteen features. F1–F8 are the pipeline: without them there is no submission. F9–F12 are the demonstration and diagnostic layer: they are what make this a *product* rather than a script, and F10 in particular is the feature that will make people remember your entry. F13–F15 are engineering hygiene that becomes critical in the last 48 hours.

For each: **what it is**, **why this specific problem demands it**, and **how to build it, step by step.**

---

## F1. Data Ingestion and Integrity Layer

**What.** A module that loads metadata and images, validates them, caches them, and exposes a clean interface to everything downstream.

**Why here.** Because you have two metadata files, 9,854 images, a strict submission ID format, and hidden test labels. A single mismatch between the metadata index and the image index — one missing file, one ID with a stripped extension — propagates silently into a zero-scoring submission. Integrity checking is cheap insurance on a problem where you get exactly one shot.

**How.**
1. `load_metadata(split)` reads the CSV, asserts the expected column names exactly (`image_id, sun_azimuth_angle, label` for train; `image_id, sun_azimuth_angle` for test), asserts the expected row counts (7,854 / 2,000), asserts no nulls, asserts no duplicate IDs.
2. `verify_images(df, image_dir)` iterates the IDs, confirms each file exists, opens it, asserts 256×256, records mode and dtype, and returns a report of anomalies. Run once and cache the report.
3. `build_cache()` decodes all images once into `uint8` arrays and writes `train_images.npy` (7854, 256, 256) and `test_images.npy` (2000, 256, 256), plus an `index.json` mapping array position to `image_id`. **This index is the contract that guarantees your predictions line up with the right IDs.**
4. `compute_stats()` returns dataset mean and std; write them to config.
5. Expose `PareidoliaDataset(split, folds, transform, config)` returning `(image, azimuth_sincos, label)`.
6. Write pytest tests for each of the above. This module changes rarely and breaks everything when it does.

---

## F2. Illumination Canonicalization Engine

**What.** The module implementing C1: calibration of the azimuth convention, and rotation of arbitrary images into the canonical sun-at-top frame.

**Why here.** This is the intellectual core of your solution. It is what converts an ill-posed problem into a well-posed one. It is also the component your app will visualize to explain the whole approach to a viewer in ten seconds.

**How.**
1. **`calibrate(images, azimuths, labels) -> (delta, handedness, confidence)`.** Implement the shadow-centroid estimator from A4/Phase 1.5. For each image compute `dark_centroid - bright_centroid` and its angle. For each candidate handedness in {+1, −1}, compute Δ per image, split by class, and measure the circular concentration R of each class's mode. Return the handedness that maximizes the summed concentration, the resulting δ, and R as a confidence score. **Log R prominently — it quantifies the strength of the shortcut and validates the entire physical model.**
2. **`sun_vector_in_image(azimuth, delta, handedness) -> (sx, sy)`.** Returns the unit vector, in image array coordinates, along which sunlight travels. Document the axis convention in a comment; this is where sign errors live.
3. **`canonicalize(img, azimuth) -> img_canon`.** Compute θ_img, upscale the canvas by √2 with reflect padding, rotate by −θ_img with bicubic interpolation, centre-crop back to 256. Cache the result for the whole dataset so you do not recompute it every epoch.
4. **`decanonicalize(img_canon, azimuth)`** — the inverse, needed to map Grad-CAM overlays back onto the original image for display in the app.
5. **Validation function** producing the per-class mean-image figure (before/after). Run it in CI or at least as a manual gate; if the means stop separating, something upstream broke.
6. **Edge cases:** handle azimuth values that are NaN or out of range (clip or flag), and expose a `canonicalize=False` path so ablations are a config flag.

---

## F3. Azimuth-Aware Augmentation Engine

**What.** The transform pipeline implementing the C2 algebra, where every geometric operation returns a consistently updated azimuth and, where applicable, an updated label.

**Why here.** Because the default augmentation stack of every computer vision codebase on Earth is *wrong for this problem* (B6). This module is what stops your team from silently injecting label noise for a fortnight.

**How.**
1. Define a `Sample` dataclass carrying `image`, `azimuth`, `label` so transforms can modify all three.
2. Implement each operator from C2 as a class with a `p` probability and an `apply(sample) -> Sample` method: `RotateWithAzimuth`, `HFlipWithAzimuth`, `CanonicalHFlip` (safe, no updates needed), `CanonicalVFlipLabelSwap`, `PhotometricNegateLabelSwap`, plus the physically-neutral ones (`ShiftScale`, `BrightnessContrastJitter`, `GammaJitter`, `GaussNoise`, `CoarseDropout`, `ShadowMask`).
3. Compose them into named policies in config: `policy_safe`, `policy_with_labelflip`, `policy_aggressive`. Ablations become one-line config edits.
4. **Unit tests, as specified in Phase 3.1.** Round-trip identities, canonicalization invariance under rotate-then-update, double-negation identity.
5. **Build a visual debugger**: a script that renders a grid of 16 augmented versions of one image, annotated with the resulting azimuth and label. **Look at this output before you train anything.** Augmentation bugs are invisible in loss curves and obvious in a picture.

---

## F4. Model Construction and Training Orchestrator

**What.** Config-driven construction of backbones with optional conditioning, plus the training loop.

**Why here.** With 15 days and a five-person team you will run dozens of experiments. If each one requires editing code, you will lose a day to merge conflicts and another to irreproducibility. Everything must be a config parameter.

**How.**
1. `build_model(cfg)`: `timm.create_model(cfg.backbone, pretrained=cfg.pretrained, in_chans=cfg.in_chans, num_classes=2, drop_path_rate=cfg.drop_path)`, wrapped by an optional `FiLMWrapper` that registers forward hooks on each backbone stage and applies per-channel γ/β derived from an MLP over (sin a, cos a).
2. `build_loss(cfg)`: weighted cross-entropy with label smoothing, plus optional `EquivarianceLoss` and `NegationConsistencyLoss` (C4) with their λ coefficients and a linear warmup schedule.
3. `train_one_fold(cfg, fold)`: standard loop with AMP autocast, `channels_last`, cosine LR with warmup, gradient clipping, EMA, per-epoch validation computing BA at both t=0.5 and t=π₁, checkpointing on best validation BA, and early stopping.
4. `train_cv(cfg)`: loops folds, assembles the OOF prediction vector, computes OOF BA and the optimal threshold, saves `oof_{run_id}.npy` and all fold checkpoints.
5. Every run writes a `run_manifest.json` with the config, the git SHA, the seed, and all metrics. **This manifest is what makes an experiment citable three days later when someone asks "which run was that?"**

---

## F5. Cross-Validation and Experiment Tracking

**What.** Frozen folds, a single metric implementation, and a tracker that every run writes to.

**Why here.** Hidden test labels mean CV is your only signal (A2). Group-aware folds mean the signal is honest (B10). A shared tracker means five people's work is comparable.

**How.**
1. `make_folds()` runs once, writes `folds.csv`, and is then **deleted from the workflow** — or at least guarded by an assertion that refuses to overwrite an existing file.
2. `metrics.py` exposes `balanced_accuracy(y_true, y_pred)`, `best_threshold(y_true, y_prob)` returning both the argmax and the plateau centre, and `slice_report(y_true, y_prob, meta)` producing the breakdown by azimuth octant, brightness quartile, contrast quartile, and shortcut-correctness.
3. W&B integration: `wandb.init(config=cfg)`, log per-epoch curves, log the final slice report as a table, log a media panel of the 32 worst OOF errors as images. **The error panel is the single most useful thing in your tracker.**
4. A `compare_runs.py` script that pulls all runs and prints a sorted table of OOF BA with per-fold standard deviations. Run it every morning as the team's standup artifact.

---

## F6. Inference and Test-Time Augmentation Engine

**What.** The path from a trained artifact to calibrated probabilities on new images.

**Why here.** This code runs exactly once on the thing that matters. It must be identical in behaviour to your validation path, or your OOF score means nothing.

**How.**
1. **Enforce path identity.** The inference preprocessing must call the *same functions* as training preprocessing, from the same module, with parameters read from the same saved config. Do not reimplement preprocessing in `infer.py`. This is the single most common source of train/serve skew.
2. `predict(model, images, azimuths, tta_policy)`: for each TTA transform, apply it (with correct azimuth update), forward, invert any label-flipping transform's effect on the output (for the negation TTA, take `1 − p`), and average in logit space.
3. Ensemble across fold checkpoints and across architectures with the weights frozen in config.
4. Save raw probabilities to `test_probs.npy` **before** thresholding, and never overwrite. If a thresholding bug is found at 10 PM on Day 20, you regenerate the CSV in two seconds instead of retraining for three hours.
5. Apply temperature scaling if calibration was fitted.

---

## F7. Balanced-Accuracy Threshold Optimizer

**What.** A module that selects and applies the decision threshold.

**Why here.** Because the metric is Balanced Accuracy and the optimal threshold is π₁, not 0.5 (A6). This is a free point or more that a large fraction of entrants will leave on the table.

**How.**
1. `sweep(y_true, y_prob)` over 197 thresholds, returning the BA curve.
2. `plateau_center(curve)` — find the maximal contiguous run of thresholds within 0.1% of the peak BA and return its midpoint. **Use this rather than the raw argmax**, which is noise-sensitive.
3. `per_fold_optima(...)` returning the spread, as a stability diagnostic.
4. Report the theoretical π₁ alongside the empirical optimum and flag a warning if they diverge by more than 0.1, since that indicates miscalibration.
5. Freeze the chosen threshold into config. **Never recompute it at submission time**; a threshold silently recomputed on a different data slice is a bug you will not notice.

---

## F8. Submission Builder and Validator

**What.** Deterministic generation of the CSV plus every check in A7.

**Why here.** A7 explains why. This is the component with the highest ratio of catastrophic-failure-prevented to lines-of-code in the entire project.

**How.**
1. `build_submission(test_probs, threshold, test_meta, out_path)`: constructs the DataFrame by **joining on `image_id`** — never by positional assumption — casts labels to `int`, writes with `index=False` and `lineterminator='\n'`.
2. `validate_submission(path, test_meta)`: assert line count 2,001; assert header string equality; assert the ID set matches exactly; assert no duplicates; assert dtype and value domain {0,1}; assert no nulls; assert `.png` suffix present on every ID. **Raise on any failure and refuse to return a path.**
3. `sanity_report(submission, test_meta, shortcut_preds, previous_submission)`: predicted class balance, shortcut agreement rate, agreement with the previous submission, probability histogram.
4. `label_inversion_check(model_artifact, train_images, train_labels)`: run 20 known-label training images through the production path and assert correctness. **Call this inside `build_submission` so it cannot be skipped.**
5. Every generated file is named with the run ID and timestamp and is never overwritten.

---

## F9. Explainability — Grad-CAM and Saliency Analysis

**What.** Visual attribution overlays showing what the model looked at, displayed in both the canonical and the original frame.

**Why here.** Two reasons, one scientific and one presentational. Scientifically, it is how you detect shortcut reliance (B2): if every heatmap sits on the shadow boundary, you know your model is a polarity detector with extra steps. Presentationally, in a competition about a *perceptual illusion*, being able to show that your model attends to the crater rim rather than to the shadow is a compelling claim that no leaderboard number can make on its own.

**How.**
1. Use `pytorch-grad-cam` with `GradCAM` or `HiResCAM` on the last convolutional stage. For transformer backbones use attention rollout or `XGradCAM` on the final block's reshaped tokens.
2. Generate the CAM in the canonical frame, then apply `decanonicalize` (F2.4) to warp the heatmap back onto the original image so the user sees it in the orientation they uploaded.
3. Overlay with a perceptually uniform colormap at ~40% alpha over the grayscale image.
4. **Add the quantitative version**: compute the fraction of CAM mass that falls inside the thresholded shadow mask. Average it over the validation set. **This single number is your "shortcut reliance score"** and you should track it across models. A model with equal BA but lower shadow-mass fraction is the more robust model and is the one to submit.
5. In the app, display the CAM next to the prediction, with the shortcut-reliance number.

---

## F10. The Sun Simulator — Interactive Illumination Explorer ★ FLAGSHIP

**What.** A control (slider or radial dial) that sweeps the assumed `sun_azimuth_angle` from 0° to 360° while the app live-updates three things: the canonicalized view of the image, the model's predicted class, and the model's confidence. Optionally, a polar plot of predicted probability as a function of azimuth.

**Why here.** This is the feature that *is* the competition. The Pareidolia Paradox is the claim that appearance without illumination context is ambiguous. This widget lets a user hold a single image fixed and watch the model's interpretation change as the stated lighting changes — which is exactly the illusion, made interactive. And it lets you demonstrate the converse, which is the real payoff: when you rotate the image **and** update the azimuth together, the prediction **does not change**, which is a live, visible proof of the equivariance property your training objective enforced. There is no more persuasive way to show that you understood the problem.

**How.**
1. Backend: `POST /simulate` accepting an image and returning predictions across a sweep of azimuths (say 72 values at 5° steps). Batch all 72 into one forward pass — it costs a few hundred milliseconds and makes the frontend feel instantaneous.
2. Frontend: a slider bound to azimuth. On change, read from the precomputed sweep — no round-trip per tick.
3. Display three panels: original image, canonicalized image at the current azimuth, and a probability gauge.
4. Add a **polar plot** of P(Rise) versus azimuth. For a well-behaved model on an unambiguous image this curve should be fairly flat with a sharp transition; the shape of this curve is itself a diagnostic and it looks impressive.
5. Add the **coupled-rotation mode**: a second control that rotates the *image* and automatically updates the azimuth in lockstep. Show that the prediction stays fixed. Label this panel something like "Equivariance check" and put the prediction's variance across the sweep on screen as a number. **A near-zero variance here is the strongest possible demonstration that your model learned the physics rather than the appearance.**
6. Preload three or four hand-picked example images (a fresh crater, a boulder, a degraded ambiguous case) so a viewer can experience the feature in one click without uploading anything.

---

## F11. Batch Prediction and Confidence Review Queue

**What.** Upload a folder or zip plus a metadata CSV; get back predictions for everything, sorted by confidence, with the least-confident cases surfaced first for human review; export as CSV.

**Why here.** It is literally the competition workflow (2,000 images, one CSV out), so it makes your app a real tool rather than a toy. And the review queue reflects the actual operational use case for this technology: planetary scientists cataloguing terrain would want the model to triage and to flag what needs human eyes.

**How.**
1. `POST /predict/batch` accepting a zip of PNGs plus a CSV of azimuths; stream results.
2. Process in batches of 64 with a progress indicator.
3. Sort output ascending by `|p − threshold|` so genuinely ambiguous cases sit at the top.
4. Render a review grid: image, canonicalized image, prediction, confidence bar, and an override control.
5. Export in the exact competition schema, running F8's validator on the way out. **This means your app can literally produce the submission**, which is a satisfying thing to demonstrate.

---

## F12. Robustness and Shortcut Audit Dashboard

**What.** An internal-facing page that runs the Phase 5 diagnostics against the current model artifact and displays them.

**Why here.** Because A5 means robustness is not a nice-to-have, it is the strategic variable. Making these numbers visible on a dashboard means the whole team internalizes them and stops treating OOF BA as the only score that matters.

**How.**
1. Panels for: OOF BA overall; BA on the shortcut-wrong subset; synthetic inversion stress-test BA; azimuth-shift stress-test degradation; shadow-mask occlusion degradation; the CAM shadow-mass fraction; and the per-slice table.
2. A model-comparison view showing all these metrics side by side for every candidate. **Select your submission from this view, not from a single number.**
3. A calibration reliability diagram.
4. A confusion matrix with clickable cells that open the corresponding images.

---

## F13. Model Registry and Artifact Versioning

**What.** A structured, versioned store of model artifacts, each self-contained.

**Why here.** With five people, dozens of runs, and a hard deadline, "which checkpoint produced the 0.94?" is a question you will be asked, and the wrong answer costs hours.

**How.**
1. Each artifact is a directory: `weights.pt` (or the fold checkpoints), `config.yaml`, `norm_stats.json`, `calibration.json` (δ, s), `threshold.json`, `metrics.json`, `git_sha.txt`.
2. Directory names include the run ID and the OOF BA, e.g. `artifacts/20260914_convnext_t_film_ba0.9412/`.
3. A `registry.json` index with a `production` pointer that the app reads. Promoting a model is a one-line change to that pointer.
4. Sync artifacts to shared cloud storage nightly. **A GPU failure on Day 18 must not be able to destroy your work.**

---

## F14. API and Deployment Layer

**What.** The FastAPI service and its container.

**Why here.** It decouples the model from the UI, lets the app run anywhere, and makes the demo reliable.

**How.**
1. FastAPI with Pydantic request/response models. Load the model once at startup into module state, never per request.
2. Endpoints: `/health`, `/model-info`, `/predict`, `/predict/batch`, `/explain`, `/simulate`.
3. Input validation: reject non-256×256 images with a clear message or auto-resize with a warning; validate azimuth is in [0, 360).
4. `Dockerfile` on a slim Python base, CPU-only Torch for the demo container (much smaller image, and inference on a single 256×256 image is fast on CPU).
5. `docker-compose.yml` bringing up API and UI together with one command.
6. Test the container on a machine that has never seen the project. **"It works on my laptop" is not a deployment.**

---

## F15. Reproducibility Harness

**What.** A single command that regenerates the final submission from a clean checkout.

**Why here.** Because many competitions require reproducibility for prize validation, because it is the only real proof your result is not an artifact of an undocumented manual step, and because it is what lets you confidently rebuild after a disaster.

**How.**
1. `make reproduce` (or `python -m src.reproduce`) that: verifies data placement, rebuilds the cache, loads the frozen `folds.csv`, trains all ensemble members with fixed seeds, generates OOF predictions, verifies the OOF BA matches the recorded value within tolerance, runs inference with TTA, applies the frozen threshold, and writes plus validates the CSV.
2. Full determinism: `set_seed`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True)` where possible, and pinned library versions.
3. Log the total runtime. If reproduction takes eight hours, document that so nobody starts it at 9 PM on Day 20.
4. **Run it at least once, from a fresh clone, before Day 19.** A reproducibility harness that has never been executed is a hypothesis, not a feature.

---
---

# PART F — COMPLETE TECH STACK

Organized as **Core** (you will use these), **Recommended** (clear value, adopt if time permits), and **Optional/Advanced** (differentiators or contingencies). Each entry states what specifically you need from it, why, and how to use it here.

---

## CORE

### Python 3.10 or 3.11
**What:** The language runtime. **Why:** Universal ML support; 3.10+ gives you structural pattern matching and better typing, while 3.12 still has occasional wheel-availability gaps for CV libraries. **How:** Create an isolated environment (`python -m venv .venv` or `conda create -n pareidolia python=3.11`) on Day 0 and pin every dependency in `requirements.txt` with `==` versions. **Detail:** Pin. Not `>=`. A minor version bump in albumentations mid-competition can silently change augmentation behaviour and invalidate your comparisons.

### PyTorch 2.x + torchvision
**What:** The deep learning framework. **Why:** Best ecosystem for transfer learning; timm integrates natively; `torch.compile` gives free speedups; AMP is mature. **How:** Install the CUDA build matching your driver from the official index. Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)` on Ampere or newer (bf16 avoids the loss-scaling fiddliness of fp16), `model.to(memory_format=torch.channels_last)` for convnets, and `torch.compile(model)` once your code is stable. **Detail:** Enable `torch.backends.cudnn.benchmark = True` for fixed input sizes — which yours are — but note it conflicts with strict determinism, so switch it off for your final reproducibility run.

### timm (PyTorch Image Models)
**What:** ~1,000 pretrained vision backbones with a uniform API. **Why:** It is the single highest-leverage library for this problem. It handles `in_chans=1` correctly by folding RGB stem weights (B7), exposes `drop_path_rate` for stochastic depth, and lets you swap architectures by changing a string. **How:** `timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True, in_chans=1, num_classes=2, drop_path_rate=0.1)`. Use `timm.list_models(pretrained=True)` to browse. **Detail:** Prefer IN-22k-pretrained variants where available; they transfer better to non-natural imagery. `timm.data.resolve_data_config` gives you the model's expected preprocessing, but **override the normalization with your own lunar statistics** rather than using ImageNet's.

### NumPy
**What:** Array computing. **Why:** Everything. Your cached image arrays, your OOF matrices, your threshold sweeps. **How:** Cache images as a single `uint8` array; use `np.memmap` if RAM is tight. **Detail:** `np.float32` everywhere in the pipeline; float64 doubles your memory for no benefit.

### pandas
**What:** Tabular handling for metadata, folds, and submissions. **Why:** The CSV format is the deliverable. **How:** `pd.read_csv` with explicit `dtype={'image_id': str}` so IDs are never coerced. Build submissions with `df.merge(...)` on `image_id`, never by row position. **Detail:** `to_csv(path, index=False)` — the missing `index=False` is a classic zero-score bug (A7).

### OpenCV (`opencv-python-headless`)
**What:** Image processing: rotation, interpolation, morphology, CLAHE, thresholding, contours. **Why:** Your canonicalization engine (F2) and your physical feature extraction (C5) both live on it. **How:** `cv2.warpAffine` with `cv2.getRotationMatrix2D` for canonicalization with `INTER_CUBIC` and `BORDER_REFLECT_101`; `cv2.createCLAHE`; `cv2.threshold` with `THRESH_OTSU` for shadow masks; `cv2.findContours` plus `cv2.convexHull` for the containment ratio. **Detail:** Install the `headless` variant on servers to avoid GUI dependency bloat. Remember OpenCV reads as BGR and uses (x, y) ordering while NumPy uses (row, col) — **this is exactly where your azimuth sign errors will come from.** Write the convention down in a comment at the top of `canonical.py`.

### Albumentations
**What:** Fast augmentation library. **Why:** Best-in-class speed and a huge operator set for the physics-neutral augmentations. **How:** Use it for `ShiftScaleRotate` (with rotate limited or disabled), `RandomBrightnessContrast`, `RandomGamma`, `GaussNoise`, `CoarseDropout`, `ElasticTransform`. **Detail (critical):** **Wrap it, do not use it raw for anything orientation-changing.** Albumentations has no concept of your azimuth. Your `transforms.py` (F3) owns all geometry; Albumentations handles only the photometric and mild-spatial operators that carry no illumination semantics. Set `HorizontalFlip` and `VerticalFlip` probabilities to zero inside any Albumentations pipeline and handle flips yourself.

### scikit-learn
**What:** Metrics, splitting, calibration, classical models. **Why:** `balanced_accuracy_score` is your metric; `StratifiedGroupKFold` is your validation protocol; `CalibratedClassifierCV` and `LogisticRegression` handle calibration and stacking. **How:** `from sklearn.model_selection import StratifiedGroupKFold; sgkf.split(X, y, groups=group_ids)`. **Detail:** `StratifiedGroupKFold` requires scikit-learn ≥ 1.0 and cannot always perfectly balance both constraints; check the resulting per-fold class ratios and fold sizes after generating and confirm they are acceptable.

### Matplotlib (+ seaborn)
**What:** Plotting. **Why:** Every diagnostic in Phase 1 and Phase 5 is a plot: azimuth histograms, circular Δ histograms, per-class mean images, BA-versus-threshold curves, reliability diagrams, confusion matrices, polar probability plots. **How:** Write a `src/viz.py` with named functions for each standard figure so everyone produces identical, comparable plots. **Detail:** For circular histograms use `plt.subplot(projection='polar')`. The Δ-histogram from Phase 1.5 is the single most important figure in your project — make it good.

### Git + GitHub
**What:** Version control. **Why:** Five people, fifteen days, shared infrastructure. **How:** Feature branches, PRs for anything touching `dataset.py`, `transforms.py`, `metrics.py`, or `folds.csv`. Tag the commit that produced each submission (`git tag sub-v3`). **Detail:** Log the git SHA into every run manifest (F4.5). When you later ask "what code produced this score," the answer must be mechanical.

---

## RECOMMENDED

### Weights & Biases
**What:** Experiment tracking. **Why:** With dozens of runs across five people, an untracked experiment is a wasted experiment. **How:** `wandb.init(project='pareidolia', config=cfg, name=run_id)`; `wandb.log()` per epoch; `wandb.Table` for slice reports; `wandb.Image` for the worst-error panel. **Detail:** The free tier is ample. The single highest-value thing to log is not the loss curve — it is the grid of the 32 worst OOF errors, refreshed every run.

### PyTorch Lightning (or a hand-written loop)
**What:** Training loop abstraction. **Why:** Removes boilerplate around AMP, checkpointing, early stopping, multi-GPU. **How:** A `LightningModule` wrapping model/loss/optimizer, a `LightningDataModule` for folds, `Trainer(precision='bf16-mixed', callbacks=[EarlyStopping('val_ba', patience=5), ModelCheckpoint(monitor='val_ba', mode='max')])`. **Detail:** Honest trade-off — if nobody on the team already knows Lightning, a 120-line hand-written loop is faster to get right in a 15-day window than learning a framework's abstractions. **Use what your team already knows.**

### LightGBM
**What:** Gradient boosting. **Why:** Trains in seconds on your C5 hand-engineered features, gives interpretable feature importances that tell you which physical signals actually carry information, and provides a **decorrelated ensemble member** (Phase 6.2). **How:** `LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31, class_weight='balanced')` on the same folds. **Detail:** Its value is diversity and interpretability, not raw accuracy. Do not expect it to beat the CNN; do expect it to add a fraction of a point in the ensemble and to teach you something in Phase 2.

### pytorch-grad-cam
**What:** CAM-family attribution. **Why:** Feature F9, and the shortcut-reliance metric. **How:** `GradCAM(model=model, target_layers=[model.stages[-1]])`, then `cam(input_tensor, targets=[ClassifierOutputTarget(cls)])`. **Detail:** Target-layer selection differs per architecture; for ConvNeXt use the last stage's final block, for Swin you need a reshape transform (the library documents this). Budget an hour for wiring, not five minutes.

### FastAPI + Uvicorn
**What:** The API layer (F14). **Why:** Async, fast, automatic OpenAPI docs, Pydantic validation. **How:** `@app.post("/predict")` with a Pydantic model; load the model at startup via a lifespan handler; `uvicorn src.app:app --host 0.0.0.0 --port 8000`. **Detail:** The auto-generated `/docs` page is a free, credible-looking API explorer for your demo.

### Streamlit (or Gradio)
**What:** The UI (F14/F10). **Why:** Turns Python into a web app in an afternoon, which is the correct trade-off when modelling is the priority. **How:** `st.file_uploader`, `st.slider` for the Sun Simulator azimuth, `st.image` for the three panels, `st.plotly_chart` for the polar plot, `@st.cache_resource` to load the model once. **Detail:** Gradio is even faster for a single-function demo and gives you a free public share link, which is excellent for a remote presentation. **Streamlit if you need multiple pages (including the F12 dashboard); Gradio if you need one great demo.**

### Docker + docker-compose
**What:** Containerization. **Why:** Reproducibility and a demo that works on someone else's machine. **How:** Multi-stage build; CPU-only Torch wheel for the demo image (`--index-url https://download.pytorch.org/whl/cpu`) to keep the image around 1 GB rather than 6 GB. **Detail:** Bake the model artifact into the image or mount it as a volume; decide deliberately, since baking makes the demo portable but the image large.

### Hydra or OmegaConf
**What:** Hierarchical configuration. **Why:** Every experiment is a config, and you need composition (`base.yaml` + `model/convnext.yaml` + `aug/labelflip.yaml`) and command-line overrides. **How:** `@hydra.main(config_path='configs')`, then `python -m src.train model=convnext aug=labelflip aug.vflip_p=0.25`. **Detail:** Hydra's automatic per-run output directory is genuinely useful. If the team finds it heavy, plain OmegaConf with manual composition gets you 80% of the value at 20% of the learning cost.

### tqdm, imagehash, FAISS
**What:** Progress bars; perceptual hashing; fast nearest-neighbour search. **Why:** `imagehash` and FAISS together power the near-duplicate detection in Phase 1.8, which protects your entire validation protocol (B10). **How:** `imagehash.phash(Image.open(p), hash_size=16)`; then `faiss.IndexFlatIP` over normalized embeddings for the semantic pass; then `scipy.sparse.csgraph.connected_components` over the thresholded similarity graph.

---

## OPTIONAL / ADVANCED

### escnn or e2cnn — rotation-equivariant CNNs
**What:** Networks with rotation equivariance built into the architecture rather than learned from augmentation. **Why:** This problem has explicit rotational structure — the label is invariant under joint rotation of image and azimuth. A steerable CNN encodes that symmetry in its weights. **How:** Build an `escnn` model over the C8 or C16 cyclic group and combine it with azimuth conditioning. **Detail:** High risk, high reward, and **no pretrained ImageNet weights**, which is a serious drawback at 7,854 samples. Attempt only if you have a spare person and are ahead of schedule. It would be a genuinely distinctive contribution if it worked.

### Self-supervised pretraining (SimCLR / DINO / MAE)
**What:** Pretraining on your own unlabeled data. **Why:** You have 9,854 lunar images and ImageNet pretraining transfers imperfectly to grayscale planetary imagery. **How:** `lightly` for SimCLR/DINO, or a small MAE implementation; pretrain for a few hundred epochs on train+test images, then fine-tune. Use only physics-neutral augmentations for the contrastive views, or you will teach the encoder to be invariant to exactly the illumination information you need. **Detail:** This is the strongest fallback if **pretrained weights turn out to be disallowed** (Step 0.1). Keep it in your back pocket.

### Optuna
**What:** Hyperparameter optimization. **Why:** Automates the search. **How:** TPE sampler over LR, weight decay, drop_path, and augmentation probabilities, with a median pruner. **Detail:** Be honest about the noise floor (B12). With run-to-run σ around 0.5 BA points, an HPO run over 40 trials will "find" configurations whose apparent superiority is noise. Use HPO on a single fold with a fixed seed for coarse ranges, then confirm the winner with full multi-seed CV. **In a 15-day window, hand-designed ablations usually beat automated search** because they teach you something.

### ONNX Runtime
**What:** Portable, optimized inference. **Why:** Fast CPU inference for the demo container; no Torch dependency at serve time. **How:** `torch.onnx.export(model, dummy, 'model.onnx', opset_version=17)`, then `onnxruntime.InferenceSession`. **Detail:** **Always verify numerical parity** between the PyTorch and ONNX outputs on 100 images before trusting the export. A silent opset incompatibility that shifts outputs by 1e-2 can move predictions across your threshold.

### DVC
**What:** Data version control. **Why:** Keeps large artifacts out of git while keeping them versioned alongside code. **How:** `dvc add data/raw`, `dvc remote add -d storage s3://...`. **Detail:** Valuable for a long project; possibly over-engineering for 15 days. A shared Drive folder with a strict naming convention may serve you better.

### Plotly
**What:** Interactive charts. **Why:** The Sun Simulator's polar probability plot (F10.4) is far more compelling interactive than static. **How:** `plotly.graph_objects.Scatterpolar`, embedded via `st.plotly_chart`.

### React + Vite + Tailwind + shadcn/ui
**What:** A production-grade frontend. **Why:** Only if presentation is explicitly scored or you are building this beyond the competition. **How:** Vite scaffold, Tailwind for styling, shadcn/ui for components, `fetch` against the FastAPI backend. **Detail:** **This costs 2–3 days that would otherwise go to modelling.** In a 15-day window with a hidden-label metric-scored competition, that is almost always the wrong trade. Choose it only with clear eyes.

### pytest
**What:** Testing. **Why:** Your transform algebra (F3.4) and submission validator (F8.2) are exactly the kind of code where a silent bug costs the competition and a test costs ten minutes. **How:** Test round-trip identities, canonicalization invariance, submission schema, and metric correctness against hand-computed examples. **Detail:** You do not need broad coverage. You need tests on the four or five functions where a silent error is unrecoverable.

### Kaggle Notebooks / Google Colab Pro / Lightning AI Studios
**What:** GPU access. **Why:** If the team lacks local GPUs. **How:** Kaggle gives ~30 GPU-hours per week free with P100/T4s and 12-hour sessions; Colab Pro gives better GPUs and longer runtimes for a modest monthly fee. **Detail:** Session timeouts are the enemy. **Checkpoint every epoch to persistent storage** and write your training loop to resume from the last checkpoint automatically. Losing an eleven-hour run to a disconnect on Day 17 is a preventable disaster.

---
---

# PART G — RESOURCES, RISKS, AND CHECKLISTS

## G1. Reading List

**On the illusion itself — read these first, they are short and they are the problem:**
- Wikipedia, "Crater illusion" — the concise statement of the phenomenon and its cause.
- EarthSky, "The crater-dome illusion" — includes the Victoria Crater rotation demonstration, which is the canonical visual.
- CosmoQuest, "Lighting Effects Guide" and "Why Does It Look Like That? Illumination & Optical Illusions" — written to train human lunar-crater annotators, which makes them unusually directly relevant. The second one walks through exactly the shading reasoning in A3.
- Popular Science, "What is the crater illusion?" — good plain-language account of why orbital geometry produces the effect.
- Bernabé-Poveda & Çöltekin, "Prevalence of the terrain reversal effect in satellite imagery" (International Journal of Digital Earth) — the peer-reviewed treatment, and the source for the claim that both 180° rotation and photometric negation remove the illusion. **This paper is the citation for your augmentation strategy.**
- Çöltekin et al., "Sunshine around the middle Earth" (2024) — on how illumination direction changes the prevalence of the illusion.

**On lunar terrain and illumination physics:**
- Search terms: "shape from shading lunar", "Lunar-Lambert photometric function", "Hapke model", "LRO NAC illumination".
- Recent work on lunar topographic reconstruction under varying solar geometry documents that brightness gradients constrain surface normals strongly *along* the illumination direction and weakly perpendicular to it — the justification for the directional-derivative input channels in C3 Level 4.

**On crater detection and classification (read for domain insight, not architecture):**
- Silburt et al., "Lunar crater identification via deep learning" (Icarus, 2019) — the standard reference; note it works on DEMs rather than optical images, which is precisely why illumination is not their problem and is yours.
- Reviews of automated crater detection note that variations in image resolution, illumination conditions, and viewing angles routinely cause discrepancies in detection — you are being scored on exactly this failure mode.
- LROC-PANGU-GAN (arXiv 2310.02781) — on synthesizing realistic lunar imagery, relevant if you consider synthetic data augmentation.

**On the ML techniques:**
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" — the conditioning method in C3 Level 3.
- Geirhos et al., "Shortcut Learning in Deep Neural Networks" — the theoretical frame for A5 and B2. **Assign this to whoever owns robustness.**
- Cohen & Welling, "Group Equivariant Convolutional Networks" — background for the equivariance framing in C4 and for escnn.
- Guo et al., "On Calibration of Modern Neural Networks" — temperature scaling, needed for Phase 5.5.
- Xie et al., "Self-training with Noisy Student" — if you pursue pseudo-labelling.
- The timm documentation and Ross Wightman's training recipes — the practical fine-tuning defaults.

**Competition craft:**
- Read winning solution writeups from any Kaggle image-classification competition with a small dataset and a domain-physics twist. The recurring pattern in all of them is: trustworthy CV, domain-appropriate augmentation, diverse ensembles, careful threshold handling. That is exactly this plan.

---

## G2. Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Submission format rejected | Medium | Fatal | F8 validator; dry-run submission on Day 3 | OPS |
| R2 | Class mapping inverted | Low | Fatal | Explicit assertion + known-label check (Phase 8.6) | OPS |
| R3 | Azimuth convention miscalibrated | Medium | Severe | Phase 1.5 empirical calibration; mean-image validation gate | DATA |
| R4 | CV optimistic due to near-duplicates | High | Severe | Grouped folds (Phase 1.8/1.9); compare grouped vs random | DATA |
| R5 | Model relies purely on shadow shortcut | High | Severe if test is adversarial | Phase 5 stress tests; consistency losses; label-flip augmentation | MODEL |
| R6 | Train/test distribution shift | Medium | Severe | Adversarial validation; canonicalization; uniform-azimuth augmentation | DATA |
| R7 | Augmentation corrupts azimuth-label physics | High if unmanaged | Severe | F3 module + unit tests + visual debugger | MODEL |
| R8 | Overfitting on 7.8k samples | High | Moderate | Pretraining, regularization, early stopping, ensembling | MODEL |
| R9 | Chasing noise (<1 pt differences) | High | Moderate (wasted days) | Multi-seed protocol; publish the noise floor to the team | MODEL |
| R10 | GPU/session loss | Medium | Moderate | Per-epoch checkpointing to cloud; auto-resume | OPS |
| R11 | Team divergence in preprocessing/folds | Medium | Severe | Shared modules; frozen `folds.csv`; PR review on core files | OPS |
| R12 | Pretrained weights disallowed | Low | Severe | Confirm Day 1; SSL pretraining fallback ready | OPS |
| R13 | Running out of time | Medium | Severe | Valid submission by Day 3; two-day end buffer; feature priority order | All |
| R14 | Threshold overfit on OOF | Medium | Moderate | Plateau selection, per-fold spread check, π₁ fallback | MODEL |
| R15 | Train/serve preprocessing skew | Medium | Severe | Shared preprocessing functions; parity test in Phase 8.6 | APP |

---

## G3. Daily Standup Template

Fifteen minutes, every day, same time. Four questions:
1. What is our current best OOF Balanced Accuracy, on the frozen folds, and which run produced it?
2. What is our current shortcut-wrong-subset accuracy and synthetic-inversion stress-test score?
3. What is blocking anyone?
4. Is our latest valid submission file still current, and would it still upload successfully today?

Question 4 is not paranoia; it is the discipline that makes the deadline a non-event.

---

## G4. Final Pre-Submission Checklist

Print this. Tick every box, out loud, with a second person watching.

**Format**
- [ ] Exactly 2,001 lines
- [ ] Header exactly `image_id,label`
- [ ] All `image_id` values carry the `.png` suffix
- [ ] ID set matches `test_metadata.csv` exactly; no duplicates; no missing
- [ ] Labels are integers in {0, 1}; no floats, no strings, no nulls
- [ ] No index column; UTF-8; no BOM; single trailing newline
- [ ] File is a single CSV, correctly named per platform requirements

**Correctness**
- [ ] Predictions generated by the exact production inference path
- [ ] Known-label training images pass through that path correctly (inversion check)
- [ ] Threshold read from frozen config, not recomputed
- [ ] Preprocessing constants (normalization, δ, s) match those used in training
- [ ] `test_probs.npy` saved and backed up before thresholding

**Sanity**
- [ ] Predicted class balance plausible relative to training prior
- [ ] Shortcut-baseline agreement rate recorded and understood
- [ ] Agreement with previous submission recorded and explained
- [ ] 40 predicted images visually spot-checked by a human
- [ ] Probability distribution is bimodal, not piled at the threshold

**Process**
- [ ] Submitted at least 6 hours before the deadline
- [ ] Platform acceptance confirmed and screenshotted
- [ ] Rollback submission file retained
- [ ] Repo tagged at the producing commit
- [ ] README allows a stranger to reproduce the file

---

## G5. Traceability — Every Element of the Brief, and Where It Is Addressed

| Element of the problem statement | Where addressed |
|---|---|
| Online ML challenge, image classification / computer vision | A2 (task framing; why not detection/segmentation) |
| 256 × 256 images | A2, F1.2, Part F (resolution sweep 4.2) |
| Grayscale | A2, B7, C3 Level 4, F (timm `in_chans`) |
| Lunar surface imagery | A2 (airless-body shadow physics), A3, G1 |
| Class 0 — Depth: craters, holes, depressions | A2 (sub-populations), A3 (shading signature), C5 |
| Class 1 — Rise: mounds, hills, rocks, boulders | A2 (sub-populations), A3 (shading signature), C5 |
| Binary classification | Throughout; A6 (threshold), F7 |
| Appearance changes with sunlight direction | A3 in full, C1, C2, F10 |
| Crater can look like a depression or a rise | A3, B1, C4, Phase 5.2 |
| Topographic inversion (named phenomenon) | A3, G1 (literature under all four names) |
| `sun_azimuth_angle` in metadata | A4, C1, C3, Phase 1.4, Phase 1.5 |
| "Use this information appropriately" | A4, A2 (clause analysis), C1–C4 |
| 7,854 training images | A2 (consequences), B5, Phase 1, F1.3 |
| 2,000 evaluation images | A2 (noise floor, unlabeled corpus), C4, Phase 4.8 |
| `train_metadata.csv`: image_id, sun_azimuth_angle, label | F1.1, Phase 1.1 |
| `test_metadata.csv`: image_id, sun_azimuth_angle | F1.1, A4 (azimuth available at inference) |
| Evaluation labels hidden | A2, Phase 2, Phase 5, F5 (CV is the only signal) |
| Must train on the provided training dataset | Phase 0.1 (rules check on external data/pretraining) |
| Predict for all 2,000 evaluation images | A2, A6, F8 |
| Submission CSV: exactly `image_id,label` | A7 in full, F8, G4 |
| Example rows `eval_00001.png,0` | A7 item 3 (the `.png` suffix trap) |
| Balanced Accuracy metric | A6 in full (derivation of t = π₁), B4, F7 |
| "Focus on underlying terrain, not solely shadow patterns" | A5 in full, B2, C5, Phase 5, F9, F12 |
| Dataset composition / metadata structure per competition doc | Phase 0.1, Phase 1.1–1.3 |
| Opens 1 Sep 2026 | Part D header (5 days already elapsed) |
| Closes / deadline 21 Sep 2026, 11:59 PM | Part D timeline, Phase 8.9, Phase 0.1 (timezone) |
| Competition round: access data, develop, train, submit | Phases 1–8 |
| Final submission must contain all 2,000 predictions in the prescribed format | A7, F8, G4 |

---

## G6. If You Only Do Five Things

If the fortnight compresses and you must triage, these five, in this order, capture most of the available value:

1. **Calibrate the azimuth convention empirically and canonicalize every image** (Phase 1.5, Phase 1.6, C1). This is the idea the competition is built to reward.
2. **Freeze group-aware, azimuth-stratified folds on Day 1 and never deviate** (Phase 1.8–1.9). Everything you learn is only as good as this.
3. **Build the azimuth-aware augmentation module and never use a raw library flip or rotation** (C2, F3). This alone puts you ahead of most of the field.
4. **Threshold at the class prior, tuned on out-of-fold predictions** (A6, F7). A free point that many entrants will miss.
5. **Produce a validated submission by Day 3 and re-validate before every upload** (Phase 2.7, A7, F8, G4). This converts a catastrophic risk into a non-event.

Everything else in this document — the consistency losses, the ensembles, the TTA, the Sun Simulator — is upside on top of a solution that these five things already make competitive.

---

*End of playbook.*
