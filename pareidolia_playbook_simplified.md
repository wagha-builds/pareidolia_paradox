# The Pareidolia Paradox — Complete Team Playbook (Simplified Edition)

**A start-to-finish guide for telling apart "dips" (craters, holes) from "bumps" (mounds, hills, boulders) on the Moon's surface — even though the same shape can *look* like either one, depending on which way the sunlight is falling.**

Prepared: 6 September 2026 · Competition window: 1–21 September 2026 · **Days remaining: 15**

---

## Table of Contents

**PART A — Understanding the Problem**
- A1. The short version
- A2. Going through the problem statement, sentence by sentence
- A3. Why lighting can make a hole look like a hill (the real heart of the challenge)
- A4. What `sun_azimuth_angle` actually is, and why they gave it to you
- A5. Reading between the lines of the organizers' warning
- A6. Balanced Accuracy: how it's calculated, and what that means for you
- A7. A close look at the submission file format

**PART B — The Pain Point Register**
- B1. Fourteen common ways teams get stuck, each with a cause and a fix

**PART C — The Core Strategy**
- C1. Putting every image into one "standard lighting"
- C2. The rules for how the sun's direction changes when you flip or rotate an image
- C3. Ways to tell the network which direction the sun is coming from
- C4. Teaching the model the underlying physics directly
- C5. Building features by hand that reflect the actual geometry

**PART D — Day-by-Day Plan**
- Phase 0 through Phase 9, each with concrete steps, an owner, something to hand in, and a clear "you're done when..." checkpoint

**PART E — Features to Build Into the App**
- Fifteen features, each explained: what it is, why you need it, and how to build it

**PART F — The Full Toolkit**
- Must-have tools, nice-to-have tools, and stretch-goal tools, with what/why/how for each

**PART G — Reading List, Risk Tracker, and Final Checklists**

---
---

# PART A — UNDERSTANDING THE PROBLEM

## A1. The Short Version

Take away all the framing, and this is a **yes/no picture-sorting task**: 7,854 labelled training photos, 2,000 photos you'll be scored on (without knowing the right answers), each photo 256×256 pixels and black-and-white, plus one extra number per photo, and a scoring method called Balanced Accuracy. On the surface, that's something anyone who's built an image classifier before could knock out in a couple of hours.

It isn't that simple, and the reason is hiding in one sentence of the brief: *"A crater illuminated from one direction can appear like a depression, while the same formation under different lighting can resemble a raised surface."*

In plain terms: **the same picture doesn't always mean the same thing.** The exact same 256×256 pixels can honestly be labelled "dip" or "bump" depending on which direction the sunlight was coming from (that's the `sun_azimuth_angle` value). Some pairs of images in this dataset look almost pixel-for-pixel identical but have opposite correct answers. Any model that only looks at pixels has a hard limit on how well it can possibly do — and that limit depends on how often the dataset happens to mix up "what it looks like" with "which way the light was shining." The true answer depends on **both the image and the sun's direction together**, never on the image alone.

That one fact changes almost everything about how you should approach this project:

1. The sun-direction number isn't a nice-to-have extra feature — it's **essential for resolving a genuinely confusing signal.** Treating it as optional is the single most common way teams lose this competition.
2. The usual "bag of tricks" for augmenting images is actually dangerous here. Randomly rotating or flipping images — three of the most common tricks in all of computer vision — quietly break the real-world link between the image and its recorded sun direction. Do this carelessly and you're teaching your model that lighting direction is meaningless noise, which is exactly backwards.
3. There's a **shortcut** available (just check which side is dark and which is light) that will look great on a naive train/validation split but might fall apart on the real hidden test set. The organizers have flat-out warned you about it. How much you lean on this shortcut versus try to beat it is the biggest strategic call your team will make.
4. Because you never see the real answers for the evaluation set, and you likely only get one shot at submitting, **your own internal testing setup is the only feedback you will ever get.** Building a trustworthy way to test your model on your own is not a side task — it's the single most valuable piece of engineering in this whole project, and it needs to happen before any serious training begins.

The winning approach, in one paragraph: **figure out, from the data itself, exactly how the sun-direction number lines up with the pixels; rotate every image so the sun always appears to come from the same direction; train a group of pretrained image models on these "standardized" images, using an augmentation strategy that respects the lighting physics and a loss function that enforces that physics; add a trick where you flip the brightness of an image to instantly create a second, correctly-labelled example (doubling your usable data for free); test using a validation setup that respects which images come from the same source photo; pick your decision cutoff to match the true class balance; apply lighting-aware test-time tricks; and wrap it all in an app whose star feature is an interactive "move the sun" slider that visibly proves your model isn't fooled by the very illusion the competition is named after.**

Everything below unpacks that paragraph in detail.

---

## A2. Going Through the Problem Statement, Sentence by Sentence

Below, I'll walk through every part of the brief you were given and explain what it actually means for what you'll do. Nothing here is just commentary — each point changes something real about your plan.

### "an online machine learning challenge focused on image classification and computer vision"

This tells you your work is judged by a number, not by how clever or polished it looks. It also tells you this is **not** a "find and outline the object" (detection) or "trace its exact edges" (segmentation) task. You're not drawing boxes around craters or tracing their outlines — each 256×256 photo is assumed to show **one main feature**, and your job is just to name what kind it is. That has a real design consequence: you want a simple "look at the whole picture, then decide" model, not a "find objects in the scene" model. Much of the published research on lunar craters is about finding and outlining them, and if your team reads that material without filtering it, you'll waste days building the wrong kind of system. Read that research for what it teaches about lighting and shapes, not for which architecture to copy.

### "classify 256 × 256 grayscale lunar surface images"

Four separate practical facts are packed into this sentence.

**256×256** means every image is the same fixed size — a real gift, since you don't have to handle different image shapes or sizes. It also happens to sit nicely between the 224×224 size most pretrained models expect and the 384×384 size many can be fine-tuned at just as well. By default, train at the native 256 size (almost all modern models handle this size just fine), and try one experiment where you scale up to 320 or 384, since fine shadow and rim details sometimes benefit from a bit more resolution. Don't shrink down to 224 "to match the pretrained model" — that throws away nearly a quarter of your pixels for no real reason.

**Grayscale** means each image has just one channel of brightness info, not three (red/green/blue). Most pretrained models expect three channels. You have three real options here, and they're not the same. The simplest is to copy the single channel three times — it works, it's what most people do, and it wastes two-thirds of the model's first layer on duplicate information. The cleaner option is to use `timm.create_model(..., in_chans=1)`, which automatically combines the pretrained red/green/blue filters into a single-channel filter, keeping the useful learned patterns while using a third of the input space. **The cleverest option, worth testing, is to build your own 3-channel input**, where one channel is the plain image, and the other two are measures of how brightness changes along and across the sun's direction. This hands the network the physically meaningful information directly. More on this in Part C5.

**Lunar surface** tells you the setting: no air, so no haze softening the light, meaning **shadows here are extremely dark and extremely crisp-edged.** On Earth, scattered skylight fills in shadows with a soft fade. On the Moon, a shadowed patch only gets a little light bounced off nearby lit slopes, so it's nearly black with a hard edge. This is actually helpful — shadow regions are easy to separate out, and simple shape analysis of the shadow (its position, whether it's inside the feature or spills outside it, its shape) tells you a lot. It also means **there's no colour or plant/water/building cue to rely on** — the only information in the image is shading, shadow, and surface texture. Your model has less to latch onto than with everyday photos, which is why starting from a model trained on everyday photos (ImageNet) helps less here than usual, and why manually engineering features tailored to this problem helps more than usual.

**Images**, treated as separate and unconnected, with no mention of which photo each was cut from. This is a trap covered more in B10: even though the brief doesn't say so, small tiles cut out from a handful of larger orbital photos are almost certainly going to include near-duplicates and overlapping content. If two crops of the same crater land in different train/test splits during your own testing, your test scores will look better than they really are, and you'll end up making bad decisions all week based on that false confidence. You need to actively check for this.

### "into one of two categories: Class 0: Depth — Craters, holes, and surface depressions. Class 1: Rise — Mounds, hills, rocks, and boulders"

Notice carefully: these are **not two simple categories — they're two broad umbrella categories**, and the things grouped inside each one can look very different from one another.

Inside Class 0 (Depth): a fresh crater is a near-perfect circular bowl with a raised edge, a sharp curved shadow inside, and maybe a bright ring of debris around it. An older, worn-down crater is a shallow, soft-edged dip with no raised edge. A "hole" (like a collapsed lava tube opening) is a steep, often irregular, almost pitch-black shaft. A general surface dip might not even have a clear boundary.

Inside Class 1 (Rise): a boulder is small, high-contrast, just a few dozen pixels across, with a long thin shadow stretching out from it. A mound or dome is large, smooth, gently shaded, and might not cast any visible shadow at all. A hill is a large, steep, positive bump with a big triangular shadow. A rocky outcrop is a cluster of many small shadow-casting rocks.

**This has real, important consequences that most teams overlook.**

First, **there's a lot more variety within each class than a "two categories" framing suggests.** A boulder and a mound are both labelled "Rise," but they look nothing alike; a boulder and a small fresh crater at similar scale might actually resemble each other more than they resemble other members of their own class. This means your model needs enough capacity to learn several very different "looks" per class, and it argues strongly against using tiny or overly simple models as your main approach. If you have the time, **sorting your training images into rough sub-types by eye and checking accuracy per sub-type is one of the most useful things you can do** — it'll quickly show you, say, that your model gets 96% right on fresh craters but only 71% on worn-down ones, telling you exactly where to focus your remaining time.

Second, the two classes are **not mirror images of each other in how their shadows behave**, and this is the single most useful piece of real-world physics in the whole problem. For a dip, the shadow is **cast by the near edge down into the inside of the feature**, so it's *bounded by and stays within* the feature's own outline. For a bump, the shadow is **cast by the feature outward onto the flat ground around it**, so it *spills beyond* the feature's outline. This "does the shadow stay inside or spill out" property stays true no matter which direction the light comes from — unlike simple "is it dark or light" — and I'll show you how to turn it into a usable feature in C5. If the organizers have deliberately messed with simple dark/light patterns to punish teams relying on shortcuts, this "stays inside vs. spills out" property is likely to survive.

Third, **scale matters, and you haven't been told how much ground each pixel covers.** A 256×256 tile might cover 50 metres or 5 kilometres of terrain. If the dataset mixes scales, then how big a feature looks in pixels doesn't tell you much, and it's fine to randomly resize images during training. If the dataset is all one scale, then apparent size becomes a real, useful clue (boulders are small; craters vary widely), and aggressive resizing during training would throw away useful information. **Figure this out for yourself early on** by checking whether feature sizes cluster into distinct groups.

### "The appearance of lunar terrain changes depending on the direction of sunlight... This phenomenon is known as topographic inversion."

This is the core idea behind the whole competition, and I cover it fully in A3 below. Worth noting: this effect is also known in the literature as the crater illusion or dome illusion — it happens because humans are used to seeing light come from above, so under unusual lighting, a bowl-shaped dip can look like a raised dome. It's also called the terrain reversal effect, and researchers have noted that **either rotating an image 180° or turning it into its brightness-negative can remove or create the illusion.** Those two facts — rotation and brightness inversion both flip the apparent shape — aren't just trivia. **They're the two techniques at the very center of the approach in this playbook.** Look this up under any of these four names: topographic inversion, relief inversion, terrain reversal effect, crater/dome illusion.

### "To account for this, each image is accompanied by a sun_azimuth_angle in the metadata. Participants are expected to use this information appropriately while developing their models."

Read the last five words again: **"use this information appropriately."** This isn't filler text. The organizers are telling you two things at once: that using the sun-direction number is expected (so a model that only looks at pixels is knowingly incomplete), and that there's a *right* and a *wrong* way to use it. The wrong way is to just tack the raw number onto your list of features and hope the model figures it out — this doesn't work well, for reasons explained in A4 and C3. The right way is to use it as a **geometric adjustment** that puts every image into a shared, standard lighting setup, and/or as a **steering signal** fed into the model in a smart way (explained in C3). This sentence is basically the organizers pointing you toward the intended solution.

### "7,854 training images / 2,000 evaluation images"

7,854 is a **small dataset by deep-learning standards**, and that shapes nearly every decision you'll make. Splitting it 5 ways for testing leaves you training on roughly 6,283 images per split. At that size:

- Training a large model completely from scratch is out of the question — you'll overfit within a few passes through the data. **Starting from a model already pretrained on everyday photos is a must**, not optional.
- Keep your model modest in size. Medium-sized models (ConvNeXt-Tiny, EfficientNetV2-S, ResNet-50, Swin-Tiny) are the right ballpark. A huge model will just memorize your training set and teach you nothing useful.
- **You need strong safeguards against overfitting**: heavy but sensible data augmentation, a decent amount of weight decay, some "stochastic depth" regularization, light label smoothing, and stopping training early based on a proper validation score.
- **Use cross-validation (multiple train/test splits), not just one split.** A single 80/20 split gives you only about 1,570 validation images, and the natural statistical wiggle in an accuracy score at that size is around ±1.2 percentage points — bigger than the differences you're actually trying to detect between models. Splitting the data 5 different ways and combining the results gives you a score based on all 7,854 images and a much steadier estimate. Use it. **Repeating this with different random seeds, if you have the compute budget, is worth doing too.**
- The whole training set fits comfortably in memory (about 515 MB as raw bytes). **Load the whole dataset into memory once, as a single array.** Don't build a pipeline that re-reads image files from disk every training pass — you'll be needlessly slow, and your training runs could take five times longer than necessary. This one decision alone could buy you 3–5× more experiments over the two weeks, which matters a lot given the deadline.

2,000 evaluation images is a **small test set**, and that has a consequence people often overlook: the **natural statistical wiggle in your final Balanced Accuracy score is roughly ±1 percentage point.** With 1,000 images per class, that means **differences under about 1.5 points between two submissions are basically noise.** Don't burn three days chasing a 0.4-point improvement in your own testing. Spend that time instead making your model robust to unexpected differences in the real test set, which has a much bigger potential payoff. It also means luck plays a real role in the final ranking, so the smart strategy is to aim for the best *expected* score with a solid, reliable method rather than squeezing the last drop out of something fragile.

Also worth noting: those 2,000 unlabeled evaluation images are a substantial **extra resource** — roughly a quarter the size of your labelled set. Legitimate ways to use them include test-time tricks (averaging predictions across small transformations of the same image), adapting your model's internal statistics to match the test images, and using confident predictions on the test set as extra "soft" training data. Double-check the competition rules for anything that prohibits using test data, but if there's no such rule, these techniques are standard practice and can be worth a point or more.

### "train_metadata.csv containing image_id, sun_azimuth_angle, and label / test_metadata.csv containing image_id and sun_azimuth_angle"

Three columns and two columns, respectively. The key structural fact is that **`sun_azimuth_angle` shows up in both files.** That means it's available when you make predictions on the test set too, which is what makes it legal to use for lining up images and possible to standardize the test set the same way. If it had only been available for training, you'd have had to estimate it from the pixels instead. (You should build that estimator anyway, as a cross-check — see Phase 1, Step 6.)

Just as important is what's **missing**. There's no sun elevation (how high in the sky the sun is). This is a real limitation — both direction and elevation matter for how a photo looks, and without elevation, **shadow length is an unpredictable extra variable.** A high sun gives short shadows and low contrast; a low sun gives long shadows and dramatic contrast. Your model needs to handle both. Two practical responses: include brightness/contrast randomization in your training augmentation so the model doesn't rely on absolute contrast, and consider **estimating a stand-in for sun elevation from each image** (like the fraction of very dark pixels, or the overall contrast) and feeding that to your model as an extra clue. This is a genuine edge — most teams won't think to reconstruct this missing piece of information.

Also missing: any way to tell which source photo a tile came from, any location or timestamp, and any size information. If there were a source-photo ID, it would solve your "avoid near-duplicates in different splits" problem for free. Since there isn't one, you'll need to **figure out that grouping yourself** by detecting near-duplicate images (Phase 1, Step 8).

### "The evaluation labels will remain hidden and will be used for final scoring."

No feedback loop, no public leaderboard to check yourself against. **Your own internal testing score is your entire source of information about how good your model is.** Everything in Part D, Phase 2 exists to make that internal testing trustworthy. If your validation setup is flawed, every decision you make for the next two weeks is built on bad information, and you won't find out until the results come back.

One consequence: **don't tune anything based on the test set, and don't let your team fall into the habit of "just checking" a submission's predicted class balance and adjusting until it looks right.** That's essentially probing a leaderboard you can't see, and it's how teams accidentally overfit to a set they can't actually observe. The one legitimate exception is a **check for whether train and test data were drawn differently** — comparing the *statistical patterns* (not labels) of train vs. test, like sun-direction distributions or brightness levels. This uses no labels and is entirely sound.

### "Participants must train a classification model on the provided training dataset"

Note "on the provided training dataset." Check your competition rules carefully to see whether using outside data or pretrained models is allowed. Pretrained models (like ones trained on ImageNet) are almost always allowed, and I'm assuming that's the case here — but **if pretrained weights turn out to be banned, your whole plan changes**: you'd need heavier augmentation, self-supervised pretraining on your own combined train+test images (very doable with ~9,854 unlabeled lunar images, and it would set you apart), and smaller models. **Confirm this on Day 1.** It's the single biggest open question in the brief.

### "generate predictions for all 2,000 evaluation images"

**All 2,000.** Not just the ones you feel confident about — there's no option to skip a guess. Every row needs a firm 0 or 1. This connects to Balanced Accuracy in a specific way, covered in A6: since you're forced to guess on every image, and the scoring method averages accuracy per class, how you handle uncertain cases directly affects your score — and the best cutoff point usually isn't the default 0.5.

### "The submission must be a single CSV file containing exactly: image_id,label"

Covered thoroughly in A7. The word "exactly" matters — assume the system checking your file is strict, with no room for near-misses.

### "Submissions will be evaluated using Balanced Accuracy."

Covered in A6.

### "Participants are encouraged to focus on the underlying terrain rather than relying solely on visual shadow patterns."

Covered in A5. Along with the topographic-inversion sentence, this is the most important line in the whole document, and it deserves its own section.

### "Competition Opens: 1 September 2026 ... Closes: 21 September 2026, 11:59 PM"

Today is 6 September. **You have 15 days** — about 15 working evenings if this is happening alongside other commitments. The plan in Part D is scheduled against that. The single most important scheduling rule: **have a valid, submittable answer file ready by the end of Day 2, not Day 15.** A so-so submission in hand on Day 2 that you improve a dozen times beats a great model on Day 21 that fails to upload properly at 11:47 PM. Build the whole pipeline end-to-end first, improve it second.

Also note: the deadline is 11:59 PM on the 21st, and the text doesn't say which timezone. **Confirm the timezone on Day 1**, and set your own internal deadline at least six hours earlier than the official one. Upload systems go down, internet connections fail, files get corrupted. Treat your real deadline as 21 September, 6:00 PM your time.

---

## A3. Why Lighting Can Flip How a Feature Looks — Explained Fully

Everything strategic in this document comes from this section, so I'll walk through the reasoning carefully instead of just stating it.

### How the photos are taken

Orbital photos of the Moon are taken **looking straight down**: the camera points directly at the ground below it. Meanwhile, the sun is typically **low on the horizon** in photos used to study terrain shape, because low sun angles create long shadows and strong shading that reveal the bumps and dips clearly. Photos of craters are taken from directly overhead, and a shadow usually only appears inside a crater when the sun's rays come in nearly parallel to the ground; when light comes from the side rather than from directly above, it changes how we perceive the scene.

So: **camera straight down, light from the side.** The `sun_azimuth_angle` tells you which side the light is coming from.

### What a "bump" looks like, in terms of light and shadow

Picture a mound, hill, or boulder — something raised up. Let's say sunlight is arriving from direction **s**.

The side of the mound facing the sun catches more direct light and looks **bright**. The far side, facing away from the sun, gets weak or no direct light and looks **dark**. Beyond that far side, the mound itself blocks light from reaching the ground behind it, so a **shadow stretches out onto the surrounding flat ground, away from the sun.**

**A bump's signature: bright on the side facing the sun, dark on the side away from it, and the dark region extends *outside* the feature's own footprint.**

### What a "dip" looks like, in terms of light and shadow

Now picture a crater — a bowl-shaped dip.

The inner wall on the *sun-facing side of the rim* faces away from the sun and down into the bowl. What's more, the raised rim on that side physically blocks the light, casting a shadow *down into the bowl*. So that near-side interior is **dark**.

The far interior wall — the one on the opposite side from the sun — faces back toward the sun and catches nearly direct light. It looks **bright**. As one lunar-imagery explainer puts it: the light shines over the rim of the crater, lighting up the inside on the far side and casting shadows on the near side.

Crucially, the shadow is **trapped inside the crater's rim**. It can't escape outward — it stays as a crescent shape contained within the feature's own outline.

**A dip's signature: dark on the side facing the sun, bright on the side away from it, and the dark region stays *inside* the feature's footprint.**

### The inversion, stated clearly

Compare these two signatures. Going from the sun-facing side to the far side, **the bright/dark order is exactly reversed between the two classes.** A bump goes bright-then-dark; a dip goes dark-then-bright.

Now think about what happens if you rotate a crater photo 180° without telling anyone. The dark part, which was on the sun-facing side, is now on the far side of the frame. The image now shows the *exact same pattern* as a bump. **A 180°-rotated photo of a dip is pixel-for-pixel indistinguishable from a photo of a bump — unless you also know the sun direction changed by 180° too.**

This is the whole Pareidolia Paradox. The illusion happens because our brains are used to seeing images lit from above, and one quick way to make an image "pop" into its correct shape is to rotate it until the light source appears to come from above. In training materials for people who identify lunar craters by hand, the same pair of images lit from the top-left versus the bottom-right will look like craters to most people in one orientation, and like hills and bumps in the other — the "hills" being an illusion created by our brains' preference for a certain lighting direction.

A neural network falls for exactly the same illusion, for exactly the same reason: it learns a fixed pattern of light and dark as its clue, and that pattern is only meaningful if you also know which way the light was coming from.

### The second way to flip a shape: inverting brightness

There's a second way to flip apparent shape, less well known but just as important. Turning an image into its brightness-negative also removes (or creates) the terrain-reversal illusion, with no rotation needed.

Here's why, in simple terms. An image's brightness roughly follows how directly a surface faces the light. Now imagine the **mirror-image terrain** — every crater becomes a mound of the exact same shape, and every mound becomes a crater. Its brightness pattern turns out to be exactly the *negative* of the original image's brightness.

**So: inverting an image's brightness while keeping the sun direction the same produces the image you'd get from the physically flipped terrain — and that flips the correct label.**

This gives you a way to create label-flipped training examples "for free," which turns out to be extremely useful for teaching a model that appearance alone doesn't determine the class. It's an approximation, not perfectly exact — real shadows aren't perfectly symmetric under this trick, the Moon's surface doesn't reflect light in a perfectly simple way, and brightness variation due to the surface material itself isn't inverted correctly either. Use it with moderate frequency and check empirically that it actually helps rather than just assuming it will. But it's grounded in real physics, it's cheap, and it directly targets the exact failure mode this competition is built around.

### The relationship you need to internalize

| What you do to the image | What you do to the sun direction | Effect on the correct label |
|---|---|---|
| Rotate by θ | Update the sun direction by θ (correct sign) | **Unchanged** |
| Rotate by θ | Leave sun direction unchanged | **Flipped if θ = 180°**; otherwise invalid |
| Rotate by 180° | Leave sun direction unchanged | **Flipped** |
| Invert brightness | Leave sun direction unchanged | **Flipped** |
| Invert brightness | Update sun direction by 180° | **Unchanged** (approximately) |
| Mirror along the sun's direction | Unchanged | **Unchanged** |
| Mirror across the sun's direction | Reflect the sun direction too | **Unchanged** |
| Mirror across the sun's direction | Leave sun direction unchanged | **Invalid — don't do this** |

Print this table. Put it on the wall. Every decision your team makes about augmenting images for the next two weeks needs to be checked against it. Most teams entering this competition will apply a standard random horizontal flip without updating the sun direction — that's row seven in the table, which is *physically wrong*, and they'll end up feeding their model bad, misleading examples while thinking they're just adding helpful variety.

---

## A4. What `sun_azimuth_angle` Actually Is, and Why They Gave It To You

### Definition and the convention used

Solar azimuth is normally measured **in degrees clockwise from north**, so 0° = north, 90° = east, 180° = south, 270° = west. But there's a real ambiguity in any dataset like this that you'll need to resolve before you can use the number properly:

1. **Does the angle point *toward* the sun, or is it the direction the light is *traveling*?** These are 180° apart.
2. **Is the image oriented with north at the top?** Map-style images usually are; raw camera images often aren't.
3. **Does "clockwise from north" in the real world match clockwise or counter-clockwise in the image's pixel grid?** Because image row numbers increase *downward*, a real-world clockwise rotation can end up looking counter-clockwise in the pixel grid, depending on how the image was originally set up.

**You can't figure this out just by reading documentation — you have to work it out from the data itself.** This is Phase 1, Step 5, and it's the single most important experiment in the whole project. Here's the approach:

For each training image, work out a **shadow-direction estimate**: find the average position of the darkest 10% of pixels, subtract the average position of the brightest 10% of pixels, and turn that into a direction. Call this angle φ (measured in the image's own coordinates). Then, separately for each class, average the difference between φ and the recorded sun-direction number.

If the physics holds up and the convention is consistent, you'll see **two tight clusters roughly 180° apart** — one per class. That one chart tells you three things at once: (a) confirms the sun-direction number is meaningful and correctly matched to the pixels, (b) tells you the exact offset and "handedness" needed to convert between the two coordinate systems, and (c) tells you how strong the simple shadow-based shortcut is. If instead you see random scatter, something is off — maybe the sun-direction values are scrambled, the images aren't oriented consistently, or the organizers have deliberately broken the pattern — and you need to know that on Day 1, not Day 12.

### Why you should never feed the raw number directly to a network

The value is **circular**. 359° and 1° are only two degrees apart in reality, but as plain numbers they're 358 apart. A network given the raw number will learn that there's a sharp jump at the "wrap-around" point, which doesn't actually exist in real life. **Always encode it as two numbers: the sine and cosine of the angle.** This maps the angle onto a circle in a way that correctly captures how close two directions really are, and it's not optional. Any time your pipeline touches this sun-direction number — for feeding the model, for grouping images, for organizing your validation splits, for logging — make sure you're handling the wrap-around correctly.

Similarly, when you compute averages or spreads of these angles (for exploring the data, for setting augmentation ranges), use tools designed for circular data, not plain arithmetic averages.

### Why they gave it to you

Three reasons, from least to most important.

The trivial reason: it's a useful piece of information, and useful information helps. The real reason: **without it, the problem doesn't have one right answer per image** — the label genuinely depends on more than just the pixels, so any model that only sees pixels has a built-in error rate determined by how often the dataset contains these confusing, illumination-ambiguous pairs. The deepest reason: **it's the key to the intended solution.** It's not primarily meant to be treated as an extra feature — it's meant to be used as an instruction for how to rotate each image into a standard, shared lighting setup. Teams that treat it as just another feature get a modest improvement. Teams that treat it as an instruction for rotating images get a much bigger one.

---

## A5. Reading Between the Lines of the Organizers' Warning

> *"Participants are encouraged to focus on the underlying terrain rather than relying solely on visual shadow patterns."*

Take this seriously — read it as a warning, not just friendly advice.

### What the shortcut is

From A3: rotate the image so the sun points toward the top; check whether the top half of the central feature is darker or lighter than the bottom half; if darker, guess "dip," otherwise guess "bump." This is a few lines of code, needs no training at all, and on clean, easy data could plausibly score in the high 80s or low 90s percent.

### Why the organizers warned you about it

There are two possibilities, and you should prepare for both.

**Possibility one: it's a genuine hint about where the real difficulty lies.** The dataset likely contains a good number of images where the simple dark/light pattern is weak, missing, or misleading — high-sun photos with barely any shadow, badly worn-down features with soft shading, overlapping features whose shadows interfere with each other, or features so tiny that the shadow is just a handful of pixels. On these, the simple shortcut fails, and a model that genuinely understands shapes wins.

**Possibility two, and the one I'd bet on: the evaluation set was deliberately built to punish the shortcut.** The most natural way for organizers to design this kind of competition is to build the test set so its sun-direction pattern differs from training, or so it contains an unusually high number of confusing, illumination-ambiguous cases, or — most aggressively — so that some test images are rotated versions of training-like images with correspondingly updated sun-direction values, specifically so a model relying on raw appearance gets them wrong. A competition literally named "The Pareidolia Paradox," which hands you the sun direction and then warns you not to lean on shadows, is basically pointing this out directly.

### What "underlying terrain" means, concretely

Signals that are *not* the simple dark/light shortcut, which you should deliberately build your model's ability to use:

- **Does the shadow stay inside the feature, or spill outside it?** (A3) — a crater's shadow is trapped by its own rim; a mound's shadow escapes onto the surrounding ground. This is about shape and structure, and stays true regardless of which direction the light comes from.
- **Rim structure.** Fresh craters have a raised edge that creates a bright ring on the sun-facing outer slope and a faint dark ring on the far outer slope — a *ring* pattern that mounds don't have.
- **How sharp and closed the boundary is.** Impact craters are nearly circular with clean, closed outlines; boulders are angular; mounds are irregular and often don't have a clean closed boundary at all.
- **Debris and surroundings.** Fresh craters may have bright rays of debris or a ring of nearby smaller rocks; mounds don't.
- **Interior texture.** Crater floors are often smooth and flat (filled in over time); mound tops often blend seamlessly into the surrounding surface texture.
- **Size patterns.** Boulders tend to be small and within a narrow size range; craters span a much wider range of sizes.

---

## A6. Balanced Accuracy — How It's Calculated, and What That Means

### The definition

Balanced Accuracy is simply the average of how well you do on each class separately:

```
BA = ½ · ( TP/(TP+FN)  +  TN/(TN+FP) )
   = ½ · ( recall on class 1 + recall on class 0 )
```

In other words: it's your accuracy as if both classes appeared equally often, no matter how common each one actually is in reality.

### What this means #1: an imbalanced training set doesn't hurt on its own, but ignoring the imbalance does

If your training set is lopsided — say, 65% dips and 35% bumps — a model trained the normal way will lean toward guessing "dip" more often, since that's usually right. Its plain accuracy might look fine, while its Balanced Accuracy suffers a lot, because it's doing badly on the smaller class.

Three complementary fixes — use at least two of them:

1. **Weight the training loss.** Give more importance to the smaller class during training. Simple and effective, with no extra data handling needed.
2. **Balance how you sample data during training** so each training pass sees roughly equal numbers of both classes.
3. **Adjust your decision cutoff at prediction time.** Covered below — this is often the cheapest fix and, on its own, often the single most effective one.

**Check the actual class balance on Day 1** — it's a one-line calculation — and pick your fixes accordingly. If it turns out to be roughly 50/50, all of this becomes a non-issue, but don't just assume that.

### What this means #2: the best decision cutoff isn't 0.5, it's the training-set class balance

This is a precise, provable result worth knowing.

If your model's predicted probability is well-calibrated (meaning a "70% confident" prediction really is right about 70% of the time), then a bit of math shows:

**The Balanced-Accuracy-optimal cutoff is t = π₁, the fraction of the training set that belongs to class 1.** If your training set is 65% dips / 35% bumps, set your cutoff at 0.35, not 0.5. If it's balanced, the cutoff stays at 0.5.

In practice your model's probabilities won't be perfectly calibrated, so **do both**: use t = π₁ as your default starting point, and also test a range of cutoffs on your own validation predictions to find what actually works best. If the two agree closely, great — you're well calibrated. If the empirical best is wildly different from π₁, your model's probabilities are poorly calibrated, and you should apply a calibration fix before choosing a cutoff.

**Watch out for overfitting your cutoff choice.** The "best" cutoff found on your validation data is itself just an estimate, with some randomness in it. Two safeguards: (a) look at the accuracy-versus-cutoff chart and pick a point in the middle of a *flat, stable stretch* rather than a sharp, narrow spike — sharp spikes are usually just noise; (b) compute the best cutoff separately for each validation split and check how much it varies. If it swings wildly from split to split, don't trust any single value — fall back to the class-balance-based cutoff instead.

### What this means #3: every mistake costs the same within its class, but classes aren't weighted equally

There's no asymmetry to exploit — a wrong "bump" guess and a wrong "dip" guess cost the same relative to their own class's total. But because they're measured relative to their own class size, **a mistake on the smaller class costs more overall than a mistake on the bigger class.** If bumps are 35% of the data, each bump mistake costs more toward your score than each dip mistake. This is exactly what the cutoff adjustment compensates for, and it's why you should spend proportionally more of your error-checking time on the smaller class.

### What this means #4: implement the metric correctly, once

Use the standard `balanced_accuracy_score` function from scikit-learn — don't build your own version by hand. Don't accidentally optimize for a different metric by mistake, just because that's what your training code prints by default. **Every early-stopping decision, every checkpoint choice, every comparison between settings, and every number you log should be Balanced Accuracy on your held-out validation predictions, using your tuned cutoff.** Optimizing for one metric while being scored on a different one is one of the most common and avoidable ways teams lose competitions.

A useful detail: ROC-AUC (a threshold-independent ranking score) is still worth tracking alongside Balanced Accuracy, because it separates "my model ranks things correctly but my cutoff is wrong" from "my model just ranks things badly." Log both — use AUC to diagnose, use Balanced Accuracy to choose your final model.

---

## A7. A Close Look at the Submission File Format

The brief specifies:

```
image_id,label

eval_00001.png,0
eval_00002.png,1
```

Assume the system reading your file is strict. Here is the complete checklist your submission needs to pass, and every single item on it has cost some team, somewhere, a competition:

1. **Exactly 2,001 lines**: one header line plus 2,000 data rows. Count them.
2. **The header must be exactly `image_id,label`** — lowercase, no extra spaces, no odd characters.
3. **`image_id` must include the `.png` extension.** The example makes this explicit. A pipeline that strips file extensions internally for convenience, and forgets to add them back, produces 2,000 unmatched rows and a score of zero. This is the single most common mistake.
4. **The `image_id` values must exactly match `test_metadata.csv`**, as a complete set. Check this with code, and also check there are no duplicates.
5. **`label` values must be plain integers, 0 or 1.** Not `0.0`. Not `"0"` in quotes. Not `True`/`False`. Not probabilities. Convert to plain integers before saving.
6. **No extra index column** in the file. Forgetting to disable it adds an unwanted extra column that breaks the expected format.
7. **No stray whitespace, no extra blank line at the end** beyond a single closing newline, and consistent line endings.
8. **Ordering**: safest to match the order in `test_metadata.csv` exactly. Most systems match rows by `image_id` regardless of order, but matching costs nothing and removes one source of risk.
9. **No missing values** anywhere in the label column.
10. **A single CSV file**, not a zip or folder, unless the platform specifically asks for that.

Beyond just being technically valid, run these **sanity checks** on every submission before you upload it:

- **Predicted class balance.** If your training set is roughly balanced and your submission is 94% one class, something's broken — most likely a cutoff bug or the labels got swapped somewhere.
- **The label-swap check.** This is the worst-case scenario: somewhere in your code, the two classes accidentally get swapped, and you submit a genuinely good model with every single prediction flipped — scoring roughly the *opposite* of what you actually achieved. Protect against this by explicitly checking your class mapping in code, and by running your final model on a **handful of training images with known correct answers**, through the exact same process you use for real predictions, and confirming it gets them right. Do this as the literal last step before uploading.
- **Agreement with your previous best submission.** If your new model only agrees with your last submission on 60% of rows, either you've made a huge breakthrough, or (much more likely) you've introduced a bug. Investigate before uploading.

**Build this checking tool as its own script on Day 2**, before you even have a good model. Run it automatically as the last step of every prediction run, so it's physically impossible to produce an invalid file.

---
---

# PART B — THE PAIN POINT REGISTER

Every real difficulty hidden in this problem, described as a symptom, a diagnosis, and a fix. Assign each one to a specific person on your team so none of them falls through the cracks.

## B1. Lighting means the same picture can honestly have two different answers

**Symptom.** Your model plateaus somewhere in the low 90s percent, and you notice a stubborn pattern: a group of dip images confidently predicted as bumps, and a group of bump images confidently predicted as dips, with nothing obviously wrong about the images themselves.

**Diagnosis.** Your model only looks at pixels. For genuinely confusing, illumination-ambiguous images, no pixels-only approach can get both members of a confusing pair right. You've hit a hard limit built into the problem itself, not a limit of your particular model.

**Fix.** The sun direction has to be incorporated into your model somehow. In order of how well they work: (1) **standardize** by rotating every image into a shared, common lighting direction (Part C1); (2) **feed the sun direction into the model** in a smart way, using techniques covered in C3; (3) simply attach the sun direction to your model's final feature vector before the last decision layer. Do (1). Optionally also add (2) as a fine-tuning correction on top. Don't rely on (3) alone — tacking it on at the very end gives it almost no chance to influence the earlier, more important parts of the model.

**How you'll know it worked.** That stubborn, confidently-wrong pattern shrinks noticeably, and your accuracy improves specifically on images you've flagged as having low shadow contrast.

## B2. Relying too heavily on the simple dark/light shortcut

**Symptom.** Great validation score, but a nagging feeling you can't shake. Or: your model's "what am I looking at" visualizations consistently point right at the shadow edge and nowhere else.

**Diagnosis.** The model has found the cheapest possible pattern that works. Training naturally finds the easiest signal that separates the classes; if the simple dark/light pattern is enough, there's no pressure on the model to learn about rim shapes, whether shadows stay contained, or texture.

**Fix.** Four steps, used together.

1. **Randomly hide the shadow region during training**, forcing the model to classify without it sometimes. This directly attacks the shortcut and pushes the model to find other clues.
2. **Use the brightness-inversion, label-flip trick** (from A3). This creates pairs of images with near-identical structure but opposite correct answers, so the simple dark/light pattern is still useful, but the model has to also track the actual physics rather than memorizing a simple appearance rule.
3. **Randomize contrast and brightness during training** so the model can't just rely on how dark the shadow looks in absolute terms.
4. **Run "the shortcut audit"** (F12): explicitly check your model's accuracy specifically on the images where the simple shortcut gets it wrong. If your fancy model does no better than random guessing on exactly those images, it hasn't actually learned anything the simple shortcut doesn't already know — and you have no real protection if the test set is designed to punish the shortcut. This one number is the best single summary of whether you've actually solved the real problem or just the easy version of it.

## B3. Train and test data having different lighting patterns

**Symptom.** You can't directly observe this — that's the whole problem. You can only detect it indirectly.

**Diagnosis.** The evaluation set might have been built with a different mix of sun directions, sun heights, feature types, or source photos than the training set.

**Fix.** Detect it, and protect against it.

*Detecting it*: (a) Plot the sun-direction histograms of your training data and your test data on the same chart. A visible difference is immediate, useful information, and takes ninety seconds to check. (b) Run a **"can you tell them apart" test**: label train images 0 and test images 1, and see if a small classifier can learn to distinguish them. If it can barely do better than chance, the two sets look statistically similar and you can relax. If it can distinguish them easily, there's a real difference worth investigating — look at which training images the classifier thinks look most "test-like." (c) Compare simple statistics: average brightness, how many near-black pixels there are (a stand-in for sun height), and general image sharpness.

*Protecting against it*: Standardizing the lighting direction is itself your strongest defense, because after standardization, the original sun-direction distribution stops mattering — every image looks like it was lit from the same direction. This is a major, often-overlooked argument in favor of this approach: **it makes lighting differences between train and test a non-issue.** On top of that, training with images rotated across the full circle of directions (with correctly updated sun-direction values) ensures your model handles the full range, no matter what the original training data looked like.

## B4. Class imbalance interacting with Balanced Accuracy

**Symptom.** High plain accuracy, disappointing Balanced Accuracy, weak performance on one class.

**Diagnosis and fix.** Fully covered in A6. Measure the class balance on Day 1; use a weighted loss or balanced sampling; set your cutoff based on the class balance; double-check with a cutoff sweep on your own validation predictions.

## B5. Small dataset and overfitting

**Symptom.** Training accuracy climbs smoothly toward perfect, while validation accuracy peaks early and then gets worse. A gap of more than about 8 points between training and validation accuracy.

**Diagnosis.** 7,854 images isn't a lot for a model with tens of millions of adjustable parameters.

**Fix.** Start from a pretrained model (a must). Keep model size modest. Use sensible weight decay. Add some regularization tricks like stochastic depth and light label smoothing. Use a gradually-decreasing learning rate schedule with a short warmup. Use heavy but *physically correct* data augmentation. Stop training early based on validation Balanced Accuracy. Techniques like Mixup and CutMix (which blend two images and their labels together) are worth testing, but be aware they might make less sense here than with everyday photos — test rather than assume. Keeping a running average of your model's weights over training is a cheap, reliable, small improvement.

## B6. The augmentation trap — standard tricks silently break the physics

**Symptom.** You add the usual random flips and rotations, and your validation score gets *worse*, or improves but doesn't generalize well.

**Diagnosis.** You're training the model to treat certain transformations as meaningless when the correct label actually changes under those transformations. Every flip that doesn't update the sun direction is essentially feeding the model a mislabeled example.

**Fix.** Build a **custom, sun-direction-aware transform pipeline** where every geometric change also updates the sun direction correctly. Never use a stock augmentation tool that touches image orientation without wrapping it in this logic. The rules are in the table at the end of A3, and the code approach is in C2. If you standardize the lighting direction first, the rules become wonderfully simple: once the sun is always at the top, **horizontal flips are completely safe** (they mirror across the lighting axis, keeping both the label and the lighting setup correct), **vertical flips flip the label**, and **any other rotation is forbidden** unless you re-standardize afterward.

## B7. Single-channel images versus models expecting 3-channel color

**Symptom.** Minor, but it wastes compute and occasionally causes normalization bugs.

**Fix.** Use `timm.create_model(name, pretrained=True, in_chans=1)`, which correctly combines the model's color-channel weights into one. Compute your **own** normalization numbers from your training set rather than using the defaults meant for everyday color photos — lunar grayscale images have a very different brightness distribution. Calculate the dataset's average and spread once, save them in a config file, and use the exact same values at prediction time. A mismatch between training-time and prediction-time normalization is a silent, serious, and surprisingly common bug.

## B8. Confusing, worn-down, and possibly mislabeled images

**Symptom.** A stubborn 2–5% of training images that every model in your group gets wrong with high confidence.

**Diagnosis.** Some are genuinely ambiguous (a shallow, worn-down feature, or a high-sun image with barely any shadow). Some are probably just mislabeled — hand-labelled datasets of this kind routinely have 1–3% label errors.

**Fix.** Pull out the 200 training images your models are most confidently wrong about, and **look at them with your own eyes.** This is the single most informative use of your time and most teams skip it. You'll learn more in forty minutes of looking at your model's mistakes than in a full day of tuning settings. Then: if they're clearly mislabeled, consider removing them or applying label smoothing; if they're genuinely hard cases, they mark the real frontier of the problem and tell you what to build next. Don't blindly delete every high-error example — you might be deleting the hard-but-correct cases that make up much of the real evaluation set.

## B9. Normalization and contrast pitfalls

**Symptom.** Model performs inconsistently between bright and dark images; sensitive to overall contrast.

**Diagnosis.** Absolute brightness depends on sun height and local surface material, neither of which relates to the actual class. But the **relative** brightness pattern within an image is the whole signal, so you need to be careful about what you normalize away.

**Fix.** Use dataset-wide normalization (subtract the dataset's average, divide by its spread) as your default, since it preserves relative patterns across images. **Try per-image normalization as a test** — it removes brightness as a distraction, but might also remove useful information about sun height. Also try contrast-enhancing preprocessing; it often helps a lot on low-contrast planetary images, but can also amplify noise in shadowed areas. Never use an augmentation that *inverts* contrast, unless you're deliberately using the label-flip trick.

## B10. Validation scores inflated by near-duplicate tiles

**Symptom.** Your validation score is 0.97; you have a bad feeling about it; there's no obvious way to check.

**Diagnosis.** Tiles cut from the same source photo may overlap or show the same feature from a slightly different angle or offset. If you split your data randomly, near-duplicates can end up in different splits, meaning your model has effectively already seen the "unseen" validation data.

**Fix.** Detect these near-duplicates and group them together. Compute a similarity fingerprint for every training image, connect similar pairs, and treat connected groups as a single unit. Then split your data so that entire groups stay together within one split. **Compare your validation score with grouping versus without.** If the grouped version is 4 points lower, you've just discovered your whole model-selection process was running on inflated numbers — and you've saved your competition. If the two scores are within a point of each other, you've cheaply bought peace of mind. Either result justifies the couple hours it takes.

## B11. Submission format errors

**Symptom.** A score of 0.0 or 0.5 on a model you know is good.

**Fix.** The checklist in A7, wired directly into your pipeline as a required final step. Plus the label-swap check. Plus a test upload done early in the window, if the platform allows it, to confirm your file is accepted.

## B12. Reproducibility and run-to-run variance

**Symptom.** You rerun the exact same setup and get a different score; you can't tell whether a change actually helped.

**Diagnosis.** Run-to-run variation on a dataset this size is typically 0.3–0.8 points on Balanced Accuracy. Many "improvements" you think you've found are actually just noise.

**Fix.** Fix your random seeds everywhere. Then, critically: **don't trust any single comparison of under about 1 point.** For decisions that matter, run the same setting 3 times with different seeds and compare the averages. Log the spread alongside the average, so everyone on the team understands the noise floor and stops arguing over 0.2-point differences.

## B13. Compute and time constraints

**Symptom.** Day 12, and you've only run four experiments.

**Fix.** Decide your compute plan on Day 0. A single mid-range GPU is enough for this dataset if you engineer for speed: preload the whole dataset into memory, use mixed-precision training, and tune your data-loading settings. A single training pass through 6,300 images at 256×256 should take well under a minute on any modern GPU. If your training passes take five minutes, you have a data-loading bug, not a compute shortage — **check this before you consider buying more hardware.** Also: run a **fast, cut-down configuration** (small model, few epochs, single split) for quick exploratory experiments, and save full multi-split, multi-seed runs for settings that have already shown promise.

## B14. Team coordination

**Symptom.** Three people have three different notebooks, two of them have drifted apart on preprocessing, and nobody can reproduce the best score anymore.

**Fix.** On Day 0, non-negotiable: a shared code repository, one shared settings file, one shared data-loading module that everyone imports, one shared scoring module, and a shared experiment log. **The rule is: preprocessing and validation splits are shared infrastructure, changed only by group agreement; individual models and augmentation choices are personal workstreams.** Generate your validation splits **once**, save them to a file in the repository, and have everyone use that exact file. If two team members use different splits, their scores aren't comparable, and every conversation about which model is better becomes meaningless.

---
---

# PART C — THE CORE STRATEGY

This is the technical heart of the document. Everything here is specific to this problem — none of it is generic advice you'd find in any machine learning guide.

## C1. Standardizing the Lighting Direction

### The idea

If the correct label depends on both the image *and* the lighting direction together, then instead of teaching your model to handle every possible lighting direction, **just remove that variable entirely**: rotate every image so the sun always appears to come from the same direction. After doing this, the lighting direction becomes a constant, the label becomes something you can determine from the pixels alone, and the problem turns into ordinary image classification.

This is the same trick a planetary scientist uses by hand: one quick way to make a flat image "pop" into its correct shape is to rotate it until the light source appears to come from above. You're automating exactly that, consistently, for every image in the dataset.

### Why this is the strongest move available

- It **removes the confusion entirely**, rather than asking the model to work around it.
- It makes **differences in lighting direction between train and test irrelevant**, because after standardizing, both sets end up looking like they were lit from the same single direction (see B3).
- It **simplifies what the model actually has to learn**, which matters a lot with only 7,854 examples. The model no longer needs to spend its capacity figuring out how the decision changes with lighting direction.
- It **simplifies the augmentation rules** down to two clean guidelines (C2).
- It makes the model **easier to explain and demonstrate** — your app can show the original image, the standardized image, and the prediction side by side, which is compelling to watch.

### The step-by-step process

1. **Figure out the exact convention.** Determine how the recorded sun-direction number maps to an actual direction within the image's pixel grid. Do this empirically, as described in A4 and Phase 1, Step 5. You're solving for two unknowns: an offset and a "handedness" (whether the mapping is mirrored or not).
2. **Rotate.** For each image, rotate it so the sun ends up pointing from the top of the frame downward (or any consistent direction you choose — the exact choice doesn't matter, only consistency does).
3. **Handle the corners.** Rotating a square image by an angle that isn't a multiple of 90° leaves some corner areas undefined. Three options: (a) **crop to the largest inscribed square**, simple but loses roughly half the area; (b) **pad the image before rotating** so the corners are filled with plausible surrounding terrain, preserving area but introducing some mirrored artifacts; (c) **rotate on an enlarged canvas, then crop back down** — this preserves the full original view with no undefined pixels, at the cost of a bit of blurring from resizing. **Option (c) is usually best**; try option (a) as a comparison, since the central feature is what matters most, and cropping to it may even reduce distractions.
4. **Interpolate carefully.** Use smooth interpolation methods rather than a blocky one, or you'll introduce jagged artifacts that the model might latch onto. Make sure you use the exact same method at training and prediction time.
5. **Double-check your work.** After standardizing, compute the average image for each class separately. **You should see a clear, easily interpretable difference**: the average dip image dark on top and bright on the bottom, the average bump image the reverse (or vice versa, depending on your chosen direction). If the two average images look basically identical, your calibration is wrong, and you need to fix it before moving on. **This chart is your proof that the whole approach is working**, and it belongs in your final presentation.

### Belt and braces

Standardizing isn't perfect: the sun-direction number might have measurement error, the convention might not be perfectly consistent across the dataset, and rotation introduces small artifacts. **So still feed the sine and cosine of the sun direction into the model as extra signal, even after standardizing.** It costs almost nothing, and lets the model learn small corrections. And **keep a non-standardized model in your group of models** as a hedge, in case your calibration turns out to be subtly wrong somewhere — having variety in your model group is cheap insurance.

## C2. The Rules for How the Sun's Direction Changes Under Each Transformation

### In the original (non-standardized) frame

Build a transform class where every geometric operation also updates the sun-direction value correctly. The core rule: **rotating the image content by an angle also rotates the apparent sun direction by that same angle.**

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

The `handedness` and `delta` values come from your Phase 1 calibration. Wrap all of this into one shared module that everyone imports — don't let each team member reimplement it their own way.

### In the standardized frame (much simpler — prefer this)

Once every image is standardized, with the sun always fixed at the top, the rules become much simpler:

- **Horizontal flip (left–right): SAFE.** This mirrors across the lighting direction. Both the label and lighting setup are preserved exactly. Use freely — this is free regularization.
- **Vertical flip (up–down): FLIPS THE LABEL.** It turns a dark-top/bright-bottom crater into a bright-top/dark-bottom mound pattern under unchanged lighting. This is approximately physically valid. Use it moderately, and double-check empirically that it helps.
- **Inverting brightness: FLIPS THE LABEL.** Same reasoning as above. Use it moderately.
- **Arbitrary rotation: FORBIDDEN**, since it breaks the standardization. Small rotations (±10°) are fine as minor jitter — they represent measurement uncertainty in the sun-direction value — but anything bigger destroys the setup.
- **Shifting position, resizing, mild distortion, adding noise, mild blur, brightness/contrast randomization (as long as it's not reversed), and random small dropout patches: SAFE.** Standard tricks with no physical implications.

The two label-flipping tricks are your secret weapon. They effectively **double your dataset with perfectly balanced, physically valid counterfactual examples**, and they directly teach the network the flip relationship the whole competition is named after. Test them carefully — measure with and without, across multiple splits and seeds — but expect them to be worth real points.

## C3. Ways to Feed the Model the Sun Direction

If you condition the model on the sun direction (instead of, or in addition to, standardizing), do it properly. Four approaches, roughly from weakest to strongest:

**Level 1 — Just attach it at the end.** Add the sine and cosine of the sun direction to the model's final pooled feature vector, right before the last decision layer. Trivial to implement, but largely ineffective — by the time the model has boiled the image down to a small feature vector, the spatial detail the sun direction needs to interact with is already gone. Use only as a baseline comparison.

**Level 2 — Extra constant input channels.** Add two extra input channels, filled entirely with the sine and cosine values. Now the sun direction is visible from the very first layer, and the model's early processing can start combining it with local image structure. Cheap and works surprisingly well.

**Level 3 — Proper conditioning layers (recommended if you go this route).** A technique called FiLM (Feature-wise Linear Modulation): run the sine/cosine through a small extra network to produce scaling and shifting values, and apply those to the model's internal features at each stage. This lets the lighting direction actually influence how the model processes the image at every level, not just at the end. Roughly 30 lines of extra code on top of a standard pretrained model. This is how conditioning is properly done in more sophisticated computer vision systems.

**Level 4 — Physically meaningful extra input channels (best, and works well alongside standardizing).** Rather than just telling the network the sun direction as an abstract number, hand it a derived quantity that's directly meaningful. Build a 3-channel input:
- Channel 0: the plain normalized image.
- Channel 1: **how brightness changes along the sun's direction**, computed with a standard edge-detection filter. Under simple shading assumptions, this is roughly a measure of the surface's curvature along the sun's axis — which is *exactly* the quantity that tells apart a bowl from a dome.
- Channel 2: how brightness changes perpendicular to the sun's direction, which carries weaker, secondary information. This asymmetry is well established — brightness patterns give strong shape information along the lighting direction, but much less across it.

This construction fits naturally with a standard pretrained model's expected 3-channel input, hands the physics to the network directly, and is a genuinely distinctive idea. **Test it.**

## C4. Teaching the Model the Underlying Physics Directly

This is the most sophisticated part of the approach, and the one most likely to set you apart from other teams. It doesn't need any extra labels, and it can even be applied to the unlabeled test images.

**The "should stay the same" rule.** The physics says: rotating an image and correspondingly updating its sun direction should never change the label. So for any image, sun direction, and rotation angle, the model's prediction should be identical whether or not you rotate (as long as you update the sun direction to match).

Add an extra training penalty that punishes the model whenever it violates this rule: pick a random rotation each training step, compute both predictions, and penalize the difference between them. Weight this penalty with a tunable coefficient.

**The "should flip" rule.** The physics also says: inverting an image's brightness while keeping the sun direction fixed should flip the label. Penalize violations of this rule too, with its own tunable coefficient.

**Why this is powerful.** These rules bake the underlying physics into the training process as a direct penalty, rather than just as another data example. They tell the model not just "here's another example" but "here's a *relationship* your predictions must respect." They provide useful training signal on every single image, including ones the model already gets right. And critically, **they can be applied to the 2,000 unlabeled evaluation images too**, which is a free 25% boost to your effective training data, and directly adapts the model to the evaluation set's own distribution — all without ever seeing a single evaluation label.

Implement this as an extra loss term added on top of your normal training loss, gradually phased in over the first few training passes so it doesn't destabilize early training. Test it carefully. If it works — and I expect it will — it's the centerpiece of your writeup.

## C5. Building Features by Hand, Based on the Actual Physics

Even if your main model is a neural network, build these as (a) your simple shortcut baseline, (b) interpretable inputs for a separate, simpler model that adds useful variety to your final ensemble, and (c) diagnostic tools for understanding your model's mistakes.

**Sun-axis brightness difference.** In the standardized frame, split the central region into a top half and bottom half and compute the difference in average brightness. This *is* the shortcut, boiled down to a single number. Its sign gives you the prediction, and its size gives you a confidence level.

**Shadow containment ratio.** Roughly separate out the shadow region and the "feature" region, and compute what fraction of the shadow falls *inside* the feature's outer boundary. Close to 1.0 suggests a dip (the shadow is trapped by the rim); noticeably below 1.0 suggests a bump (the shadow spills outward). **This is likely the single most valuable hand-built feature in this problem**, because it captures a structural property that survives changes in lighting direction.

**Shadow shape and offset.** Fit an ellipse to the shadow region. A boulder's cast shadow is long, thin, and offset well away from the object's center. A crater's interior shadow is a crescent shape centered close to the feature's own center. Compute the shadow's length-to-width ratio, and how far its center is from the feature's center.

**Rim ring pattern.** In the standardized frame, compute how average brightness changes as you move outward from the feature's center, ring by ring. A fresh crater shows a characteristic bright-rim, dark-interior, bright-far-wall pattern; a mound shows a simple, steady falloff. Even a rough version of this profile is a useful input.

**Sun-height stand-in.** The fraction of pixels below a fixed brightness threshold, and the overall spread of brightness in the image. These recover some of the missing "how high was the sun" information (A2), and are worth feeding to your model as extra input.

**Roundness and boundary closure.** Roughly separate out the feature, and compute how close to a perfect circle its outline is. Impact craters are strikingly round; mounds and rock clusters usually aren't.

Feed all of these into a simple gradient-boosted model alongside the sun-direction features. It won't beat your main neural network, but it will be **usefully different** from it, which is exactly what you want when combining multiple models — and it only takes an afternoon to build.

---
---

# PART D — DAY-BY-DAY PLAN

Fifteen days remain (6–21 September). The plan below is scheduled out, with each phase broken into numbered steps, something concrete to hand in, and a clear **"you're done when..."** checkpoint. Don't move past a checkpoint just because you're behind schedule — a phase left half-finished causes problems for everything that comes after it.

Assume a team of three to five people. The role labels below are suggestions: **DATA** (exploring the data, preprocessing, setting up validation splits), **MODEL** (training, model design, combining models), **APP** (the application, API, and user interface), **OPS** (infrastructure, reproducibility, submissions).

---

## PHASE 0 — Getting Set Up
**Day 1 (6 September), first half. Owner: OPS, with the whole team present.**

Nothing in this phase produces a score. All of it prevents disasters later.

**Step 0.1 — Resolve the open questions about the rules.** Read the official competition rules end to end and answer, in writing, in a shared document: (a) Are pretrained models allowed? (b) Is outside data allowed? (c) Can you use the unlabeled test images (for test-time tricks or pseudo-labelling)? (d) What exact timezone is the deadline in? (e) How many submissions are allowed, and is there any feedback in between? (f) Is a written report required alongside the CSV file? Each of these meaningfully changes your plan. If you can't find an answer in the rules, **email the organizers on Day 1** — you won't get a reply on Day 14.

**Step 0.2 — Set up your code repository.** Create a git repository with a clear folder structure:

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

Add rules to exclude data files, checkpoints, and cache folders from version control. Pin your dependency versions exactly in a requirements file.

**Step 0.3 — Set shared ground rules.** Write these into the README as team law: preprocessing, the validation splits, and the scoring function are shared infrastructure, changed only by group agreement; the validation split file is generated once and saved into the repo; every experiment gets its own settings file and a logged run; nobody reports a score that wasn't computed by the shared scoring code on the shared validation splits.

**Step 0.4 — Set up experiment logging.** A tool like Weights & Biases (free tier) works well. For every run, log: the exact settings used, the code version, per-split scores, the overall score, the chosen cutoff, training curves, and a sample of misclassified images. If that's not an option, a shared spreadsheet with a strict, consistent set of columns works too.

**Step 0.5 — Confirm your compute access.** Make sure each team member can access a GPU and run a quick test training pass. Agree on where trained model files live and how they're shared. Don't discover on Day 9 that only one person has a working GPU.

**Step 0.6 — Fix your random seeds everywhere.** Write a small utility function that seeds every relevant random number generator and enables deterministic behavior, and call it at the start of every script.

> **Hand in:** a working repo, a pinned environment, and written answers to the rules questions.
> **You're done when:** every team member can clone the repo, run a basic training command, and watch a loss number go down.

---

## PHASE 1 — Understanding the Data
**Day 1 (second half) through Day 2. Owner: DATA. This is the highest-value phase in the whole project.**

Don't skip any step here. What you find here shapes everything that follows.

**Step 1.1 — Load and sanity-check the data.** Read both metadata files. Confirm: 7,854 rows in training, 2,000 in test; no duplicate image IDs; every listed image actually exists as a file and vice versa; no missing values. Report any mismatch right away — a missing file discovered on Day 14 is a crisis; on Day 1 it's just an email.

**Step 1.2 — Check the image properties.** For a sample of 200 images (and, in the background, all of them): confirm they're exactly 256×256, confirm they're single-channel, note the value range, and check for any corrupted files. Also note how many unique brightness values appear — if it's low, the images may have been compressed in a way worth knowing about.

**Step 1.3 — Check the class balance.** Compute what fraction of training images fall into each class. This single number determines how you weight your training loss and how you set your decision cutoff (A6). Save it in your settings file as a fixed constant.

**Step 1.4 — Analyze the sun-direction distribution.** Plot the training and test sun-direction histograms **on the same chart**. Then plot the training sun-direction histogram, **split by class**. Answer three questions:
   - *Is the sun direction spread evenly across the full circle, or clustered?* Clustering suggests the images come from a small number of source photos.
   - *Do training and test differ noticeably?* This is your early warning sign for train/test differences.
   - ***Is the sun direction correlated with the label in the training set?*** This is critical. If, say, dip images tend to have sun directions in one range and bump images in a different range, then **the sun direction alone is a "leaky" clue** — a model could exploit that pattern instead of actually learning shapes, and it likely won't generalize. If you find this, you need to actively break that correlation: either train with augmentation that randomizes the sun direction across the full circle, or avoid using sun direction as a direct feature and use it only for standardizing lighting (another strong argument for that approach). Check this by measuring how strongly a simple model trained only on the sun direction can predict the label. **If that sun-direction-only model scores meaningfully better than chance, you've found a trap.**

**Step 1.5 — CALIBRATE THE SUN-DIRECTION CONVENTION. (The single most important experiment.)**
   1. For every training image, compute the shadow-direction estimate: find the average position of the darkest 10% of pixels, subtract the average position of the brightest 10% of pixels, and turn that into an angle in the image's own coordinates.
   2. Compute the difference between this angle and the recorded sun-direction value, for every image.
   3. Plot the circular histogram of that difference, **split by class**.
   4. **Expected result:** two tight clusters roughly 180° apart. Read off the typical difference for each class — this gives you the offset. The fact that the classes separate confirms which "handedness" (mirrored or not) is correct. If the separation isn't clean, try the opposite handedness and see if it gives tighter clusters.
   5. **Note how tight the clusters are.** Very tight clusters mean the simple dark/light shortcut is a near-perfect predictor on its own — the shortcut is very strong. Loose clusters mean it's noisy, and your model has real, meaningful work to do beyond the shortcut. **This tells you how much of the problem the shortcut solves, and therefore how to spend your remaining two weeks.**
   6. Write the resulting offset and handedness into your settings file. Everything downstream depends on it.

**Step 1.6 — Build and check your standardization function.** Implement the function that rotates any image into the standard lighting direction, using your calibrated offset and handedness. Then check it visually: compute and display the **average image per class, before and after standardizing.** Before: both averages should look like a featureless blur. After: **they should look clearly different and easy to interpret** (dark-top/bright-bottom versus the reverse). Save this chart — it's your proof that things are working correctly, and it belongs in your final presentation. If the averages don't clearly separate, stop and debug before moving forward.

**Step 1.7 — Look at the images yourself.** Build a simple grid viewer and look at, at minimum: 50 random images of each class, the 20 brightest, the 20 darkest, the 20 lowest-contrast, and 30 spread across the range of sun directions. **Actually look at them.** Write down, in plain language, in your shared doc, what the different sub-types look like (fresh craters, worn-down craters, pits, boulders, mounds, rock fields), roughly how common each is, and which pairs look easy to confuse. This mental map will guide all your later error analysis.

**Step 1.8 — Find near-duplicate images and group them.** Compute a similarity fingerprint for every training image, and also compute similarity using a pretrained model's internal representations. Connect similar pairs above some threshold, and treat connected groups as a single unit. Report: how many groups there are, how big they get, and the biggest one. **Then also check for near-duplicates between training and test images.** If you find any, that's useful information about how the split was made (and, if they're exact duplicates, part of the test set is effectively already solved).

**Step 1.9 — Create and lock in your validation splits.** Produce a file listing which validation split each image belongs to, using a method that keeps both label balance and image groups intact across splits. If the sun-direction distribution is uneven, also try to balance that across splits. **Save this file, and don't ever regenerate it.** Every model everyone trains from now on uses these exact same splits.

**Step 1.10 — Check whether train and test look statistically different.** Train a quick classifier to try to distinguish training images from test images (without using any labels). A score close to random guessing is reassuring. If it can tell them apart well, investigate what's driving the difference by looking at the training images it thinks look most "test-like."

**Step 1.11 — Compute normalization numbers.** The overall average and spread of pixel brightness across the training set. Save these in your settings file.

**Step 1.12 — Cache the data for speed.** Save the entire training set as one single array on disk, plus the test set. This takes up a few hundred MB and lets every future training run load the data in seconds instead of re-reading thousands of individual image files every time.

> **Hand in:** a written summary of your findings, the locked validation-split file, the calibrated offset/handedness values, the cached data arrays, and the per-class average-image chart.
> **You're done when:** the per-class standardized average images clearly and interpretably differ; the validation-split file exists and is saved to the repo; the team can state in one sentence how strong the shortcut is, and whether training and test data look different.

---

## PHASE 2 — Simple Baselines and the Full Pipeline
**Days 2–3. Owner: MODEL + OPS.**

The point of this phase isn't to score well. It's to make sure a valid submission is possible, and to set reference points you'll measure everything else against.

**Step 2.1 — Baseline 0: guess the same answer every time.** Predict all 0. Compute the score. It should be exactly 0.5 by construction. This is your floor, and it confirms your scoring code is correct.

**Step 2.2 — Baseline 1: the simple shortcut.** Implement the sun-axis brightness difference rule from C5, with no training required at all: standardize the lighting, compare the average brightness of the top versus bottom half of the central region, and use zero as the cutoff. Test this on the full training set. **Write this number down prominently.** It's the number to beat, and the gap between it and your final model is the honest measure of what your machine learning actually contributed. If this baseline scores 0.93, you're likely in the "the last 7% is the real challenge" scenario from A5. If it scores 0.71, the problem is much richer than pure brightness patterns, and your model has real room to improve on it.

**Step 2.3 — Baseline 2: hand-built features plus a simple model.** Compute the C5 feature set (brightness difference, containment ratio, shadow shape, rim profile, roundness, sun-height stand-in, sun direction), train a simple gradient-boosted model with proper cross-validation, and report the score. Fast, easy to understand, and something you'll genuinely combine with your other models later. Look at which features mattered most — that tells you what actually matters in this problem.

**Step 2.4 — Baseline 3: a plain, deliberately-naive neural network.** Train a standard, medium-sized model on **raw, non-standardized images with no sun-direction input at all**, using ordinary augmentation. Report its score. **This is the number every other team will get on their very first day, and how it compares to Baseline 1 and to your later models tells you exactly what the physics-aware approach buys you.** Keep it for your writeup.

**Step 2.5 — Build the full prediction and submission pipeline.** Write the scripts that run your best baseline over all 2,000 test images and produce a submission file.

**Step 2.6 — Build the submission checker**, implementing every check from A7, and wire it into your submission-building script as a required final step.

**Step 2.7 — SUBMIT.** Upload this baseline submission to the platform (if intermediate submissions are allowed). **The goal is just to confirm the platform accepts your file format.** If it's accepted, you've eliminated the single biggest catastrophic risk in the whole project on Day 3, instead of at 11:47 PM on Day 21.

> **Hand in:** four baseline scores in your log, a checked submission file, and confirmation from the platform that it was accepted.
> **You're done when:** a valid CSV has been produced automatically by a script and accepted by the platform. **Don't move on until this is true.**

---

## PHASE 3 — The Standardized Model
**Days 3–5. Owner: MODEL.**

**Step 3.1 — Build the sun-direction-aware augmentation module**, implementing the rules from C2 exactly. Write **tests** for it: check that rotating by an angle and then by its opposite gets you back to the original sun direction; check that a standardized image, when rotated with the sun direction correctly updated and then re-standardized, comes back to the same result; check that flipping brightness twice gets you back to the original. Bugs in these transforms are silent and can be devastating — test them.

**Step 3.2 — Build the dataset class.** It should support settings for: whether to standardize, how to handle rotated corners, which augmentation policy to use, the normalization numbers, and which input-channel setup to use (single channel versus the 3-channel setup from C3). It should return the image, the sun direction (as sine/cosine), and the label.

**Step 3.3 — Build the model class.** A pretrained model with an optional conditioning layer and an optional simple attach-at-the-end head. Make the conditioning approach a settings toggle, so comparing them is a one-line change.

**Step 3.4 — Build the training loop.** Standard setup: a good optimizer, a gradually decreasing learning rate with warmup, mixed precision, a weighted loss with light label smoothing, gradient clipping, a running average of the model's weights, and early stopping based on validation score, saving the best checkpoint. Log everything to your experiment tracker.

**Step 3.5 — Train your first real model.** A medium-sized backbone, standardized inputs, safe augmentations only (no label-flip tricks yet), a reasonable number of training passes, and proper multi-split validation. **This is your reference model.** Record its score and how much it varies across splits.

**Step 3.6 — The critical comparison: standardizing on versus off.** Train the exact same setup but with standardization turned off. The difference between these two numbers is the real, measured value of the central idea in this whole document. Expect it to be large. If it isn't, something's wrong with your calibration, and you should go back to Step 1.5.

**Step 3.7 — Sanity-check with attention/heatmap visualization.** Generate "what is the model looking at" visualizations for 30 correctly-classified and 30 misclassified validation images. Are the highlighted areas on the actual feature, or on a corner artifact from rotation? Are they focused only on the shadow edge, or do they also cover the rim and surroundings? This is your first read on how much the model is relying on the shortcut.

> **Hand in:** a trained, properly validated, standardized model, with its predictions saved to disk.
> **You're done when:** the score meaningfully beats both the simple shortcut baseline and the naive neural network baseline, and you've clearly measured the standardization comparison.

---

## PHASE 4 — Trying Different Models and Settings
**Days 5–9. Owner: MODEL, split across team members.**

Run these as separate workstreams, one per person, all writing to the same log and using the same validation splits. **Change one thing at a time.** Remember the noise floor from B12: treat any difference under 1 point as unproven until confirmed with multiple seeds.

**Step 4.1 — Try different pretrained backbones.** A handful of medium-sized options (some convolutional, some transformer-based). Same training setup, same validation splits for all of them. Record the score for each. **Don't just pick the single winner** — pick the top three or four *diverse* ones for your final combined model, since variety matters more than raw individual strength when combining models.

**Step 4.2 — Try different input setups.** Single-channel standardized, 3-channel duplicated, 3-channel derivative setup (from C3, Level 4), with and without contrast-enhancing preprocessing, and a couple of different resolutions.

**Step 4.3 — Try different conditioning approaches.** No conditioning (standardization only), simple attach-at-the-end, extra constant channels, and full FiLM conditioning. On standardized inputs, conditioning should help only a little; on raw inputs, it should help a lot.

**Step 4.4 — Test the label-flip augmentation.** Vertical flip with label flip at a few different probabilities; brightness inversion with label flip at a few different probabilities. **Run each with three different random seeds**, since this is an important claim and you want to be confident. Report averages and spreads.

**Step 4.5 — Test the physics-based training penalty.** The "should stay the same" penalty at a few different strengths; the "should flip" penalty at a few different strengths. Then try applying that penalty to the **unlabeled test images** too. Run three seeds for whichever setting wins.

**Step 4.6 — Tune your regularization settings.** Weight decay, stochastic depth strength, label smoothing, mixup/cutmix on or off. You can automate this search if you like, but honestly, given the dataset size and the deadline, a hand-picked set of a dozen configurations run by one person is often a better use of a day than setting up an automated search tool.

**Step 4.7 — Look at your errors, mid-phase — this is mandatory.** Take your current best model's validation predictions. Sort by how wrong it is, worst first. **Look at the top 200 mistakes with your own eyes.** Sort them into the sub-types you identified in Step 1.7. Build a table of accuracy by sub-type. This tells you, concretely, where the remaining error actually lives. Then aim the rest of the phase at fixing that. Also break down accuracy by: sun-direction range, brightness level, contrast level, and whether the simple shortcut got it right or wrong. **The last one is the shortcut audit, and it's the single most important row in the whole table.**

**Step 4.8 — Pseudo-labelling (if the rules allow it, and you have time).** Take your best combined model's test predictions, keep only the ones it's very confident about, add them to your training data as extra soft or hard labels, retrain, and re-check your score on your *original* validation splits. Only keep this trick if it actually improves your score. Pseudo-labelling can help meaningfully on small datasets, but it can also amplify existing biases, so treat it as an experiment rather than a sure thing.

> **Hand in:** a ranked table of at least 20 tracked experiments, 4–6 selected diverse configurations, and a written error analysis.
> **You're done when:** you can name your top four models and explain, with evidence, why each one made the cut.

---

## PHASE 5 — Robustness and Reducing Shortcut Reliance
**Days 7–11, overlapping Phase 4. Owner: MODEL + DATA.**

This phase exists because of A5. It's what you do to protect yourself against the possibility that the evaluation set is designed to punish the simple shortcut.

**Step 5.1 — The shortcut audit.** Split your validation predictions by whether the simple shortcut baseline got them right or wrong. Compute your model's score on each group separately. **Target: noticeably above chance level on the group where the shortcut is wrong.** If you're at chance there, your model has learned nothing beyond the shortcut, and you have no protection if the real test set is designed to punish it. Make improving this number the explicit goal of this phase.

**Step 5.2 — A synthetic worst-case test.** Build a stress-test set from your validation images by applying the two label-flip tricks — images whose correct labels are now the *opposite* of what they superficially look like. Measure your score on it. **A model that's genuinely learned the physics does well here; a model that just memorized appearance scores near zero.** This is your sharpest single diagnostic, and it's essentially a simulation of the worst-case adversarial test set.

**Step 5.3 — A sun-direction-shift stress test.** Hold out a validation split, rotate every image in it by a random angle with the sun direction correctly updated, and re-check your score. A correctly standardizing, correctly conditioned model should show **almost no drop in performance.** Any significant drop points to a bug or a leak somewhere.

**Step 5.4 — Hiding tests.** Hide the shadow region and re-check the score. Hide the center of the feature and re-check the score. How much the score drops in each case tells you what the model is actually relying on.

**Step 5.5 — Calibration check.** Plot a chart comparing your model's predicted confidence to its actual accuracy. If it's badly off, apply a standard calibration fix based on your validation predictions. This matters because your cutoff logic (A6) assumes your probabilities are well-calibrated.

**Step 5.6 — Build a breakdown report.** Build a standing report that breaks your validation score down by sun-direction range, brightness level, contrast level, sub-type, and shortcut-correctness. Regenerate it for every candidate model. **Choose your final combined model based on its worst-case slice, not just its average.** If there's a real chance of train/test differences, worst-case performance predicts real-world behavior better than the average does.

> **Hand in:** the robustness report.
> **You're done when:** the synthetic worst-case test (5.2) is passed convincingly, and the shortcut-wrong accuracy (5.1) is well above chance level.

---

## PHASE 6 — Combining Models, Test-Time Tricks, Calibration, Cutoff
**Days 11–13. Owner: MODEL.**

**Step 6.1 — Gather your validation predictions.** For each of your candidate models, you should already have a set of validation-only predictions covering all 7,854 training images, all computed on the exact same splits. Combine them into one table. **If any model used different splits, drop it** — you can't fairly combine predictions that aren't directly comparable.

**Step 6.2 — Check how correlated your models are.** Compute how similarly your models' predictions agree with each other. Prefer combining models that don't agree too closely — a simpler feature-based model that disagrees a fair amount with your neural networks may add more value than a fifth neural network that behaves almost identically to the others.

**Step 6.3 — Start with simple averaging.** Average the predicted probabilities across your models (or average them in a slightly different mathematical space, which often works a touch better). Check the score. This is your baseline for combining models.

**Step 6.4 — Try weighted averaging.** Search for the best combination of weights based on your validation predictions. **Keep the weights non-negative and summing to one, and watch out for overfitting** — with several models and only a few thousand samples, optimizing weights has real overfitting risk. Compare weighted averaging against simple averaging fairly; if the improvement is under half a point, **stick with simple averaging**, which is more reliable.

**Step 6.5 — Try stacking (optional).** Train a simple model on top of your models' predicted probabilities, plus the sun-direction features. Evaluate it honestly, using nested validation. Only adopt this if it clearly beats simple averaging.

**Step 6.6 — Test-time augmentation.** Design a small set of safe transformations to average predictions over, using the rules from C2. In the standardized frame, the safe options are: no change, horizontal flip, and small rotations (with re-standardization). You can also include **a physics-based trick**: predict on the brightness-inverted image and flip that prediction, then average it in — this is an unusually strong trick because it probes a different part of the model's reasoning. Test every combination on your validation predictions before adopting it.

**Step 6.7 — Test-time statistics adaptation (optional).** Run the test images through the model once to update its internal statistics before making final predictions. Cheap, and occasionally a real gain if train and test differ statistically.

**Step 6.8 — Optimize your cutoff.** On your final combined model's validation predictions, sweep across a fine range of possible cutoffs and plot the score against each one. Look for the flat, stable stretch, not a narrow spike. Check how much the best cutoff varies across different validation splits. Compare against the theoretical value from A6. **Pick a cutoff from the middle of a stable stretch.** Save the chosen value in your settings.

**Step 6.9 — Final model selection meeting.** As a team, review: each candidate combined model's validation score, its worst-case slice performance, the synthetic worst-case test result, and how well it agrees with the simple shortcut on the test set. Make a decision. Write down why, in one paragraph, in your shared doc.

> **Hand in:** the final combined model setup and cutoff, locked into your settings.
> **You're done when:** your final combined model is chosen, and its validation score, slice breakdown, and stress-test results are all written down.

---

## PHASE 7 — The Application
**Days 8–15, running in parallel throughout. Owner: APP.**

The application track should start on Day 8 and run alongside the modelling work, using whatever model is currently available. Design it so it loads whatever model is current, so the app is never stuck waiting on the final model. Full feature details are in Part E; here's the build order:

**Step 7.1 —** Define exactly what a "model package" contains: the weights, the settings, the normalization numbers, the calibration values, and the cutoff. The app should load this package and nothing else. Version it.

**Step 7.2 —** Export your final models in a portable format for fast prediction. Plain checkpoints are fine too if you're deploying on the same setup you trained on.

**Step 7.3 —** Build the backend API with endpoints for: predicting on a single image, predicting on a batch of images, a health check, model info, and an explanation endpoint (returning a visual "what the model looked at" overlay).

**Step 7.4 —** Build the front end. A quick tool like Streamlit or Gradio if you want it done in a day; a full custom web app if you want it to look polished and have someone who can build it. Given a 15-day window where modelling is the priority, **a quick tool is the right choice unless the competition explicitly scores presentation.**

**Step 7.5 —** Build the features in the order given in Part E, prioritizing the pipeline-critical ones before the demonstration ones.

**Step 7.6 —** Build the **Sun Simulator** (Feature F10). This is the flagship feature. Don't skip it.

**Step 7.7 —** Package everything into a container with a one-command startup. Test it on a clean machine.

**Step 7.8 —** Record a 3-minute demo video, as insurance in case a live demo fails.

> **Hand in:** a running, packaged application.
> **You're done when:** someone who has never seen the project can open the app, upload an image, set a sun direction, and get an explained prediction.

---

## PHASE 8 — Final Submission Engineering
**Days 14–15. Owner: OPS. Treat this phase with more caution than it seems to need.**

**Step 8.1 — Do a full, clean retrain.** From a fresh copy of the repository, retrain your final combined model setup end to end with fixed seeds. Confirm the score matches what you selected on. If it doesn't, you have some unfixed source of randomness, and you need to find it before submitting.

**Step 8.2 — Decide deliberately how to use your training splits.** You have a choice: submit the average of your split-based models (each trained on 80% of the data), or retrain a single model on all of the data. **The split-based ensemble is the safer choice** — it's itself a combination of models, it's exactly what your validation score measured, and it carries no risk of behaving differently from what you validated. The full-data version sees more data but is unvalidated. **Recommended: use the split-based ensemble.** If you want the extra data too, do both and average them.

**Step 8.3 — Generate test predictions** with your full test-time trick set, saving the raw probabilities to disk before applying the cutoff. Never overwrite this file.

**Step 8.4 — Apply your locked-in cutoff** and generate the final CSV.

**Step 8.5 — Run your checker.** Every check from A7.

**Step 8.6 — Run the label-swap check.** Push 20 training images with known correct answers through the *exact same production prediction process* — same script, same model package, same preprocessing — and confirm the predictions match the known answers. This catches a class-swap bug that would otherwise cost you everything.

**Step 8.7 — Sanity statistics.** Predicted class balance versus the training class balance. Agreement rate with the simple shortcut baseline (A5 — record this, it's your best intelligence about the hidden test set). Agreement with your previous submission. Distribution of predicted probabilities (it should have two clear peaks, not be clustered right around the cutoff).

**Step 8.8 — Look at the actual images.** Display 20 test images predicted as dips and 20 predicted as bumps, along with their sun directions. Look at them. Do they look right to a human who now understands the physics? This takes ten minutes and catches more bugs than any automated check.

**Step 8.9 — Submit, early.** Aim for **21 September, 6:00 PM at the very latest**, ideally Day 14. Confirm the platform's acceptance message. Take a screenshot of it.

**Step 8.10 — Keep a backup submission.** Hold onto your previous known-good submission file, so if something goes wrong you can resubmit a known-valid alternative.

> **You're done when:** your final submission has been accepted, and you have proof of that, well before the deadline.

---

## PHASE 9 — Documentation and Backup Plans
**Days 14–15, in parallel.**

**Step 9.1 —** Write the README: how to set things up, where to put the data, how to reproduce the final submission with one command, and which versions of everything you used.

**Step 9.2 —** Write a model summary: architecture, training data, validation approach, score, breakdown by slice, known limitations, and what it's intended to be used for.

**Step 9.3 —** Write the technical report, if one is required: the physics of the lighting flip, your standardization approach, the augmentation rules, the physics-based training penalties, your comparison table, and your robustness results. **Your comparison table is your differentiator** — most teams will just report a score; you can report a clear explanation of *why* the score is what it is.

**Step 9.4 —** Plan for things going wrong. Write down, in advance: what you'll do if the GPU dies on Day 19 (answer: you already have a valid submitted CSV from Day 3, and your checkpoints are backed up); what you'll do if the platform rejects your file (answer: you already tested this on Day 3); what you'll do if a serious bug is found on Day 20 (answer: roll back to your last tagged, working version and its saved raw probabilities).

---

## Timeline at a Glance

| Days | Phase | Main outcome |
|---|---|---|
| 6 Sep (D1) | 0 + 1 start | Repo set up, rules confirmed, data exploration begun |
| 6–7 Sep | 1 | Calibration done, validation splits locked, data cached |
| 7–8 Sep | 2 | Baselines done + **first valid submission** |
| 8–10 Sep | 3 | Standardized model built, standardization comparison measured |
| 10–14 Sep | 4 | Model/augmentation/loss experiments, error analysis |
| 12–16 Sep | 5 | Robustness testing, stress tests, reducing shortcut reliance |
| 13–17 Sep | 7 | Application (running in parallel from Day 8) |
| 16–18 Sep | 6 | Combining models, test-time tricks, cutoff selection |
| 19–20 Sep | 8 | Final retrain, validation, **submit** |
| 20–21 Sep | 9 | Documentation, buffer time, backup plans |

Notice the deliberate two-day buffer at the end. **Use it as buffer, not as extra modelling time.** Any competition schedule that plans to finish exactly on the deadline actually finishes after it.

---
---

# PART E — FEATURES TO BUILD INTO THE APP

Fifteen features. F1–F8 are the core pipeline: without them, there's no submission at all. F9–F12 are the demonstration and diagnostic layer: they turn this from a script into a real *product*, and F10 especially is the feature people will remember your entry for. F13–F15 are engineering practices that become critical in the final 48 hours.

For each: **what it is**, **why this specific problem needs it**, and **how to build it, step by step.**

---

## F1. Data Loading and Integrity Layer

**What.** A module that loads metadata and images, checks them for problems, caches them, and gives everything downstream a clean, simple way to access the data.

**Why here.** Because you have two metadata files, thousands of images, a strict submission ID format, and hidden test labels. A single mismatch between the metadata list and the actual image files — one missing file, one ID with a stripped extension — quietly propagates into a zero-scoring submission. Checking for these problems is cheap insurance on a problem where you only get one real shot.

**How.**
1. A function that reads the metadata files, checks the expected column names, checks the expected row counts, and checks for missing values or duplicate IDs.
2. A function that goes through every listed image, confirms the file exists, opens it, checks it's 256×256, and records its format — returning a report of anything unusual. Run this once and cache the report.
3. A function that decodes all images once into a single array and saves it to disk, along with an index mapping array position back to image ID. **This index is the contract that guarantees your predictions line up with the right IDs.**
4. A function that computes and returns the dataset's average and spread of brightness, saved into your settings.
5. A dataset class that returns the image, the sun direction (as sine/cosine), and the label, based on your chosen settings.
6. Write basic tests for all of the above. This module changes rarely, and breaks everything when it does.

---

## F2. Lighting Standardization Engine

**What.** The module implementing C1: figuring out the sun-direction convention, and rotating any image into the standardized, sun-at-top frame.

**Why here.** This is the intellectual core of your solution. It's what turns an unsolvable, ambiguous problem into a solvable one. It's also the piece your app will visualize to explain your whole approach to a viewer in ten seconds.

**How.**
1. **A calibration function.** Implement the shadow-centroid estimator from A4/Phase 1.5. For each image, compute the darkest-minus-brightest centroid difference and its angle. Try both possible "handedness" options, and for each, compute how tightly the classes' angle differences cluster. Return whichever handedness gives tighter clustering, the resulting offset, and how tight the clusters are as a confidence score. **Log that confidence score prominently — it tells you how strong the shortcut is, and confirms the whole physical model.**
2. **A function converting sun direction into an actual direction vector** in the image's own pixel coordinates. Document the axis convention clearly in a comment — this is where sign errors tend to hide.
3. **The standardization function itself.** Rotate the image so the sun points in the chosen standard direction, handling the corners as discussed in C1, using smooth interpolation. Cache the result for the whole dataset so you don't have to recompute it every training pass.
4. **The reverse function** — needed to map "what the model looked at" overlays back onto the original image for display in the app.
5. **A checking function** that produces the per-class average-image chart (before/after). Run it whenever you make changes; if the averages stop clearly separating, something upstream broke.
6. **Handle edge cases:** sun-direction values that are missing or out of range, and a toggle to turn standardization off entirely so comparisons are just a settings change.

---

## F3. Sun-Direction-Aware Augmentation Engine

**What.** The transform pipeline implementing the C2 rules, where every geometric operation returns a correctly updated sun direction and, where relevant, an updated label.

**Why here.** Because the default augmentation approach used in nearly every computer vision codebase is *wrong for this problem* (B6). This module is what stops your team from silently feeding bad, mislabeled examples into training for two weeks straight.

**How.**
1. Define a simple data structure carrying the image, the sun direction, and the label together, so transforms can update all three at once.
2. Implement each operation from C2 as its own transform, with a probability and an "apply" method: rotate-with-update, horizontal-flip-with-update, the safe canonical horizontal flip, the label-flipping vertical flip, the label-flipping brightness inversion, plus the physically-neutral ones (shift/scale, brightness/contrast randomization, gamma randomization, noise, small dropout patches, shadow masking).
3. Combine these into named policies in your settings file (a "safe" policy, a policy that includes the label-flip tricks, and an "aggressive" policy). Comparisons then become one-line settings changes.
4. **Write tests, as described in Phase 3.1.** Round-trip checks, standardization consistency checks, double-negation checks.
5. **Build a visual debugging tool**: a script that shows a grid of 16 augmented versions of one image, labeled with the resulting sun direction and label. **Look at this before you train anything.** Augmentation bugs are invisible in loss curves and obvious in a picture.

---

## F4. Model Construction and Training Orchestrator

**What.** A settings-driven way to build models with optional conditioning, plus the training loop itself.

**Why here.** With 15 days and a multi-person team, you'll run dozens of experiments. If each one requires editing code, you'll lose time to merge conflicts and lost reproducibility. Everything should be a settings parameter.

**How.**
1. A function that builds the model from settings, wrapped by an optional conditioning module that applies sun-direction-based adjustments to the model's internal features.
2. A function that builds the training loss: weighted cross-entropy with label smoothing, plus optional physics-based penalties (from C4) with their own strengths and a gradual warmup.
3. A function that trains one validation split: the standard loop with mixed precision, a gradually decreasing learning rate with warmup, gradient clipping, a running average of weights, per-pass validation scoring at both the default and tuned cutoffs, checkpointing the best version, and early stopping.
4. A function that loops over all splits, collects the combined validation predictions, computes the overall score and best cutoff, and saves everything to disk.
5. Every run should write a small summary file with its settings, code version, seed, and all metrics. **This summary is what makes an experiment identifiable three days later when someone asks "which run was that?"**

---

## F5. Cross-Validation and Experiment Tracking

**What.** Locked-in validation splits, one shared scoring implementation, and a tracker every run writes to.

**Why here.** Since test labels are hidden, your own validation is your only signal (A2). Group-aware splits keep that signal honest (B10). A shared tracker keeps everyone's work comparable.

**How.**
1. A function that generates the splits once, saves them to a file, and is then either removed from the regular workflow or protected so it can't accidentally overwrite the existing file.
2. A shared scoring module exposing the Balanced Accuracy calculation, a function that finds the best cutoff (both the raw best and the stable-plateau center), and a function that produces the breakdown by sun-direction range, brightness level, contrast level, and shortcut-correctness.
3. Logging integration: log per-pass training curves, log the final breakdown as a table, log a panel of the worst validation mistakes as images. **This error panel is the single most useful thing in your tracker.**
4. A script that pulls all runs and prints a sorted table of scores with their variation across splits. Run it every morning as your team's daily check-in material.

---

## F6. Prediction and Test-Time Augmentation Engine

**What.** The path from a trained model to calibrated probabilities on new images.

**Why here.** This code runs exactly once on the thing that actually matters. It must behave identically to your validation process, or your validation score means nothing.

**How.**
1. **Make sure the prediction path is identical to the training path.** Prediction-time preprocessing must call the *exact same functions* as training-time preprocessing, from the same shared module, using the same saved settings. Don't reimplement preprocessing separately for predictions. This is the single most common source of a mismatch between what you validated and what you actually submit.
2. A prediction function that, for each test-time trick, applies it (with the sun direction correctly updated), runs the model, reverses any label-flipping effect on the output, and averages the results.
3. Combine predictions across your different validation-split models and across different architectures, with weights locked into your settings.
4. Save the raw probabilities to disk **before** applying the cutoff, and never overwrite this file. If a cutoff bug is found at 10 PM on Day 20, you can regenerate the CSV in two seconds instead of retraining for three hours.
5. Apply calibration fixes if you've set them up.

---

## F7. Balanced-Accuracy Cutoff Optimizer

**What.** A module that chooses and applies the decision cutoff.

**Why here.** Because the scoring method is Balanced Accuracy, and the best cutoff is the training class balance, not 0.5 (A6). This is a free point or more that a lot of entrants will leave on the table.

**How.**
1. A function that sweeps across a fine range of possible cutoffs and returns the resulting score curve.
2. A function that finds the middle of the widest flat, high-scoring stretch on that curve. **Use this rather than the single raw best point**, which is noise-sensitive.
3. A function that computes the best cutoff separately per validation split, returning how much it varies — a stability check.
4. Report the theoretical class-balance-based cutoff alongside the empirical best one, and flag a warning if they're noticeably different, since that points to poor calibration.
5. Lock the chosen cutoff into your settings. **Never recompute it at submission time** — a cutoff silently recomputed on a different slice of data is a bug you won't notice until it's too late.

---

## F8. Submission Builder and Checker

**What.** A reliable, repeatable way to generate the CSV, plus every check from A7.

**Why here.** A7 explains why. This is the piece with the highest ratio of disasters-prevented to lines-of-code in the whole project.

**How.**
1. A function that builds the submission by **matching on image ID** — never by assuming row order — casts labels to plain integers, and saves the file correctly.
2. A checking function that verifies: the line count, the exact header text, that the ID set matches exactly, no duplicates, correct value types and range, no missing values, and that every ID has the `.png` suffix. **It should fail loudly and refuse to return a valid path if any check fails.**
3. A sanity-report function: predicted class balance, agreement rate with the shortcut, agreement with the previous submission, and a histogram of predicted probabilities.
4. A label-swap check function: run 20 known-label training images through the production prediction path and confirm they're correct. **Call this automatically as part of building the submission, so it can't be skipped.**
5. Every generated file should be named with a run ID and timestamp, and never overwritten.

---

## F9. Explainability — Visual "What Did the Model Look At" Overlays

**What.** Visual overlays showing which parts of the image the model relied on, shown both in the standardized and the original view.

**Why here.** Two reasons — one scientific, one for presentation. Scientifically, it's how you detect shortcut reliance (B2): if every overlay just highlights the shadow edge, you know your model is essentially a dark/light detector in disguise. For presentation, in a competition about a *visual illusion*, being able to show that your model pays attention to the crater rim rather than just the shadow is a compelling claim that no single score can make on its own.

**How.**
1. Use a standard visual-explanation library on the model's last convolutional layer (or the equivalent for transformer-based models).
2. Generate the overlay in the standardized frame, then map it back onto the original image so the user sees it in the orientation they uploaded.
3. Overlay it on the grayscale image with a clear, easy-to-read color scale at moderate transparency.
4. **Add a quantitative version**: compute what fraction of the overlay's "attention" falls inside the shadow region. Average this over your validation set. **This single number is your "shortcut reliance score,"** and you should track it across your models. Given equal scores, a model with a lower shadow-reliance number is the more robust one, and the one to prefer.
5. In the app, display this overlay next to the prediction, along with the shortcut-reliance number.

---

## F10. The Sun Simulator — Interactive Lighting Explorer ★ FLAGSHIP

**What.** A slider or dial that sweeps the assumed sun direction from 0° to 360°, live-updating three things: the standardized view of the image, the model's predicted class, and its confidence. Optionally, a circular chart showing predicted probability as a function of sun direction.

**Why here.** This is the feature that *is* the competition. The Pareidolia Paradox is the claim that appearance alone, without knowing the lighting direction, is ambiguous. This tool lets a user hold one image fixed and watch the model's interpretation change as the stated lighting changes — literally showing the illusion, interactively. And it lets you demonstrate the flip side, which is the real payoff: when you rotate the image **and** update the sun direction together, the prediction **stays the same** — a live, visible proof that your training approach actually worked. There's no more convincing way to show that you understood the problem.

**How.**
1. Backend: an endpoint that takes an image and returns predictions across a sweep of sun directions (say, 72 values). Run all 72 in a single batch — it costs a few hundred milliseconds and makes the interface feel instant.
2. Frontend: a slider tied to sun direction. On change, read from the precomputed sweep — no round trip needed per movement.
3. Show three panels: the original image, the standardized image at the current sun direction, and a confidence gauge.
4. Add a **circular chart** of predicted "bump" probability versus sun direction. For a well-behaved model on an unambiguous image, this should be fairly flat with a sharp transition — the shape of this curve is itself a useful diagnostic, and it looks impressive.
5. Add a **linked-rotation mode**: a second control that rotates the *image* and automatically updates the sun direction to match. Show that the prediction stays fixed. Label this panel something like "Consistency check," and display the prediction's variation across the sweep as a number. **A near-zero variation here is the strongest possible demonstration that your model learned the physics, not just the appearance.**
6. Preload a few hand-picked example images (a fresh crater, a boulder, a worn-down ambiguous case), so a viewer can try the feature in one click without uploading anything.

---

## F11. Batch Prediction and Confidence Review Queue

**What.** Upload a folder or zip file plus a metadata CSV; get back predictions for everything, sorted by confidence, with the least-confident cases shown first for a human to double-check; export as a CSV.

**Why here.** It's literally the competition's own workflow (2,000 images in, one CSV out), so it makes your app a real, usable tool rather than a toy. And the review queue reflects how this kind of technology would actually be used: someone cataloguing terrain would want the model to sort things and flag what needs a human's attention.

**How.**
1. An endpoint accepting a zip of images plus a CSV of sun directions, streaming back results.
2. Process in batches with a progress indicator.
3. Sort the output so the most uncertain cases sit at the top.
4. Show a review grid: the image, the standardized image, the prediction, a confidence bar, and a way to override the prediction.
5. Export in the exact competition format, running the F8 checker on the way out. **This means your app can literally produce the submission file**, which is a satisfying thing to demonstrate.

---

## F12. Robustness and Shortcut Audit Dashboard

**What.** An internal page that runs the Phase 5 diagnostics against the current model and displays them.

**Why here.** Because A5 means robustness isn't a nice-to-have, it's a core strategic factor. Making these numbers visible on a dashboard keeps the whole team focused on them, instead of treating the overall validation score as the only number that matters.

**How.**
1. Panels for: overall validation score; score on the shortcut-wrong subset; synthetic worst-case test score; sun-direction-shift stress-test drop; shadow-hiding test drop; the shortcut-reliance number from F9; and the full breakdown table.
2. A model-comparison view showing all these numbers side by side for every candidate. **Choose your final submission from this view, not from a single number.**
3. A calibration chart.
4. A confusion matrix where you can click on a cell to see the corresponding images.

---

## F13. Model Registry and Version Tracking

**What.** A structured, versioned store of model packages, each self-contained.

**Why here.** With several people, dozens of runs, and a hard deadline, "which checkpoint produced that 0.94?" is a question you will be asked, and answering it wrong costs hours.

**How.**
1. Each model package is a folder containing: the weights, the settings, the normalization numbers, the calibration values, the cutoff, the score summary, and the code version.
2. Folder names include the run ID and the score.
3. A simple index file with a "production" pointer that the app reads. Promoting a new model is a one-line change to that pointer.
4. Sync these packages to shared cloud storage regularly. **A GPU failure on Day 18 must not be able to destroy your work.**

---

## F14. API and Deployment Layer

**What.** The backend service and its container.

**Why here.** It separates the model from the interface, lets the app run anywhere, and makes your demo reliable.

**How.**
1. A backend framework with clear request/response validation. Load the model once at startup, never on every request.
2. Endpoints for: health check, model info, single prediction, batch prediction, explanation, and simulation.
3. Input checking: reject or auto-resize non-256×256 images with a clear message; validate the sun direction is a real angle.
4. A lightweight container setup, using a CPU-only version of your model framework for the demo container (much smaller image, and prediction on a single small image is fast even on CPU).
5. A one-command way to bring up both the API and the interface together.
6. Test the container on a machine that's never seen the project before. **"It works on my laptop" isn't a real deployment.**

---

## F15. Reproducibility Harness

**What.** A single command that regenerates the final submission from a completely clean copy of the repo.

**Why here.** Because many competitions require reproducibility to validate prizes, because it's the only real proof your result isn't the product of an undocumented manual step, and because it's what lets you confidently rebuild everything after a disaster.

**How.**
1. A single command that: checks the data is in place, rebuilds the cache, loads the locked validation splits, trains all your combined-model members with fixed seeds, generates validation predictions, checks that the score matches your recorded value within a small tolerance, runs prediction with test-time tricks, applies the locked cutoff, and writes and checks the final CSV.
2. Full determinism: fixed seeds everywhere, deterministic settings turned on, and pinned library versions.
3. Log how long the whole thing takes. If it takes eight hours, write that down so nobody starts it at 9 PM on Day 20.
4. **Run it at least once, from a completely fresh copy of the repo, before Day 19.** A reproducibility setup that's never actually been run is just a guess, not a real feature.

---
---

# PART F — THE FULL TOOLKIT

Organized as **Core** (you will use these), **Recommended** (clearly useful, adopt if time allows), and **Optional/Advanced** (differentiators or backup plans). Each entry explains what specifically you need from it, why, and how to use it here.

---

## CORE

### Python 3.10 or 3.11
**What:** The programming language. **Why:** Broad support across the machine learning ecosystem, with good typing support, while newer versions still occasionally have gaps in library support for computer vision tools. **How:** Create an isolated environment on Day 0, and pin every dependency to an exact version in your requirements file. **Detail:** Pin exact versions, not "at least" versions. A minor version bump in an augmentation library mid-competition can silently change how your augmentations behave and invalidate your comparisons.

### PyTorch 2.x + torchvision
**What:** The deep learning framework. **Why:** The best ecosystem for transfer learning; integrates smoothly with pretrained-model libraries; gives free speed improvements; mixed-precision training is mature and reliable. **How:** Install the version matching your GPU driver. Use mixed precision on modern GPUs, an efficient memory layout for convolutional models, and compile your model once your code is stable. **Detail:** Enable a performance-tuning flag for fixed input sizes — which yours are — but note it conflicts with strict reproducibility, so turn it off for your final reproducibility run.

### timm (PyTorch Image Models)
**What:** A library with roughly 1,000 pretrained vision models under one consistent interface. **Why:** It's the single highest-value library for this problem. It correctly handles single-channel input by combining the pretrained color weights (B7), supports stochastic depth, and lets you swap architectures by changing a single text string. **How:** Create a model with your chosen backbone name, pretrained weights, single-channel input, and your chosen regularization. Browse available pretrained models through the library's listing function. **Detail:** Prefer variants pretrained on a broader dataset where available; they tend to transfer better to non-everyday imagery. Use the library's default preprocessing settings as a starting point, but **override the normalization numbers with your own lunar-specific ones** rather than using the defaults meant for everyday photos.

### NumPy
**What:** Array computing. **Why:** Everything — your cached image arrays, your combined prediction tables, your cutoff sweeps. **How:** Cache images as a single array of raw bytes; use memory-mapped arrays if RAM is tight. **Detail:** Use a compact numeric type throughout your pipeline where possible; using an unnecessarily precise type doubles your memory use for no real benefit.

### pandas
**What:** Handling tabular data — metadata, splits, and submissions. **Why:** The CSV format is your deliverable. **How:** Read CSVs with explicit string types for IDs so they're never accidentally converted to numbers. Build submissions by matching on image ID, never by row position. **Detail:** Remember to disable the automatic index column when saving — forgetting this is a classic zero-score bug (A7).

### OpenCV
**What:** Image processing: rotation, interpolation, shape detection, contrast enhancement, thresholding, contour finding. **Why:** Your standardization engine (F2) and your hand-built features (C5) both rely on it. **How:** Use its rotation functions with smooth interpolation and sensible border handling for standardization; its contrast-enhancement tool; its thresholding tools for shadow masks; its contour and convex-hull tools for the containment ratio. **Detail:** Install the headless variant on servers to avoid unnecessary dependencies. Remember that this library reads colors and coordinates in a different order than NumPy does — **this is exactly where sun-direction sign errors tend to come from.** Write down your chosen convention in a comment at the top of your standardization module.

### Albumentations
**What:** A fast, widely-used augmentation library. **Why:** Best-in-class speed and a huge set of operations for the physically-neutral augmentations. **How:** Use it for shifting/scaling (with rotation limited or disabled), brightness/contrast randomization, gamma randomization, noise, small dropout patches, and mild distortion. **Detail (critical):** **Wrap it, don't use it raw for anything involving orientation.** This library has no concept of your sun-direction value. Your own custom module (F3) should own all geometric transforms; let this library handle only the brightness and mild-spatial operations that carry no lighting-direction meaning. Disable its built-in flip functions entirely and handle flips yourself.

### scikit-learn
**What:** Metrics, data splitting, calibration, and simple classical models. **Why:** Its balanced-accuracy function is your scoring metric; its group-aware stratified splitting is your validation approach; its calibration and logistic regression tools handle calibration and stacking. **How:** Use its group-aware, stratified splitting function, passing in your group IDs. **Detail:** This splitting function requires a recent version of the library and can't always perfectly balance both constraints at once; check the resulting per-split class ratios and sizes after generating them, and confirm they're acceptable.

### Matplotlib (and seaborn)
**What:** Plotting. **Why:** Every diagnostic in Phase 1 and Phase 5 is a chart: sun-direction histograms, calibration histograms, per-class average images, score-versus-cutoff curves, calibration charts, confusion matrices, circular probability plots. **How:** Write a shared visualization module with named functions for each standard chart, so everyone produces identical, comparable plots. **Detail:** Use a polar plot for circular histograms. The calibration histogram from Phase 1.5 is the single most important chart in your whole project — make it good.

### Git + GitHub
**What:** Version control. **Why:** Multiple people, fifteen days, shared infrastructure. **How:** Use feature branches and reviews for anything touching your core shared modules or the validation-split file. Tag the commit that produced each submission. **Detail:** Log the code version into every run's summary file (F4.5). When you later ask "what code produced this score," the answer should be exact and automatic.

---

## RECOMMENDED

### Weights & Biases
**What:** Experiment tracking. **Why:** With dozens of runs across several people, an untracked experiment is a wasted one. **How:** Initialize a project, log metrics per training pass, log your breakdown tables, and log a gallery of your worst mistakes. **Detail:** The free tier is plenty. The single highest-value thing to log isn't the loss curve — it's the gallery of your 32 worst validation mistakes, refreshed with every run.

### PyTorch Lightning (or a hand-written training loop)
**What:** A framework for structuring your training loop. **Why:** Removes a lot of repetitive boilerplate around mixed precision, checkpointing, early stopping, and multi-GPU support. **How:** Wrap your model, loss, and optimizer in the framework's structure, and configure early stopping and checkpointing through it. **Detail:** Honest trade-off — if nobody on your team already knows this particular framework, a shorter, hand-written training loop is often faster to get right within a 15-day window than learning a new framework's conventions. **Use whatever your team already knows well.**

### LightGBM
**What:** A gradient-boosting library. **Why:** Trains in seconds on your hand-built C5 features, gives you interpretable feature importances that tell you which physical signals actually matter, and provides a **usefully different ensemble member** (Phase 6.2). **How:** Train it with class-balanced weighting on the same validation splits as everything else. **Detail:** Its value is variety and interpretability, not raw accuracy. Don't expect it to beat your neural network; do expect it to add a small boost when combined, and to teach you something useful in Phase 2.

### A visual-explanation library
**What:** Tools for generating "what did the model look at" overlays. **Why:** Feature F9, and your shortcut-reliance metric. **How:** Apply it to the model's last convolutional layer (or equivalent for transformer models). **Detail:** Which internal layer to target differs by architecture; budget an hour for getting this wired up correctly, not five minutes.

### FastAPI + Uvicorn
**What:** The backend API layer (F14). **Why:** Fast, with automatic documentation and built-in input validation. **How:** Define your endpoints with clear input/output models; load the model once at startup; run the server. **Detail:** The automatically-generated API documentation page is a free, credible-looking demo tool on its own.

### Streamlit (or Gradio)
**What:** The user interface (F14/F10). **Why:** Turns Python into a working web app in an afternoon, which is the right trade-off when modelling is the priority. **How:** Use built-in file upload, slider, and image display components; use a plotting library for the circular chart; cache your model loading so it only happens once. **Detail:** One of these tools is even faster for a single-purpose demo and gives you a free shareable link, which is great for a remote presentation. **Use the one that supports multiple pages if you need the diagnostic dashboard too; use the simpler one if you just need one great demo.**

### Docker + docker-compose
**What:** Containerization. **Why:** Reproducibility, and a demo that works on someone else's machine. **How:** Use a multi-stage build, and a CPU-only version of your deep learning framework for the demo container to keep it small. **Detail:** Decide deliberately whether to bundle the model directly into the container image or mount it separately — bundling makes the demo more portable, but makes the image larger.

### Hydra or OmegaConf
**What:** Structured configuration management. **Why:** Every experiment is a settings file, and you'll want to combine pieces (a base config plus a model-specific config plus an augmentation-specific config) and override individual values from the command line. **How:** Set up your main script to load composed configs, with the ability to override specific values when launching a run. **Detail:** The automatic per-run output folder that some of these tools provide is genuinely useful. If your team finds the tool too heavy, simpler manual config composition gets you most of the benefit at a fraction of the learning cost.

### Progress bars, image-hashing, and fast nearest-neighbour search tools
**What:** Progress indicators; perceptual image fingerprinting; fast similarity search. **Why:** Fingerprinting and fast similarity search together power the near-duplicate detection in Phase 1.8, which protects your entire validation setup (B10). **How:** Compute a perceptual fingerprint for each image, then use a fast nearest-neighbour search over embeddings for the more thorough pass, then find connected groups over the resulting similarity graph.

---

## OPTIONAL / ADVANCED

### Rotation-equivariant neural networks
**What:** Networks with rotational symmetry built directly into their architecture, rather than learned only from augmentation. **Why:** This problem has an explicit rotational structure — the label doesn't change under joint rotation of the image and the sun direction. A network built with this symmetry baked in encodes that directly. **How:** Build such a model and combine it with sun-direction conditioning. **Detail:** High risk, high reward, and there are **no pretrained weights available for this kind of architecture**, which is a real drawback given only 7,854 training examples. Attempt this only if you have a spare person and are ahead of schedule. It would be a genuinely distinctive contribution if it worked.

### Self-supervised pretraining
**What:** Pretraining a model on your own unlabeled data before fine-tuning it on the labelled task. **Why:** You have about 9,854 lunar images total, and pretraining on everyday photos transfers imperfectly to grayscale planetary imagery. **How:** Use a self-supervised learning library, pretrain for a good while on your combined train+test images, then fine-tune. Use only physically-neutral augmentations for this step, or you risk teaching the model to ignore exactly the lighting information you need. **Detail:** This is your strongest backup plan if **pretrained weights turn out to be disallowed** (Step 0.1). Keep it in your back pocket.

### Optuna (or similar automated tuning tools)
**What:** Automated hyperparameter search. **Why:** Automates trying different settings. **How:** Search over learning rate, weight decay, regularization strength, and augmentation probabilities, with early stopping for unpromising trials. **Detail:** Be honest about the noise floor (B12). With run-to-run variation around half a point, a large automated search will "find" configurations whose apparent advantage is really just noise. Use automated search on a single validation split with a fixed seed for rough exploration, then confirm any winner with a full multi-split, multi-seed check. **In a 15-day window, hand-designed comparisons usually beat automated search**, because they teach you something along the way.

### ONNX Runtime
**What:** Portable, optimized model inference. **Why:** Fast CPU-based prediction for the demo container, with no need for the full deep learning framework at prediction time. **How:** Export your model to a portable format, then run predictions using a lightweight runtime. **Detail:** **Always double-check that the exported model produces the same outputs** as the original, on a sample of images, before trusting it. A small numerical mismatch introduced during export can shift predictions across your cutoff.

### DVC (Data Version Control)
**What:** Version control specifically for large data files. **Why:** Keeps large files out of your code repository while still versioning them alongside your code. **How:** Track your raw data folder and configure a remote storage location. **Detail:** Valuable for a longer project; possibly more setup than you need for 15 days. A shared cloud folder with a strict naming convention may serve you just as well.

### Plotly
**What:** Interactive charts. **Why:** The Sun Simulator's circular probability plot (F10.4) is far more compelling as an interactive chart than a static one. **How:** Use its polar scatter chart type, embedded in your chosen UI framework.

### A full custom frontend (React or similar)
**What:** A production-grade, fully custom web interface. **Why:** Worth it only if presentation is explicitly scored, or you're building this beyond the competition itself. **How:** Set up a modern frontend project, style it, and connect it to your backend API. **Detail:** **This costs 2–3 days that would otherwise go toward modelling.** In a 15-day window with a hidden-label, metric-scored competition, that's almost always the wrong trade-off. Choose it only with your eyes open.

### pytest
**What:** A testing framework. **Why:** Your augmentation rules (F3.4) and submission checker (F8.2) are exactly the kind of code where a silent bug costs you the whole competition, and a test takes ten minutes to write. **How:** Test the round-trip consistency checks, the standardization consistency checks, the submission format checks, and your scoring calculation against a hand-computed example. **Detail:** You don't need broad test coverage overall. You need tests specifically on the handful of functions where a silent error would be unrecoverable.

### Kaggle Notebooks / Google Colab Pro / other cloud GPU platforms
**What:** GPU access. **Why:** If your team lacks local GPUs. **How:** Free platforms typically offer a limited number of GPU hours per week with time-limited sessions; paid tiers offer better GPUs and longer runtimes for a modest monthly fee. **Detail:** Session timeouts are the enemy. **Save a checkpoint after every training pass to persistent storage**, and write your training loop to automatically resume from the last checkpoint. Losing an eleven-hour run to a disconnect on Day 17 is a completely preventable disaster.

---
---

# PART G — RESOURCES, RISKS, AND CHECKLISTS

## G1. Reading List

**On the illusion itself — read these first, they're short and they explain the problem directly:**
- Wikipedia's article on the "crater illusion" — a concise explanation of the phenomenon and its cause.
- EarthSky's piece on the crater-dome illusion — includes a famous rotating-crater demonstration that's a great visual reference.
- CosmoQuest's lighting-effects and illumination-illusion guides — written specifically to train human lunar-crater identifiers, which makes them unusually directly relevant. One of them walks through exactly the shading reasoning covered in A3.
- Popular Science's explainer on the crater illusion — a good plain-language account of why the imaging setup produces this effect.
- Bernabé-Poveda & Çöltekin's peer-reviewed paper on the "terrain reversal effect" in satellite imagery — the academic source for the claim that both 180° rotation and brightness inversion remove the illusion. **This paper is the citation behind your augmentation strategy.**
- Çöltekin and colleagues' more recent work on how lighting direction changes how often the illusion occurs.

**On lunar terrain and lighting physics:**
- Search terms worth trying: "shape from shading lunar," "Lunar-Lambert photometric function," "Hapke model," "LRO NAC illumination."
- Recent research on reconstructing lunar terrain shape under varying sunlight shows that brightness changes give strong shape information *along* the lighting direction but much weaker information perpendicular to it — this is the justification for the directional-derivative input channels described in C3, Level 4.

**On crater detection and classification (read these for domain insight, not for architecture ideas):**
- Silburt and colleagues' widely-cited paper on identifying lunar craters via deep learning — note that it works on elevation maps rather than photos, which is exactly why lighting direction isn't a problem for them, but is for you.
- Reviews of automated crater detection note that differences in image resolution, lighting conditions, and viewing angle routinely cause inconsistencies in detection — you're being scored specifically on this kind of failure.
- Research on synthesizing realistic lunar imagery, relevant if you're considering generating synthetic training data.

**On the relevant machine learning techniques:**
- The original FiLM paper — the conditioning method described in C3, Level 3.
- Geirhos and colleagues' paper "Shortcut Learning in Deep Neural Networks" — the theoretical backdrop for A5 and B2. **Assign this to whoever owns robustness.**
- Cohen & Welling's paper on group-equivariant convolutional networks — background reading for the physics-consistency idea in C4, and for rotation-equivariant architectures.
- Guo and colleagues' paper on calibrating modern neural networks — the basis for the calibration step in Phase 5.5.
- The "Noisy Student" self-training paper — relevant if you pursue pseudo-labelling.
- The documentation for your chosen pretrained-model library, and any published fine-tuning recipes — practical, tested default settings.

**On competition craft in general:**
- Read winning-solution writeups from any image-classification competition with a small dataset and a domain-physics twist. The recurring pattern in all of them is: trustworthy validation, domain-appropriate augmentation, diverse combined models, and careful cutoff handling. That's exactly this plan.

---

## G2. Risk Tracker

| # | Risk | Likelihood | Impact | How to prevent it | Owner |
|---|---|---|---|---|---|
| R1 | Submission file format gets rejected | Medium | Fatal | The F8 checker; a test submission on Day 3 | OPS |
| R2 | Classes accidentally get swapped | Low | Fatal | An explicit check, plus the known-label check (Phase 8.6) | OPS |
| R3 | Sun-direction convention miscalibrated | Medium | Severe | Phase 1.5 calibration; the average-image check | DATA |
| R4 | Validation score inflated by near-duplicates | High | Severe | Grouped validation splits (Phase 1.8/1.9); compare grouped vs. ungrouped | DATA |
| R5 | Model relies purely on the simple shortcut | High | Severe if the test set is adversarial | Phase 5 stress tests; physics-based training penalties; label-flip augmentation | MODEL |
| R6 | Train/test lighting or feature-mix differences | Medium | Severe | "Can you tell them apart" test; standardization; full-circle augmentation | DATA |
| R7 | Augmentation breaks the sun-direction/label relationship | High if unmanaged | Severe | The F3 module, its tests, and the visual debugger | MODEL |
| R8 | Overfitting on a small dataset | High | Moderate | Pretraining, regularization, early stopping, combining models | MODEL |
| R9 | Chasing noise (differences under 1 point) | High | Moderate (wasted days) | Multi-seed testing; share the noise floor with the team | MODEL |
| R10 | GPU or session loss | Medium | Moderate | Frequent checkpointing to cloud storage; auto-resume | OPS |
| R11 | Team members drift apart on preprocessing/splits | Medium | Severe | Shared modules; locked-in validation split file; review process for core files | OPS |
| R12 | Pretrained weights turn out to be disallowed | Low | Severe | Confirm on Day 1; have a self-supervised-pretraining backup ready | OPS |
| R13 | Running out of time | Medium | Severe | Valid submission by Day 3; two-day buffer at the end; feature priority order | All |
| R14 | Cutoff chosen based on noisy validation data | Medium | Moderate | Stable-stretch selection, per-split variation check, class-balance fallback | MODEL |
| R15 | Mismatch between training-time and prediction-time preprocessing | Medium | Severe | Shared preprocessing functions; a matching check in Phase 8.6 | APP |

---

## G3. Daily Standup Template

Fifteen minutes, every day, at the same time. Four questions:
1. What's our current best validation score, on the locked splits, and which run produced it?
2. What's our current score on the shortcut-wrong subset, and on the synthetic worst-case test?
3. What's blocking anyone?
4. Is our latest valid submission file still current, and would it still upload successfully today?

Question 4 isn't paranoia — it's the habit that makes the deadline a non-event.

---

## G4. Final Pre-Submission Checklist

Print this. Check every box, out loud, with a second person watching.

**Format**
- [ ] Exactly 2,001 lines
- [ ] Header exactly `image_id,label`
- [ ] Every `image_id` value has the `.png` suffix
- [ ] ID set matches `test_metadata.csv` exactly; no duplicates; nothing missing
- [ ] Labels are plain integers, 0 or 1; no decimals, no text, no missing values
- [ ] No index column; correct text encoding; single trailing newline
- [ ] A single CSV file, correctly named per the platform's requirements

**Correctness**
- [ ] Predictions generated by the exact production prediction process
- [ ] Known-label training images pass correctly through that same process (the swap check)
- [ ] Cutoff read from your locked settings, not recomputed
- [ ] Preprocessing settings match exactly what was used during training
- [ ] Raw probabilities saved and backed up before applying the cutoff

**Sanity**
- [ ] Predicted class balance looks reasonable given the training class balance
- [ ] Shortcut-agreement rate recorded and understood
- [ ] Agreement with your previous submission recorded and explained
- [ ] 40 predicted images visually spot-checked by a human
- [ ] Predicted probability distribution has two clear peaks, not a pile-up at the cutoff

**Process**
- [ ] Submitted at least 6 hours before the deadline
- [ ] Platform acceptance confirmed and screenshotted
- [ ] A backup submission file kept on hand
- [ ] The repository tagged at the exact commit that produced this file
- [ ] The README lets a stranger reproduce this file from scratch

---

## G5. Traceability — Every Part of the Brief, and Where It's Covered

| Part of the problem statement | Where it's covered |
|---|---|
| Online ML challenge, image classification / computer vision | A2 (why not detection/segmentation) |
| 256 × 256 images | A2, F1.2, Part F (resolution comparison) |
| Grayscale | A2, B7, C3 Level 4 |
| Lunar surface imagery | A2 (shadow physics on an airless body), A3, G1 |
| Class 0 — Depth: craters, holes, depressions | A2 (sub-types), A3 (shading signature), C5 |
| Class 1 — Rise: mounds, hills, rocks, boulders | A2 (sub-types), A3 (shading signature), C5 |
| Binary classification | Throughout; A6 (cutoff), F7 |
| Appearance changes with sunlight direction | A3 in full, C1, C2, F10 |
| Crater can look like a depression or a rise | A3, B1, C4, Phase 5.2 |
| Topographic inversion (the named phenomenon) | A3, G1 (reading list under all four names) |
| `sun_azimuth_angle` in metadata | A4, C1, C3, Phase 1.4, Phase 1.5 |
| "Use this information appropriately" | A4, A2, C1–C4 |
| 7,854 training images | A2 (consequences), B5, Phase 1, F1.3 |
| 2,000 evaluation images | A2 (uncertainty and the unlabeled corpus), C4, Phase 4.8 |
| `train_metadata.csv`: image_id, sun_azimuth_angle, label | F1.1, Phase 1.1 |
| `test_metadata.csv`: image_id, sun_azimuth_angle | F1.1, A4 (available at prediction time) |
| Evaluation labels hidden | A2, Phase 2, Phase 5, F5 (your own testing is the only signal) |
| Must train on the provided training dataset | Phase 0.1 (rules check on outside data/pretraining) |
| Predict for all 2,000 evaluation images | A2, A6, F8 |
| Submission CSV: exactly `image_id,label` | A7 in full, F8, G4 |
| Example rows `eval_00001.png,0` | A7 item 3 (the `.png` suffix trap) |
| Balanced Accuracy metric | A6 in full (deriving the cutoff = class balance rule), B4, F7 |
| "Focus on underlying terrain, not solely shadow patterns" | A5 in full, B2, C5, Phase 5, F9, F12 |
| Dataset composition / metadata structure | Phase 0.1, Phase 1.1–1.3 |
| Opens 1 Sep 2026 | Part D header (5 days already elapsed) |
| Closes / deadline 21 Sep 2026, 11:59 PM | Part D timeline, Phase 8.9, Phase 0.1 (timezone) |
| Competition round: access data, develop, train, submit | Phases 1–8 |
| Final submission must contain all 2,000 predictions, in the prescribed format | A7, F8, G4 |

---

## G6. If You Only Do Five Things

If the two weeks get tight and you have to prioritize ruthlessly, these five, in this order, capture most of the available value:

1. **Calibrate the sun-direction convention from the data, and standardize every image's lighting** (Phase 1.5, Phase 1.6, C1). This is the core idea the whole competition is built to reward.
2. **Lock in group-aware, sun-direction-balanced validation splits on Day 1, and never change them** (Phase 1.8–1.9). Everything you learn afterward is only as trustworthy as this.
3. **Build the sun-direction-aware augmentation module, and never use a raw library flip or rotation** (C2, F3). This alone puts you ahead of most other teams.
4. **Set your cutoff based on the class balance, fine-tuned on your own validation predictions** (A6, F7). A free point that many entrants will miss.
5. **Have a validated submission ready by Day 3, and re-check it before every upload** (Phase 2.7, A7, F8, G4). This turns a potential disaster into a non-event.

Everything else in this document — the physics-based training penalties, combining multiple models, test-time tricks, the Sun Simulator — is upside on top of a solution that these five things already make competitive.

---

*End of playbook.*
