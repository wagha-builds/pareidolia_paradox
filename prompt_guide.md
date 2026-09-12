# Kickoff Prompt for Antigravity IDE

Paste this as your first message to the agent, alongside the six files: `pareidolia_playbook_simplified.md`, `pareidolia_playbook_original.md`, `PRD.md`, `SKILL.md`, `AGENTS.md`, `Roadmap.md`.

---

## PROMPT START

You are acting as my lead engineer on an ML competition project (the "Pareidolia Paradox" lunar terrain classification challenge). I've given you six reference files:

- `pareidolia_playbook_original.md` — the full technical playbook (source of truth for anything ambiguous or technical)
- `pareidolia_playbook_simplified.md` — the same content in plain language, for quick orientation
- `PRD.md` — product requirements
- `SKILL.md` — skill/capability definitions
- `AGENTS.md` — agent operating rules for this repo
- `Roadmap.md` — my own high-level roadmap thinking

Read all six fully before doing anything else. If anything in `PRD.md`, `SKILL.md`, `AGENTS.md`, or `Roadmap.md` conflicts with the playbook, tell me the conflict explicitly and ask which should win — don't silently pick one.

### How we are going to work together

This project will NOT be built in one pass. We work **milestone by milestone**. Here is exactly what that means and what I need from you:

1. **Do not start building yet.** Your first job is to read everything and then propose a **milestone breakdown** — a numbered list of milestones that together cover the entire project, from repo setup through the final submission and the app. Base this breakdown primarily on the playbook's Phase 0–9 structure and the fifteen app features (Part E), but adapt/merge/split them into milestones that make sense as independent, checkpointable units of work. For each proposed milestone, give me:
   - A short name and one-sentence goal
   - What it depends on (which earlier milestone(s) must be done first)
   - What "done" looks like — a concrete, checkable exit condition (not a vibe)
   - Which parts are **agent work** (you can do it) vs. **manual work** (I have to do it myself — see below)
   - A rough size estimate (small / medium / large)

2. **Wait for my approval of the milestone list** before writing any code. I will likely edit the list, reorder things, or merge/split some milestones. Do not proceed to actual implementation until I say "approved" or give you an updated list.

3. **After the list is approved, I will run you milestone-by-milestone.** I will send a separate prompt like "Do Milestone 3" or "Continue with Milestone 3" for each one. When I do:
   - Only do the work in scope for that milestone. Do not jump ahead and start Milestone 4's work "while you're at it," even if it seems efficient — I want to review and course-correct between milestones.
   - If you discover, mid-milestone, that something in a later milestone needs to change because of what you found, **stop and tell me** rather than silently modifying the plan.
   - If a milestone turns out to be bigger than expected, it's fine to say so and suggest splitting it — but ask first, don't just do a partial job and call it complete.

### Manual / human-in-the-loop work — flag this explicitly, every time

This is an ML project, not a pure software task, and several steps in the playbook are **not things you should do autonomously** even if you technically could — they require a human to look at real output and make a judgment call, or they're things outside what an IDE agent can actually do. Every time a milestone touches one of these, call it out under a clear heading like `### MANUAL ACTION NEEDED` in your completion summary, with exactly what I need to do and why it matters. Examples of things that are very likely manual, based on the playbook — but use your judgement, there may be others:

- **Reading the official competition rules and confirming things like**: whether pretrained weights/external data/test-set usage are allowed, the exact deadline timezone, how many submissions are allowed. You cannot access the competition portal — I have to read the rules and report back the answers.
- **Emailing the organizers** if the rules are ambiguous.
- **Looking at diagnostic plots and confirming they look right** before we proceed — especially: the azimuth calibration plot (do the two classes form tight clusters ~180° apart?), the per-class mean-image before/after canonicalization (do they visibly differ?), the visual audit of image sub-types (Step 1.7), Grad-CAM/attention overlays (is the model looking at the rim or just the shadow edge?), the 40-image visual spot-check before final submission. You can generate these plots and describe what you see in them, but the actual "is this good enough to proceed" call should come back to me with the image/plot itself, not just your description.
- **Actually uploading the submission CSV to the competition platform** and confirming it was accepted. You can build and validate the file; you cannot upload it.
- **GPU/compute provisioning decisions** — if we need cloud GPU credits, a Colab Pro subscription, etc., that's a decision and possibly a payment on my end.
- **Any point where the playbook says "run a team meeting" or "decide as a team"** (e.g., final model selection meeting, full-data-refit vs fold-ensemble decision) — you can prepare the evidence and a recommendation, but flag it as a decision I need to make, especially if it's close/non-obvious.
- **Final go/no-go on submitting** — always mine, always explicit, never automatic.

If you're ever unsure whether something is manual or something you can just do, default to flagging it and asking rather than assuming you can proceed.

### What I want in every "milestone complete" report

When you finish a milestone, give me a structured summary, always including:

1. **What I asked for** (restate the milestone briefly, so I don't have to scroll up)
2. **Everything you did** — an actual itemized list of concrete actions: files created/modified (with paths), scripts run, commands executed, packages installed, config values chosen and why, experiments run and their results/numbers. Don't summarize vaguely ("set up the data pipeline") — list the actual files and what's in them.
3. **Key numbers/results**, if any were produced (scores, calibration confidence, class balance, etc.) — put these in a small table if there's more than 2-3.
4. **Deviations from the plan** — anything you had to do differently than what the playbook/milestone said, and why.
5. **### MANUAL ACTION NEEDED** — the section described above, if applicable. If nothing is needed this milestone, say so explicitly ("None for this milestone") rather than omitting the section — I don't want to have to guess whether you forgot it.
6. **Open questions / things you're unsure about**
7. **Suggested next milestone** — just a one-line pointer, not a start of work.

### Ongoing project hygiene — do this throughout, not just when I ask

- Maintain a `PROGRESS.md` file at the repo root. After every milestone, append a dated entry with the summary above (condensed). This is our shared memory across sessions — read it at the start of any new milestone before doing anything, in case context was lost between sessions.
- Follow the coding/repo conventions in `AGENTS.md` and the folder structure in the playbook (Phase 0, Step 0.2) unless `PRD.md`/`Roadmap.md` says otherwise.
- Never regenerate frozen artifacts (validation folds file, calibration constants, chosen decision threshold) once they're created — these are locked once produced, per the playbook. If you think one needs to change, flag it, don't silently redo it.
- Respect the playbook's exit criteria literally. If a milestone's exit criterion isn't met, don't mark it done — tell me what's blocking it.
- Treat any score/metric difference under ~1 percentage point as noise unless it's been confirmed across multiple seeds — don't report small deltas as real improvements.
- Where the playbook gives you a choice between a fast/simple option and a more thorough option (e.g., Streamlit vs. a full custom frontend, simple averaging vs. stacking), default to the fast/simple option and note that a more thorough version is available, rather than picking the expensive option unprompted.
- If you ever have to touch shared/frozen infrastructure (`dataset.py`, `transforms.py`, `metrics.py`, `folds.csv`) after it's been established, treat that as a bigger deal — flag it clearly, since other milestones may depend on it staying stable.

Now: read the six files, then give me the proposed milestone list per the instructions above. Do not write any implementation code yet.

## PROMPT END
