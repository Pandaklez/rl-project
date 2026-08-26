# Session summary: experiment (E), the supervised regression benchmark

2026-08-26. Adds a new experiment, not present before this session: a plain
supervised regressor trained by direct gradient descent against GT, as a
benchmark alongside (A)'s lifted baseline and (B)/(C)'s RL/GAIL correction.
Full writeup is in `report.md` under **(E) Supervised regression** and the
updated **Primary metric** / **Seed sweep** tables — this file is the
shorter "what happened and why" version.

## Headline result

| variant | PA-MPJPE, test (mm) | Δ vs (A) |
|---|---|---|
| (A) lifted baseline (no correction) | 34.396 | 0 |
| (B1)/(B2)/(B3) PPO correction | 34.48 – 34.51 | +0.08 to +0.12 (worse) |
| (C) PPO + GAIL | 34.45 – 34.53 | +0.05 to +0.14 (worse) |
| **(E) supervised regression** | **29.297 ± 0.196** | **-5.099 ± 0.196 (better)** |

**(E) is the only condition in this document that beats (A).** Every RL/GAIL
variant tried so far ((B), (C)) comes out slightly *worse* than doing nothing;
a plain per-frame MLP trained with a supervised MSE loss against GT — no PPO,
no reward, no discriminator, no rollout — closes 5.1mm (≈15%) of the 34.4mm
gap in one pass. Statistically decisive: t = -20.4 over 1,122 paired
clip-camera deltas, three seeds, seed spread (sd 0.196mm) a small fraction of
the effect size.

**What this does and doesn't mean.** (E) is not a fair head-to-head against
(B)/(C) — it gets GT directly at training time, which (B)/(C) deliberately
don't (the entire point of the GT-free reprojection reward). So this isn't
"(E) beats (B)/(C) as a method" — it's "the ceiling, when GT is available at
all, is far below 34mm, so the failure of (B)/(C) isn't about a small ceiling."
Read together with the existing finding that the reprojection reward's own
displacement from GT is small (+0.31px, see *Reading these numbers* in
report.md), the likely story is that the reprojection reward is just a much
weaker training signal for joint-angle accuracy than a direct GT gradient —
independent of that small displacement. Full argument is in report.md.

**This reframes experiment (D).** (D) — PPO with an MSE-to-GT term folded into
its reward (`scripts/run_exp_d.sh`) — is the one condition that gives PPO
access to the same kind of signal (E) shows is this informative. **(D) has not
been run to completion in this repo — no checkpoints, no eval dumps, no result
in report.md.** It's worth running: (E) shows the signal is there, (D) is the
test of whether PPO can actually use it.

## What was built

New files:

| file | purpose |
|---|---|
| `src/models/supervised.py` | `SupervisedPoseRegressor` (the model) + `gt_space_mse` (the loss) |
| `src/train_supervised.py` | training script — i.i.d. frame SGD, not PPO rollouts |
| `scripts/run_exp_e.sh` | 3-seed train+eval launcher, resumable, GPU training / CPU eval |

Modified:

| file | change |
|---|---|
| `src/evaluate.py` | `eval_supervised_model` + `--supervised_checkpoint` flag, so (E) scores through the same PA-MPJPE pipeline as (A)-(D) |
| `src/viz_pose.py` | `SupervisedPoseVizLogger` — same lifted/corrected/GT skeleton grid as `PoseVizLogger`, but calls the regressor directly instead of going through `src.env.rollout_policy` (the regressor is stateless, no env to step) |
| `scripts/sweep_stats.py` | `E_VARIANTS` entry so `--table pampjpe` picks up (E)'s dumps automatically |
| `report.md` | (E) column/rows added to Primary metric + Seed sweep tables, new `## (E) Supervised regression` section, closing paragraph in `## Reading these numbers`; every "no variant beats doing nothing" claim re-scoped to RL/GAIL so it isn't contradicted by (E) |

## Key design decisions

**Architecture: plain per-frame MLP, no RNN (yet).** Same `hidden_dims=(512,
256)` as `PoseActor`/(B3) — same capacity, so any gap between (E) and (B3) is
attributable to the training objective, not model size. Input is the lifted
pose alone (66-d, no `trans`/`betas`/evidence/`corrected_{t-1}` — there's
nothing analogous to that last one since the regressor is stateless). Output
is a residual delta added to the input, same convention as the RL action. A
recurrent variant was discussed and deliberately deferred to a separate,
later experiment — building it alongside this one would have conflated "does
supervision help" with "does temporal context help."

**Loss: MSE, but in one shared normalised space.** `data/processed_movi.h5`
stores GT and lifted poses both normalised, but with *different* per-camera
affine maps. Computing MSE directly on the raw stored tensors — which is what
(D)'s `r_mse` reward term does (`src/env.py::_apply_mse`) — compares two
different affine encodings of the same 66 numbers, the same bug
`src/models/discriminator.py::PoseSpace` exists to fix for the GAIL
discriminator (report.md: ~5.1° RMS / 31° root bias if uncorrected).
Tolerable inside a baselined PPO reward (the mismatch mostly cancels
relatively); not tolerable for a loss gradient descent is driven toward
directly. `gt_space_mse` reuses `PoseSpace(exclude_joints=())` to remap the
model's output into GT-normalised space before the subtraction. Still just
elementwise MSE — no geodesic/rotation-matrix loss, no per-joint reweighting.

## Bugs found and fixed along the way (not specific to this experiment)

Windows' console defaults to the cp1252 codepage, which chokes on a handful of
Unicode glyphs — **not** most non-ASCII (`±`, `—` both encode fine), just
box-drawing dashes (U+2500) and the arrow (U+2192)/delta (U+0394) glyphs. Three
`print()` calls in `src/evaluate.py` and one in `scripts/sweep_stats.py` used
these and would crash **any** `--dump_scores` or `--table pampjpe` run on
Windows — this affected (A)-(D) too, not just (E), and was only caught because
this session ran (E) for real on a Windows machine. Fixed to ASCII, matching
the convention `src/train.py`'s own comments already established for this
exact issue.

## Data note

`data/processed_movi.h5` was regenerated mid-session (`data/norm_upsample.py`)
to the current 22-joint format — the local copy had been a stale 52-joint
snapshot. Verified before running anything real: `n_joints` attrs, pose
shapes, norm stats, and frame counts (1,894,445 train frames) all match what
`report.md`'s existing (A)-(D) numbers were built from.

## Suggested follow-ups

- **Run (D) to completion** — see above, it's the natural next question.
- **RNN variant of (E)** — deferred on purpose, not forgotten.
- `scripts/sweep_stats.py`'s `--table rollout`/`--table diagnostics`/
  `--table config` don't apply to (E) (no PPO rollout, no TensorBoard training
  curve in that format) — only `--table pampjpe` does. (E)'s own TensorBoard
  scalars/images live directly in `checkpoints/exp_e/s*/` instead.
