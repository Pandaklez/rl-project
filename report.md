# Results

All figures on the **test** split (187 clips x 2 cameras = 374 clip-cameras),
except the reprojection rows, which are training-rollout and held-out-validation
diagnostics read from TensorBoard.

## Primary metric

| | (A) SMPLer-X, no correction | (B1) PPO, trans frozen | (B2) PPO, du,dv image shift | (B3) PPO, pose-only state | (C) PPO + GAIL |
|---|---|---|---|---|---|
| **PA-MPJPE, test (mm), lower is better** | **34.40** | 34.49 ± 0.01 | 34.52 ± 0.01 | 34.51 ± 0.07 | |
| **Δ vs (A), same clip-cameras (mm)** | 0 *(by definition)* | +0.094 ± 0.011 | +0.121 ± 0.015 | +0.118 ± 0.074 | |

**Reproduce.**
```bash
# 1. score every checkpoint on the test split (~13 min each, parallelisable)
mkdir -p eval_scores
for v in frozen uv notrans; do for sd in 42 43 44; do
  python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --joint_set j14 --split test --device cpu \
    --checkpoint checkpoints/sweep/${v}_s${sd}/actor_final.pt \
    --dump_scores eval_scores/${v}_s${sd}.json
done; done
# 2. aggregate, pairing each corrected score against its own clip-camera
python scripts/sweep_stats.py --table pampjpe --scores_dir eval_scores
```

The ± is the standard deviation over **three seeds** (42, 43, 44); see *Seed
sweep* below for the per-seed values and the paired test. **No (B) variant beats
doing nothing**, and the ordering is stable: every variant, at every seed, scores
slightly worse than the uncorrected lifted pose.

**How PA-MPJPE is calculated.** The predicted and the ground-truth pose are each
run through SMPL-X forward kinematics (`src/smplx_fk.joints_from_poses`) to give
3D joint positions in metres. Each frame is then Procrustes-aligned — rotation,
uniform scale and translation, solved by SVD (`src/evaluate.procrustes_align`) —
so global position and orientation are irrelevant, which is the point of the
metric. The score is the Euclidean distance between corresponding joints,
averaged over all frames and joints, reported in **millimetres**. Measured over
the **14-joint H3.6M convention** with **ground-truth betas on both sides**; both
choices move the number and are quantified under *Measurement convention*.

## Training diagnostics

All three conditions at **matched 128,000 PPO timesteps**, as **mean ± sd over
three seeds** (42, 43, 44) from `checkpoints/sweep/`. Rollout scalars are averaged
over the final 20,000 timesteps rather than read at a single point, because the
per-update value is noisy enough that one sample is not representative.

Within a seed all three variants see the identical clip sequence — the lifted
reprojection error agrees to four decimals across them — so variant comparisons
are exact. Across seeds the clips differ, which is what the ± measures.

| Metric | Unit | (A) | (B1) frozen | (B2) du,dv | (B3) pose-only | (C) |
|---|---|---|---|---|---|---|
| Reprojection error, corrected | px, lower better | 8.13 ± 0.68 | 8.63 ± 0.68 | 13.64 ± 3.52 | 8.62 ± 0.74 | |
| Reprojection error, lifted (same rollout) | px | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 | |
| Improvement over lifted, train rollout | px, higher better | 0 *(by definition)* | **-0.466 ± 0.050** | **-5.479 ± 3.046** | **-0.444 ± 0.026** | |
| Improvement over lifted, held-out val | px, higher better | 0 *(by definition)* | **-0.023 ± 0.027** | **+0.008 ± 0.085** | **-0.051 ± 0.043** | |
| Smoothness reward `exp(-a/sigma^2)` | dimensionless, (0, 1] | 0.9824 | 0.9253 ± 0.0018 | 0.9252 ± 0.0019 | 0.9252 ± 0.0018 | |
| Mean discriminator reward, test | dimensionless | — | — | — | — | |
| RMSE, corrected vs GT | normalised pose units | — | not measured | not measured | not measured | |

**Reproduce.** Rollout rows from the sweep event files; the held-out row from
the checkpoints, via the dump the second command writes; the (A) smoothness cell
by scanning the train split (`--h5_path ''` skips it).

```bash
python scripts/heldout_eval.py --n_clips 80 --n_frames 12 --clip_seed 42 \
    --dump eval_scores/heldout.json
python scripts/sweep_stats.py --table diagnostics --heldout eval_scores/heldout.json
```

> This table was rebuilt from the sweep on 2026-08-21;

**Reprojection error, corrected / lifted (px).** The pose is unnormalised with
the per-camera lifted statistics, its root rotated back into the camera frame
(`src/camera_frame.uncorrect_root`), passed through SMPL-X forward kinematics,
placed at the clip's metric translation, and projected through the real camera
intrinsics **with radial distortion** (`src/reproject.project`). The projected
joints are compared against the ViTPose 2D keypoints for that video frame, the
per-joint systematic offset is subtracted (see below), and the remaining
distances are averaged weighted by detector confidence. Reported in **pixels** on
the original 800 x 600 frame. "Corrected" scores the policy's output; "lifted"
scores the untouched SMPLer-X input on the same frames.

**Improvement over lifted (px).** `err_lifted - err_corrected`, computed on the
same frame against the same targets, so every clip-specific factor cancels.
Positive means the policy moved the pose closer to the 2D evidence. The
train-rollout version is measured on the stochastic policy during training; the
held-out version is measured on the **policy mean** over validation clips the
policy never trains on, so it carries no exploration noise.

**Smoothness reward (dimensionless).** `exp(-a/sigma^2)`, where `a` is the mean
squared **acceleration** of the corrected pose — the second finite difference
over three consecutive frames, in normalised pose units — with sigma = 0.5.
Acceleration rather than velocity: penalising velocity would reward a subject who
stops moving, which is wrong for a dataset of people walking and crawling.
Bounded to (0, 1]. (`a` is already a squared quantity; earlier versions of this
table wrote the formula as `exp(-a^2/sigma^2)`, which double-counts the squaring.
`src/rewards.smoothness_reward` returns `exp(-a/sigma^2)`.)

**(A) is the untouched lifted trajectory**, scored with the same function over the
whole train split — 1,891,664 frames, frame-weighted, per-clip spread 0.9822 ±
0.0136. It is a property of the data, not of a run: `GymMoviEnv._lifted_smoothness`
computes it per step and the env subtracts it from the reward, but it is never
logged as a scalar, which is why this cell was blank until now.

**The corrected pose is rougher than its own input — and it is all exploration
noise.** (A) 0.9824 against (B) 0.9252 looks like the policy adding jitter, but
the (B) rows are measured on the **stochastic** policy, which adds i.i.d.
`N(0, sigma^2)` noise to every frame independently. That noise has acceleration
`eps_t - 2*eps_{t-1} + eps_{t-2}`, of variance `6*sigma^2`, so a policy whose
*mean* were exactly the identity would still score
`exp(-(0.004439 + 6*0.0498^2)/0.25) = 0.9256`. Observed: 0.9253. The residual is
**-0.0003** — the learned mean adds no measurable roughness, and the smoothness
row says nothing about what the policy learned. Measuring that needs a
deterministic rollout, which `scripts/heldout_eval.py` does for reprojection but
does not currently track for smoothness.

**Mean discriminator reward (dimensionless).** `sigmoid(D(pose))`, the probability
the discriminator assigns to a single corrected pose being real GT motion.
Bounded to (0, 1). Condition (C) is **built but not yet run** — `src/gail_train.py`,
`src/gail_env.py`, `src/models/discriminator.py`, merged from `g34` at `c8c5617`.

One measurement to settle before running it: the discriminator's real samples are
`gt/poses`, normalised with **GT** statistics, while its fake samples are the
policy's output in **lifted per-camera** space. MoVi GT has no finger
articulation, so the 90 hand dimensions have sigma ~1e-7 and normalise to exactly
±1, which SMPLer-X's predicted fingers never do — the rule "are all 90 hand dims
±1?" separates real from fake at 100.0% on `data/processed_movi.h5`. Excluding
hands, the 66 body dimensions remain ~2.66 sigma apart from the stats mismatch
alone. Expect the discriminator to saturate immediately and `r_gail` to sit near a
constant 0, i.e. (C) reproducing (B).

**RMSE, corrected vs GT (normalised pose units).** Root-mean-square error between
the corrected and ground-truth pose vectors. This is a **diagnostic only** for
(B): it is the `reward_mode="gt"` supervised objective, deliberately disabled for
(B) and (C) so the reward stays ground-truth-free.

### Configuration

| | Unit | (B1) frozen | (B2) du,dv | (B3) pose-only |
|---|---|---|---|---|
| Observation width | dimensions | 362 | 362 | 356 |
| Action width | dimensions | 156 | 158 | 156 |
| PPO steps completed | environment timesteps | 128,000 | 128,000 | 128,000 |
| Pose action bound | normalised pose units | +/-0.3 | +/-0.3 | +/-0.3 |
| Image-shift bound (du, dv) | bbox heights | — | +/-0.1 | — |
| Reward width sigma | bbox heights | 0.0225 | 0.0225 | 0.0225 |

**Reproduce.** Widths and step counts are read back out of the checkpoints, so
they cannot drift from what actually ran. Bounds and sigma come from the same
configs (`trans_mode` -> `src.models.policy.action_bounds`).

```bash
python scripts/sweep_stats.py --table config
```

The observation is the lifted pose, the previous corrected pose, and a 44-dim
block of 2D evidence (per-joint reprojection residual, confidences, scalar error,
episode progress, camera one-hot, bbox geometry). (B3) is 6 dims narrower because
it drops the translation from the state; the translation is kept inside the
environment and used only by the reward.

**One earlier (B3) run went further.** Before the sweep, a single-seed (B3) run in
`checkpoints/biasfix/notrans/` reached 288,000 timesteps — where it logged
**+0.064 px** held-out improvement and 8.22 px corrected error, the strongest
(B3) figure on record. It is not in the table above: it is one seed, its
diagnostics do not reproduce (see the note under *Training diagnostics*), and the
three-seed sweep at 128,000 puts (B3) at **-0.051 ± 0.043 px** held-out, negative
at every seed. Whether the gain is real or is that run's seed is untested —
running the sweep out to 288,000 would settle it, and is the one experiment that
could still rescue (B3).

---

## Seed sweep

Nine runs — three variants x seeds 42/43/44 — each 40 updates x 3,200 = **128,000
timesteps**, in `checkpoints/sweep/`. This is the sweep the previous version of
this document said (B3) needed before its advantage could be claimed.

**PA-MPJPE (mm), test split, 374 clip-cameras, j14, GT betas.** Each corrected
score is paired against the lifted score on the *same* clip-camera, so the delta
carries no aggregation difference.

| variant | s42 | s43 | s44 | mean ± sd | Δ vs lifted | clip-cameras improved |
|---|---|---|---|---|---|---|
| (A) lifted baseline | 34.396 | 34.396 | 34.396 | **34.396** | 0 *(by definition)* | — |
| (B1) frozen | 34.495 | 34.477 | 34.497 | 34.490 ± 0.011 | +0.094 ± 0.011 | 37.7% |
| (B2) du,dv | 34.522 | 34.527 | 34.500 | 34.517 ± 0.015 | +0.121 ± 0.015 | 37.7% |
| (B3) pose-only | 34.430 | 34.548 | 34.565 | 34.514 ± 0.074 | +0.118 ± 0.074 | 34.0% |

**Reproduce.**
```bash
# 1. score every checkpoint on the test split (~13 min each, parallelisable)
mkdir -p eval_scores
for v in frozen uv notrans; do for sd in 42 43 44; do
  python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --joint_set j14 --split test --device cpu \
    --checkpoint checkpoints/sweep/${v}_s${sd}/actor_final.pt \
    --dump_scores eval_scores/${v}_s${sd}.json
done; done
# 2. aggregate, pairing each corrected score against its own clip-camera
python scripts/sweep_stats.py --table pampjpe --scores_dir eval_scores
```

The paired t-statistics below are printed by the same command.

**The degradation is significant, not noise.** Paired over all 1,122 clip-camera
deltas per variant (3 seeds x 374), a one-sample t-test against zero gives
**t = +9.1 (B1), +9.2 (B2), +11.1 (B3)** — all far outside sampling noise, all in
the *worse* direction. Roughly a third of clip-cameras improve and two thirds
degrade, in every variant.

**(B3)'s apparent advantage was seed 42.** At seed 42 it is the best run in the
sweep (34.430, Δ +0.034); at seeds 43 and 44 it is the worst (34.548, 34.565).
Its across-seed spread is **7x** (B1)'s — 0.074 vs 0.011 mm — and its mean is
indistinguishable from (B1)'s. Dropping the translation from the state does not
help; it mainly makes the run less reproducible.

**Training-rollout reprojection**, mean over the final 20,000 timesteps:

| variant | error corrected (px) | improvement over lifted (px) | s42 / s43 / s44 |
|---|---|---|---|
| (B1) frozen | 8.63 ± 0.68 | **-0.466 ± 0.050** | -0.408 / -0.489 / -0.501 |
| (B2) du,dv | 13.64 ± 3.52 | **-5.479 ± 3.046** | -5.153 / -2.609 / -8.674 |
| (B3) pose-only | 8.62 ± 0.74 | **-0.444 ± 0.026** | -0.468 / -0.417 / -0.446 |

**Reproduce.** Reads the nine event files directly; `--window 0` would point-read
the final step instead of averaging, which is noisier per update.

```bash
python scripts/sweep_stats.py --table rollout --window 20000
```

The per-seed lifted errors quoted below are printed by the same command.

Improvement is negative for every variant at every seed, consistent with the
PA-MPJPE result. (B2) is both far worse and wildly seed-dependent — its spread
(±3.0 px) is six times (B1)'s entire deficit — which is the clearest argument yet
against giving the policy an image-plane shift.

**Seeds see different clips; variants within a seed do not.** The lifted
reprojection error is identical to four decimals across the three variants at a
given seed (8.6394 s42 / 7.3602 s43 / 8.3901 s44) and differs between seeds. So
variant comparisons within a seed are exact, and the across-seed spread above is
genuine seed variance rather than a clip-sampling artefact.

**Held-out validation improvement (px), higher is better.** All nine runs were
launched with `viz_interval = 0`, so no `pose/*` scalars were written during
training. The metric does not need them: it is a pure function of the final actor
weights and the validation clips (`ImagePoseVizLogger._rollout` just calls
`rollout_policy` with the actor), so it was recovered by rolling the nine saved
`actor_final.pt` over the val split with the **policy mean**, through
`correction_magnitude()` — the same code path that logged the `biasfix` figures.
Only the endpoint is recoverable this way, not the curve, but the endpoint at
matched steps is what the table reads.

Measured over **80 clips x 12 frames** at a fixed clip seed, so all nine
checkpoints are scored on identical clips — the lifted error is 12.0454 px in
every one of them, which confirms it. That is roughly 20x the sample behind the
`biasfix` figures, which used `viz_clips = 3`, `viz_frames = 4`.

| variant | s42 | s43 | s44 | mean ± sd | sign |
|---|---|---|---|---|---|
| (B1) frozen | -0.052 | +0.002 | -0.018 | **-0.023 ± 0.027** | - + - |
| (B2) du,dv | -0.015 | -0.063 | +0.103 | **+0.008 ± 0.085** | - - + |
| (B3) pose-only | -0.048 | -0.010 | -0.095 | **-0.051 ± 0.043** | - - - |

**Reproduce.** No retraining — this rolls each saved actor over the val split.
About a minute per checkpoint. The script prints the lifted error per run and
warns if it is not constant, which would mean the runs saw different clips and
the comparison is void.

```bash
python scripts/heldout_eval.py --n_clips 80 --n_frames 12 --clip_seed 42
```

**Held-out improvement is indistinguishable from zero for every variant.** For
(B1) and (B2) the sign is not even stable across seeds, and the across-seed
spread exceeds the mean in both (|mean|/sd = 0.83 and 0.10). (B2)'s best run
(+0.103 at s44) and its worst (-0.063 at s43) differ by more than the entire
effect anyone is trying to measure. (B3) is the only variant with a consistent
sign, and it is **negative at all three seeds** — which is what finally settles
the +0.016 px figure: measured over a 20x larger sample and three seeds instead
of one, (B3) does not improve the held-out image fit.

> These are **raw** pixel errors — `correction_magnitude()` computes distances
> directly and applies no keypoint-bias subtraction, unlike the training-rollout
> reprojection rows above. That is why the level reads ~12 px here against ~8-9 px
> there. The bias is common to the lifted and corrected sides, so the improvement
> column is the comparable one; the levels are not.

---

## Measurement convention

**Joint set changes the headline by a third.** SMPL-X's 22 body joints include
the pelvis, three spine joints, two collars, neck and head — all close to the
body axis and nearly free to match once Procrustes has aligned the pose. Same
poses, same alignment, on the lifted baseline:

| Joint set | PA-MPJPE (mm) |
|---|---|
| all 22 SMPL-X body joints | 30.2 |
| 14-joint H3.6M convention *(reported above)* | 34.1 |
| limbs only (knees, ankles, elbows, wrists) | 40.7 |

**Reproduce.**
```bash
for js in all22 j14 limbs; do
  python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --split test --device cpu --joint_set $js
done
```

> Note the 34.1 mm here is the 25-clip subset, not the 34.40 mm full-split
> baseline reported above — see the last paragraph of this section. Re-running
> the command replaces all three with full-split numbers.

The **30.0 mm** figure in `summary-of-changes.md` is the all-22 number and is not
comparable to published PA-MPJPE.

**Ground-truth betas are given to the prediction**, so bone lengths match exactly
and the metric reflects joint angles alone. That is correct for comparing
correction policies — the policy only moves joint angles, and pose cannot alter
bone lengths in SMPL-X — but it is **not** the benchmark setting, which uses the
predicted shape. `src/evaluate.py --betas lifted` measures that; it has not been
run.

**Aggregation is not a confound — measured.** (A) averages the two cameras per
clip, then over 187 clips; (B) averages over all 374 clip-cameras. Computing the
lifted baseline both ways gives **34.396 mm either way** (agreeing to three
decimals), because every test clip has both cameras and so the two averages carry
identical weights. The *Seed sweep* deltas are additionally paired per
clip-camera, so they are unaffected regardless.

**(A) is the full test split**: 34.40 mm over 187 clips. An earlier 34.1 mm came
from a 25-clip subset.

---

## Reading these numbers

**No (B) variant beats doing nothing.** 34.49 / 34.52 / 34.51 mm against the
34.40 mm baseline — 0.09 to 0.12 mm *worse*, or 0.3% relative. The seed sweep
shows this is not baseline noise: paired per clip-camera over three seeds it is
significant at t = +9 to +11, and only about a third of clip-cameras improve. The
effect is small but it is real, and it points the wrong way. The next paragraph
is why.

**The reprojection reward's optimum is displaced from ground truth.** On held-out
validation the GT pose scores *worse* on reprojection than the lifted pose —
**9.38 px vs 7.21 px** — because SMPLer-X and ViTPose were both fit to the same
image and share correlated error, while ground truth is independent of it.
Maximising this reward therefore moves the pose *away* from GT, which is why (B1)
lands slightly above the (A) baseline in PA-MPJPE rather than below it. That is a
property of the objective, not a training failure, and it is the argument for (C).

**Reprojection rows carry a fitted bias correction.** All reprojection figures
subtract a systematic per-joint COCO-to-SMPL-X offset (`scripts/fit_kp_bias.py`,
fitted on the train split only), which accounts for ~85% of the raw error —
0.0262 of 0.0309 bbox heights. Without it the same runs read ~14 px instead of
~7 px. The row measures **agreement with ViTPose**, not reconstruction accuracy.

**Held-out improvement is the honest (B) headline, not reprojection error.**
Reprojection error alone rewards agreement with the detector; improvement over
the lifted input is the only column that says whether the policy did anything.
The earlier single-seed reading of that row put (B3) at +0.016 px — the one
positive number anywhere in (B), and the basis for preferring it.

**That (B3) result did not survive the seed sweep.** It was measured at seed 42,
which the sweep now shows is (B3)'s lucky seed: seed 42 gives its best PA-MPJPE
(34.430 mm) and seeds 43/44 its worst two (34.548, 34.565). Across three seeds
(B3) is 34.514 ± 0.074 mm — no better than (B1)'s 34.490 ± 0.011, with seven
times the spread — and its training-rollout improvement is negative at all three
seeds (-0.444 ± 0.026 px).

**And the held-out figure does not reproduce either.** Rolling the nine
checkpoints over the val split — 80 clips x 12 frames, roughly 20x the sample the
+0.016 px came from — puts (B3) at **-0.051 ± 0.043 px, negative at all three
seeds**. (B1) and (B2) are likewise indistinguishable from zero, and their sign
is not even stable across seeds. So all three lines of evidence agree: PA-MPJPE,
training-rollout improvement, and held-out improvement. **The (B3) advantage is
withdrawn.**

**Lifter identity.** The experiment description names MotionBERT for condition
(A), but nothing in this repository references it; the lifted poses come from
SMPLer-X throughout. Column (A) reports SMPLer-X.
