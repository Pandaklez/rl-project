# Results

All figures on the **test** split (187 clips x 2 cameras = 374 clip-cameras),
except the reprojection rows, which are training-rollout and held-out-validation
diagnostics read from TensorBoard.

> **Dataset change, 2026-08-21.** The 30 SMPL-X finger joints were removed: the
> pose vector is now 22 joints / 66 dims. In GT they were a constant that
> normalised to exactly ±1 while SMPLer-X's genuine finger predictions never did,
> making 90 of 156 dimensions a perfect "GT or lifted" label; they also carried
> 91.6% of the smoothness reward's acceleration energy while feeding no other
> reward term and no metric. See `summary-of-changes.md`. **Experiment (A) is
> unaffected** — verified bit-identical at 0.03439604253931479, because finger
> pose cannot move a body joint. **Everything under (B) has been re-run and every
> table below is 22-joint** — checkpoints in `checkpoints/sweep/`, test-split
> dumps in `eval_scores/`. The superseded 52-joint sweep is kept for comparison
> in `checkpoints/sweep_52joint/` and `eval_scores_52joint/`; where the two
> disagree in a way that matters, this document says so.

## Primary metric

| | (A) SMPLer-X, no correction | (B1) PPO, trans frozen | (B2) PPO, du,dv image shift | (B3) PPO, pose-only state | (C) PPO + GAIL |
|---|---|---|---|---|---|
| **PA-MPJPE, test (mm), lower is better** | **34.40** | 34.51 ± 0.03 | 34.50 ± 0.06 | 34.48 ± 0.06 | |
| **Δ vs (A), same clip-cameras (mm)** | 0 *(by definition)* | +0.119 ± 0.030 | +0.101 ± 0.057 | +0.084 ± 0.058 | |

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
| Reprojection error, corrected | px, lower better | 8.13 ± 0.68 | 8.56 ± 0.70 | 11.74 ± 1.29 | 8.56 ± 0.71 | |
| Reprojection error, lifted (same rollout) | px | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 | |
| Improvement over lifted, train rollout | px, higher better | 0 *(by definition)* | **-0.384 ± 0.026** | **-3.561 ± 0.786** | **-0.386 ± 0.039** | |
| Improvement over lifted, held-out val | px, higher better | 0 *(by definition)* | **+0.022 ± 0.066** | **-0.007 ± 0.056** | **+0.004 ± 0.085** | |
| Smoothness reward `exp(-a/sigma^2)` | dimensionless, (0, 1] | 0.9949 | 0.9371 ± 0.0012 | 0.9369 ± 0.0012 | 0.9370 ± 0.0012 | |
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
whole train split — 1,891,664 frames over 2,781 clip-cameras, frame-weighted,
per-clip spread 0.9952 ± 0.0078. It is a property of the data, not of a run: the
22-joint value is much closer to 1 than the 0.9824 the 52-joint data gave,
because the finger dimensions carried 91.6% of the acceleration energy and are
now gone. `GymMoviEnv._lifted_smoothness`
computes it per step and the env subtracts it from the reward, but it is never
logged as a scalar, which is why this cell was blank until now.

**The corrected pose is rougher than its own input — and it is all exploration
noise.** (A) 0.9949 against (B) 0.9371 looks like the policy adding jitter, but
the (B) rows are measured on the **stochastic** policy, which adds i.i.d.
`N(0, sigma^2)` noise to every frame independently. That noise has acceleration
`eps_t - 2*eps_{t-1} + eps_{t-2}`, of variance `6*sigma^2`, so a policy whose
*mean* were exactly the identity would still score
`exp(-(0.002362 + 6*0.0498^2)/0.25) = 0.9334` — where `0.002362` is the lifted
mean squared acceleration on the train split and `sigma = 0.0498` is read from
`log_std` in the saved actors. Observed: 0.9371, a residual of **+0.0037**.

That residual is the size of the approximation, not of an effect: the prediction
exponentiates the mean acceleration while the table averages the per-frame
exponential, and on the lifted data that same Jensen gap is +0.0043
(`exp(-0.002362/0.25) = 0.9906` against the table's 0.9949). So the learned mean
adds no measurable roughness, and the smoothness row says nothing about what the
policy learned. Measuring that needs a deterministic rollout, which
`scripts/heldout_eval.py` does for reprojection but does not currently track for
smoothness.

**Mean discriminator reward (dimensionless).** `sigmoid(D(pose))`, the probability
the discriminator assigns to a single corrected pose being real GT motion.
Bounded to (0, 1). Condition (C) is **built but not yet run** — `src/gail_train.py`,
`src/gail_env.py`, `src/models/discriminator.py`, merged from `g34` at `c8c5617`.

The discriminator's real samples are `gt/poses`, normalised with **GT**
statistics, while its fake samples are the policy's output in **lifted
per-camera** space, so a stats mismatch alone can label them. On the 52-joint
data this was fatal: MoVi GT has no finger articulation, so the 90 hand
dimensions had sigma ~1e-7 and normalised to exactly ±1 where SMPLer-X's
predicted fingers never did, and the rule "are all 90 hand dims ±1?" separated
real from fake at 100.0%.

**The finger removal closed that leak.** Re-measured on the current 22-joint
`data/processed_movi.h5`, no dimension saturates — at most 0.01% of real frames
sit at exactly ±1 in any single dim. Per-dimension separation
`d' = |mu_r - mu_f| / sqrt((sd_r^2 + sd_f^2)/2)` averages **0.005** over the 66
dims and never exceeds 0.02, and the best *linear* rule — which bounds the whole
family, not one dimension at a time — reaches `d' = 0.06`, or **51.2%**
accuracy against 50% chance. The "discriminator saturates immediately and
`r_gail` sits at a constant" prediction is **withdrawn**: on this data there is
no trivial rule left to find.

**But the two spaces still do not mean the same thing, and standardisation
hides it.** Both sides are stored already-normalised — GT with the GT stats,
lifted with the per-camera stats — so each has mean 0 and sd 1 in its own space
and the marginals match by construction. That is why `d'` is ~0. It is not
evidence the spaces agree: the same coordinate denotes a *different physical
pose* on each side, because the two affine maps differ (per-joint sd ratios run
0.12 to 3.22, median 0.68). A latent that is realistic in GT space denotes
`mu_gt + sd_gt * z`, while the policy emitting that latent in lifted space
produces `mu_lifted + sd_lifted * z`. **If the policy satisfied this
discriminator perfectly it would be driven to a pose wrong by 0.088 rad RMS per
axis-angle component — 5.1°, against a GT spread of 12.6° — and by 31° on the
root.** So the risk for (C) is not a saturated discriminator any more; it is a
well-behaved discriminator rewarding the wrong pose. Put both sides in one
space before running (C).

**Reproduce.**
```bash
python scripts/disc_separability.py --physical
```

> Both poses in `data/processed_movi.h5` are stored **already normalised**.
> Re-applying `data/normalization*.json` to them normalises twice and
> manufactures a mismatch the discriminator never sees — an earlier version of
> this section did exactly that and reported a spurious root-dimension
> separation of `d' = 1.50`.

**RMSE, corrected vs GT (normalised pose units).** Root-mean-square error between
the corrected and ground-truth pose vectors. This is a **diagnostic only** for
(B): it is the `reward_mode="gt"` supervised objective, deliberately disabled for
(B) and (C) so the reward stays ground-truth-free.

### Configuration

| | Unit | (B1) frozen | (B2) du,dv | (B3) pose-only |
|---|---|---|---|---|
| Observation width | dimensions | 182 | 182 | 176 |
| Action width | dimensions | 66 | 68 | 66 |
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
environment and used only by the reward. Widths are 22-joint: they were 362 /
362 / 356 and 156 / 158 / 156 before the finger removal.

**One earlier (B3) run went further.** Before the sweep, a single-seed (B3) run in
`checkpoints/biasfix/notrans/` reached 288,000 timesteps — where it logged
**+0.064 px** held-out improvement and 8.22 px corrected error, the strongest
(B3) figure on record. It is not in the table above: it is one seed, its
diagnostics do not reproduce (see the note under *Training diagnostics*), and the
three-seed sweep at 128,000 puts (B3) at **+0.004 ± 0.085 px** held-out — a mean
twenty times smaller than its own seed spread, with the sign splitting 2-1. That
is weaker evidence against the run than the 52-joint sweep gave (-0.051 ± 0.043,
negative at every seed), but it is not evidence for it either: the column simply
cannot resolve an effect this size at three seeds. Whether the gain is real or is
that run's seed is untested — running the sweep out to 288,000 would settle it,
and is the one experiment that could still rescue (B3).

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
| (B1) frozen | 34.549 | 34.504 | 34.492 | 34.515 ± 0.030 | +0.119 ± 0.030 | 38.2% |
| (B2) du,dv | 34.476 | 34.453 | 34.561 | 34.497 ± 0.057 | +0.101 ± 0.057 | 40.7% |
| (B3) pose-only | 34.416 | 34.529 | 34.494 | 34.480 ± 0.058 | +0.084 ± 0.058 | 40.1% |

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
**t = +9.2 (B1), +8.7 (B2), +7.7 (B3)** — all far outside sampling noise, all in
the *worse* direction. About 40% of clip-cameras improve and 60% degrade, in
every variant.

**The variant ordering is not stable, which is the finding.** On this data (B3)
has the lowest mean (34.480) and (B1) the highest (34.515). On the 52-joint
sweep the order was the reverse — (B1) 34.490 best, (B2) 34.517 worst, (B3)
34.514 — so the same nine seeds, changed only by dropping dimensions that fed no
reward, reshuffle the ranking end to end. The spread between variant means here
is 0.035 mm against per-variant across-seed sds of 0.030-0.058 mm, so the
ranking is inside its own noise and no variant is distinguishable from another.
What survives both sweeps is the sign: **every variant, at every seed, in both
datasets, is worse than doing nothing.**

**(B3)'s apparent advantage was seed 42.** Seed 42 is still (B3)'s best run
(34.416, the best single number anywhere in the sweep), and it was the seed the
original single-run claim was measured at — but at 43 and 44 it gives back 0.113
and 0.078 mm. Its across-seed spread is now comparable to the other variants'
rather than 7x (B1)'s as on the 52-joint data, which weakens the earlier
"(B3) is merely less reproducible" reading: the honest statement is that (B3) is
not separable from (B1), in either direction, at three seeds.

**Training-rollout reprojection**, mean over the final 20,000 timesteps:

| variant | error corrected (px) | improvement over lifted (px) | s42 / s43 / s44 |
|---|---|---|---|
| (B1) frozen | 8.56 ± 0.70 | **-0.384 ± 0.026** | -0.384 / -0.410 / -0.359 |
| (B2) du,dv | 11.74 ± 1.29 | **-3.561 ± 0.786** | -4.461 / -3.212 / -3.009 |
| (B3) pose-only | 8.56 ± 0.71 | **-0.386 ± 0.039** | -0.361 / -0.366 / -0.431 |

**Reproduce.** Reads the nine event files directly; `--window 0` would point-read
the final step instead of averaging, which is noisier per update.

```bash
python scripts/sweep_stats.py --table rollout --window 20000
```

The per-seed lifted errors quoted below are printed by the same command.

Improvement is negative for every variant at every seed. (B2) is still far worse
than the other two, but the finger removal helped it more than anything else in
this report: its deficit fell from -5.479 to -3.561 px and its across-seed spread
collapsed from ±3.046 to ±0.786, a factor of four. That fits the mechanism —
dropping 90 action dimensions that fed no reward term but did feed exploration
noise makes the policy's random walk much less damaging. (B1) and (B3) improved
too (-0.466 → -0.384 and -0.444 → -0.386) and remain indistinguishable from each
other.

**Seeds see different clips; variants within a seed do not.** The lifted
reprojection error is identical to four decimals across the three variants at a
given seed (8.6394 s42 / 7.3602 s43 / 8.3901 s44) and differs between seeds. So
variant comparisons within a seed are exact, and the across-seed spread above is
genuine seed variance rather than a clip-sampling artefact.

**Held-out validation improvement (px), higher is better.** The 22-joint runs
*were* launched with pose logging on — the saved configs record
`viz_interval = 5`, `viz_mode = image`, and each event file carries eight
`pose/img_improvement_px` points — unlike the 52-joint sweep, which ran at
`viz_interval = 0` and logged none. The table does not read those scalars
anyway. They are computed at `viz_clips = 3`, `viz_frames = 4`, which is 36
frames per point and far too few to resolve an effect of this size; the logged
final values scatter from -0.087 to +0.147 px, an order of magnitude wider than
the column below.

The table instead rolls each saved `actor_final.pt` over the val split with the
**policy mean**, through `correction_magnitude()` — a pure function of the final
weights and the validation clips (`ImagePoseVizLogger._rollout` just calls
`rollout_policy` with the actor), so it is the same code path, at a usable
sample size. Only the endpoint is measurable this way, not the curve, but the
endpoint at matched steps is what the table reads.

Measured over **80 clips x 12 frames** at a fixed clip seed, so all nine
checkpoints are scored on identical clips — the lifted error is 12.0454 px in
every one of them, which confirms it. That is roughly 20x the sample behind the
in-training scalars.

> Note this means `scripts/run_seed_sweep.sh` as committed (`--viz_interval 0`)
> is **not** how the 22-joint checkpoints were produced; they were launched with
> logging on, per the project's training-run convention. The flag does not change
> which data a run sees: the per-seed lifted rollout error is identical to four
> decimals across both sweeps (8.6394 s42 / 7.3602 s43 / 8.3901 s44) despite the
> different setting. Whether it leaves the weight updates bit-identical has not
> been checked. Reconcile the script before using it to reproduce this sweep.

| variant | s42 | s43 | s44 | mean ± sd | sign |
|---|---|---|---|---|---|
| (B1) frozen | -0.048 | +0.031 | +0.083 | **+0.022 ± 0.066** | - + + |
| (B2) du,dv | +0.049 | -0.064 | -0.004 | **-0.007 ± 0.056** | + - - |
| (B3) pose-only | -0.094 | +0.053 | +0.053 | **+0.004 ± 0.085** | - + + |

**Reproduce.** No retraining — this rolls each saved actor over the val split.
About a minute per checkpoint. The script prints the lifted error per run and
warns if it is not constant, which would mean the runs saw different clips and
the comparison is void.

```bash
python scripts/heldout_eval.py --n_clips 80 --n_frames 12 --clip_seed 42
```

**Held-out improvement is indistinguishable from zero for every variant, and now
no variant even holds its sign.** Each of the three splits 2-1 across seeds, and
the across-seed spread swamps the mean in all of them — |mean|/sd is 0.33 (B1),
0.13 (B2), 0.05 (B3). (B3)'s best and worst seeds differ by 0.147 px, roughly ten
times the +0.016 px effect this column was once claimed to show.

This is a *weaker* result than the 52-joint sweep gave, not a stronger one. There,
(B3) was negative at all three seeds (-0.051 ± 0.043); here it is +0.004 ± 0.085.
The finger removal moved every variant toward zero and widened the seed spread,
so what was a consistent small negative is now noise centred on nothing. The
honest reading is that this column cannot resolve an effect of the size being
looked for at three seeds — not that (B) improves the held-out image fit.

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

**No (B) variant beats doing nothing.** 34.51 / 34.50 / 34.48 mm against the
34.40 mm baseline — 0.08 to 0.12 mm *worse*, or 0.3% relative. The seed sweep
shows this is not baseline noise: paired per clip-camera over three seeds it is
significant at t = +7.7 to +9.2, and only about 40% of clip-cameras improve. The
effect is small but it is real, and it points the wrong way. It also survived the
finger removal unchanged in sign and magnitude, on independently retrained
weights — which is the strongest evidence in this report that it is a property of
the objective rather than of a particular run. The next paragraph is why.

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
which the sweep shows is (B3)'s lucky seed: seed 42 gives its best PA-MPJPE
(34.416 mm) and seeds 43/44 give back 0.113 and 0.078 mm. Across three seeds
(B3) is 34.480 ± 0.058 mm. That is nominally the lowest variant mean, but the
gap to (B1)'s 34.515 ± 0.030 is smaller than either variant's own seed spread,
and the ordering reverses between the 52-joint and 22-joint sweeps — so it is a
ranking inside the noise, not a result. Its training-rollout improvement is
negative at all three seeds (-0.386 ± 0.039 px).

**And the held-out figure does not reproduce either.** Rolling the nine
checkpoints over the val split — 80 clips x 12 frames, roughly 20x the sample the
+0.016 px came from — puts (B3) at **+0.004 ± 0.085 px**, with the sign splitting
2-1 across seeds. It is not negative, but it is not the +0.016 px either: the
spread is twenty times the mean, so the column resolves nothing at this sample
size. Training-rollout improvement remains negative at every seed
(-0.386 ± 0.039 px). **The (B3) advantage is withdrawn** — not because the
held-out column refutes it, but because nothing reproduces it and the one
measurement that is stable across seeds points the other way.

**Lifter identity.** The experiment description names MotionBERT for condition
(A), but nothing in this repository references it; the lifted poses come from
SMPLer-X throughout. Column (A) reports SMPLer-X.
