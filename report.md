# Results

All figures on the **test** split (187 clips x 2 cameras = 374 clip-cameras),
except the training-rollout reprojection rows, which are read from TensorBoard.
The held-out reprojection row is on the same test split as PA-MPJPE, and so is
the GT-vs-lifted reprojection comparison that motivates (C).

> **`val` is not used as an evaluation split anywhere in this report.** Nothing
> was ever selected on it -- no early stopping, no checkpoint choice, no
> hyperparameter search -- so scoring on it only cost comparability with the
> PA-MPJPE table. Every reported reprojection figure is now on `test`.
>
> One trap survives this move and is worth stating once: the **training-time**
> logger still scores `val`. `src/train.py:380` and `src/gail_train.py:726` both
> build their `ImagePoseVizLogger` with `split="val"`, so the
> `pose/img_improvement_px` scalar in every run's TensorBoard file is a **val**
> number. Its lifted error reads 11.68 px against the test split's 12.14 px,
> which is the quickest way to tell the two apart. No table below reads that
> scalar -- every held-out row rolls the saved actor over `test` via
> `scripts/heldout_eval.py` -- but numbers quoted from TensorBoard in the prose
> are flagged where they appear, and are not comparable to the tables.

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

> **Condition (C) ran on 2026-08-24.** Six runs in `checkpoints/gail_c/` — two
> discriminator conditions x seeds 42/43/44 — with test-split dumps in
> `eval_scores/gail_feet_*.json` and `eval_scores/heldout_test_gail_c.json`.
> **No (B) number in this document changed**: (C) is a separate sweep against the
> same data, split, clips and seeds, and every (B) table still reproduces
> unchanged from the commands printed beneath it. Two *pre-run* claims about (C)
> did change, and are marked where they appear — the "no trivial rule left to
> find" separability reading is **reinstated as a saturation prediction** (it was
> measured in the wrong space), and the discriminator reward is `amp_reward`, not
> `sigmoid(D)`. See *(C) PPO + GAIL*.

## Primary metric

| | (A) SMPLer-X, no correction | (B1) PPO, trans frozen | (B2) PPO, du,dv image shift | (B3) PPO, pose-only state | (C) PPO + GAIL, feet in | (C) PPO + GAIL, feet out |
|---|---|---|---|---|---|---|
| **PA-MPJPE, test (mm), lower is better** | **34.40** | 34.51 ± 0.03 | 34.50 ± 0.06 | 34.48 ± 0.06 | 34.45 ± 0.06 | 34.53 ± 0.03 |
| **Δ vs (A), same clip-cameras (mm)** | 0 *(by definition)* | +0.119 ± 0.030 | +0.101 ± 0.057 | +0.084 ± 0.058 | +0.050 ± 0.055 | +0.138 ± 0.026 |

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
# 1b. the same for (C), whose checkpoints live under checkpoints/gail_c. The
#     dumps must be named gail_<cond>_s<seed>.json -- that is what
#     sweep_stats.py's GAIL_VARIANTS looks for. scripts/run_eval_c_light.sh
#     runs exactly this, core-pinned and re-runnable.
for c in feet_in feet_out; do for sd in 42 43 44; do
  python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --joint_set j14 --split test --device cpu \
    --checkpoint checkpoints/gail_c/${c}_s${sd}/actor_final.pt \
    --dump_scores eval_scores/gail_${c}_s${sd}.json
done; done
# 2. aggregate, pairing each corrected score against its own clip-camera
python scripts/sweep_stats.py --table pampjpe --scores_dir eval_scores
```

The ± is the standard deviation over **three seeds** (42, 43, 44); see *Seed
sweep* below for the per-seed values and the paired test. **No variant beats
doing nothing — (C) included**, and the ordering is stable: every variant, at
every seed, scores slightly worse than the uncorrected lifted pose.

**(C) does not change that verdict, but it does move the number.** Adding the
discriminator to the (B1) recipe takes the gap to (A) from +0.119 to
**+0.050 mm**, the smallest degradation in this report, and lifts the fraction of
clip-cameras that improve from 38.2% to 45.5% — while remaining significantly
worse than doing nothing (paired t = +4.3 over 1,122 clip-camera deltas).
**Hiding the feet from the discriminator reverses that gain and then some**:
+0.138 mm, the *largest* degradation here, at t = +11.3. The two (C) arms differ
from each other by 0.088 mm paired per clip-camera (t = -5.5, feet-in better at
all three seeds), so the feet ablation is the one comparison in this report that
is cleanly resolved at three seeds — and it comes out opposite to its motivation.
See *(C) PPO + GAIL* below.

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

| Metric | Unit | (A) | (B1) frozen | (B2) du,dv | (B3) pose-only | (C) GAIL, feet in | (C) GAIL, feet out |
|---|---|---|---|---|---|---|---|
| Reprojection error, corrected | px, lower better | 8.13 ± 0.68 | 8.56 ± 0.70 | 11.74 ± 1.29 | 8.56 ± 0.71 | 8.52 ± 0.70 | 8.52 ± 0.69 |
| Reprojection error, lifted (same rollout) | px | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 | 8.13 ± 0.68 |
| Improvement over lifted, train rollout | px, higher better | 0 *(by definition)* | **-0.384 ± 0.026** | **-3.561 ± 0.786** | **-0.386 ± 0.039** | **-0.340 ± 0.023** | **-0.360 ± 0.029** |
| Improvement over lifted, held-out test | px, higher better | 0 *(by definition)* | **+0.002 ± 0.067** | **-0.003 ± 0.067** | **-0.005 ± 0.090** | **+0.049 ± 0.052** | **-0.031 ± 0.072** |
| Smoothness reward `exp(-a/sigma^2)` | dimensionless, (0, 1] | 0.9949 | 0.9371 ± 0.0012 | 0.9369 ± 0.0012 | 0.9370 ± 0.0012 | 0.9370 ± 0.0012 | 0.9369 ± 0.0012 |
| Discriminator reward `amp_reward(D)`, train rollout | dimensionless, [0, 1] | — | — | — | — | 0.5662 ± 0.0138 | 0.6494 ± 0.0142 |
| Discriminator reward, improvement over lifted | dimensionless | — | — | — | — | +0.0049 ± 0.0004 | +0.0033 ± 0.0009 |
| Discriminator term as fed to PPO (scaled) | reward units | — | — | — | — | +0.0346 ± 0.0035 | +0.0274 ± 0.0072 |
| Discriminator accuracy, real | fraction | — | — | — | — | 0.9938 ± 0.0018 | 0.9675 ± 0.0049 |
| Discriminator accuracy, fake | fraction | — | — | — | — | 0.9841 ± 0.0099 | 0.9674 ± 0.0087 |
| Discriminator accuracy, probe (unseen GT) | fraction | — | — | — | — | 0.9963 ± 0.0005 | 0.9616 ± 0.0061 |
| Memorisation gap (bank - probe) | fraction | — | — | — | — | -0.0025 ± 0.0014 | +0.0059 ± 0.0014 |
| RMSE, corrected vs GT | normalised pose units | — | not measured | not measured | not measured | not measured | not measured |

**Reproduce.** Rollout rows from the sweep event files; the held-out row from
the checkpoints, via the dump the second command writes; the (A) smoothness cell
by scanning the train split (`--h5_path ''` skips it). `--gail_dir` adds the two
(C) columns and the discriminator rows; without it the table is (B)-only and the
trailing (C) column is left blank, as it was before (C) ran.

```bash
python scripts/heldout_eval.py --split test --n_clips 400 --n_frames 12 \
    --clip_seed 42 --dump eval_scores/heldout_test.json
python scripts/heldout_eval.py --sweep_dir checkpoints/gail_c --split test \
    --n_clips 400 --n_frames 12 --clip_seed 42 \
    --dump eval_scores/heldout_test_gail_c.json
python scripts/sweep_stats.py --table diagnostics \
    --heldout eval_scores/heldout_test.json \
    --gail_dir checkpoints/gail_c \
    --heldout_gail eval_scores/heldout_test_gail_c.json
```

> The five discriminator rows exist only in the (C) event files, so they read
> "—" for (A)/(B) — those runs have no discriminator, not a discriminator
> scoring zero. The reward row is named for `amp_reward`, the least-squares
> score `max(0, 1 - 0.25*(D - 1)^2)` the code actually uses; **the TensorBoard
> tag is still called `Reward / GAIL raw sigmoid(D)` and is misnamed.** The
> reward stopped being a sigmoid when the discriminator moved to AMP's
> least-squares objective (`src/models/discriminator.py::amp_reward`); the tag
> string was not renamed, because doing so would have orphaned the scalar in the
> event files these tables read.

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
held-out version is measured on the **policy mean** over test clips the policy
never trains on, so it carries no exploration noise.

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

**Discriminator reward (dimensionless).** `amp_reward(D(pose)) = max(0, 1 - 0.25*(D - 1)^2)`,
AMP eq. 5 (Peng et al. 2021) — 1 when the pose is indistinguishable from GT
motion, 0 once the discriminator is confidently against it. Bounded to [0, 1].
**Condition (C) has now been run**: six runs in `checkpoints/gail_c/`, two
discriminator conditions x seeds 42/43/44, at the same 128,000 timesteps as (B).
See *(C) PPO + GAIL* below for the results; this section covers only what the
discriminator sees, which is where the pre-run prediction went wrong.

> This row used to be specified as `sigmoid(D(pose))`. The reward changed to the
> least-squares form above when the discriminator moved to AMP's objective,
> precisely because a sigmoid reward goes flat the moment the discriminator can
> classify — which, as the rest of this section shows, is what happens here
> within ten updates.

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
accuracy against 50% chance.

**That measurement was in the wrong space, and the conclusion drawn from it was
wrong.** This document previously withdrew the "discriminator saturates
immediately" prediction on the strength of that 51.2%, concluding there was "no
trivial rule left to find." **The prediction is reinstated.** The figures above
are computed with each side standardised in *its own* normalisation, which
*forces* the marginals to agree — `d' ≈ 0` is a property of having standardised
twice, not evidence that the two distributions overlap. The very next paragraph
said as much, and the two claims were never reconciled.

Measured instead in the single space `PoseSpace` maps both sides into — the
space the discriminator in (C) actually sees — the classes are far apart:

| space the samples are in | joints scored | mean per-dim d' | best linear (LDA) | implied accuracy |
|---|---|---|---|---|
| each side in its own normalisation *(the old measurement)* | 22 / 66 dims | 0.005 | 0.06 | 51.2% |
| common GT space, `exclude_joints 0` *(the (C) "feet in" arm)* | 21 / 63 dims | 0.374 | 8.83 | ~100% |
| common GT space, `exclude_joints 0 10 11` *(the (C) "feet out" arm)* | 19 / 57 dims | 0.296 | 6.29 | ~99.9% |

A *linear* rule already separates real from fake essentially perfectly. The
trained discriminators duly saturate: 99.4% / 98.4% accuracy (real / fake) in the
feet-in arm and 96.8% / 96.7% in the feet-out arm, at the R1 penalty of 50 that
`src/models/discriminator.py::_r1_gradient_penalty` documents as buying headroom
rather than preventing this. Excluding the feet is the only thing that measurably
moves it, and it moves it from ~100% to ~99.9%.

**What saved the experiment from that is the reward, not the discriminator.**
`amp_reward`'s least-squares score is unsquashed, so it keeps *ranking* fakes
long after it could *classify* them. The (C) runs bear that out: the
discriminator's improvement-over-lifted term stays positive and stable at
**+0.0049 ± 0.0004** (feet in) and **+0.0033 ± 0.0009** (feet out) across all six
runs, rather than collapsing to the constant a sigmoid reward would have given.
The saturation prediction was right about the discriminator and wrong about the
consequence.

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
root.**

**That fix was made before (C) ran.** `src/models/discriminator.py::PoseSpace`
maps the fake side into the GT-normalised space the real bank already lives in
(`physical = mu_lifted + sd_lifted * z`, then `z_gt = (physical - mu_gt) / sd_gt`,
per camera), and `GAILRewardProvider` *requires* one rather than accepting it as
an option — scoring a lifted-space pose with a GT-space critic now raises instead
of silently costing 5.1°. Joint 0 (`global_orient`) is excluded by default, both
because where the subject faced is not a plausibility cue and because it
contributed 31° of that 5.1° RMS.

So the risk this section warned about — "a well-behaved discriminator rewarding
the wrong pose" — was addressed. The risk it had *withdrawn* is the one that
materialised: once both sides share a space, they are trivially separable, and
every (C) discriminator saturates.

**Reproduce.** The first command is the old, own-space measurement (51.2%); the
`--common_space` ones are the space the discriminator actually sees, and take
`--exclude_joints` exactly as `gail_train --disc_exclude_joints` does.

```bash
python scripts/disc_separability.py --physical
python scripts/disc_separability.py --common_space --exclude_joints 0
python scripts/disc_separability.py --common_space --exclude_joints 0 10 11
```

> `scripts/run_gail_feet_ablation.sh`'s header comment quotes this ablation as
> cutting separability from `d' = 8.3` to `d' = 5.9`. The reproducible figures
> are **8.83 → 6.29** on the train split (9.81 → 6.87 on test); the comment
> predates the current measurement and is stale. The direction and the size of
> the cut are unchanged, and neither figure gets the discriminator below 99.9%.

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

| | Unit | (B1) frozen | (B2) du,dv | (B3) pose-only | (C) feet in | (C) feet out |
|---|---|---|---|---|---|---|
| Observation width | dimensions | 182 | 182 | 176 | 182 | 182 |
| Action width | dimensions | 66 | 68 | 66 | 66 | 66 |
| PPO steps completed | environment timesteps | 128,000 | 128,000 | 128,000 | 128,000 | 128,000 |
| Pose action bound | normalised pose units | +/-0.3 | +/-0.3 | +/-0.3 | +/-0.3 | +/-0.3 |
| Image-shift bound (du, dv) | bbox heights | — | +/-0.1 | — | — | — |
| Reward width sigma | bbox heights | 0.0225 | 0.0225 | 0.0225 | 0.0225 | 0.0225 |
| Discriminator weight `w_gail` | relative to base reward sd | — | — | — | 0.5 | 0.5 |
| Joints hidden from discriminator | joint indices | — | — | — | [0] | [0, 10, 11] |
| Discriminator input width | dimensions | — | — | — | 63 | 57 |
| Discriminator hidden dims | — | — | — | — | (256, 128) | (256, 128) |
| R1 gradient penalty | weight | — | — | — | 50.0 | 50.0 |
| Real bank / policy clips disjoint | `demo_frac` | — | — | — | no (0.0) | no (0.0) |
| Memorisation probe | `demo_probe_frac` | — | — | — | 0.05 | 0.05 |

**(C) is (B1) plus a discriminator, and nothing else.** Comparing the two saved
configs key by key, every hyperparameter they share is identical — same
`hidden_dims`, `learning_rate`, `rollouts`, `mini_batches`, `total_updates`,
`reward_mode=reproj`, `trans_mode=none`, `reproj_sigma`, `w_reproj`,
`w_smoothness`, `kp_bias_path`, seeds. The only differences are the keys a (B1)
run does not have at all (`w_gail`, `disc_*`, `demo_*`, `gail_*`). That is why
the (C) columns of the diagnostics table can be read against (B1) directly.

**Reproduce.** Widths and step counts are read back out of the checkpoints, so
they cannot drift from what actually ran. Bounds and sigma come from the same
configs (`trans_mode` -> `src.models.policy.action_bounds`). The second command
prints the discriminator settings as a second table; it stays silent for run sets
that have no `w_gail` key, i.e. for (B).

```bash
python scripts/sweep_stats.py --table config
python scripts/sweep_stats.py --table config --sweep_dir checkpoints/gail_c \
    --runs feet_in_s42 feet_in_s43 feet_in_s44 \
           feet_out_s42 feet_out_s43 feet_out_s44
```

The observation is the lifted pose, the previous corrected pose, and a 44-dim
block of 2D evidence (per-joint reprojection residual, confidences, scalar error,
episode progress, camera one-hot, bbox geometry). (B3) is 6 dims narrower because
it drops the translation from the state; the translation is kept inside the
environment and used only by the reward. Widths are 22-joint: they were 362 /
362 / 356 and 156 / 158 / 156 before the finger removal.

**One earlier (B3) run went further — but its number cannot be compared to this
table, and cannot be made comparable.** Before the sweep, a single-seed (B3) run
in `checkpoints/biasfix/notrans/` reached 288,000 timesteps, where it logged
**+0.064 px** held-out improvement and 8.22 px corrected error, the strongest
(B3) figure on record. That figure is **`pose/img_improvement_px` read from its
TensorBoard file**, which the trainer computes on **`val`**, not `test` — and the
checkpoint is **52-joint** (action width 156), from before the finger removal.
It is therefore two changes away from the -0.005 ± 0.090 px it used to be quoted
against: different split *and* different data.

Re-scoring it on `test` would settle the first difference but is not possible:
`scripts/heldout_eval.py` rejects the checkpoint outright —
`ValueError: action width 156 is none of 66 / 68 / 69` — because a 52-joint actor
cannot be rolled over the 22-joint dataset. Only a retrained run at 288,000 steps
could produce a comparable number.

Setting that aside, it is not in the table above for the reasons that held
before: it is one seed, its diagnostics do not reproduce (see the note under
*Training diagnostics*), and the three-seed sweep at 128,000 puts (B3) at
**-0.005 ± 0.090 px** held-out — a mean eighteen times smaller than its own seed
spread, with the sign splitting 2-1. That
is weaker evidence against the run than the 52-joint sweep gave (-0.051 ± 0.043,
negative at every seed), but it is not evidence for it either: the column simply
cannot resolve an effect this size at three seeds. Whether the gain is real or is
that run's seed is untested — running the sweep out to 288,000 would settle it,
and is the one experiment that could still rescue (B3).

---

## Seed sweep

Nine runs — three variants x seeds 42/43/44 — each 40 updates x 3,200 = **128,000
timesteps**, in `checkpoints/sweep/`. This is the sweep the previous version of
this document said (B3) needed before its advantage could be claimed. The (C)
rows below add six more runs on the same schedule and the same seeds, from
`checkpoints/gail_c/`.

**PA-MPJPE (mm), test split, 374 clip-cameras, j14, GT betas.** Each corrected
score is paired against the lifted score on the *same* clip-camera, so the delta
carries no aggregation difference.

| variant | s42 | s43 | s44 | mean ± sd | Δ vs lifted | clip-cameras improved |
|---|---|---|---|---|---|---|
| (A) lifted baseline | 34.396 | 34.396 | 34.396 | **34.396** | 0 *(by definition)* | — |
| (B1) frozen | 34.549 | 34.504 | 34.492 | 34.515 ± 0.030 | +0.119 ± 0.030 | 38.2% |
| (B2) du,dv | 34.476 | 34.453 | 34.561 | 34.497 ± 0.057 | +0.101 ± 0.057 | 40.7% |
| (B3) pose-only | 34.416 | 34.529 | 34.494 | 34.480 ± 0.058 | +0.084 ± 0.058 | 40.1% |
| (C) GAIL, feet in | 34.426 | 34.509 | 34.404 | **34.446 ± 0.055** | **+0.050 ± 0.055** | **45.5%** |
| (C) GAIL, feet out | 34.505 | 34.556 | 34.541 | 34.534 ± 0.026 | +0.138 ± 0.026 | 34.2% |

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
| (C) GAIL, feet in | 8.52 ± 0.70 | **-0.340 ± 0.023** | -0.334 / -0.321 / -0.366 |
| (C) GAIL, feet out | 8.52 ± 0.69 | **-0.360 ± 0.029** | -0.381 / -0.372 / -0.328 |

**Reproduce.** Reads the nine event files directly; `--window 0` would point-read
the final step instead of averaging, which is noisier per update. The (C) runs
live under a different sweep directory and are named by condition rather than by
variant, so they are passed explicitly.

```bash
python scripts/sweep_stats.py --table rollout --window 20000
python scripts/sweep_stats.py --table rollout --window 20000 \
    --sweep_dir checkpoints/gail_c \
    --runs feet_in_s42 feet_in_s43 feet_in_s44 \
           feet_out_s42 feet_out_s43 feet_out_s44
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

**This holds across (B) and (C) too, which is what makes them comparable.** The
(C) runs report the same three lifted errors to four decimals — 8.6394 / 7.3602 /
8.3901 — despite being launched separately, weeks apart, from a different script.
They can be, because `demo_frac = 0.0` leaves the policy on 100% of the train
split, exactly the clips (B) used; the discriminator's real bank overlaps those
clips rather than being carved out of them. So a (C)-vs-(B1) comparison at a
fixed seed is exact in the same sense a (B1)-vs-(B3) one is.

**Held-out improvement (px), higher is better.** The 22-joint runs
*were* launched with pose logging on — the saved configs record
`viz_interval = 5`, `viz_mode = image`, and each event file carries eight
`pose/img_improvement_px` points — unlike the 52-joint sweep, which ran at
`viz_interval = 0` and logged none. The table does not read those scalars
anyway, and two independent reasons say it should not. First, **they are on
`val`**: the trainer hard-codes `split="val"` for its viz logger
(`src/train.py:380`), so those points describe different clips from every table
in this report. Second, they are computed at `viz_clips = 3`, `viz_frames = 4`,
which is 36 frames per point and far too few to resolve an effect of this size;
the logged final values scatter from -0.087 to +0.147 px, an order of magnitude
wider than the column below.

The table instead rolls each saved `actor_final.pt` over the **test** split with
the **policy mean**, through `correction_magnitude()` — a pure function of the
final weights and the scored clips (`ImagePoseVizLogger._rollout` just calls
`rollout_policy` with the actor), so it is the same code path, at a usable
sample size. Only the endpoint is measurable this way, not the curve, but the
endpoint at matched steps is what the table reads.

**Scored on `test`, the same split as the PA-MPJPE table**, over all **372
aligned clip-cameras x 12 frames** = 4,464 frames at a fixed clip seed, so all
nine checkpoints see identical clips — the lifted error is 12.1368 px in every
one of them, which confirms it. (372 of 374: two clip-cameras have no aligned
2D targets.)

> **This row used to be measured on `val`, at 80 clips x 12 frames.** That was
> inherited from `ImagePoseVizLogger`'s role as a training-time logger, not
> chosen for this metric — and nothing was ever selected on val (no early
> stopping, no checkpoint choice, no hyperparameter search), so the separation
> bought nothing while making this row and the PA-MPJPE row describe different
> clips. The move also raises the sample 4.6x. The old val numbers were
> +0.022 ± 0.066 (B1), -0.007 ± 0.056 (B2), +0.004 ± 0.085 (B3);
> `--split val --n_clips 80` reproduces them. **The conclusion is unchanged** —
> all three are still indistinguishable from zero with the sign splitting
> across seeds — which is itself worth noting: the reading did not depend on
> which held-out split was used.

> Note this means `scripts/run_seed_sweep.sh` as committed (`--viz_interval 0`)
> is **not** how the 22-joint checkpoints were produced; they were launched with
> logging on, per the project's training-run convention. The flag does not change
> which data a run sees: the per-seed lifted rollout error is identical to four
> decimals across both sweeps (8.6394 s42 / 7.3602 s43 / 8.3901 s44) despite the
> different setting. Whether it leaves the weight updates bit-identical has not
> been checked. Reconcile the script before using it to reproduce this sweep.

| variant | s42 | s43 | s44 | mean ± sd | sign |
|---|---|---|---|---|---|
| (B1) frozen | -0.074 | +0.025 | +0.054 | **+0.002 ± 0.067** | - + + |
| (B2) du,dv | +0.052 | -0.078 | +0.016 | **-0.003 ± 0.067** | + - + |
| (B3) pose-only | -0.110 | +0.049 | +0.045 | **-0.005 ± 0.090** | - + + |
| (C) GAIL, feet in | +0.074 | -0.012 | +0.083 | **+0.049 ± 0.052** | + - + |
| (C) GAIL, feet out | +0.001 | -0.114 | +0.020 | **-0.031 ± 0.072** | + - + |

**Reproduce.** No retraining — this rolls each saved actor over the test split.
About a minute per checkpoint. The script prints the lifted error per run and
warns if it is not constant, which would mean the runs saw different clips and
the comparison is void. It reads **12.1368 px for all fifteen checkpoints**,
(C) included.

```bash
python scripts/heldout_eval.py --split test --n_clips 400 --n_frames 12 --clip_seed 42
python scripts/heldout_eval.py --sweep_dir checkpoints/gail_c \
    --split test --n_clips 400 --n_frames 12 --clip_seed 42
```

**Held-out improvement is indistinguishable from zero for every variant, and now
no variant even holds its sign.** Each of the three splits 2-1 across seeds, and
the across-seed spread swamps the mean in all of them — |mean|/sd is 0.33 (B1),
0.13 (B2), 0.05 (B3). (B3)'s best and worst seeds differ by 0.147 px, roughly ten
times the +0.016 px effect this column was once claimed to show.

This is a *weaker* result than the 52-joint sweep gave, not a stronger one. There,
(B3) was negative at all three seeds (-0.051 ± 0.043); here it is -0.005 ± 0.090,
i.e. the same sign but a mean the seed spread swamps. The finger removal moved
every variant toward zero and widened the seed spread, so what was a consistent
small negative is now noise centred on nothing. The
honest reading is that this column cannot resolve an effect of the size being
looked for at three seeds — not that (B) improves the held-out image fit.

> These are **raw** pixel errors — `correction_magnitude()` computes distances
> directly and applies no keypoint-bias subtraction, unlike the training-rollout
> reprojection rows above. That is why the level reads ~12 px here against ~8-9 px
> there. The bias is common to the lifted and corrected sides, so the improvement
> column is the comparable one; the levels are not.

---

## (C) PPO + GAIL

Six runs in `checkpoints/gail_c/` — two discriminator conditions x seeds
42/43/44 — each 40 updates x 3,200 = **128,000 timesteps**, launched by
`scripts/run_gail_feet_ablation.sh`. Every hyperparameter shared with **(B1)
frozen** is identical (see *Configuration*), so (C) is (B1) plus a discriminator
term, and the per-seed lifted errors match (B)'s to four decimals, so
within-seed comparisons against (B) are exact.

The two conditions differ only in what the discriminator is allowed to see:

* **feet in** — 21 joints / 63 dims, excluding joint 0 (`global_orient`).
* **feet out** — 19 joints / 57 dims, also excluding joints 10 and 11 (the feet).

**Why the feet were suspect.** SMPLer-X under-articulates them badly. Measured on
the physical axis-angle pose over the train split, the GT-to-lifted ratio of
per-joint standard deviations is **4.6x at joint 10 and 5.4x at joint 11** — the
two largest of any body joint, against 2.5x for the ankles, 2.2x for the wrists
and **1.48x averaged over the other nineteen body joints**. A discriminator can
read that without learning anything about pose plausibility, which is the same
shape as the 52-joint finger leak. Hiding the feet cuts best-linear separability
from `d' = 8.83` to `d' = 6.29`. The hypothesis was that removing a spurious cue
would force the discriminator onto genuine plausibility and improve the policy.

```bash
python scripts/disc_separability.py --joint_sd
```

**The ablation came out backwards, and it is the one comparison in this report
that three seeds cleanly resolve.** Hiding the feet made every metric worse on
the mean, and both test-split metrics worse at every individual seed:

| | PA-MPJPE (mm) | Δ vs (A) | clip-cams improved | held-out test (px) | train rollout (px) |
|---|---|---|---|---|---|
| (C) feet in | **34.446 ± 0.055** | **+0.050 ± 0.055** | **45.5%** | **+0.049 ± 0.052** | **-0.340 ± 0.023** |
| (C) feet out | 34.534 ± 0.026 | +0.138 ± 0.026 | 34.2% | -0.031 ± 0.072 | -0.360 ± 0.029 |

Paired per clip-camera, feet-in beats feet-out by **0.088 mm** (t = -5.5 over
1,122 deltas), at **all three seeds**, on 59.1% of clip-cameras; it is also ahead
at all three seeds on the held-out row. (The train-rollout row is the one that
splits, 2-1 — but that row measures the reward the policy was optimising, not
held-out quality, and it is negative for every variant in this report.) Nothing
else in this document separates two conditions as consistently. **The joints the discriminator was
suspected of cheating on were carrying signal it needed** — plausible enough in
hindsight, since the feet are where a physically implausible pose shows up first,
and "SMPLer-X under-articulates them" is a fact about the pose being wrong, not
only about which model produced it.

**Against (A), (C) still loses.** Feet-in is the best variant in this report on
both test-split metrics — the smallest PA-MPJPE degradation (+0.050 mm against
(B3)'s +0.084 and (B1)'s +0.119), the highest share of clip-cameras improved
(45.5% against 38.2-40.7%), and the only held-out mean that is comparable to its
own seed spread (|mean|/sd = 0.94, against 0.03-0.33 for the three (B) variants).
It is still **significantly worse than doing nothing**: paired t = +4.3 over
1,122 clip-camera deltas, in the *worse* direction. The finding of this report
survives (C) intact.

**And the margin over (B) is not settled at three seeds.** Paired per
clip-camera, feet-in is ahead of (B1) by 0.069 mm (t = -4.0) and of (B3) by
0.034 mm (t = -2.0) — but both split **2-1 across seeds**, so the pooled t is
reading within-seed clip variation, not a reproducible ranking. This document
has already retracted one result of exactly that shape ((B3)'s seed-42
advantage), so the same standard applies here: **(C) feet-in is not claimed to
beat (B)**. What is claimed is the feet ablation, which holds at every seed.

**The discriminator saturates, as originally predicted.** 99.4% / 98.4%
(real / fake) for feet-in, 96.8% / 96.7% for feet-out, at an R1 penalty of 50.
The withdrawal of that prediction was based on a separability measured with each
side standardised in its own space; in the common space `PoseSpace` builds, a
linear rule already gets ~100%. See *Training diagnostics* above.

**It is not memorising the bank.** With `demo_frac = 0.0` the real bank is drawn
from the same clips the policy rolls out, so 5% of train clips are held out of the
bank only and scored as a probe. Accuracy on that unseen GT tracks accuracy on the
bank — the gap is **-0.0025 ± 0.0014** (feet in) and **+0.0059 ± 0.0014** (feet
out), i.e. nothing. The discriminator is separating SMPLer-X-shaped pose from
MoVi-shaped pose in general, not recognising particular clips.

**The reward kept working anyway, which is the reason to keep the AMP form.**
`amp_reward`'s least-squares score is unsquashed, so it ranks fakes after it can
classify them. Improvement-over-lifted stays positive and tight across all six
runs — **+0.0049 ± 0.0004** (feet in), **+0.0033 ± 0.0009** (feet out) — so the
policy does make its pose measurably more GT-like than the pose it was handed. A
sigmoid reward would have been a constant here. Note the sign: the arm whose
discriminator is *harder* to fool (feet in, 99.4%) earns the *larger* plausibility
gain, and its scaled contribution to PPO's reward is larger too
(+0.0346 ± 0.0035 against +0.0274 ± 0.0072).

**Reproduce.** Training, then the two evaluation tables. `run_eval_c_light.sh`
pins each process to a fixed core set and runs at most two at a time; the
unpinned version of this sweep drove load average to 425 on 32 cores and took the
machine down. Every step skips work whose output already exists, so an
interrupted run resumes.

```bash
bash scripts/run_gail_feet_ablation.sh     # 6 runs, ~18 of 32 cores
bash scripts/run_eval_c_light.sh           # PA-MPJPE + held-out, core-pinned
```

**What would settle the open question.** The (C)-over-(B) margin needs more
seeds, not more steps — its sign is stable pooled and unstable per seed, which is
what three samples of a small effect look like. The feet result needs neither: it
is already consistent at every seed on both test-split metrics. The more
interesting
follow-up is the one the ablation's failure suggests — that the discriminator's
value here is coming from the joints where SMPLer-X is *most* obviously wrong, in
which case the productive direction is to give it more of them (a 2-frame AMP
window, per `WINDOW` in `src/models/discriminator.py`), not fewer.

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

**The reprojection reward's optimum is displaced from ground truth — but by far
less than this report used to claim.** The GT pose does reproject *worse* than
the lifted pose, on `test`, over the same 372 clip-cameras x 12 frames the
held-out row uses: **12.45 px vs 12.14 px**, a gap of **+0.31 px**. SMPLer-X and
ViTPose were both fit to the same image and share correlated error, while ground
truth is independent of it, so maximising this reward does move the pose *away*
from GT. That is a property of the objective, not a training failure, and it was
the argument for (C) — which (C)'s result is consistent with, in direction and in
size.

| pose | betas | reprojection error, test (px, raw) |
|---|---|---|
| lifted (SMPLer-X) | lifted | 12.14 |
| ground truth | GT | 12.45 |
| ground truth | lifted | 12.47 |

**The previous figure — "9.38 px vs 7.21 px on held-out validation", a +2.17 px
gap — was an artefact of the bias correction, not the split.** Two measurements
separate the causes. Re-running the same comparison on `val` gives 12.38 vs
12.05, a +0.33 px gap: statistically the same as `test`, so the split was never
the issue. Re-running it on `test` *with the fitted keypoint bias subtracted* —
the convention of the bias-corrected rows — gives **10.82 px vs 5.96 px, a
+4.86 px gap**, which is the regime the old number lived in.

The mechanism is that `data/kp_bias.json` is fitted on **lifted** poses
(`scripts/fit_kp_bias.py`, train split), and `scripts/fit_kp_bias.py`'s own
docstring already records that GT carries a *different* offset — PG1 global
(+0.0147, +0.0235) for GT against (+0.0112, +0.0208) for lifted. Subtracting the
lifted offset therefore removes 51% of the lifted error (12.14 -> 5.96) but only
13% of the GT error (12.45 -> 10.82). The residue is a correspondence mismatch
being scored as a pose difference. Holding betas fixed instead of letting each
pose use its own narrows it to +1.36 px, still four times the raw gap.

**What survives.** The sign, which is what the (C) argument needs: reprojection
prefers the lifted pose over ground truth, on both splits, under both
conventions. What does not survive is the magnitude — the displacement is
**+0.31 px, not +2.17 px**, roughly 2.5% of the lifted error rather than 30% of
it. Any claim that the reprojection reward is *strongly* mis-aimed rests on the
bias artefact. (C) was worth running; the headroom it was chasing is small.

**(C) recovered a little of that headroom and did not close it.** Adding the
discriminator to the (B1) recipe cuts the PA-MPJPE degradation from +0.119 to
+0.050 mm and is the only arm whose held-out improvement (+0.049 ± 0.052 px) is
comparable to its own seed spread. Both are consistent with a reward that is
mis-aimed by about a third of a pixel: a term that does not use the detector at
all pulls the pose back toward GT, but not far enough to make correction pay.
**No variant in this report beats doing nothing**, and (C) is worse than (A) at
t = +4.3. See *(C) PPO + GAIL*.

**Reproduce.** Scores the same clips as `heldout_eval.py` at the same
`--clip_seed`; no policy is involved on either side.

```bash
python scripts/gt_reproj_check.py --split test --n_clips 400 --clip_seed 42
python scripts/gt_reproj_check.py --split val  --n_clips 80  --clip_seed 42
python scripts/gt_reproj_check.py --split test --n_clips 400 --bias data/kp_bias.json
```

**Reprojection rows carry a fitted bias correction.** All reprojection figures
subtract a systematic per-joint COCO-to-SMPL-X offset (`scripts/fit_kp_bias.py`,
fitted on the train split only), which accounts for ~85% of the raw error —
0.0262 of 0.0309 bbox heights. Without it the same runs read ~14 px instead of
~7 px. The row measures **agreement with ViTPose**, not reconstruction accuracy.

**Held-out improvement is the honest (B) headline, not reprojection error.**
Reprojection error alone rewards agreement with the detector; improvement over
the lifted input is the only column that says whether the policy did anything.
The earlier single-seed reading of that row put (B3) at +0.016 px — the one
positive number anywhere in (B), and the basis for preferring it. That reading
was on **`val`**, at 80 clips x 12 frames; the column it is compared against
below is on `test` at 372 clip-cameras x 12 frames.

**That (B3) result did not survive the seed sweep.** It was measured at seed 42,
which the sweep shows is (B3)'s lucky seed: seed 42 gives its best PA-MPJPE
(34.416 mm) and seeds 43/44 give back 0.113 and 0.078 mm. Across three seeds
(B3) is 34.480 ± 0.058 mm. That is nominally the lowest variant mean, but the
gap to (B1)'s 34.515 ± 0.030 is smaller than either variant's own seed spread,
and the ordering reverses between the 52-joint and 22-joint sweeps — so it is a
ranking inside the noise, not a result. Its training-rollout improvement is
negative at all three seeds (-0.386 ± 0.039 px).

**And the held-out figure does not reproduce either.** Rolling the nine
checkpoints over the test split — 372 clip-cameras x 12 frames, roughly 4.6x the
sample the val-split +0.016 px came from — puts (B3) at **-0.005 ± 0.090 px**, with the
sign splitting 2-1 across seeds. It is not negative, but it is not the +0.016 px either: the
spread is twenty times the mean, so the column resolves nothing at this sample
size. Training-rollout improvement remains negative at every seed
(-0.386 ± 0.039 px). **The (B3) advantage is withdrawn** — not because the
held-out column refutes it, but because nothing reproduces it and the one
measurement that is stable across seeds points the other way.

**Lifter identity.** The experiment description names MotionBERT for condition
(A), but nothing in this repository references it; the lifted poses come from
SMPLer-X throughout. Column (A) reports SMPLer-X.
