# Results

All figures on the **test** split (187 clips x 2 cameras = 374 clip-cameras),
except the reprojection rows, which are training-rollout and held-out-validation
diagnostics read from TensorBoard.

## Primary metric

| | (A) SMPLer-X, no correction | (B1) PPO, trans frozen | (B2) PPO, du,dv image shift | (B3) PPO, pose-only state | (C) PPO + GAIL |
|---|---|---|---|---|---|
| **PA-MPJPE, test (mm), lower is better** | **34.40** | 34.49 | *running* | *running* | |

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

Read at **matched 128,000 PPO timesteps** so the three conditions are comparable.
All three use seed 42 and see the identical clip sequence — the lifted
reprojection error is 8.4047 px in all three, which confirms it.

| Metric | Unit | (A) | (B1) frozen | (B2) du,dv | (B3) pose-only | (C) |
|---|---|---|---|---|---|---|
| Reprojection error, corrected | px, lower better | 8.40 | 8.89 | 13.59 | 8.94 | |
| Reprojection error, lifted (same rollout) | px | 8.40 | 8.40 | 8.40 | 8.40 | |
| Improvement over lifted, train rollout | px, higher better | 0 *(by definition)* | -0.42 | -5.14 | -0.45 | |
| Improvement over lifted, held-out val | px, higher better | 0 *(by definition)* | -0.009 | -0.053 | **+0.016** | |
| Smoothness reward `exp(-a^2/sigma^2)` | dimensionless, (0, 1] | — | 0.9248 | 0.9248 | 0.9248 | |
| Mean discriminator reward, test | dimensionless | — | — | — | — | |
| RMSE, corrected vs GT | normalised pose units | — | not measured | not measured | not measured | |

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

**Smoothness reward (dimensionless).** `exp(-a^2/sigma^2)`, where `a` is the mean
squared **acceleration** of the corrected pose — the second finite difference
over three consecutive frames, in normalised pose units — with sigma = 0.5.
Acceleration rather than velocity: penalising velocity would reward a subject who
stops moving, which is wrong for a dataset of people walking and crawling.
Bounded to (0, 1].

**Mean discriminator reward (dimensionless).** Not yet defined — condition (C) is
not built.

**RMSE, corrected vs GT (normalised pose units).** Root-mean-square error between
the corrected and ground-truth pose vectors. This is a **diagnostic only** for
(B): it is the `reward_mode="gt"` supervised objective, deliberately disabled for
(B) and (C) so the reward stays ground-truth-free.

### Configuration

| | Unit | (B1) frozen | (B2) du,dv | (B3) pose-only |
|---|---|---|---|---|
| Observation width | dimensions | 362 | 362 | 356 |
| Action width | dimensions | 156 | 158 | 156 |
| PPO steps completed | environment timesteps | 128,000 | 128,000 | 288,000 |
| Pose action bound | normalised pose units | +/-0.3 | +/-0.3 | +/-0.3 |
| Image-shift bound (du, dv) | bbox heights | — | +/-0.1 | — |
| Reward width sigma | bbox heights | 0.0225 | 0.0225 | 0.0225 |

The observation is the lifted pose, the previous corrected pose, and a 44-dim
block of 2D evidence (per-joint reprojection residual, confidences, scalar error,
episode progress, camera one-hot, bbox geometry). (B3) is 6 dims narrower because
it drops the translation from the state; the translation is kept inside the
environment and used only by the reward.

**(B3) ran longer than the other two** (288,000 vs 128,000 timesteps). The table
reads all three at 128,000 for comparability. At its full 288,000 steps (B3)
reaches **+0.064 px** held-out improvement and 8.22 px corrected error.

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

The **30.0 mm** figure in `summary-of-changes.md` is the all-22 number and is not
comparable to published PA-MPJPE.

**Ground-truth betas are given to the prediction**, so bone lengths match exactly
and the metric reflects joint angles alone. That is correct for comparing
correction policies — the policy only moves joint angles, and pose cannot alter
bone lengths in SMPL-X — but it is **not** the benchmark setting, which uses the
predicted shape. `src/evaluate.py --betas lifted` measures that; it has not been
run.

**Aggregation differs slightly between columns.** (A) averages the two cameras
per clip, then over 187 clips; (B) averages over all 374 clip-cameras. To be
equalised before publication.

**(A) is the full test split**: 34.40 mm over 187 clips. An earlier 34.1 mm came
from a 25-clip subset.

---

## Reading these numbers

**(B1) does not beat doing nothing.** 34.49 mm against the 34.40 mm baseline —
0.09 mm *worse*, or 0.3% relative. Within noise of the baseline, and certainly
not an improvement. The next paragraph is why.

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
Only (B3) is positive, and by 0.016 px at matched steps — small enough that it
needs a seed sweep before it is claimed.

**Lifter identity.** The experiment description names MotionBERT for condition
(A), but nothing in this repository references it; the lifted poses come from
SMPLer-X throughout. Column (A) reports SMPLer-X.
