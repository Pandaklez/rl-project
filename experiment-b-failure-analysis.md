# Why experiment (B) failed, and what to change

Companion to `summary-of-changes.md`, which lists (B) as an open item with the
truncation/bootstrap fix as "the leading explanation, not a proven one". The two
runs on disk now settle it.

Evidence: `checkpoints/expB/` (bootstrap fixed) and
`checkpoints/expB_failed_no_bootstrap/` (before the fix), read with the
TensorBoard event accumulator.

---

## 1. The bootstrap fix is not the explanation

The two runs fail identically. Same shape, same rate, same endpoint.

| timesteps | no bootstrap | bootstrap fixed |
|---|---|---|
| 0 | reward 0.741, +0 px | reward 0.741, +0 px |
| ~350k | 0.596, −6.6 px | 0.588, −4.7 px |
| ~680k | 0.157, −6.2 px | 0.249, −6.9 px |
| ~1.36M | 0.174, **−35.8 px** | 0.239, **−48.3 px** |

The fix was correct on its own terms — reporting clip exhaustion as truncation
and enabling `time_limit_bootstrap` is right — but it addressed a value-target
bias that was never what killed the run. **Cross it off.**

The bootstrap-fixed run then ran on to 2.77M steps and died with

```
src/rewards.py:214 -> src/camera_frame.py:89
ValueError: Found zero norm quaternions in `quat`
```

i.e. the policy produced a NaN pose. All three losses read `nan` at the last
write. That is the end state of a divergence, not its cause.

## 2. What the curves actually say

`Reprojection / error lifted (px)` sits at **10–15 px for the entire run**, on both
runs, from step 0 to step 2.77M. The environment, the calibration, the 2D targets
and the reward are all stable and correct. Only the policy moves.

And it moves one way. Monotonically. `error corrected` goes 12.7 → 20 → 35 → 55 →
90 → 103 px. Improvement over lifted goes −0.06 → −89 px. The held-out image
figures agree independently: `pose/img_improvement_px` −3.0 → −53.0.

**A PPO agent that gets monotonically worse at its own reward is not a weak-signal
problem. It is a broken-gradient problem.** The reward is doing its job; the
optimiser is not attached to it.

Two more readings pin it down:

- **`Loss / Policy loss` is 0.18 at the very first update** and stays 0.2–0.4
  throughout. With normalised advantages and an unchanged policy, PPO's surrogate
  loss at the first minibatch of the first epoch should be ≈ 0. It never is.
- **`Policy / Standard deviation` is flat at 0.0498 for the first ~1M steps**, then
  climbs 0.05 → 0.11 and diverges. So the early collapse (0.74 → 0.20 by 215k) is
  *mean* drift, and the late collapse is σ runaway. Two phases, one cause.

## 3. Root cause: dropout corrupts the PPO importance ratio

`Config.dropout = 0.1` (`src/train.py:153`), applied to both actor and critic via
`_mlp(..., dropout=...)` in `src/models/policy.py:44-49`.

skrl switches model mode around the update:

- `skrl/agents/torch/ppo/ppo.py:202` — `set_mode("eval")` at init, so **rollout runs
  with dropout off**. `log_prob` and `values` are stored from a clean forward pass.
- `skrl/agents/torch/ppo/ppo.py:344` — `set_mode("train")` before `_update`, so the
  **update recomputes `log_prob` with dropout on**.

So `ratio = exp(log_prob_new − log_prob_old)` compares a dropout-perturbed network
against a clean one. It is not measuring a policy change at all.

Measured at initialisation (actor 318→512→256→156, dropout 0.1, σ = 0.0498):

```
|mu_train − mu_eval| per dim   0.00381      (mu scale 0.00793)
log-ratio                      mean −0.75   std 1.23
ratio                          median 0.481   p5 0.063   p95 3.41
outside PPO's clip range [0.8, 1.2]          88.9%
critic V, train vs eval        mean |dV| 0.364 on |V| 0.768   (47% noise)
```

**89% of transitions are clipped on the first update, before the policy has
changed at all.** The median ratio is 0.48, not 1.0. Every gradient PPO takes is
dominated by dropout noise rather than by which actions earned reward — which is
exactly a random walk in a 156-dimensional unbounded action space, and a random
walk is monotone in `|correction|`, hence monotone in error. That is phase one.

The critic is corrupted the same way: value regression targets are fit under a
47% noise floor.

### The σ runaway follows from the same bug

For a Gaussian policy with a mean perturbation Δμ and std σ:

```
log-ratio ~ N( −|Δμ|²/2σ² ,  |Δμ|²/σ² )
```

Substituting the measured Δμ: mean −0.725, std 1.21 — against −0.75 / 1.23
measured. The formula is exact.

Both terms shrink as σ grows. **PPO's clip penalty therefore creates a direct
gradient pressure to inflate σ**, because a wider action distribution is the only
way to make the dropout noise look small in log-prob units. Add the
`entropy_loss_scale = 0.01` bonus, whose gradient w.r.t. `log_std` is a constant
+1 per dim regardless of reward, and σ has two independent upward pressures and no
downward one that works. Hence flat-then-runaway: 0.05 → 0.11 → NaN. That is
phase two.

This also means σ inflation here is **not exploration**. It is the optimiser
escaping its own noise.

## 4. Three more things that would have to be fixed anyway

**No observation or value normalisation.** `state_preprocessor` and
`value_preprocessor` are both left at `None`. Per-step reward ≈ 0.7 at γ = 0.99
gives returns ≈ 70, and episode returns of 883 are logged. `Loss / Value loss`
starts at 99 and takes ~500k steps to come down. For that entire period GAE
advantages are dominated by value error, not by action quality.

**The action space is unbounded.** `spaces.Box(low=-inf, high=inf)` with
`clip_actions=False`. Nothing bounds the correction, so the random walk of §3 has
nothing to walk into — and nothing catches the NaN before it reaches
`Rotation.from_rotvec`.

**The policy cannot see what it is scored on.** The observation is
`(lifted_t, corrected_{t−1})` — 318 numbers of pose. The reward is reprojection
error against ViTPose keypoints, using this clip's camera, bbox and metric
translation. **None of that is in the observation.** The policy is asked to reduce
a 2D error it cannot measure. Even with every bug above fixed, the best learnable
policy is the identity, which is where it starts.

Minor, but real: `MoviEnv.reset` sets `corrected_state` to `zeros_like`, so the
first observation of every episode claims a corrected pose of zero — the mean pose
in normalised units — where every subsequent step has `corrected ≈ lifted`. Every
episode starts off-distribution.

## 5. The headroom problem is separate, and still true

`summary-of-changes.md` already measured it: identity scores 0.678, GT scores
0.711, so the entire action-attributable range of the reward is **~5%, sitting on
a floor of ~0.68 that the policy cannot influence**. Meanwhile instantaneous
reward varies 0.46–0.92 across frames for reasons that have nothing to do with the
action. The advantage signal-to-noise ratio is well below 1 before any bug is
considered.

So: the bugs in §3–4 explain why (B) actively diverged. §5 explains why fixing
them buys stability, not accuracy. Both are true and they are not the same claim.

## 6. What to change

### P0 — the run cannot work until these are done

1. **Remove dropout from the actor and critic.** `--dropout 0` at minimum; better,
   drop the parameter from the RL models entirely. Dropout in a policy network is
   wrong independent of the skrl mode-switch: it makes the behaviour policy
   stochastic in a way the log-prob does not model.
2. **Add `RunningStandardScaler` for both state and value.**
   ```python
   from skrl.resources.preprocessors.torch import RunningStandardScaler
   ppo_cfg["state_preprocessor"] = RunningStandardScaler
   ppo_cfg["state_preprocessor_kwargs"] = {"size": obs_space, "device": device}
   ppo_cfg["value_preprocessor"] = RunningStandardScaler
   ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
   ```
3. **Bound the action.** Squash to a fixed range (±0.3 normalised pose units is
   ~6σ of the current policy and far more than any plausible correction), or set
   finite `action_space` bounds with `clip_actions=True`. Then assert finiteness on
   the pose before `uncorrect_root` and end the episode on violation rather than
   raising 8 hours in.
4. **Set `entropy_loss_scale = 0.0`.** There is nothing to explore: the action is a
   residual around a good initial guess. The bonus only inflates σ.

### P1 — give the run something to learn from

5. **Subtract the lifted baseline from the reward.** `r = r_corrected − r_lifted`,
   computed every step rather than every 10th (`baseline_every=1`). This is
   GT-free — the lifted pose is the policy's own input — and it removes the ~0.68
   constant that dominates the return, turning the reward into pure improvement
   signal. Costs a second SMPL-X forward per step (3.9 → ~7.8 ms/step).
6. **Put the 2D evidence in the observation.** Append the 12 ViTPose keypoints in
   bbox-normalised coordinates plus their confidences (36 dims) to the state. This
   is the principled fix for §4: it is the difference between a policy that cannot
   learn and one that can. It stays GT-free — the same evidence the reward uses.
7. **Set `kl_threshold` and a `KLAdaptiveRL` scheduler** so a divergence throttles
   the learning rate instead of running for eight hours.

### P2 — variance

8. Episode lengths span 200–1448 frames and neither `t` nor `T` is observable, so
   the critic cannot be right near episode ends. Either append normalised `t/T` to
   the observation or accept the bias explicitly.
9. `num_envs=1` with a 3200-step rollout is 2–6 clips per update. Vectorise, or
   shorten episodes to fixed-length windows, so one clip's idiosyncrasies stop
   dominating an update.

### Run this first

**A positive control.** Train with `--reward_mode gt`, high learning rate, short.
The GT similarity reward has a strong, dense, learnable signal. If PPO still
degrades under it, the remaining fault is in the plumbing, not the reward — and
you find that out in twenty minutes instead of eight hours. It is a supervised
objective and not valid for (B); it is valid as a test that the optimiser works.

Then re-run (B) with P0 applied and the honest hypothesis stated up front:
**the target is a flat improvement curve, not a positive one.** Given §5, "(B) does
not degrade the lifted pose" is the result (B) can deliver. (C) is where the 3D
gains have to come from.
