from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
from gymnasium import spaces

from src.models.policy import (TRANS_MODE_NONE, action_bounds, action_dim,
                               flatten_state, state_dim, unflatten_action)


def _to_tensor(x: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


class MoviEnv:

    def __init__(self, device: str = "cpu", keys: tuple = ("poses", "trans")):
        self.device = torch.device(device)
        self.keys = keys

    def reset(self, init_state: dict) -> dict:
        """
        Initialise episode from the first lifted frame.
        init_state: dict of {key: tensor or ndarray}, one frame each.

        The corrected slot starts as a *copy of the lifted frame*, i.e. the
        identity correction. It used to start at `zeros_like`, which in
        normalised units is the mean pose of the dataset — a T-pose-ish average
        that no frame of any episode ever looks like. Every subsequent step has
        `corrected ~= lifted`, so the first observation of every episode was the
        only one drawn from a different distribution than the rest, and the
        policy had to spend capacity on a transient that means nothing.
        """
        lifted = {k: _to_tensor(init_state[k], self.device) for k in self.keys}
        self.trans_delta = None
        self.state = {
            "lifted_state": lifted,
            "corrected_state": {k: v.clone() for k, v in lifted.items()},
        }
        return self.state

    def step(self, action: dict, next_lifted: dict) -> dict:
        """
        Apply correction action and advance to the next lifted frame.

        action       : {key: tensor}  — correction deltas, same shape as one frame
        next_lifted  : {key: tensor}  — lifted frame at t+1

        Returns the new state dict. Reward is computed externally in the training loop.
        """
        for k in self.keys:
            if k == "trans":
                # Never corrected additively. The lifted `trans` is virtual-camera
                # depth, not metres, so a delta on it has no geometric meaning;
                # the policy's translation action is `(du, dv, dlog_tz)` in
                # `action["trans_delta"]`, applied to the *metric* translation at
                # projection time. Passing the lifted value through unchanged
                # keeps the observation and FK identical in every trans mode.
                self.state["corrected_state"][k] = self.state["lifted_state"][k].clone()
            else:
                self.state["corrected_state"][k] = (
                    self.state["lifted_state"][k] + action[k].to(self.device))
            self.state["lifted_state"][k] = _to_tensor(next_lifted[k], self.device)
        self.trans_delta = action.get("trans_delta")
        return self.state


# TODO: this metrics should be pulled from evaluate.py because we want the metrics to be the same
def compute_reward(
    corrected:      dict[str, torch.Tensor],
    gt:             dict[str, torch.Tensor],
    prev_corrected: dict[str, torch.Tensor],
    w_similarity:   float = 1.0,
    w_smoothness:   float = 0.1,
    reward_scale:   float = 10.0,
    keys:           tuple = ("poses", "trans"),
) -> float:
    # `keys` drops to ("poses",) under pose-only training. Scoring `trans` there
    # would add the lifted-vs-GT translation error to every reward — a constant
    # the policy cannot affect, which leaves the gradient alone but shifts the
    # value target by a large per-clip offset for no reason.
    reward = 0.0
    for key in keys:
        # TODO: add Procrustes-Aligned instead of raw RMSE for similarity
        rmse    = (corrected[key] - gt[key]).pow(2).mean().sqrt()
        reward -= w_similarity * rmse.item()
        # TODO: change it from velocity to acceleration 
        vel     = corrected[key] - prev_corrected[key]
        reward -= w_smoothness * vel.pow(2).mean().item()
    return reward / reward_scale


class GymMoviEnv(gymnasium.Env):
    """
    Gymnasium-compatible wrapper around MoviEnv for use with skrl.

    On each reset() a clip is sampled at random from the dataset.
    The agent sees a flat `state_dim(use_evidence)` observation and outputs a
    bounded correction delta.  Episodes end when the clip runs out of frames.

    The observation is the lifted frame, the previous corrected frame, and —
    under the reprojection reward — the 2D evidence block: the per-joint
    residual between where the lifted pose projects and where ViTPose saw the
    joint, its confidences, the scalar error, episode progress, and the camera /
    bbox context needed to turn an image-space residual into a joint-space
    correction. Without that block the reward is computed from quantities the
    policy cannot observe, and the identity is the best policy it can represent.

    Two reward modes:

    * `reward_mode="gt"` (default) — the original GT-based similarity term. This
      is a supervised objective, so it is **not** a valid reward for experiments
      (B) or (C); keep it for ablations and debugging.
    * `reward_mode="reproj"` — reprojection + smoothness, using no ground truth
      (`src/rewards.py`). This is the reward experiment (B) calls for. It needs
      the dataset to have been built with `reproj_path=`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset,
        device:       str   = "cpu",
        keys:         tuple = ("poses", "trans"),
        w_similarity: float = 1.0,
        w_smoothness: float = 0.1,
        reward_scale: float = 10.0,
        reward_mode:  str   = "gt",
        reproj_reward=None,
        w_reproj:     float = 1.0,
        baseline_every: int = 1,
        trans_mode: str = TRANS_MODE_NONE,
        reward_baseline: bool = True,
        use_evidence: bool = True,
    ):
        super().__init__()
        self.dataset      = dataset
        self.device       = torch.device(device)
        self.keys         = keys
        # Pose-only by default (§2): the lifted `trans` is virtual-camera depth,
        # not metres, so it is passed through rather than corrected. The state
        # still carries it; only the action shrinks.
        # `trans` is never corrected as a normalised delta in any mode, so it is
        # never scored as one either.
        self.trans_mode    = trans_mode
        self.predict_trans = trans_mode != TRANS_MODE_NONE
        self.reward_keys   = tuple(k for k in keys if k != "trans")
        self.w_similarity = w_similarity
        self.w_smoothness = w_smoothness
        self.reward_scale = reward_scale
        self.reward_mode  = reward_mode
        self.w_reproj     = w_reproj
        # Scoring the *unmodified* lifted frame alongside the corrected one is the
        # only way to see whether the policy is actually improving the fit rather
        # than just collecting a high reward the lifted pose already earned.
        #
        # It is now needed every frame regardless, because the same projection
        # produces the observation's residual (fix 06), so `baseline_every`
        # only controls how often the comparison is *logged*. The forward pass
        # for frame t+1 is computed once at the end of step t, cached, and reused
        # as step t+1's baseline — one SMPL-X forward per frame, not two.
        self.baseline_every = int(baseline_every)

        # Fix 05: r = r_corrected - r_lifted. Identity scores ~0.678 and GT
        # ~0.711, so without this the return is dominated by a ~0.68 constant the
        # policy cannot influence, and the advantage is the difference of two
        # large, nearly equal numbers. Subtracting the lifted pose's own score
        # leaves pure improvement signal centred on zero. Still GT-free: the
        # lifted pose is the policy's own input.
        self.reward_baseline = bool(reward_baseline)

        # Fix 06: put the 2D evidence the reward is computed from into the
        # observation. Only possible under the reprojection reward — there are no
        # 2D targets to show under `reward_mode="gt"`.
        self.use_evidence = bool(use_evidence) and reward_mode == "reproj"

        if reward_mode not in ("gt", "reproj"):
            raise ValueError(f"reward_mode must be 'gt' or 'reproj', got {reward_mode!r}")
        if reward_mode == "reproj" and reproj_reward is None:
            raise ValueError("reward_mode='reproj' requires a ReprojectionReward instance")
        self._reproj = reproj_reward
        self._reproj_active = False

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim(use_evidence=self.use_evidence),), dtype=np.float32,
        )
        # Fix 03: finite bounds. This was `[-inf, inf]` with `clip_actions=False`,
        # so a corrupted gradient had nothing to walk into — the correction grew
        # without limit until SMPL-X received a non-finite pose 2.77M steps in.
        # Per-dimension: the pose delta and the image-plane shift are on
        # different scales (see policy.action_bounds).
        self._act_low, self._act_high = action_bounds(trans_mode)
        self.action_space = spaces.Box(
            low=self._act_low, high=self._act_high,
            shape=(action_dim(trans_mode),), dtype=np.float32,
        )

        self._inner          = MoviEnv(device=device, keys=keys)
        self._x              = None
        self._y              = None
        self._t:   int       = 0
        self._T:   int       = 0
        self._prev_corrected = None
        self._state          = None
        # (frame index, reward, info) for the untouched lifted pose. Written when
        # the observation for a frame is built and read back as that frame's
        # baseline on the next step, so the projection is paid for once.
        self._lifted_eval    = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx          = int(self.np_random.integers(0, len(self.dataset)))
        sample       = self.dataset[idx]
        self._x      = sample["x"]
        self._y      = sample["y"]
        self._T      = self._y["poses"].shape[0]
        self._t      = 0

        state = self._inner.reset({k: self._x[k][0] for k in self.keys})
        self._prev_corrected = {k: state["corrected_state"][k].clone() for k in self.keys}
        # Acceleration needs two frames of history, not one.
        self._prev2_corrected = {k: v.clone() for k, v in self._prev_corrected.items()}
        self._state          = state

        self._lifted_eval = None
        if self.reward_mode == "reproj":
            self._reproj_active = self._reproj.reset(sample)

        return self._observation(), {}

    # TODO: plot the delta updates and the corrected poses to see if they are reasonable to Tensorboard
    def step(self, action: np.ndarray):
        # skrl's GaussianMixin already clips to the action space, but this env is
        # also driven directly by viz_pose.py and evaluate.py, which sample or
        # take the mean without going through the mixin. Clipping here means the
        # bound holds on every path rather than only the training one.
        action = np.clip(np.asarray(action, dtype=np.float32),
                         self._act_low, self._act_high)
        action_dict = unflatten_action(torch.from_numpy(action).float())
        next_lifted = {k: self._x[k][self._t + 1] for k in self.keys}

        self._state = self._inner.step(action_dict, next_lifted)
        corrected_t = {k: self._state["corrected_state"][k] for k in self.keys}

        # Fix 03, second half. With the action bounded and the lifted input
        # finite this is unreachable, which is the point: it is the assertion
        # that says so, and it ends one episode instead of raising eight hours
        # into a run from inside `Rotation.from_rotvec`.
        terminated = not bool(torch.isfinite(corrected_t["poses"]).all())

        if self.reward_mode == "reproj":
            # GT is deliberately not read on this path.
            reward, info = self._reproj_step(corrected_t)
        else:
            gt_current = {k: self._y[k][self._t].to(self.device) for k in self.keys}
            reward = compute_reward(
                corrected_t, gt_current, self._prev_corrected,
                self.w_similarity, self.w_smoothness, self.reward_scale,
                keys=self.reward_keys,
            )
            info = {}

        if terminated:
            info["nonfinite_pose"] = True

        self._t += 1
        # Running out of frames is a *time limit*, not a terminal state. Nothing
        # failed and no goal was reached — the clip simply has no frame t+1.
        #
        # Reporting it as `terminated` tells GAE that the value after the last
        # step is zero. With ~500-frame episodes and a per-step reward near 0.7
        # at discount 0.99, the true remaining return is ~60, so every episode
        # boundary injected a value target off by that much. Reporting
        # `truncated` instead lets skrl add the bootstrap (`ppo.py:299`,
        # `rewards += discount_factor * values * truncated`) — which also
        # requires `time_limit_bootstrap=True` in the PPO config, set in
        # src/train.py.
        truncated = self._t >= self._T - 1
        self._prev2_corrected = self._prev_corrected
        self._prev_corrected = {k: corrected_t[k].clone() for k in self.keys}

        return self._observation(), reward, terminated, truncated, info

    # ── observation ──────────────────────────────────────────────────────────
    def _t_frac(self) -> float:
        """Episode progress. Fix 08: lengths span 200-1448 frames and neither t
        nor T was observable, so the critic could not be right near an episode
        end — it had no way to know one was coming."""
        return float(self._t) / max(self._T - 1, 1)

    def _eval_lifted(self, t: int) -> tuple[float, dict]:
        """
        Score the *untouched* lifted frame t, memoised on t.

        This single projection does double duty: it is the baseline subtracted
        from the reward (fix 05) and the source of the residual the policy
        observes (fix 06). Because the observation emitted at the end of step t
        describes frame t+1, and step t+1's baseline is also frame t+1, the cache
        hit rate is 100% in a normal rollout — one SMPL-X forward per frame.
        """
        from src.rewards import empty_info

        if self._lifted_eval is not None and self._lifted_eval[0] == t:
            return self._lifted_eval[1], self._lifted_eval[2]
        if self._reproj_active and t < self._T:
            lifted_t = {k: self._x[k][t] for k in self.keys}
            reward, info = self._reproj.step(lifted_t, t)
        else:
            reward, info = float("nan"), empty_info()
        self._lifted_eval = (t, reward, info)
        return reward, info

    def _observation(self) -> np.ndarray:
        """
        Flat observation for the frame the next action will address.

        Fix 06. Without the evidence block this is 318 numbers of pose against a
        reward computed from ViTPose keypoints, this clip's camera, bbox and
        metric translation — none of which the policy could see. It was being
        asked to reduce a 2D error it had no way to measure, and the best
        learnable policy under that observation is the identity.
        """
        evidence = None
        if self.use_evidence:
            from src.rewards import empty_evidence, pack_evidence

            if self._reproj_active:
                _, info = self._eval_lifted(self._t)
                evidence = pack_evidence(info, self._reproj.context(self._t),
                                         self._t_frac())
            else:
                evidence = empty_evidence(self._t_frac())
        return flatten_state(self._state, evidence=evidence).detach().cpu().numpy()

    # ── reward ───────────────────────────────────────────────────────────────
    def _lifted_smoothness(self, t: int) -> float:
        """Acceleration of the raw lifted trajectory at t — the smoothness the
        policy inherits for free, and therefore the baseline its own smoothness
        is measured against."""
        from src.rewards import smoothness_reward

        poses = self._x["poses"]
        return smoothness_reward(
            poses[t].detach().cpu().numpy(),
            poses[max(t - 1, 0)].detach().cpu().numpy(),
            poses[max(t - 2, 0)].detach().cpu().numpy(),
        )

    def _reproj_step(self, corrected_t: dict) -> tuple[float, dict]:
        """Reprojection + smoothness, no ground truth involved."""
        from src.rewards import combine, empty_info, smoothness_reward

        t = self._t
        smooth = smoothness_reward(
            corrected_t["poses"].detach().cpu().numpy(),
            self._prev_corrected["poses"].detach().cpu().numpy(),
            self._prev2_corrected["poses"].detach().cpu().numpy(),
        )
        if self._reproj_active:
            r_proj, info = self._reproj.step(corrected_t, t, self._inner.trans_delta)
        else:
            r_proj, info = float("nan"), empty_info()

        # The per-joint arrays are for the observation, not for skrl's info dict,
        # which is forwarded per step and expects scalars.
        info.pop("resid", None)
        info.pop("conf", None)
        info["r_smooth"] = smooth
        info["r_reproj"] = r_proj
        if self.trans_mode != TRANS_MODE_NONE and self._inner.trans_delta is not None:
            d = self._inner.trans_delta.detach().cpu().numpy().reshape(3)
            info["trans_du"] = float(d[0])
            info["trans_dv"] = float(d[1])
            info["trans_dlogz"] = float(d[2])

        # Free now: this is the projection the previous step already computed to
        # build the observation.
        r_base, base = self._eval_lifted(t)
        if base["valid"] and self.baseline_every and t % self.baseline_every == 0:
            info["r_reproj_lifted"] = r_base
            info["err_px_lifted"] = base["err_px"]

        if not self.reward_baseline:
            return combine(r_proj, smooth, self.w_reproj, self.w_smoothness), info

        # Fix 05. Both terms are differenced against the lifted pose, so the
        # reward is improvement rather than level: a clip that is inherently
        # well-detected or inherently smooth no longer pays more than one that is
        # not, and the ~0.68 floor the policy cannot influence drops out. `combine`
        # falls back to the smoothness term alone when either projection is
        # missing, so an undetected frame contributes no spurious improvement.
        return combine(r_proj - r_base,
                       smooth - self._lifted_smoothness(t),
                       self.w_reproj, self.w_smoothness), info


# ─── Shared rollout ───────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_policy(
    sample:        dict,
    actor,
    keys:          tuple = ("poses", "trans"),
    device:        str   = "cpu",
    reproj_reward        = None,
    use_evidence:  bool  = False,
    max_steps:     int | None = None,
    deterministic: bool  = True,
    trans_mode:    str   = TRANS_MODE_NONE,
) -> list:
    """
    Roll a policy over one clip and return the corrected poses, `(52, 3)` each,
    still normalised.

    **This exists so that training, `src/viz_pose.py` and `src/evaluate.py`
    cannot disagree about the observation.** They all build one from the lifted
    frame, the previous corrected frame and — under the reprojection reward —
    the 2D evidence block, and a policy fed 318 numbers when it was trained on
    362 does not raise, it silently produces nonsense that looks like a bad
    result rather than a bug. One assembly, three callers.

    `reproj_reward` is required when `use_evidence` is set: the evidence *is* the
    reward's own projection of the lifted pose. Rolling out without it is only
    correct for a policy trained without it.
    """
    from src.rewards import empty_evidence, pack_evidence

    if use_evidence and reproj_reward is None:
        raise ValueError(
            "use_evidence=True needs a ReprojectionReward — the observation the "
            "policy was trained on contains that reward's 2D residual, and "
            "there is no way to reconstruct it here without one.")

    x = sample["x"]
    T = sample["y"]["poses"].shape[0]
    steps = (T if max_steps is None else min(max_steps, T)) - 1
    if steps < 1:
        return []

    active = bool(reproj_reward.reset(sample)) if use_evidence else False
    lo, hi = action_bounds(trans_mode)
    act_low, act_high = torch.from_numpy(lo), torch.from_numpy(hi)
    inner  = MoviEnv(device=device)
    state  = inner.reset({k: x[k][0] for k in keys})
    dev    = torch.device(device)

    def observe(t):
        evidence = None
        if use_evidence:
            t_frac = float(t) / max(T - 1, 1)
            if active:
                lifted_t = {k: x[k][t] for k in keys}
                _, info = reproj_reward.step(lifted_t, t)
                evidence = pack_evidence(info, reproj_reward.context(t), t_frac)
            else:
                evidence = empty_evidence(t_frac)
        return flatten_state(state, evidence=evidence).to(dev)

    corrected = []
    for t in range(steps):
        obs = observe(t)
        if deterministic:
            # The policy mean, not a sample: frame-to-frame differences between
            # epochs then reflect learning rather than exploration noise.
            action = actor.net(obs.unsqueeze(0)).squeeze(0)
        else:
            action, _ = actor.act(obs)
        action = torch.max(torch.min(action.cpu(), act_high), act_low)
        state = inner.step(unflatten_action(action), {k: x[k][t + 1] for k in keys})
        corrected.append(state["corrected_state"]["poses"].cpu())
    return corrected
