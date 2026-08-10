from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
from gymnasium import spaces

from src.models.policy import (STATE_DIM, action_dim, flatten_state,
                               unflatten_action)


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
        Corrected state starts as zeros (no correction applied yet).
        """
        self.state = {
            "lifted_state": {
                k: _to_tensor(init_state[k], self.device) for k in self.keys
            },
            "corrected_state": {
                k: torch.zeros_like(_to_tensor(init_state[k], self.device)) for k in self.keys
            },
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
            self.state["corrected_state"][k] = self.state["lifted_state"][k] + action[k].to(self.device)
            self.state["lifted_state"][k] = _to_tensor(next_lifted[k], self.device)
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
    The agent sees a flat STATE_DIM observation and outputs a flat FRAME_DIM action
    (correction delta).  Episodes end when the clip runs out of frames.

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
        baseline_every: int = 10,
        predict_trans: bool = False,
    ):
        super().__init__()
        self.dataset      = dataset
        self.device       = torch.device(device)
        self.keys         = keys
        # Pose-only by default (§2): the lifted `trans` is virtual-camera depth,
        # not metres, so it is passed through rather than corrected. The state
        # still carries it; only the action shrinks.
        self.predict_trans = bool(predict_trans)
        self.reward_keys   = keys if predict_trans else tuple(k for k in keys if k != "trans")
        self.w_similarity = w_similarity
        self.w_smoothness = w_smoothness
        self.reward_scale = reward_scale
        self.reward_mode  = reward_mode
        self.w_reproj     = w_reproj
        # Scoring the *unmodified* lifted frame alongside the corrected one is the
        # only way to see whether the policy is actually improving the fit rather
        # than just collecting a high reward the lifted pose already earned. It
        # costs a second forward pass, so it runs on every Nth frame (0 = off).
        self.baseline_every = int(baseline_every)

        if reward_mode not in ("gt", "reproj"):
            raise ValueError(f"reward_mode must be 'gt' or 'reproj', got {reward_mode!r}")
        if reward_mode == "reproj" and reproj_reward is None:
            raise ValueError("reward_mode='reproj' requires a ReprojectionReward instance")
        self._reproj = reproj_reward
        self._reproj_active = False

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim(self.predict_trans),), dtype=np.float32,
        )

        self._inner          = MoviEnv(device=device, keys=keys)
        self._x              = None
        self._y              = None
        self._t:   int       = 0
        self._T:   int       = 0
        self._prev_corrected = None
        self._state          = None

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

        if self.reward_mode == "reproj":
            self._reproj_active = self._reproj.reset(sample)

        obs = flatten_state(state).detach().cpu().numpy()
        return obs, {}

    # TODO: plot the delta updates and the corrected poses to see if they are reasonable to Tensorboard
    def step(self, action: np.ndarray):
        action_dict = unflatten_action(torch.from_numpy(action).float())
        next_lifted = {k: self._x[k][self._t + 1] for k in self.keys}

        self._state = self._inner.step(action_dict, next_lifted)
        corrected_t = {k: self._state["corrected_state"][k] for k in self.keys}

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

        self._t += 1
        terminated = self._t >= self._T - 1
        self._prev2_corrected = self._prev_corrected
        self._prev_corrected = {k: corrected_t[k].clone() for k in self.keys}

        obs = flatten_state(self._state).detach().cpu().numpy()
        return obs, reward, terminated, False, info

    def _reproj_step(self, corrected_t: dict) -> tuple[float, dict]:
        """Reprojection + smoothness, no ground truth involved."""
        from src.rewards import combine, smoothness_reward

        smooth = smoothness_reward(
            corrected_t["poses"].detach().cpu().numpy(),
            self._prev_corrected["poses"].detach().cpu().numpy(),
            self._prev2_corrected["poses"].detach().cpu().numpy(),
        )
        if self._reproj_active:
            r_proj, info = self._reproj.step(corrected_t, self._t)
        else:
            r_proj, info = float("nan"), {"valid": False}

        info["r_smooth"] = smooth
        info["r_reproj"] = r_proj

        if (self.baseline_every and self._reproj_active
                and self._t % self.baseline_every == 0):
            # self._x is the raw clip, so index _t is still the current frame even
            # though the inner env has already advanced its lifted_state to _t+1.
            lifted_t = {k: self._x[k][self._t] for k in self.keys}
            r_base, base = self._reproj.step(lifted_t, self._t)
            if base["valid"]:
                info["r_reproj_lifted"] = r_base
                info["err_px_lifted"] = base["err_px"]

        return combine(r_proj, smooth, self.w_reproj, self.w_smoothness), info
