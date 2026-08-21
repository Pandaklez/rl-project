from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# 22 SMPL-X body joints (0 global orient, 1-21 body). Joints 22-51 are the two
# hands and are dropped in data/norm_upsample.py -- MoVi has no finger mocap, so
# in GT they are a constant that normalises to exactly +/-1 and acts as a
# "GT or lifted" label for the discriminator, while contributing 91.6% of the
# smoothness reward's acceleration energy and nothing to any metric.
N_JOINTS  = 22
POSE_DIM  = N_JOINTS * 3   # 66
TRANS_DIM = 3
BETAS_DIM = 16
FRAME_DIM = POSE_DIM + TRANS_DIM        # 159 — one frame of poses + trans
STATE_DIM = 2 * FRAME_DIM               # 138 — lifted_t concat corrected_{t-1}

# ...and the same state without translation. The lifted `trans` is SMPLer-X's
# virtual-camera depth, not metres, so it is not a quantity the policy can
# interpret: it cannot be corrected (that is why the action dropped it) and it
# is not in the same space as anything else in the observation. Carrying it in
# the state only offers the network 3 numbers whose units it has no way to
# learn. `state_trans=False` removes it; the value is still kept inside the env
# and handed to the reprojection reward, which *does* need it.
FRAME_DIM_NO_TRANS = POSE_DIM           # 66
STATE_DIM_NO_TRANS = 2 * FRAME_DIM_NO_TRANS   # 132

# Width of the 2D-evidence block (src/rewards.py owns the layout). Imported by
# value rather than re-derived so the two cannot drift apart.
from src.rewards import EVIDENCE_DIM     # noqa: E402  (44)


def state_dim(use_evidence: bool = False, use_betas: bool = False,
              state_trans: bool = True) -> int:
    """
    Observation width.

    Four combinations, all of which have to line up between the trainer,
    `src/viz_pose.py` and `src/evaluate.py` — a policy fed the wrong width does
    not raise, it silently produces nonsense:

        state_trans=True,  use_evidence=False   318   the original
        state_trans=True,  use_evidence=True    362   + the 2D residual block
        state_trans=False, use_evidence=False   312   pose-only state
        state_trans=False, use_evidence=True    356   pose-only + residual

    See `src/rewards.pack_evidence` for why the evidence block is the difference
    between a policy that can learn and one whose optimum is the identity, and
    `STATE_DIM_NO_TRANS` for why dropping `trans` from the state costs nothing.
    """
    base = STATE_DIM if state_trans else STATE_DIM_NO_TRANS
    return base + (EVIDENCE_DIM if use_evidence else 0) + (BETAS_DIM if use_betas else 0)


# Bound on one correction delta, in normalised pose units (fix 03).
#
# The action is a residual around an already-good pose: the policy starts at
# sigma = 0.05 (INIT_LOG_STD below), so 0.3 is 6 sigma and further than any
# plausible correction — a joint moved 0.3 normalised units is moved a third of
# the dataset's spread for that joint. It is a guard rail, not a constraint that
# should ever bind during healthy training; if it binds, the run is diverging,
# which is exactly the case where an unbounded Box let 4.1M steps of random walk
# run off to a NaN pose instead of being caught.
ACTION_LIMIT = 0.3

# Action layout. The *state* always carries both poses and trans; only the
# action shrinks.
#
# §2 of the handover: the lifted `trans` is not a translation in metres. It is
# SMPLer-X's cam_trans in a cropped-bbox virtual camera with a 5000 px focal, so
# a subject 4.5 m away is stored at z ~= 42. There is no rigid transform from it
# to the room, and it is the field that sits at ~57% in val_rmse and does not
# improve from pose work. Predicting a correction to it asks the policy to
# regress a quantity whose units it cannot see, and PA-MPJPE is Procrustes-
# aligned so the metric never depended on it either.
#
# Pose-only is therefore the default: 66 dims, no trans delta. `trans` stays in
# the observation, because apparent size in the image is genuine information
# about depth even when the number is not metric.
ACTION_DIM_POSE_ONLY  = POSE_DIM        # 66 — poses only (default)
ACTION_DIM_UV         = POSE_DIM + 2    # 68 — poses + image-plane shift
ACTION_DIM_WITH_TRANS = POSE_DIM + 3    # 69 — poses + shift + log-depth

# ── Translation action: reparameterised into what the reward can actually see ─
#
# The old 159-d action added a delta to the lifted `trans`, i.e. to SMPLer-X's
# cam_trans in a cropped-bbox virtual camera with a 5000 px focal. Three
# problems, all fatal: the units are not metres, the mapping to the image is not
# a translation, and the depth component is the direction the reprojection
# reward is *blind* in. Letting the policy move it is letting it random-walk in
# an unobservable direction — which is exactly the failure §5 predicts, since GT
# scores no better than the lifted pose on 2D evidence precisely because the
# residual error lives in depth.
#
# The reparameterisation splits the translation into the coordinates the reward
# is sensitive in, and freezes the one it is not:
#
#   du, dv     image-plane shift in bbox-height units. Moves the projection
#              one-for-one, and is exactly what the mean of the observed
#              residual reports — so the policy reads the residual and writes
#              back its negation. Near-invertible.
#   dlog_tz    log-depth, applied along the viewing ray so u, v are unchanged
#              and only apparent size moves. Reprojection barely responds to it.
#
# `trans_mode="uv"` omits dlog_tz from the action *by width*, so depth drift is
# structurally impossible rather than merely discouraged by a reward term.
# `trans_mode="uvz"` unfreezes it: one extra scalar, a clean ablation with an
# obvious hypothesis — if depth is unobservable, unfreezing it should degrade
# 3D accuracy while barely moving the 2D reward.
TRANS_MODE_NONE = "none"
TRANS_MODE_UV   = "uv"
TRANS_MODE_UVZ  = "uvz"
TRANS_MODES     = (TRANS_MODE_NONE, TRANS_MODE_UV, TRANS_MODE_UVZ)

_TRANS_WIDTH = {
    TRANS_MODE_NONE: ACTION_DIM_POSE_ONLY,
    TRANS_MODE_UV:   ACTION_DIM_UV,
    TRANS_MODE_UVZ:  ACTION_DIM_WITH_TRANS,
}
_WIDTH_TRANS = {v: k for k, v in _TRANS_WIDTH.items()}

# Bound on the image-plane shift, in bbox-height units. The measured operating
# point of the reprojection error is 0.028, so 0.1 is ~3.5x any offset a sane
# correction needs — generous, and still finite.
TRANS_UV_LIMIT = 0.1
# Bound on dlog_tz: +-10% in depth. Only reachable in the "uvz" ablation.
TRANS_LOGZ_LIMIT = 0.1


def action_dim(trans_mode: str = TRANS_MODE_NONE) -> int:
    """Width of the policy's action vector."""
    if isinstance(trans_mode, bool):        # legacy predict_trans=True/False
        trans_mode = TRANS_MODE_UVZ if trans_mode else TRANS_MODE_NONE
    if trans_mode not in _TRANS_WIDTH:
        raise ValueError(f"trans_mode must be one of {TRANS_MODES}, got {trans_mode!r}")
    return _TRANS_WIDTH[trans_mode]


def trans_mode_from_width(width: int) -> str:
    """Which translation parameterisation an action of this width carries."""
    try:
        return _WIDTH_TRANS[int(width)]
    except KeyError:
        raise ValueError(
            f"action width {width} is none of {ACTION_DIM_POSE_ONLY} (poses only), "
            f"{ACTION_DIM_UV} (poses + du,dv) or {ACTION_DIM_WITH_TRANS} "
            f"(poses + du,dv,dlog_tz)") from None


def action_bounds(trans_mode: str = TRANS_MODE_NONE):
    """
    Per-dimension (low, high) for the action space.

    The pose dims and the translation dims are on different scales and get
    different bounds — a single box would either let the image shift run to 0.3
    bbox heights (a tenth of the frame) or squeeze the pose delta to 0.1.
    """
    import numpy as np

    width = action_dim(trans_mode)
    low = np.full(width, -ACTION_LIMIT, dtype=np.float32)
    high = np.full(width, ACTION_LIMIT, dtype=np.float32)
    if width > POSE_DIM:
        low[POSE_DIM:POSE_DIM + 2] = -TRANS_UV_LIMIT
        high[POSE_DIM:POSE_DIM + 2] = TRANS_UV_LIMIT
    if width > POSE_DIM + 2:
        low[POSE_DIM + 2] = -TRANS_LOGZ_LIMIT
        high[POSE_DIM + 2] = TRANS_LOGZ_LIMIT
    return low, high


def _mlp(dims: list[int], activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    """
    Plain MLP. **There is deliberately no dropout option** (fix 01).

    Dropout here was the cause of the (B) divergence. skrl runs the rollout under
    `set_mode("eval")` and the update under `set_mode("train")`
    (`skrl/agents/torch/ppo/ppo.py:202,344`), so `log_prob_old` was recorded from
    a clean network and `log_prob_new` recomputed from a perturbed one. The
    importance ratio then measured dropout noise rather than a policy change:
    median ratio 0.48, and 88.9% of transitions clipped on the very first update,
    before any learning had happened. The measured mean/std of the log-ratio
    matched `N(-|dmu|^2/2s^2, |dmu|^2/s^2)` exactly, and both terms shrink with
    sigma, so PPO's clip penalty also drove sigma upward until the policy
    produced a NaN pose at 2.77M steps.

    Independent of the skrl mode switch, dropout in a policy network is simply
    wrong: it makes the behaviour policy stochastic in a way the stored log-prob
    does not model. Removing the parameter rather than defaulting it to 0 is the
    point — a default can be passed over.
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def flatten_state(state: dict, use_betas: bool = False, evidence=None) -> torch.Tensor:
    """
    Convert the env state dict to a flat tensor suitable for the policy.

    Expected state structure (from MoviEnv):
        state["lifted_state"]["poses"]    (..., 22, 3)
        state["lifted_state"]["trans"]    (..., 3)
        state["corrected_state"]["poses"] (..., 22, 3)
        state["corrected_state"]["trans"] (..., 3)

    Accepts either state shape. `MoviEnv` stores each slot as a
    `{"poses", "trans"}` dict; `NoTransMoviEnv` stores a bare pose tensor and
    keeps the translation to itself. Which one it is follows from the type, so
    no caller has to pass a flag.

    `evidence` is the EVIDENCE_DIM block from `src.rewards.pack_evidence`: the
    2D residual the reward is computed from, plus the context needed to act on
    it. Passing it is what makes the reward observable at all — see fix 06.

    **Every rollout path must agree on this.** Training, `src/viz_pose.py` and
    `src/evaluate.py` all build observations through this function, and a policy
    fed 318 numbers when it was trained on 362 does not fail loudly, it just
    produces nonsense. `state_dim()` is the single source of truth for the width.

    Returns a tensor of shape (..., state_dim(...)).
    """
    lifted    = state["lifted_state"]
    corrected = state["corrected_state"]
    if isinstance(lifted, torch.Tensor):
        # Pose-only state (NoTransMoviEnv): the two slots are (..., 22, 3)
        # tensors rather than dicts, because there is no second key to hold.
        parts = [lifted.flatten(-2), corrected.flatten(-2)]
    else:
        parts = [
            lifted["poses"].flatten(-2),      # (..., 66)
            lifted["trans"],                   # (..., 3)
            corrected["poses"].flatten(-2),    # (..., 66)
            corrected["trans"],                # (..., 3)
        ]
    if use_betas:
        parts.append(lifted["betas"])      # (..., 16)
    if evidence is not None:
        if not isinstance(evidence, torch.Tensor):
            evidence = torch.as_tensor(evidence, dtype=torch.float32)
        ref = parts[0]
        parts.append(evidence.to(device=ref.device, dtype=ref.dtype)
                     .reshape(*ref.shape[:-1], EVIDENCE_DIM))
    return torch.cat(parts, dim=-1)


def extract_corrected_pose(state: torch.Tensor, state_trans: bool = True) -> torch.Tensor:
    """
    Slice the corrected-pose block back out of a flat observation built by
    `flatten_state`.

    `flatten_state`'s layout is fixed regardless of `use_evidence` (the
    evidence block, if any, is appended last), so only `state_trans` — whether
    `trans` occupies the two 3-wide gaps — moves the offset:

        state_trans=True:   [lifted(66), lifted_trans(3), corrected(66), corrected_trans(3), ...]
        state_trans=False:  [lifted(66), corrected(66), ...]

    Exists so the GAIL discriminator (`src/gail_train.py::PPOWithGAIL`) can
    read the policy's corrected poses straight out of skrl's on-policy memory
    instead of re-deriving them from the env: row t of the stored `states`
    tensor carries corrected_{t-1} — the correction the *previous* action
    produced (see `flatten_state`). Not every row is a genuine policy output
    though: the first row of each episode carries `reset()`'s identity
    placeholder instead (`corrected_state = lifted_state.clone()`, before any
    action exists) — `PPOWithGAIL._update_discriminator` masks those rows out
    rather than treating this function's output as always trustworthy. One
    definition here rather than duplicating the offset arithmetic at the call
    site keeps it from drifting out of sync with `flatten_state` the way
    `state_dim` is kept in sync everywhere else.
    """
    off = POSE_DIM + (TRANS_DIM if state_trans else 0)
    return state[..., off:off + POSE_DIM]


def unflatten_action(action: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Split a flat action vector into the parts `MoviEnv.step` consumes.

    Which layout it is follows from the width itself rather than from a flag the
    caller has to keep in sync:

        (..., 66)  poses only            -> trans_delta is exactly zero
        (..., 158)  poses + du, dv        -> dlog_tz is exactly zero (frozen)
        (..., 159)  poses + du, dv, dlog  -> all three free (ablation)

    Returns `{"poses": (..., 22, 3), "trans_delta": (..., 3)}`.

    `trans_delta` is **not** a delta on the lifted `trans`. It is
    `(du, dv, dlog_tz)`: an image-plane shift in bbox-height units and a
    log-depth change along the viewing ray, applied to the *metric* translation
    at projection time (`src/rewards.py::_project_frame`). The lifted `trans`
    itself is always passed through untouched, so nothing downstream has to know
    which mode is in force.

    Zero-padding the frozen dimensions rather than branching means the reward
    path sees the same 3-vector shape in every mode, and "frozen" is expressed
    as a structural zero that no gradient can reach — the policy has no
    parameter for it at all.
    """
    width = action.shape[-1]
    mode = trans_mode_from_width(width)
    poses_flat = action[..., :POSE_DIM]
    n_free = width - POSE_DIM
    trans_delta = action.new_zeros((*action.shape[:-1], 3))
    if n_free:
        trans_delta[..., :n_free] = action[..., POSE_DIM:]
    return {
        "poses": poses_flat.unflatten(-1, (N_JOINTS, 3)),
        "trans_delta": trans_delta,
        "trans_mode": mode,
    }


# Initial log standard deviation of the action distribution.
#
# Actions are correction *deltas* in normalised pose units, so std = 1.0 (the
# old `zeros` init) perturbs every joint by a full standard deviation of the
# pose distribution on every frame. Against the reprojection reward that lands
# in the saturated corner of exp(-e²/σ²), where there is almost no gradient and
# the first thing the policy has to learn is to undo its own initialisation.
#
# Measured on the train split (reward σ = 0.04):
#
#     action std   err_px   r_reproj   r_smooth   combined
#       0.00        15.5     0.584      0.936      0.678     <- identity
#       0.05        15.7     0.578      0.927      0.671
#       0.20        18.2     0.479      0.799      0.559
#       1.00        49.8     0.034      0.016      0.036     <- old default
#
# log(0.05) ~= -3.0 starts the policy as a small perturbation around the lifted
# pose, in the responsive part of the reward, which is what a residual policy
# wants. PPO learns log_std from there, so this is a starting point and not a
# ceiling.
INIT_LOG_STD = -3.0

# ...and the same number is *wrong* for the translation dimensions, because they
# are not in pose units.
#
# `du`/`dv` are in bbox-height units, where the reward's whole operating point is
# err = 0.028 and its sigma is 0.04. Initialising them at 0.05 as well makes the
# mean sampled shift `0.05·sqrt(2/pi)` = 0.040 bbox heights — 18.4 px on a 464 px
# bbox, larger than the 13 px error the policy is being asked to remove.
#
# Measured, in a run that shipped with the shared value: `du |abs|` settled at
# 0.0398 and `dv |abs|` at 0.0434 against a predicted 0.0397, i.e. the
# translation action was pure sampling noise and nothing had been learned; the
# corrected error sat at 32.9 px against a lifted 13.9 px, the difference being
# exactly the injected noise. The held-out figure, which uses the policy *mean*
# and so carries no sampling noise, was simultaneously the best of any run —
# which is what says the mechanism is sound and only its scale was wrong.
#
# 0.005 bbox heights is ~2.3 px: small against the 13 px operating point, and
# still two orders of magnitude above float noise, so there is a gradient.
INIT_LOG_STD_TRANS = -5.3           # log(0.005)


def init_log_std_vector(trans_mode: str = TRANS_MODE_NONE,
                        pose_value: float = INIT_LOG_STD,
                        trans_value: float = INIT_LOG_STD_TRANS):
    """
    Per-dimension initial log-sigma.

    A single scalar cannot be right for both halves of this action: the pose
    dims and the translation dims are in different units, and the sigma that is
    a small perturbation in one is a destructive one in the other. Same reason
    `action_bounds` is per-dimension.

    PPO learns `log_std` from here, so these are starting points, not ceilings.
    """
    import torch as _torch

    width = action_dim(trans_mode)
    v = _torch.full((width,), float(pose_value))
    v[POSE_DIM:] = float(trans_value)
    return v


# ─── Legacy models (used by src/evaluate.py) ──────────────────────────────────

class PoseActor(nn.Module):
    """
    Stochastic Gaussian actor for PPO pose correction.

    state  : flat vector of shape (..., STATE_DIM [+ BETAS_DIM])
    action : correction delta, shape (..., FRAME_DIM)
    """

    def __init__(
        self,
        use_betas:   bool          = False,
        hidden_dims: tuple[int, ...] = (512, 256),
        init_log_std: float        = INIT_LOG_STD,
        trans_mode:  str           = TRANS_MODE_NONE,
        use_evidence: bool         = False,
        state_trans: bool          = True,
    ):
        super().__init__()
        obs_dim      = state_dim(use_evidence=use_evidence, use_betas=use_betas,
                                 state_trans=state_trans)
        act_dim      = action_dim(trans_mode)
        self.net     = _mlp([obs_dim, *hidden_dims, act_dim])
        self.log_std = nn.Parameter(init_log_std_vector(trans_mode, init_log_std))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def distribution(self, state: torch.Tensor) -> Normal:
        mean = self.net(state)
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist     = self.distribution(state)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist     = self.distribution(state)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy


class PoseCritic(nn.Module):
    """State-value network V(s) for PPO."""

    def __init__(
        self,
        use_betas:   bool          = False,
        hidden_dims: tuple[int, ...] = (512, 256),
        use_evidence: bool         = False,
        state_trans: bool          = True,
    ):
        super().__init__()
        obs_dim  = state_dim(use_evidence=use_evidence, use_betas=use_betas,
                             state_trans=state_trans)
        self.net = _mlp([obs_dim, *hidden_dims, 1])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ─── skrl models ──────────────────────────────────────────────────────────────

class SkrlPoseActor(GaussianMixin, Model):
    """
    Gaussian policy for skrl PPO.

    The internal MLP + log_std share the same names as PoseActor so that
    evaluate.py can load a skrl-saved policy.pt into PoseActor directly:
        actor.load_state_dict(torch.load("policy.pt"))
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_dims: tuple[int, ...] = (512, 256),
        init_log_std: float          = INIT_LOG_STD,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            # Fix 03. With a finite action_space this clips the sampled delta to
            # +-ACTION_LIMIT. The log-prob is still the unclipped Gaussian's,
            # which is the standard clipped-Gaussian treatment and is fine here
            # because at sigma = 0.05 the bound is 6 sigma away and effectively
            # never binds — it exists to stop a diverging run, not to shape a
            # healthy one.
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-20.0,
            # Was 2.0, i.e. sigma up to 7.4 in normalised pose units, which is
            # not a policy so much as noise. Tied to ACTION_LIMIT instead: once
            # sigma reaches the clip bound the distribution is wider than the
            # range it is allowed to act in, so nothing above that is a policy.
            # The diverging run reached 0.118 and was still climbing.
            max_log_std=math.log(ACTION_LIMIT),
            reduction="sum",
        )
        self.net     = _mlp([self.num_observations, *hidden_dims, self.num_actions])
        self.log_std = nn.Parameter(
            init_log_std_vector(trans_mode_from_width(self.num_actions), init_log_std))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std, {}


class SkrlPoseCritic(DeterministicMixin, Model):
    """Value network V(s) for skrl PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_dims: tuple[int, ...] = (512, 256),
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _mlp([self.num_observations, *hidden_dims, 1])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
