from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

N_JOINTS  = 52
POSE_DIM  = N_JOINTS * 3   # 156
TRANS_DIM = 3
BETAS_DIM = 16
FRAME_DIM = POSE_DIM + TRANS_DIM        # 159 — one frame of poses + trans
STATE_DIM = 2 * FRAME_DIM               # 318 — lifted_t concat corrected_{t-1}


def _mlp(
    dims: list[int],
    activation: type[nn.Module] = nn.ELU,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def flatten_state(state: dict, use_betas: bool = False) -> torch.Tensor:
    """
    Convert the env state dict to a flat tensor suitable for the policy.

    Expected state structure (from MoviEnv):
        state["lifted_state"]["poses"]    (..., 52, 3)
        state["lifted_state"]["trans"]    (..., 3)
        state["corrected_state"]["poses"] (..., 52, 3)
        state["corrected_state"]["trans"] (..., 3)

    Returns tensor of shape (..., STATE_DIM) or (..., STATE_DIM + BETAS_DIM).
    """
    lifted    = state["lifted_state"]
    corrected = state["corrected_state"]
    parts = [
        lifted["poses"].flatten(-2),      # (..., 156)
        lifted["trans"],                   # (..., 3)
        corrected["poses"].flatten(-2),    # (..., 156)
        corrected["trans"],                # (..., 3)
    ]
    if use_betas:
        parts.append(lifted["betas"])      # (..., 16)
    return torch.cat(parts, dim=-1)


def unflatten_action(action: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Split a flat action vector back into the dict format expected by MoviEnv.step().

    action: (..., FRAME_DIM)  →  {"poses": (..., 52, 3), "trans": (..., 3)}
    """
    poses_flat = action[..., :POSE_DIM]
    trans      = action[..., POSE_DIM:]
    return {
        "poses": poses_flat.unflatten(-1, (N_JOINTS, 3)),
        "trans": trans,
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
        dropout:     float         = 0.0,
        init_log_std: float        = INIT_LOG_STD,
    ):
        super().__init__()
        state_dim    = STATE_DIM + (BETAS_DIM if use_betas else 0)
        self.net     = _mlp([state_dim, *hidden_dims, FRAME_DIM], dropout=dropout)
        self.log_std = nn.Parameter(torch.full((FRAME_DIM,), float(init_log_std)))
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
        dropout:     float         = 0.0,
    ):
        super().__init__()
        state_dim = STATE_DIM + (BETAS_DIM if use_betas else 0)
        self.net  = _mlp([state_dim, *hidden_dims, 1], dropout=dropout)
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
        dropout:     float           = 0.0,
        init_log_std: float          = INIT_LOG_STD,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
        )
        self.net     = _mlp([self.num_observations, *hidden_dims, self.num_actions], dropout=dropout)
        self.log_std = nn.Parameter(torch.full((self.num_actions,), float(init_log_std)))
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
        dropout:     float           = 0.0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _mlp([self.num_observations, *hidden_dims, 1], dropout=dropout)
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
