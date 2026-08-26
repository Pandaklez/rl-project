"""
Plain supervised regression baseline: correct the lifted pose by direct MSE
regression against GT, with no RL, no discriminator, no reprojection reward.

This is the direct-regression counterpart to experiment (D). (D) adds an
MSE-to-GT term *inside a PPO reward* (`src/env.py::_apply_mse`); this instead
trains a plain feedforward regressor to minimise that same error by gradient
descent. It exists to separate two questions (D) could only answer jointly:
"is there learnable signal in lifted-pose -> GT-pose at all?" and "did PPO
recover it?" — a benchmark that skips PPO entirely answers the first question
on its own.

Deliberately the simplest architecture that can be compared against (B3),
the pose-only RL variant: a per-frame MLP of the same width
(`hidden_dims=(512, 256)`, matching `src.models.policy.PoseActor`), predicting
a residual correction to the lifted pose alone. No `corrected_{t-1}`
(that only exists in the RL setup because the policy consumes its own
previous output during a rollout — there is nothing analogous here), no
trans, no betas, no 2D evidence. A recurrent variant that sees the whole clip
is a natural follow-up, but is deliberately not this file: building it later,
behind its own flag, keeps "does supervision help" and "does temporal context
help" as two separable results instead of one experiment answering both badly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.discriminator import PoseSpace
from src.models.policy import N_JOINTS, POSE_DIM, _mlp


class SupervisedPoseRegressor(nn.Module):
    """
    forward(lifted_pose_norm) -> corrected_pose_norm, same shape as the input
    (accepts either `(..., N_JOINTS, 3)` or the flat `(..., POSE_DIM)`).

    The output lives in the same **lifted-per-camera-normalised** space the
    input does — i.e. the same space `src.models.policy.PoseActor`'s action
    lives in, not GT-normalised space. That matters for two call sites:
    `gt_space_mse` below, which is why the training loss has to remap before
    comparing to GT, and `src.evaluate.eval_supervised_model`, which
    unnormalises this output with the *lifted* stats for the same reason
    `eval_model` does for the RL policy.
    """

    def __init__(self, hidden_dims: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.net = _mlp([POSE_DIM, *hidden_dims, POSE_DIM])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        # Small final-layer gain, so training starts from ~zero correction
        # (the identity map) rather than a large random delta — the same
        # reasoning as PoseActor's near-zero action init (see
        # `src.models.policy.INIT_LOG_STD`), just applied to a mean instead
        # of a distribution.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lifted_pose_norm: torch.Tensor) -> torch.Tensor:
        joint_shaped = lifted_pose_norm.dim() >= 2 and lifted_pose_norm.shape[-1] == 3
        x = lifted_pose_norm.flatten(-2) if joint_shaped else lifted_pose_norm
        out = x + self.net(x)
        return out.unflatten(-1, (N_JOINTS, 3)) if joint_shaped else out


def gt_space_mse(corrected_norm: torch.Tensor, gt_norm: torch.Tensor,
                 camera, space: PoseSpace) -> torch.Tensor:
    """
    MSE between the model's output and GT, both mapped into GT-normalised
    space first.

    `corrected_norm` and `gt_norm` are flat `(..., POSE_DIM)` tensors in their
    *own* normalised spaces (lifted-per-camera and GT respectively — exactly
    how `data/processed_movi.h5` stores them, see `src/data/datasets.py`'s
    module docstring). Computing MSE on them directly, the way experiment
    (D)'s `r_mse` term does (`src/env.py::_apply_mse`), compares two different
    affine maps over the same 66 numbers — the exact mismatch `PoseSpace`
    exists to fix for the GAIL discriminator (`report.md`: a policy that
    satisfied that discriminator perfectly would land ~5.1 deg RMS / 31 deg on
    the root away from GT). That mismatch is tolerable inside a PPO reward
    baselined against `mse_lifted` computed the same mismatched way, because
    it then cancels to first order; it is not tolerable here, where the raw
    value is what gradient descent is driven toward. So both sides are mapped
    into `space`'s shared GT-normalised space before the difference is taken.

    Pass `space = PoseSpace(exclude_joints=())` to score all 22 joints — the
    GAIL discriminator's default excludes joint 0 for reasons specific to
    plausibility scoring that do not apply to a regression loss.
    """
    pred = space.fake(corrected_norm, camera)
    tgt  = space.real(gt_norm)
    return (pred - tgt).pow(2).mean()
