"""
Tests for the pose-only action layout (§2: drop absolute translation).

The property that matters is not the width itself but what the width *does*:
a pose-only action must leave the lifted translation exactly untouched, because
downstream FK and projection still consume it. If a trans delta ever leaked in,
the reward would silently move the body in depth and nothing else would fail.

Tests that touch `src.env` are skipped when `gymnasium` is not importable (it
lives in the `smplerx` env); the action-layout tests themselves run anywhere.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from src.models.policy import (ACTION_DIM_POSE_ONLY, ACTION_DIM_WITH_TRANS,
                               N_JOINTS, PoseActor, action_dim,
                               unflatten_action)

needs_env = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="src.env imports gymnasium (smplerx env)")


def _frame(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "poses": torch.from_numpy(rng.normal(size=(N_JOINTS, 3)).astype(np.float32)),
        "trans": torch.from_numpy(rng.normal(size=(3,)).astype(np.float32)),
    }


# ── widths ───────────────────────────────────────────────────────────────────

def test_action_dim_defaults_to_pose_only():
    assert action_dim() == ACTION_DIM_POSE_ONLY == 156
    assert action_dim(predict_trans=True) == ACTION_DIM_WITH_TRANS == 159


def test_unflatten_rejects_other_widths():
    with pytest.raises(ValueError, match="neither"):
        unflatten_action(torch.zeros(157))


@pytest.mark.parametrize("width", [ACTION_DIM_POSE_ONLY, ACTION_DIM_WITH_TRANS])
def test_unflatten_shapes(width):
    out = unflatten_action(torch.zeros(width))
    assert out["poses"].shape == (N_JOINTS, 3)
    assert out["trans"].shape == (3,)


def test_unflatten_batched():
    out = unflatten_action(torch.zeros(4, ACTION_DIM_POSE_ONLY))
    assert out["poses"].shape == (4, N_JOINTS, 3)
    assert out["trans"].shape == (4, 3)


# ── the load-bearing property ────────────────────────────────────────────────

def test_pose_only_action_gives_zero_trans_delta():
    action = torch.arange(ACTION_DIM_POSE_ONLY, dtype=torch.float32)
    assert torch.equal(unflatten_action(action)["trans"], torch.zeros(3))


@needs_env
def test_pose_only_passes_lifted_trans_through_untouched():
    """corrected trans must equal lifted trans, bit for bit, after a step."""
    from src.env import MoviEnv

    env = MoviEnv()
    first, second = _frame(0), _frame(1)
    env.reset(first)

    rng = np.random.default_rng(7)
    action = torch.from_numpy(
        rng.normal(size=ACTION_DIM_POSE_ONLY).astype(np.float32))
    state = env.step(unflatten_action(action), second)

    assert torch.equal(state["corrected_state"]["trans"], first["trans"])
    # and the poses genuinely did move, so the test is not vacuous
    assert not torch.allclose(state["corrected_state"]["poses"], first["poses"])


@needs_env
def test_with_trans_action_does_move_trans():
    """The ablation path still works — pose-only is a choice, not a removal."""
    from src.env import MoviEnv

    env = MoviEnv()
    first, second = _frame(0), _frame(1)
    env.reset(first)

    action = torch.zeros(ACTION_DIM_WITH_TRANS)
    action[ACTION_DIM_POSE_ONLY:] = 0.5
    state = env.step(unflatten_action(action), second)

    assert torch.allclose(state["corrected_state"]["trans"], first["trans"] + 0.5)


# ── GT-mode reward keys ──────────────────────────────────────────────────────

@needs_env
def test_gt_reward_ignores_trans_when_pose_only():
    """
    With trans excluded, a clip whose translation is wildly off must score the
    same as one whose translation is perfect — the policy cannot affect it.
    """
    from src.env import compute_reward

    corrected, gt, prev = _frame(0), _frame(1), _frame(2)
    gt["poses"] = corrected["poses"].clone()

    near = {**gt, "trans": corrected["trans"].clone()}
    far  = {**gt, "trans": corrected["trans"] + 40.0}

    pose_only = dict(keys=("poses",))
    assert compute_reward(corrected, near, prev, **pose_only) == pytest.approx(
        compute_reward(corrected, far, prev, **pose_only))

    # the default (with trans) does notice the difference
    assert compute_reward(corrected, near, prev) != pytest.approx(
        compute_reward(corrected, far, prev))


# ── checkpoint round-trip ────────────────────────────────────────────────────

@pytest.mark.parametrize("predict_trans", [False, True])
def test_actor_head_width_and_state_dict_round_trip(predict_trans):
    actor = PoseActor(predict_trans=predict_trans)
    expected = action_dim(predict_trans)
    assert actor.log_std.shape == (expected,)

    out, _ = actor.act(torch.zeros(2, 318))
    assert out.shape == (2, expected)

    # evaluate.py infers the width from log_std; check that inference is right
    inferred = actor.state_dict()["log_std"].shape[0]
    assert (inferred == ACTION_DIM_WITH_TRANS) == predict_trans
    PoseActor(predict_trans=predict_trans).load_state_dict(actor.state_dict())
