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

from src.models.policy import (ACTION_DIM_POSE_ONLY, ACTION_DIM_UV,
                               ACTION_DIM_WITH_TRANS, N_JOINTS, TRANS_MODES,
                               PoseActor, action_bounds, action_dim,
                               trans_mode_from_width, unflatten_action)

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

def test_action_dim_per_trans_mode():
    assert action_dim() == action_dim("none") == ACTION_DIM_POSE_ONLY == 66
    assert action_dim("uv") == ACTION_DIM_UV == 68
    assert action_dim("uvz") == ACTION_DIM_WITH_TRANS == 69


def test_width_and_mode_round_trip():
    for mode in TRANS_MODES:
        assert trans_mode_from_width(action_dim(mode)) == mode


def test_unflatten_rejects_other_widths():
    with pytest.raises(ValueError, match="none of"):
        unflatten_action(torch.zeros(157))


@pytest.mark.parametrize("width", [ACTION_DIM_POSE_ONLY, ACTION_DIM_UV,
                                   ACTION_DIM_WITH_TRANS])
def test_unflatten_shapes(width):
    out = unflatten_action(torch.zeros(width))
    assert out["poses"].shape == (N_JOINTS, 3)
    assert out["trans_delta"].shape == (3,)


def test_unflatten_batched():
    out = unflatten_action(torch.zeros(4, ACTION_DIM_UV))
    assert out["poses"].shape == (4, N_JOINTS, 3)
    assert out["trans_delta"].shape == (4, 3)


# ── the load-bearing property: depth is frozen structurally ──────────────────

def test_pose_only_gives_zero_trans_delta():
    action = torch.arange(ACTION_DIM_POSE_ONLY, dtype=torch.float32)
    assert torch.equal(unflatten_action(action)["trans_delta"], torch.zeros(3))


def test_uv_mode_freezes_log_depth_no_matter_what_the_policy_emits():
    """
    The whole point of `trans_mode="uv"`: the policy has no parameter for
    log-depth, so `dlog_tz` is zero for *every* action, not just small ones.
    Reprojection is nearly blind in depth, so a free depth dimension is a random
    walk in a direction the reward cannot correct.
    """
    for value in (0.0, 1.0, -50.0, 1e6):
        action = torch.full((ACTION_DIM_UV,), float(value))
        assert unflatten_action(action)["trans_delta"][2].item() == 0.0

    # ...and it is *not* frozen in the ablation, so the test is not vacuous
    action = torch.full((ACTION_DIM_WITH_TRANS,), 0.25)
    assert unflatten_action(action)["trans_delta"][2].item() == pytest.approx(0.25)


def test_du_dv_carry_through_in_order():
    action = torch.zeros(ACTION_DIM_WITH_TRANS)
    action[ACTION_DIM_POSE_ONLY:] = torch.tensor([0.1, -0.2, 0.3])
    assert unflatten_action(action)["trans_delta"].tolist() == pytest.approx([0.1, -0.2, 0.3])


def test_translation_dims_are_bounded_more_tightly_than_pose_dims():
    """A single box would either let the image shift run a tenth of the frame or
    squeeze the pose delta to the translation's scale."""
    lo, hi = action_bounds("uvz")
    assert hi[:ACTION_DIM_POSE_ONLY].min() == pytest.approx(0.3)
    assert hi[ACTION_DIM_POSE_ONLY:ACTION_DIM_POSE_ONLY + 2].max() == pytest.approx(0.1)
    assert hi[ACTION_DIM_POSE_ONLY + 2] == pytest.approx(0.1)
    assert (lo == -hi).all()


@needs_env
@pytest.mark.parametrize("width", [ACTION_DIM_POSE_ONLY, ACTION_DIM_UV,
                                   ACTION_DIM_WITH_TRANS])
def test_lifted_trans_passes_through_untouched_in_every_mode(width):
    """
    `corrected trans` must equal `lifted trans` bit for bit in every mode.

    The translation action is applied to the *metric* translation at projection
    time, not as a delta on the normalised virtual-camera `trans` — those are
    not metres and adding to them has no geometric meaning. Downstream FK and
    the observation must see the untouched value regardless of trans mode.
    """
    from src.env import MoviEnv

    env = MoviEnv()
    first, second = _frame(0), _frame(1)
    env.reset(first)

    rng = np.random.default_rng(7)
    action = torch.from_numpy(rng.normal(size=width).astype(np.float32))
    state = env.step(unflatten_action(action), second)

    assert torch.equal(state["corrected_state"]["trans"], first["trans"])
    # and the poses genuinely did move, so the test is not vacuous
    assert not torch.allclose(state["corrected_state"]["poses"], first["poses"])
    # the delta is carried alongside, for the reward to apply
    assert env.trans_delta.shape == (3,)


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

@pytest.mark.parametrize("trans_mode", TRANS_MODES)
def test_actor_head_width_and_state_dict_round_trip(trans_mode):
    actor = PoseActor(trans_mode=trans_mode)
    expected = action_dim(trans_mode)
    assert actor.log_std.shape == (expected,)

    out, _ = actor.act(torch.zeros(2, 138))
    assert out.shape == (2, expected)

    # evaluate.py infers the mode from log_std's width; check that inference is
    # right, because getting it wrong loads a policy that silently mis-acts.
    inferred = actor.state_dict()["log_std"].shape[0]
    assert trans_mode_from_width(inferred) == trans_mode
    PoseActor(trans_mode=trans_mode).load_state_dict(actor.state_dict())
