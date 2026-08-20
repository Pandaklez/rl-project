"""
The pose-only state variant.

`NoTransMoviEnv` removes `trans` from the observation but keeps it inside the
env, because the reprojection reward still needs a translation to place the body
in the camera. Three things have to hold, and none of them fails loudly:

1. The observation width drops by exactly the translation, and every rollout
   path agrees on the new number.
2. The translation handed to the reward is the one belonging to the frame that
   was just corrected — not the next frame the clip advanced to. Getting that
   wrong scores frame t's pose against frame t+1's translation, which at walking
   speed is a few centimetres of silent error in the reward.
3. The reward is *unchanged* by dropping trans from the state. That is what
   makes this variant comparable with the trans-in-state runs at all: if the
   reward moved, the three experiments would not be measuring the same thing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

needs_env = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="src.env imports gymnasium (smplerx env)")

REPO = Path(__file__).resolve().parent.parent
H5 = REPO / "data" / "processed_movi.h5"
TARGETS = REPO / "data" / "reproj_targets.h5"
needs_data = pytest.mark.skipif(
    not (H5.exists() and TARGETS.exists()),
    reason="needs data/processed_movi.h5 and data/reproj_targets.h5")

N_JOINTS, T = 52, 8


def _frame(seed):
    rng = np.random.default_rng(seed)
    return {"poses": torch.from_numpy(rng.normal(0, .3, (N_JOINTS, 3)).astype(np.float32)),
            "trans": torch.from_numpy(rng.normal(0, 1., (3,)).astype(np.float32))}


# ── widths ───────────────────────────────────────────────────────────────────

def test_state_dim_drops_exactly_the_translation():
    from src.models.policy import TRANS_DIM, state_dim

    for evidence in (False, True):
        with_trans = state_dim(evidence, False, True)
        without = state_dim(evidence, False, False)
        assert with_trans - without == 2 * TRANS_DIM      # lifted + corrected


def test_the_four_widths_are_what_the_docstring_says():
    from src.models.policy import state_dim

    assert state_dim(False, False, True) == 318
    assert state_dim(True, False, True) == 362
    assert state_dim(False, False, False) == 312
    assert state_dim(True, False, False) == 356


@needs_env
def test_flatten_state_accepts_both_state_shapes():
    from src.models.policy import flatten_state

    poses = torch.zeros(N_JOINTS, 3)
    bare = {"lifted_state": poses, "corrected_state": poses.clone()}
    dicts = {"lifted_state": {"poses": poses, "trans": torch.zeros(3)},
             "corrected_state": {"poses": poses.clone(), "trans": torch.zeros(3)}}
    assert flatten_state(bare).shape == (312,)
    assert flatten_state(dicts).shape == (318,)


# ── the timing invariant ─────────────────────────────────────────────────────

@needs_env
def test_corrected_trans_belongs_to_the_corrected_frame():
    """
    The bug this exists to catch: advancing the clip before handing the
    translation to the reward, so pose t is scored against translation t+1.
    """
    from src.env import NoTransMoviEnv

    env = NoTransMoviEnv()
    f0, f1, f2 = _frame(0), _frame(1), _frame(2)
    env.reset(f0)
    assert torch.equal(env.corrected_trans, f0["trans"])
    assert torch.equal(env.lifted_trans, f0["trans"])

    act = {"poses": torch.zeros(N_JOINTS, 3), "trans_delta": torch.zeros(3)}
    env.step(act, f1)
    # the pose just corrected is frame 0's, so its translation must be frame 0's
    assert torch.equal(env.corrected_trans, f0["trans"])
    assert torch.equal(env.lifted_trans, f1["trans"])

    env.step(act, f2)
    assert torch.equal(env.corrected_trans, f1["trans"])
    assert torch.equal(env.lifted_trans, f2["trans"])


@needs_env
def test_reset_starts_at_the_identity_correction_not_the_mean_pose():
    from src.env import NoTransMoviEnv

    f0 = _frame(3)
    state = NoTransMoviEnv().reset(f0)
    assert torch.equal(state["corrected_state"], f0["poses"])
    assert not torch.equal(state["corrected_state"], torch.zeros_like(f0["poses"]))


@needs_env
def test_action_moves_poses_and_nothing_else_is_carried():
    from src.env import NoTransMoviEnv

    env = NoTransMoviEnv()
    f0, f1 = _frame(4), _frame(5)
    env.reset(f0)
    delta = torch.full((N_JOINTS, 3), 0.05)
    state = env.step({"poses": delta, "trans_delta": torch.zeros(3)}, f1)
    assert torch.allclose(state["corrected_state"], f0["poses"] + delta)
    assert torch.equal(state["lifted_state"], f1["poses"])
    # the state is a bare tensor -- there is no trans in it to leak
    assert isinstance(state["corrected_state"], torch.Tensor)


# ── the invariant that makes the three variants comparable ───────────────────

@needs_env
@needs_data
def test_dropping_trans_from_the_state_does_not_change_the_reward():
    """
    Same clip, same actions, both state layouts: the reward must match exactly.

    If it did not, the pose-only run could not be reported next to the
    trans-in-state runs, because the three would be optimising different things.
    """
    from src.data.datasets import MoViDataset
    from src.env import GymMoviEnv
    from src.rewards import ReprojectionReward, load_calib

    ds = MoViDataset(str(H5), str(REPO / "data/normalization.json"), split="val",
                     verbose=False, reproj_path=str(TARGETS))
    calib = load_calib(str(H5))
    rng = np.random.default_rng(0)
    actions = rng.normal(0, 0.02, (60, 156)).astype(np.float32)

    rewards = {}
    for state_trans in (True, False):
        env = GymMoviEnv(ds, device="cpu", reward_mode="reproj",
                         reproj_reward=ReprojectionReward(
                             calib, sigma=0.0225, correct_translation=False, bias=None),
                         trans_mode="none", state_trans=state_trans)
        env.reset(seed=0)
        got = []
        for a in actions:
            _, r, term, trunc, _ = env.step(a)
            got.append(r)
            if term or trunc:
                break
        rewards[state_trans] = np.array(got)

    assert len(rewards[True]) == len(rewards[False])
    np.testing.assert_allclose(rewards[True], rewards[False], rtol=0, atol=0)
