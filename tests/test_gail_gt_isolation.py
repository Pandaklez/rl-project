"""
The two routes by which GT can reach the policy in experiment (C), and the
guards that close them.

Experiment (C) is "(B) plus a discriminator". (B)'s reward is ground-truth-free,
and the discriminator is meant to be the *only* place GT enters the loop — and
to enter it distributionally, as a bank of unpaired GT poses, never as this
clip's label at this frame. Two things break that, and neither fails loudly:

1. **A second reward channel.** `reward_mode="gt"` is the supervised frame-wise
   similarity term: it reads the GT for the very frame being scored. Combined
   with an active discriminator it leaks GT through a dense channel alongside
   the sparse adversarial one, and the run's curves answer a different question
   than (C) asks. Guarded in `GymMoviEnv.__init__` rather than in a trainer,
   because the env is what reads GT and several entry points build it directly.

2. **A shared clip set.** If the discriminator's real bank is drawn from the
   same clips the policy rolls out, it can memorise a clip's GT and then reward
   the policy for reproducing the ground truth of the clip it is correcting.
   That is not the paired per-frame leak (1) covers — the sampler never lines a
   real sample up with the fake one — but it is still a route from a clip's
   label to that same clip's reward, and it is the one AMP and vanilla GAIL
   close by keeping the demonstration set disjoint.

The `split_demo_clips` tests need `data/processed_movi.h5`; the guard tests do
not touch data at all.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

H5 = Path(__file__).resolve().parent.parent / "data" / "processed_movi.h5"

needs_skrl = pytest.mark.skipif(
    importlib.util.find_spec("skrl") is None
    or importlib.util.find_spec("gymnasium") is None,
    reason="src.gail_env imports gymnasium; src.gail_train imports skrl")
needs_h5 = pytest.mark.skipif(not H5.exists(), reason=f"{H5} not present")


def _provider():
    """A provider in the shape a real run builds one: the discriminator's width
    follows the space, and the space is mandatory (see `PoseSpace`)."""
    from src.models.discriminator import (GAILRewardProvider, MotionDiscriminator,
                                          PoseSpace)
    space = PoseSpace(str(H5.parent / "normalization.json"),
                      str(H5.parent / "normalization_lifted_{cam}.json"))
    return GAILRewardProvider(MotionDiscriminator(pose_dim=space.dim), space=space)


# ── 1. the second-reward-channel guard ──────────────────────────────────────

@needs_skrl
@needs_h5
def test_gt_reward_with_active_discriminator_is_rejected():
    from src.gail_env import GymMoviEnv

    with pytest.raises(ValueError, match="two channels at once"):
        GymMoviEnv(dataset=None, reward_mode="gt", w_gail=0.5,
                   disc_reward=_provider())


@needs_skrl
@needs_h5
def test_gt_reward_with_silent_discriminator_is_allowed():
    """The infra smoke test: discriminator built and trained, but contributing
    nothing to the reward. `w_gail=0` is also a legitimate ablation value, so
    this must not be collateral damage of the guard."""
    from src.gail_env import GymMoviEnv

    GymMoviEnv(dataset=None, reward_mode="gt", w_gail=0.0, disc_reward=_provider())


@needs_skrl
def test_gt_reward_without_a_discriminator_is_allowed():
    """Experiments (A)/(B) and the labelled ablation are untouched."""
    from src.gail_env import GymMoviEnv

    GymMoviEnv(dataset=None, reward_mode="gt", w_gail=0.5, disc_reward=None)


@needs_skrl
def test_the_guard_mirrors_the_step_time_test_for_an_active_term():
    """The guard's condition and `step()`'s own "is the GAIL term active" test
    must stay the same expression, or the guard starts protecting a different
    composition than the one that runs."""
    import inspect

    from src.gail_env import GymMoviEnv

    src = inspect.getsource(GymMoviEnv)
    active = "self._disc_reward is not None and self.w_gail"
    assert src.count(active) >= 2, (
        "step() and __init__'s guard should share the activity test verbatim")


# ── 2. the disjoint demonstration set ───────────────────────────────────────

@needs_h5
def test_demo_and_policy_clips_are_disjoint():
    from src.models.discriminator import split_demo_clips

    policy, demo = split_demo_clips(str(H5), "train", demo_frac=0.2, seed=0)
    assert policy and demo
    assert not set(policy) & set(demo)
    assert len(demo) == pytest.approx(0.2 * (len(policy) + len(demo)), rel=0.02)


@needs_h5
def test_partition_is_deterministic_and_independent_of_the_run_seed():
    """The partition seed is deliberately its own knob. If it followed the run
    seed, a seed sweep would resample the training set too and could not
    separate partition variance from policy variance."""
    from src.models.discriminator import split_demo_clips

    a = split_demo_clips(str(H5), "train", demo_frac=0.2, seed=0)
    b = split_demo_clips(str(H5), "train", demo_frac=0.2, seed=0)
    c = split_demo_clips(str(H5), "train", demo_frac=0.2, seed=1)
    assert a == b
    assert a != c


@needs_h5
def test_demo_frac_zero_restores_the_overlapping_behaviour():
    from src.models.discriminator import split_demo_clips

    policy, demo = split_demo_clips(str(H5), "train", demo_frac=0.0)
    assert policy == demo


@needs_h5
def test_bank_and_policy_dataset_share_no_clip():
    """The end-to-end invariant: what the discriminator learns "real" from and
    what the policy is scored on come from different clips."""
    from src.data.datasets import MoViDataset
    from src.models.discriminator import load_gt_transitions, split_demo_clips

    stats = str(H5.parent / "normalization.json")
    policy, demo = split_demo_clips(str(H5), "train", demo_frac=0.2, seed=0)

    dataset = MoViDataset(str(H5), stats, split="train", clips=policy)
    assert not {clip for clip, _ in dataset.samples} & set(demo)

    full = load_gt_transitions(str(H5), "train")
    held = load_gt_transitions(str(H5), "train", clips=demo)
    assert 0 < held.shape[0] < full.shape[0]


@needs_h5
def test_unknown_clip_names_raise_rather_than_silently_shrinking():
    """A typo or a stale clip list would otherwise cut the training set without
    saying so."""
    from src.data.datasets import MoViDataset
    from src.models.discriminator import load_gt_transitions

    with pytest.raises(ValueError, match="not in train"):
        load_gt_transitions(str(H5), "train", clips=["no_such_clip"])
    with pytest.raises(ValueError, match="not in train"):
        MoViDataset(str(H5), str(H5.parent / "normalization.json"),
                    split="train", clips=["no_such_clip"])
