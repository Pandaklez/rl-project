"""
Episode-boundary semantics for GymMoviEnv.

A clip running out of frames is a time limit, not a terminal state. Getting this
wrong does not crash anything — it silently corrupts the value target at every
episode boundary, which is how the first experiment (B) attempt spent an hour
training itself steadily worse. Hence a test.

Skipped when `gymnasium` is not importable (it lives in the `smplerx` env).
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

needs_env = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="src.env imports gymnasium (smplerx env)")

N_JOINTS, T = 22, 6


class _StubDataset:
    """Minimal stand-in: one short clip, no HDF5, no reprojection targets."""

    def __init__(self, n_frames: int = T):
        rng = np.random.default_rng(0)
        f = lambda *s: torch.from_numpy(rng.normal(size=s).astype(np.float32))
        self._sample = {
            "x": {"poses": f(n_frames, N_JOINTS, 3), "trans": f(n_frames, 3),
                  "betas": f(16)},
            "y": {"poses": f(n_frames, N_JOINTS, 3), "trans": f(n_frames, 3),
                  "betas": f(16)},
            "meta": {"clip": "stub", "camera": "pg1", "split": "val"},
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._sample


def _run_episode(env):
    """Step until the env says stop; return the per-step (terminated, truncated)."""
    env.reset(seed=0)
    flags = []
    for _ in range(T * 3):                       # generous bound; must stop sooner
        _, _, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape[0], dtype=np.float32))
        flags.append((bool(terminated), bool(truncated)))
        if terminated or truncated:
            break
    return flags


@needs_env
def test_running_out_of_frames_is_truncation_not_termination():
    """
    The last step must report truncated=True, terminated=False.

    skrl only adds the `gamma * V(s)` bootstrap on truncated steps
    (ppo.py:299). Reporting `terminated` instead tells GAE the future is worth
    zero, understating the value target by the whole remaining return.
    """
    from src.env import GymMoviEnv

    env = GymMoviEnv(_StubDataset(), reward_mode="gt")
    flags = _run_episode(env)

    assert flags, "episode produced no steps"
    terminated, truncated = flags[-1]
    assert truncated is True, "clip running out must be reported as truncation"
    assert terminated is False, "there is no terminal state in this MDP"

    # and nothing before the end claims either flag
    assert all(not t and not tr for t, tr in flags[:-1])


@needs_env
def test_episode_ends_and_does_not_run_past_the_clip():
    from src.env import GymMoviEnv

    env = GymMoviEnv(_StubDataset(), reward_mode="gt")
    flags = _run_episode(env)
    assert len(flags) == T - 1, f"expected {T - 1} steps for a {T}-frame clip"


@needs_env
def test_time_limit_bootstrap_is_enabled_in_the_ppo_config():
    """
    The env reporting truncation only matters if skrl is told to act on it —
    `time_limit_bootstrap` defaults to False.
    """
    import inspect

    from src import train

    src = inspect.getsource(train.train)
    assert 'ppo_cfg["time_limit_bootstrap"] = True' in src, (
        "truncation reporting is inert unless time_limit_bootstrap is enabled")
