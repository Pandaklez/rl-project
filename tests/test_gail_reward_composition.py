"""
Tests for `src.gail_train.check_reward_composition`.

Experiment (C) is "(B) + a discriminator reward, and nothing else new" — the
discriminator's real-motion bank is meant to be the *only* place GT enters
the loop. `reward_mode="gt"` (the direct frame-wise GT-similarity ablation)
combined with an active discriminator (`w_gail>0`) leaks GT through a second
channel, which a real run here did by mistake once. These tests only touch
`Config`, a plain dataclass — no dataset, model or GPU needed — so they run
fast and do not need `data/processed_movi.h5`.

Skipped when `skrl`/`gymnasium` aren't importable: `src.gail_train` imports
both at module level (skrl for PPO, gymnasium transitively via
`src.gail_env`), even though the function under test touches neither.
"""
from __future__ import annotations

import importlib.util

import pytest

needs_gail_train = pytest.mark.skipif(
    importlib.util.find_spec("skrl") is None
    or importlib.util.find_spec("gymnasium") is None,
    reason="src.gail_train imports skrl and gymnasium (rl_project/smplerx env)")


@needs_gail_train
def test_gt_reward_with_active_gail_is_rejected():
    from src.gail_train import Config, check_reward_composition

    cfg = Config(reward_mode="gt", w_gail=0.5)
    with pytest.raises(ValueError, match="reward_mode='gt'"):
        check_reward_composition(cfg)


@needs_gail_train
def test_gt_reward_with_gail_disabled_is_allowed():
    """The infra-smoke-test path (no data/reproj_targets.h5 yet): discriminator
    built but contributing nothing to the reward."""
    from src.gail_train import Config, check_reward_composition

    check_reward_composition(Config(reward_mode="gt", w_gail=0.0))


@needs_gail_train
def test_reproj_reward_with_active_gail_is_allowed():
    """The actual experiment (C) composition."""
    from src.gail_train import Config, check_reward_composition

    check_reward_composition(Config(reward_mode="reproj", w_gail=0.5))


@needs_gail_train
def test_reproj_reward_alone_is_allowed():
    """Experiment (B), reproduced by w_gail=0 — must stay unaffected by this
    guard, since it isn't specific to gail_train.py's own default config."""
    from src.gail_train import Config, check_reward_composition

    check_reward_composition(Config(reward_mode="reproj", w_gail=0.0))


@needs_gail_train
def test_default_config_is_the_experiment_c_composition():
    """Regression guard for the exact mistake this file exists to catch: the
    out-of-the-box `Config()` must already be (B) + discriminator, not the
    double-GT-leakage combination a previous run used."""
    from src.gail_train import Config, check_reward_composition

    cfg = Config()
    assert cfg.reward_mode == "reproj"
    assert cfg.w_gail > 0
    check_reward_composition(cfg)  # must not raise
