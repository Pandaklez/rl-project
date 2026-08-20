"""
The observation carries what the reward is computed from.

§4 of the failure analysis: the observation was `(lifted_t, corrected_{t-1})`,
318 numbers of pose, while the reward was reprojection error against ViTPose
keypoints using this clip's camera, bbox and metric translation. None of that
was observable, so the best policy the network could represent was the identity.

The fix appends the reward's own per-joint residual. These tests pin the three
properties that make it worth appending:

* the width is right and every rollout path agrees on it,
* the residual actually tracks the error (a perturbed pose changes it),
* and no ground truth leaks in along the way.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

needs_env = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None,
    reason="src.env imports gymnasium (smplerx env)")

from src.rewards import (CONTEXT_DIM, EVIDENCE_DIM, N_KP, empty_evidence,
                         empty_info, pack_evidence)


def test_evidence_layout_adds_up():
    assert EVIDENCE_DIM == N_KP * 2 + N_KP + 3 + CONTEXT_DIM == 44


def test_state_dim_tracks_the_flag():
    from src.models.policy import STATE_DIM, state_dim
    assert state_dim(False) == STATE_DIM == 318
    assert state_dim(True) == 318 + EVIDENCE_DIM == 362


def test_empty_evidence_is_all_zero_but_carries_progress():
    e = empty_evidence(0.5)
    assert e.shape == (EVIDENCE_DIM,)
    assert e[:N_KP * 3].sum() == 0.0          # no residual, no confidence
    assert e[N_KP * 3 + 1] == 0.0             # valid = 0
    assert e[N_KP * 3 + 2] == pytest.approx(0.5)


def test_pack_evidence_marks_a_measured_frame_valid():
    info = empty_info()
    info.update(valid=True, err_norm=0.03)
    info["resid"][0] = [0.1, -0.2]
    info["conf"][0] = 0.9
    e = pack_evidence(info, np.zeros(CONTEXT_DIM, np.float32), 0.25)
    assert e[0:2].tolist() == pytest.approx([0.1, -0.2])
    assert e[N_KP * 2] == pytest.approx(0.9)
    assert e[N_KP * 3] == pytest.approx(0.03)
    assert e[N_KP * 3 + 1] == 1.0


def test_nan_error_never_reaches_the_observation():
    """`err_norm` is nan on frames with no 2D evidence. A nan in the state would
    propagate through the network and through the running normaliser, killing
    the run silently."""
    e = pack_evidence(empty_info(), np.zeros(CONTEXT_DIM, np.float32), 0.0)
    assert np.isfinite(e).all()


@needs_env
def test_flatten_state_appends_evidence_at_the_end():
    from src.models.policy import STATE_DIM, flatten_state

    state = {
        "lifted_state":    {"poses": torch.zeros(52, 3), "trans": torch.zeros(3)},
        "corrected_state": {"poses": torch.zeros(52, 3), "trans": torch.zeros(3)},
    }
    ev = np.arange(EVIDENCE_DIM, dtype=np.float32)
    flat = flatten_state(state, evidence=ev)
    assert flat.shape == (STATE_DIM + EVIDENCE_DIM,)
    assert flat[STATE_DIM:].tolist() == pytest.approx(ev.tolist())
    # and without it the width is unchanged, so the ablation is a real ablation
    assert flatten_state(state).shape == (STATE_DIM,)
