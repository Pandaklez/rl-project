"""
The reparameterised translation action.

The old translation action added a delta to the lifted `trans` — SMPLer-X's
cam_trans in a cropped-bbox virtual camera. Not metres, not a translation in the
image, and its depth component is the direction the reprojection reward is
blind in.

The reparameterisation splits it into `(du, dv, dlog_tz)`: an image-plane shift
in bbox-height units, and a log-depth change along the viewing ray. Two
properties have to hold for that to be worth anything, and both are geometric
rather than statistical, so both are testable:

1. `du`/`dv` move the projection one-for-one and independently — that is what
   makes the observed residual and the correction the same quantity.
2. `dlog_tz` barely moves the projection at all — which is *why* it is frozen
   under `trans_mode="uv"`, and the number below is the argument.

Skipped without the real dataset: the claim is about this calibration and these
bboxes, so a synthetic camera would be testing arithmetic rather than the thing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
H5 = REPO / "data" / "processed_movi.h5"
TARGETS = REPO / "data" / "reproj_targets.h5"

needs_data = pytest.mark.skipif(
    not (H5.exists() and TARGETS.exists()),
    reason="needs data/processed_movi.h5 and data/reproj_targets.h5")


@pytest.fixture(scope="module")
def bound_reward():
    """A ReprojectionReward bound to the first val clip with usable targets."""
    from src.data.datasets import MoViDataset
    from src.rewards import ReprojectionReward, load_calib

    ds = MoViDataset(str(H5), str(REPO / "data" / "normalization.json"),
                     split="val", verbose=False, reproj_path=str(TARGETS))
    rr = ReprojectionReward(load_calib(str(H5)), sigma=0.04,
                            correct_translation=False)
    for i in range(len(ds)):
        sample = ds[i]
        if rr.reset(sample):
            break
    else:
        pytest.skip("no val clip with usable 2D targets")
    t = int(np.flatnonzero(rr._clip["valid"])[0])
    frame = {"poses": sample["x"]["poses"][t], "trans": sample["x"]["trans"][t]}
    return rr, frame, t, float(rr._clip["bbox"][t][3])


@needs_data
@pytest.mark.parametrize("axis,delta", [(0, 0.10), (1, 0.10), (0, -0.06), (1, -0.06)])
def test_image_shift_is_one_for_one(bound_reward, axis, delta):
    """A du of x bbox heights moves the projection by x bbox heights, on that
    axis only. Tolerance is 3%: the pinhole term is exact, radial distortion is
    not, and the reward projects with distortion on."""
    rr, frame, t, h = bound_reward
    base = rr._project_frame(frame, t, None)

    d = np.zeros(3, np.float32)
    d[axis] = delta
    moved = rr._project_frame(frame, t, d)

    shift = (moved - base).mean(0) / h
    assert shift[axis] == pytest.approx(delta, rel=0.03)
    assert shift[1 - axis] == pytest.approx(0.0, abs=0.005)


@needs_data
def test_log_depth_is_nearly_invisible_to_the_reward(bound_reward):
    """
    The freeze, justified.

    A 10% depth change is a large 3D error — and it moves the mean projection by
    under 2% of a bbox height, against a reward sigma of 4%. The same-sized
    image-plane action moves it by 10%, an order of magnitude more. A free depth
    dimension is therefore a random walk the reward cannot correct, which is
    exactly what `trans_mode="uv"` removes by not giving the policy the
    parameter at all.
    """
    rr, frame, t, h = bound_reward
    base = rr._project_frame(frame, t, None)

    depth = np.abs((rr._project_frame(frame, t, np.array([0, 0, 0.10], np.float32))
                    - base).mean(0)).max() / h
    image = np.abs((rr._project_frame(frame, t, np.array([0, 0.10, 0], np.float32))
                    - base).mean(0)).max() / h

    assert depth < 0.02
    assert image > 5 * depth


@needs_data
def test_body_pushed_into_the_camera_is_a_miss_not_a_crash(bound_reward):
    """A depth so negative the body lands behind the lens must return None, the
    same as any other unprojectable frame — never a wild projection."""
    rr, frame, t, _ = bound_reward
    assert rr._project_frame(frame, t, np.array([0, 0, -50.0], np.float32)) is None
    assert rr._project_frame(frame, t, np.array([np.nan, 0, 0], np.float32)) is None


@needs_data
def test_zero_delta_is_bit_identical_to_no_delta(bound_reward):
    """The `uv` policy at initialisation emits ~0, so the zero path must not
    perturb anything — otherwise the baseline would not cancel and the reward
    would not start at exactly 0."""
    rr, frame, t, _ = bound_reward
    a = rr._project_frame(frame, t, None)
    b = rr._project_frame(frame, t, np.zeros(3, np.float32))
    assert np.array_equal(a, b)


# ── exploration scale ────────────────────────────────────────────────────────

def test_initial_sigma_is_per_dimension():
    """
    The pose dims and the translation dims are in different units, so one scalar
    sigma cannot be right for both.

    At the shared 0.05 the *mean sampled image shift* is 0.05·sqrt(2/pi) = 0.040
    bbox heights — 18 px on a typical bbox, against a reprojection error of 13 px
    that the policy is supposed to be removing. A run shipped that way logged
    `du |abs|` = 0.0398 and corrected error 32.9 px against a lifted 13.9 px: the
    translation action was sampling noise and nothing else.
    """
    import math

    from src.models.policy import POSE_DIM, init_log_std_vector

    v = init_log_std_vector("uv").exp()
    assert v[:POSE_DIM].min() == pytest.approx(0.0498, rel=1e-3)
    assert v[POSE_DIM:].max() == pytest.approx(0.005, rel=1e-2)

    # the property that actually matters: mean sampled shift stays well under
    # the reward's operating point of 0.028 bbox heights
    mean_shift = float(v[POSE_DIM]) * math.sqrt(2 / math.pi)
    assert mean_shift < 0.028 / 4


def test_pose_only_sigma_is_unchanged():
    """The 156-d policy must be bit-identical to before — the per-dim vector
    only differs where translation dims exist."""
    from src.models.policy import INIT_LOG_STD, init_log_std_vector

    v = init_log_std_vector("none")
    assert v.shape == (156,)
    assert (v == INIT_LOG_STD).all()
