# -*- coding: utf-8 -*-
"""
Tests for src.rewards — the GT-free reprojection and smoothness rewards.

Tests that need SMPL-X forward kinematics are skipped when `smplx` is not
importable (it lives in the `smplerx` env); everything else runs anywhere.
"""
import json

import numpy as np
import pytest

from src.rewards import (MIN_DEPTH_VIRTUAL, ReprojectionReward, combine,
                         load_lifted_stats, smoothness_reward, unscale)

K_MATLAB = np.array([[979.17889011, 0.0, 0.0],
                     [0.0, 978.10179305, 0.0],
                     [408.02731030, 291.16967878, 1.0]])
CALIB = {"pg1": {"IntrinsicMatrix": K_MATLAB,
                 "RadialDistortion": np.array([-0.18236467, 0.18388686])}}


def make_stats():
    """Lifted-style stats tuple, as load_lifted_stats returns."""
    return (("poses", (np.zeros((52, 3), np.float32), np.ones((52, 3), np.float32))),
            ("trans", (np.zeros(3, np.float32), np.ones(3, np.float32))),
            ("betas", (np.zeros(16, np.float32), np.ones(16, np.float32))))


def make_sample(T=6, valid=True, camera="pg1"):
    bbox = np.tile([300.0, 100.0, 350.0, 350.0 * 512 / 384], (T, 1)).astype(np.float32)
    kp2d = np.zeros((T, 17, 3), np.float32)
    kp2d[..., 0] = 420.0
    kp2d[..., 1] = 300.0
    kp2d[..., 2] = 0.9
    return {
        "x": {"poses": np.zeros((T, 52, 3), np.float32),
              "trans": np.tile([0.0, 0.0, 41.8], (T, 1)).astype(np.float32),
              "betas": np.zeros(16, np.float32)},
        "meta": {"clip": "Subject_1__walking", "camera": camera, "split": "test"},
        "reproj": {"kp2d": kp2d, "bbox": bbox,
                   "trans_metric": np.tile([0.0, 0.0, 4.5], (T, 1)).astype(np.float32),
                   "valid": np.full(T, valid, dtype=bool)},
    }


class TestUnscale:
    def test_inverts_normalisation(self):
        stats = make_stats()
        mu = np.full((52, 3), 0.3, np.float32)
        sigma = np.full((52, 3), 2.0, np.float32)
        stats = (("poses", (mu, sigma)),) + stats[1:]
        raw = np.random.default_rng(0).normal(size=(52, 3)).astype(np.float32)
        normed = (raw - mu) / sigma
        np.testing.assert_allclose(unscale(normed, "poses", stats), raw, rtol=1e-5)


class TestLoadLiftedStats:
    def test_rejects_stats_predating_the_root_correction(self, tmp_path):
        """
        The stale copies at the repo root have no _root_corrected flag and put
        the pose ~130 px off. That must fail loudly, not silently.
        """
        p = tmp_path / "normalization_lifted_pg1.json"
        p.write_text(json.dumps({
            "poses": {"mu": np.zeros((52, 3)).tolist(), "sigma": np.ones((52, 3)).tolist()},
            "trans": {"mu": [0, 0, 0], "sigma": [1, 1, 1]},
            "betas": {"mu": [0] * 16, "sigma": [1] * 16},
        }))
        with pytest.raises(ValueError, match="_root_corrected"):
            load_lifted_stats("pg1", str(p))

    def test_accepts_current_stats_and_guards_zero_sigma(self, tmp_path):
        p = tmp_path / "normalization_lifted_pg2.json"
        sigma = np.ones((52, 3))
        sigma[0, 0] = 0.0                      # a constant dimension
        p.write_text(json.dumps({
            "_root_corrected": True,
            "poses": {"mu": np.zeros((52, 3)).tolist(), "sigma": sigma.tolist()},
            "trans": {"mu": [0, 0, 0], "sigma": [1, 1, 1]},
            "betas": {"mu": [0] * 16, "sigma": [1] * 16},
        }))
        stats = dict(load_lifted_stats("pg2", str(p)))
        assert stats["poses"][1][0, 0] == 1.0   # zero sigma replaced, never divides by 0


class TestSmoothness:
    def test_constant_velocity_is_not_penalised(self):
        """
        The point of using acceleration: steady motion must score as highly as
        standing still, or the policy is rewarded for freezing the subject.
        """
        a = np.zeros((52, 3), np.float32)
        b = a + 0.1
        c = b + 0.1
        assert smoothness_reward(c, b, a) == pytest.approx(1.0, abs=1e-6)
        assert smoothness_reward(a, a, a) == pytest.approx(1.0, abs=1e-6)

    def test_jitter_is_penalised(self):
        a = np.zeros((52, 3), np.float32)
        jitter = np.full((52, 3), 0.5, np.float32)
        assert smoothness_reward(jitter, a, jitter) < 0.5

    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            r = smoothness_reward(*(rng.normal(size=(52, 3)) * 3 for _ in range(3)))
            assert 0.0 <= r <= 1.0


class TestCombine:
    def test_weighted_sum(self):
        assert combine(0.8, 0.5, 1.0, 0.1) == pytest.approx(0.85)

    def test_missing_reprojection_falls_back_rather_than_scoring_zero(self):
        """
        A frame the detector missed says nothing about the pose. Scoring it 0
        would teach the policy that poorly-detected clips are intrinsically bad.
        """
        nan = float("nan")
        r = combine(nan, 1.0, w_reproj=1.0, w_smooth=0.1, fallback=0.5)
        assert r == pytest.approx(0.1 + 0.5)
        assert not np.isnan(r)


class TestReprojectionRewardBinding:
    def test_reset_rejects_sample_without_targets(self):
        rw = ReprojectionReward(CALIB)
        assert rw.reset({"x": {}, "meta": {"camera": "pg1"}}) is False
        assert rw.reset({"x": {}}) is False

    def test_reset_rejects_all_invalid_clip(self):
        """The 20 unaligned cam-clips must not bind."""
        rw = ReprojectionReward(CALIB)
        assert rw.reset(make_sample(valid=False)) is False

    def test_step_on_unbound_reward_reports_invalid(self):
        rw = ReprojectionReward(CALIB)
        r, info = rw.step({"poses": np.zeros((52, 3)), "trans": np.zeros(3)}, 0)
        assert np.isnan(r) and info["valid"] is False


@pytest.mark.parametrize("t_z", [0.0, MIN_DEPTH_VIRTUAL - 0.1, 60.0, 1e6])
def test_degenerate_depth_scores_zero_not_a_wild_projection(t_z, monkeypatch):
    """
    t_z is a sigmoid output in (0, T_Z_MAX) by construction. A policy that
    pushes it outside that range has produced a meaningless pose, which should
    score 0 as a real miss — not nan (a missing target) and not a huge
    projection that swamps the rollout.
    """
    pytest.importorskip("smplx")
    import src.rewards as rewards

    rw = rewards.ReprojectionReward(CALIB, correct_translation=True)
    monkeypatch.setattr(rewards, "load_lifted_stats", lambda *a, **k: make_stats())
    assert rw.reset(make_sample())
    r, info = rw.step({"poses": np.zeros((52, 3), np.float32),
                       "trans": np.array([0.0, 0.0, t_z], np.float32)}, 0)
    assert r == 0.0
    assert info["valid"] is True


class TestReprojectionRewardScoring:
    """Behavioural tests that need real forward kinematics."""

    @pytest.fixture(autouse=True)
    def _fk(self, monkeypatch):
        pytest.importorskip("smplx")
        import src.rewards as rewards
        monkeypatch.setattr(rewards, "load_lifted_stats", lambda *a, **k: make_stats())

    def test_reward_is_bounded_and_finite(self):
        rw = ReprojectionReward(CALIB)
        assert rw.reset(make_sample())
        r, info = rw.step({"poses": np.zeros((52, 3), np.float32),
                           "trans": np.array([0.0, 0.0, 41.8], np.float32)}, 0)
        assert 0.0 <= r <= 1.0 and np.isfinite(info["err_px"])

    def test_closer_projection_scores_higher(self):
        """The property the whole reward rests on."""
        rw = ReprojectionReward(CALIB)
        sample = make_sample()
        assert rw.reset(sample)
        base = {"poses": np.zeros((52, 3), np.float32),
                "trans": np.array([0.0, 0.0, 41.8], np.float32)}
        r_near, i_near = rw.step(base, 0)

        far = {"poses": np.full((52, 3), 0.4, np.float32), "trans": base["trans"]}
        r_far, i_far = rw.step(far, 0)
        assert i_far["err_px"] > i_near["err_px"]
        assert r_far < r_near

    def test_low_confidence_keypoints_are_ignored(self):
        rw = ReprojectionReward(CALIB, min_confidence=0.5)
        sample = make_sample()
        sample["reproj"]["kp2d"][..., 2] = 0.2        # all below threshold
        assert rw.reset(sample)
        r, info = rw.step({"poses": np.zeros((52, 3), np.float32),
                           "trans": np.array([0.0, 0.0, 41.8], np.float32)}, 0)
        assert np.isnan(r) and info["valid"] is False

    def test_scale_normalisation_makes_reward_depth_invariant(self):
        """
        The same pose filmed twice as close must not be scored twice as harshly.

        A closer subject means a bigger bbox, hence a bigger f_crop, hence half
        the recovered depth — so the body projects twice as large and the raw
        pixel error doubles. Dividing by bbox height should cancel that exactly.

        The bbox *centre* is held fixed and the target keypoints placed on it,
        so the only thing changing between the two runs is scale. Letting the
        centre drift would add a constant offset that does not scale, which is
        not what this is testing.
        """
        rw = ReprojectionReward(CALIB)
        cx, cy = 475.0, 333.0
        errs_px, errs_norm = [], []
        for scale in (1.0, 2.0):
            sample = make_sample()
            w, h = 350.0 * scale, 350.0 * 512 / 384 * scale
            sample["reproj"]["bbox"][:] = [cx - w / 2, cy - h / 2, w, h]
            sample["reproj"]["kp2d"][..., 0] = cx
            sample["reproj"]["kp2d"][..., 1] = cy
            assert rw.reset(sample)
            _, info = rw.step({"poses": np.full((52, 3), 0.15, np.float32),
                               "trans": np.array([0.0, 0.0, 41.8], np.float32)}, 0)
            errs_px.append(info["err_px"])
            errs_norm.append(info["err_norm"])

        # the raw error really does scale with the subject...
        assert errs_px[1] == pytest.approx(2 * errs_px[0], rel=0.05)
        # ...and normalising by bbox height removes it
        assert errs_norm[0] == pytest.approx(errs_norm[1], rel=0.05)

    def test_invalid_frame_returns_nan_not_zero(self):
        rw = ReprojectionReward(CALIB)
        sample = make_sample()
        sample["reproj"]["valid"][2] = False
        assert rw.reset(sample)
        r, info = rw.step({"poses": np.zeros((52, 3), np.float32),
                           "trans": np.array([0.0, 0.0, 41.8], np.float32)}, 2)
        assert np.isnan(r) and info["valid"] is False
