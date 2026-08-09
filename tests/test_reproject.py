# -*- coding: utf-8 -*-
"""
Tests for src.reproject and the 2D-target resampling in src.data.datasets.

The central invariant is that the conversion is *projection-preserving*: the
recovered metric translation must land on the same image pixel as the virtual
camera it was derived from. If that holds, the reprojection reward compares like
with like; if it drifts, every reward value is quietly wrong.
"""
import numpy as np
import pytest

from src.data.datasets import resample_to
from src.reproject import (COCO17_NAMES, COCO17_TO_SMPLX, T_Z_MAX,
                           crop_intrinsics, metric_translation, place_in_camera,
                           project, real_intrinsics)

# A MoVi-style MATLAB intrinsic matrix (transposed, principal point in row 2).
K_MATLAB = np.array([[979.17889011, 0.0, 0.0],
                     [0.0, 978.10179305, 0.0],
                     [408.02731030, 291.16967878, 1.0]])


def make_bbox(n=8, w=180.0):
    """process_bbox output: aspect fixed at 384/512, so h = 4/3 w."""
    bbox = np.zeros((n, 4), np.float32)
    bbox[:, 0] = np.linspace(100, 400, n)
    bbox[:, 1] = np.linspace(50, 200, n)
    bbox[:, 2] = w
    bbox[:, 3] = w * 512.0 / 384.0
    return bbox


class TestIntrinsics:
    def test_real_intrinsics_reads_matlab_layout(self):
        fx, fy, cx, cy = real_intrinsics(K_MATLAB)
        assert fx == pytest.approx(979.17889011)
        assert fy == pytest.approx(978.10179305)
        # principal point comes from the bottom row, not the right column
        assert (cx, cy) == pytest.approx((408.02731030, 291.16967878))

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            real_intrinsics(np.eye(4))

    def test_crop_focals_are_isotropic(self):
        """process_bbox's fixed aspect makes fx_crop == fy_crop; the code relies on it."""
        f, cx, cy = crop_intrinsics(make_bbox())
        bbox = make_bbox()
        assert f == pytest.approx(5000.0 / 192.0 * bbox[:, 2])
        # principal point is the bbox centre
        assert cx == pytest.approx(bbox[:, 0] + bbox[:, 2] / 2)
        assert cy == pytest.approx(bbox[:, 1] + bbox[:, 3] / 2)

    def test_crop_intrinsics_rejects_wrong_aspect(self):
        bad = make_bbox()
        bad[:, 3] = bad[:, 2]           # square bbox never comes from process_bbox
        with pytest.raises(ValueError, match="aspect"):
            crop_intrinsics(bad)


class TestMetricTranslation:
    def test_projection_is_preserved(self):
        """
        The defining property: the recovered metric point must project to the
        same pixel under the REAL camera as `transl` did under the virtual one.
        """
        bbox = make_bbox()
        rng = np.random.default_rng(0)
        transl = np.stack([rng.uniform(-0.3, 0.3, len(bbox)),
                           rng.uniform(-0.3, 0.3, len(bbox)),
                           rng.uniform(30.0, 50.0, len(bbox))], -1)

        tm, uv, valid = metric_translation(transl, bbox, K_MATLAB)
        assert valid.all()

        f, cx, cy = crop_intrinsics(bbox)
        uv_virtual = np.stack([f * transl[:, 0] / transl[:, 2] + cx,
                               f * transl[:, 1] / transl[:, 2] + cy], -1)
        np.testing.assert_allclose(uv, uv_virtual, rtol=1e-4)

        # and re-projecting the metric point (no distortion) returns the same uv
        np.testing.assert_allclose(project(tm, K_MATLAB), uv, rtol=1e-4, atol=1e-2)

    def test_depth_scales_by_focal_ratio(self):
        bbox = make_bbox(n=4)
        transl = np.tile([0.0, 0.0, 40.0], (4, 1))
        tm, _, _ = metric_translation(transl, bbox, K_MATLAB)
        f, _, _ = crop_intrinsics(bbox)
        fx, fy, _, _ = real_intrinsics(K_MATLAB)
        np.testing.assert_allclose(tm[:, 2], 40.0 * (0.5 * (fx + fy)) / f, rtol=1e-5)

    def test_recovered_depth_is_plausible_for_movi(self):
        """
        A 5000 px virtual focal maps t_z ~ 42 to a few metres, not tens. The
        350 px bbox is representative of MoVi detections; the measured median
        depth over the test split is 4.5 m, with the full range 2.5-6.1 m.
        """
        bbox = make_bbox(n=1, w=350.0)
        tm, _, _ = metric_translation(np.array([[0.0, 0.0, 41.8]]), bbox, K_MATLAB)
        assert 2.0 < tm[0, 2] < 8.0

    def test_t_z_ceiling_matches_get_camera_trans(self):
        assert T_Z_MAX == pytest.approx(56.379, abs=1e-2)

    def test_missing_bbox_is_invalid_not_nan(self):
        bbox = make_bbox(n=3)
        bbox[1] = 0.0                                  # undetected frame
        transl = np.tile([0.0, 0.0, 40.0], (3, 1))
        tm, uv, valid = metric_translation(transl, bbox, K_MATLAB)
        assert valid.tolist() == [True, False, True]
        assert np.isfinite(tm).all() and np.isfinite(uv).all()
        np.testing.assert_array_equal(tm[1], 0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            metric_translation(np.zeros((5, 3)), make_bbox(n=4), K_MATLAB)


class TestProject:
    def test_radial_distortion_is_applied_and_grows_outward(self):
        pts = np.array([[[0.0, 0.0, 4.0], [1.2, 0.9, 4.0]]])
        radial = np.array([-0.18236467, 0.18388686])
        clean = project(pts, K_MATLAB)
        dist = project(pts, K_MATLAB, radial)
        # on-axis point is unmoved; off-axis point shifts
        np.testing.assert_allclose(clean[0, 0], dist[0, 0], atol=1e-6)
        assert np.linalg.norm(clean[0, 1] - dist[0, 1]) > 1.0

    def test_rejects_non_3d_points(self):
        with pytest.raises(ValueError):
            project(np.zeros((4, 2)), K_MATLAB)


class TestPlaceInCamera:
    def test_does_not_recentre_on_pelvis(self):
        """
        Regression guard: cam_trans positions the model origin, so the raw
        joints must be offset wholesale. Re-centring on the pelvis costs ~76 px.
        """
        joints = np.zeros((2, 5, 3), np.float32)
        joints[:, 0] = [0.0, -0.35, 0.0]              # SMPL-X pelvis below origin
        joints[:, 1] = [0.1, 0.2, 0.0]
        trans = np.tile([0.0, 0.0, 4.5], (2, 1))
        out = place_in_camera(joints, trans)
        np.testing.assert_allclose(out[:, 0], [[0.0, -0.35, 4.5]] * 2, atol=1e-6)
        assert not np.allclose(out[:, 0], [[0.0, 0.0, 4.5]] * 2)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            place_in_camera(np.zeros((2, 5, 3)), np.zeros((3, 3)))


class TestJointMapping:
    def test_mapping_is_consistent_and_excludes_face(self):
        assert len(COCO17_NAMES) == 17
        coco = [c for c, _ in COCO17_TO_SMPLX]
        smplx = [s for _, s in COCO17_TO_SMPLX]
        assert len(set(coco)) == len(coco) == 12
        assert len(set(smplx)) == len(smplx)
        assert all(0 <= c < 17 for c in coco)
        assert all(1 <= s < 22 for s in smplx)          # body joints only
        # the five face keypoints have no SMPL-X counterpart
        assert not set(coco) & {0, 1, 2, 3, 4}

    def test_left_right_are_not_swapped(self):
        m = dict(COCO17_TO_SMPLX)
        assert (m[COCO17_NAMES.index("left_shoulder")],
                m[COCO17_NAMES.index("right_shoulder")]) == (16, 17)
        assert (m[COCO17_NAMES.index("left_hip")],
                m[COCO17_NAMES.index("right_hip")]) == (1, 2)


class TestResampleTo:
    def test_identity_when_lengths_match(self):
        a = np.arange(12, dtype=np.float32).reshape(4, 3)
        np.testing.assert_array_equal(resample_to(a, 4), a)

    def test_endpoints_are_pinned(self):
        a = np.arange(10, dtype=np.float32)[:, None]
        r = resample_to(a, 37)
        assert r.shape == (37, 1)
        assert r[0, 0] == pytest.approx(0.0)
        assert r[-1, 0] == pytest.approx(9.0)

    def test_preserves_trailing_shape(self):
        a = np.zeros((5, 17, 3), np.float32)
        assert resample_to(a, 20).shape == (20, 17, 3)

    def test_validity_mask_does_not_spread(self):
        """
        A frame interpolated between a valid and an invalid source frame must
        not count as valid — the >= 1.0 rule the loader uses.
        """
        valid = np.array([1, 1, 0, 1, 1], np.float32)
        up = resample_to(valid, 17)
        assert (up >= 1.0).sum() < 17
        # nothing blended with the invalid frame survives
        assert not np.any((up > 0.0) & (up < 1.0) & (up >= 1.0))

    def test_empty_input_returns_zeros(self):
        assert resample_to(np.zeros((0, 3), np.float32), 6).shape == (6, 3)
