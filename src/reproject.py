"""
reproject.py
────────────
Convert SMPLer-X's virtual-camera translation into metres in the real MoVi
camera, and project 3D points back into the image.

SMPLer-X does not regress a metric translation. `get_camera_trans`
(`smpler-x-main/SMPLer_X.py:68-76`) produces a depth in a *virtual* camera with
a 5000 px focal over the 192x256 crop:

    k = sqrt(fx_v * fy_v * s^2 / (H_body * W_body))    s = camera_3d_size = 2.5
    t_z = k * sigmoid(...)                             so t_z in (0, 56.38)

which is why the stored `transl` has z ~ 41.8 against a GT stature of 0.86 m.
The mesh itself is metric; only the camera is virtual. Rendering works because
`inference.py:159-160` rescales the focal to the bbox:

    f_crop  = 5000 / 192 * bbox_w   (x)      5000 / 256 * bbox_h   (y)
    c_crop  = bbox_x + bbox_w / 2            bbox_y + bbox_h / 2

`process_bbox` fixes the aspect at 384/512, so bbox_h = 4/3 * bbox_w and the two
crop focals are equal — `crop_intrinsics` asserts this rather than assuming it.

Recovering metres is then a similar-triangles argument. A body of size S at
depth Z subtends f * S / Z pixels. The image evidence is fixed, so matching the
apparent size under the real focal gives

    Z_real = t_z * f_real / f_crop
    X_real = (u - c_x) * Z_real / f_real        u, v = the root projected into
    Y_real = (v - c_y) * Z_real / f_real        the original image

No ground truth is involved: bbox comes from the detector and the intrinsics
from calibration, so this introduces no GT leakage into a reprojection reward.

Distortion: MoVi's intrinsics carry two radial coefficients. `project` applies
them, so projected points land in the same distorted pixel space as the ViTPose
detections and a reward can compare the two directly with no undistortion step.
The effect is small but not negligible — ~4 px at the image corners.

**Work in the camera frame.** The whole path — lifted camera-frame pose, FK,
`place_in_camera`, `project` — needs only the intrinsics and the bbox, so a
reward never touches the extrinsics. That keeps the reward independent of a
calibration it does not need, and the path validates at 11.6 px (PG1) /
14.5 px (PG2) on the test split.

An earlier version of this docstring justified that differently, claiming the
PG2 extrinsics were measurably worse than PG1's — 65-75 px against 9.7 px.
**That is false and has been retracted.** `scripts/check_extrinsics.py`
re-measures it: with the correct convention (`X_cam = R_extᵀ · F · X_world + t`,
`t` in millimetres) GT through the calibration lands at 12.0 px (PG1) /
14.5 px (PG2) over 60 test clips — no asymmetry. A wrong convention costs
90-270 px on *both* cameras, which is why a large per-camera gap was never a
convention error to begin with.
"""
from __future__ import annotations

import numpy as np

# SMPLer-X virtual camera (config_smpler_x_b32.py:86-98)
FOCAL_VIRTUAL = (5000.0, 5000.0)
INPUT_BODY_SHAPE = (256, 192)      # H, W
CAMERA_3D_SIZE = 2.5
# Depth ceiling implied by get_camera_trans; t_z is a sigmoid times this.
T_Z_MAX = float(np.sqrt(FOCAL_VIRTUAL[0] * FOCAL_VIRTUAL[1] * CAMERA_3D_SIZE ** 2
                        / (INPUT_BODY_SHAPE[0] * INPUT_BODY_SHAPE[1])))


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# (coco_index, smplx_joint_index) for the 12 COCO joints with an unambiguous
# SMPL-X counterpart. The five face keypoints (nose, eyes, ears) are deliberately
# excluded: SMPL-X's 52-joint skeleton has only `head` (15) in that region, and
# pairing it with the nose would bake in a systematic offset.
COCO17_TO_SMPLX = (
    (5, 16), (6, 17),      # shoulders
    (7, 18), (8, 19),      # elbows
    (9, 20), (10, 21),     # wrists
    (11, 1), (12, 2),      # hips
    (13, 4), (14, 5),      # knees
    (15, 7), (16, 8),      # ankles
)
COCO_IDX = np.array([c for c, _ in COCO17_TO_SMPLX], dtype=np.int64)
SMPLX_IDX = np.array([s for _, s in COCO17_TO_SMPLX], dtype=np.int64)


# MATLAB's Camera Calibrator reports the principal point in **1-based** pixel
# coordinates: the centre of the top-left pixel is (1, 1). numpy and OpenCV are
# 0-based, with that same pixel centre at (0, 0). Using the reported value
# unshifted therefore places every projected point one pixel right and one pixel
# down of where it belongs.
#
# It is a small term next to the ~10 px systematic offset measured between
# SMPL-X joints and ViTPose keypoints (scripts/fit_kp_bias.py), and the
# empirical bias correction absorbs it either way — but it is wrong, it is in
# the same direction as that offset, and it affects everything that projects,
# including the visualisations and scripts/check_extrinsics.py.
MATLAB_PIXEL_ORIGIN = 1.0


def real_intrinsics(intrinsic_matrix: np.ndarray) -> tuple[float, float, float, float]:
    """
    (fx, fy, cx, cy) from a MoVi `cameraParams_*.npz` IntrinsicMatrix, in
    0-based pixel coordinates.

    Two MATLAB conventions have to be undone, not one:

    * K is stored transposed (row-vector convention), so the principal point
      sits in the bottom row rather than the right column. Reading it from the
      right column yields (0, 0), which is why this is easy to catch.
    * Pixel coordinates are 1-based, so the principal point is shifted by one
      pixel relative to numpy's convention. This one is *not* easy to catch: it
      is a one-pixel error that looks like nothing until it is measured.
    """
    K = np.asarray(intrinsic_matrix, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"IntrinsicMatrix must be 3x3, got {K.shape}")
    return (float(K[0, 0]), float(K[1, 1]),
            float(K[2, 0]) - MATLAB_PIXEL_ORIGIN,
            float(K[2, 1]) - MATLAB_PIXEL_ORIGIN)


def crop_intrinsics(bbox: np.ndarray, atol: float = 1e-3):
    """
    The virtual camera SMPLer-X's `transl` lives in, expressed in original-image
    pixels (`inference.py:159-160`).

    bbox : (N, 4) xywh, after `process_bbox`
    returns (f_crop, cx_crop, cy_crop), each (N,)
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    if bbox.ndim != 2 or bbox.shape[1] != 4:
        raise ValueError(f"bbox must be (N, 4) xywh, got {bbox.shape}")
    w, h = bbox[:, 2], bbox[:, 3]
    fx = FOCAL_VIRTUAL[0] / INPUT_BODY_SHAPE[1] * w
    fy = FOCAL_VIRTUAL[1] / INPUT_BODY_SHAPE[0] * h

    # process_bbox fixes the aspect, so these must agree; if they ever diverge
    # the bbox did not come from the SMPLer-X pipeline and the depth below is
    # not well defined.
    ok = bbox[:, 2] > 0
    if ok.any() and not np.allclose(fx[ok], fy[ok], rtol=atol):
        worst = np.max(np.abs(fx[ok] - fy[ok]) / fx[ok])
        raise ValueError(
            f"crop focals disagree by {worst:.2%}; bbox aspect is not 384/512")

    return fx, bbox[:, 0] + w / 2.0, bbox[:, 1] + h / 2.0


def metric_translation(transl: np.ndarray, bbox: np.ndarray,
                       intrinsic_matrix: np.ndarray):
    """
    SMPLer-X virtual-camera `transl` -> metres in the real camera frame.

    transl : (N, 3) the `transl` stored by inference.py (really `cam_trans`)
    bbox   : (N, 4) xywh, after process_bbox, same frames
    returns (trans_metric (N,3) float32, uv_root (N,2) float32, valid (N,) bool)

    `valid` is False where the bbox is absent (w == 0) or the depth is
    degenerate; those rows are returned as zeros rather than NaN so downstream
    array shapes stay uniform.
    """
    transl = np.asarray(transl, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    if transl.shape[0] != bbox.shape[0]:
        raise ValueError(f"transl {transl.shape} and bbox {bbox.shape} differ in length")

    valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) & (np.abs(transl[:, 2]) > 1e-6)
    trans_metric = np.zeros((len(transl), 3), dtype=np.float64)
    uv = np.zeros((len(transl), 2), dtype=np.float64)
    if not valid.any():
        return trans_metric.astype(np.float32), uv.astype(np.float32), valid

    f_crop, cx_crop, cy_crop = crop_intrinsics(bbox[valid])
    fx, fy, cx, cy = real_intrinsics(intrinsic_matrix)
    tx, ty, tz = transl[valid].T

    # Where the root projects in the original image under the virtual camera.
    u = f_crop * tx / tz + cx_crop
    v = f_crop * ty / tz + cy_crop

    # Same apparent size under the real focal => scale depth by the focal ratio.
    z = tz * (0.5 * (fx + fy)) / f_crop
    trans_metric[valid] = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], -1)
    uv[valid] = np.stack([u, v], -1)
    return trans_metric.astype(np.float32), uv.astype(np.float32), valid


def place_in_camera(joints: np.ndarray, trans_metric: np.ndarray) -> np.ndarray:
    """
    Put SMPL-X joints at their metric position in the real camera frame.

    joints       : (T, J, 3) the RAW body-model output, i.e. `transl=0` and
                   **not** re-centred on the pelvis
    trans_metric : (T, 3) from `metric_translation`

    The distinction matters more than it looks. SMPLer-X composes its mesh as
    `vertices + cam_trans`, so `cam_trans` positions the *model origin*, not the
    pelvis — and the SMPL-X pelvis sits ~0.35 m below that origin. Re-centring on
    the pelvis first (the natural-looking `J - J[:, :1] + trans`) therefore
    shifts the whole body by 0.35 m, which at a ~4.5 m working depth is ~76 px.

    Measured on the test split, projecting lifted poses against the ViTPose
    detections with no GT and no extrinsics:

        placement                    PG1        PG2
        raw + cam_trans (this)     11.6 px    14.5 px
        pelvis at cam_trans        83.4 px    95.4 px
    """
    joints = np.asarray(joints, dtype=np.float64)
    trans_metric = np.asarray(trans_metric, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joints must be (T, J, 3), got {joints.shape}")
    if trans_metric.shape != (joints.shape[0], 3):
        raise ValueError(
            f"trans_metric must be ({joints.shape[0]}, 3), got {trans_metric.shape}")
    return (joints + trans_metric[:, None, :]).astype(np.float32)


def project(points_cam: np.ndarray, intrinsic_matrix: np.ndarray,
            radial: np.ndarray | None = None) -> np.ndarray:
    """
    Pinhole-project metric camera-frame points into image pixels.

    points_cam : (..., 3) metres, real camera frame
    radial     : optional (2,) MoVi RadialDistortion [k1, k2]; applied so the
                 result shares the pixel space of the ViTPose detections.
    returns (..., 2) float32
    """
    p = np.asarray(points_cam, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError(f"points_cam must end in 3, got {p.shape}")
    fx, fy, cx, cy = real_intrinsics(intrinsic_matrix)

    z = np.where(np.abs(p[..., 2]) < 1e-9, 1e-9, p[..., 2])
    x, y = p[..., 0] / z, p[..., 1] / z

    if radial is not None:
        k1, k2 = np.asarray(radial, dtype=np.float64).ravel()[:2]
        r2 = x * x + y * y
        s = 1.0 + k1 * r2 + k2 * r2 * r2
        x, y = x * s, y * s

    return np.stack([fx * x + cx, fy * y + cy], -1).astype(np.float32)
