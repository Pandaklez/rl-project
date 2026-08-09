"""
camera_frame.py
───────────────
Undo the camera-frame rotation that SMPLer-X leaves in the lifted root orientation.

SMPLer-X regresses the root in the *camera* frame and never applies calibration
(`smpler-x-main/inference.py:143`; the real focal/princpt at `:159-160` are used
only for rendering). The lifted root therefore sits ~95° (PG1) / ~120° (PG2) away
from the AMASS world frame, while joints 1-51 are parent-relative and unaffected.

The correction uses only the MoVi calibration and a fixed axis relabel — no
ground truth is involved, so it introduces no GT leakage into the model input:

    R_world = F⁻¹ · (R_extᵀ)⁻¹ · R_camera

`R_ext` is transposed because MoVi's calibration comes from MATLAB's Camera
Calibrator, which uses the row-vector convention `X_cam = X_world · R + t`.

`F` is the world-convention relabel (x, y, z) -> (-y, x, z), i.e. +90° about Z,
shared by both cameras. Fitting it against GT gives 88.38° about [0, 0, 1],
1.72° from this exact permutation — so the exact matrix is used rather than the
fitted one.

Measured on the test split (median root error vs GT):

    PG1  93.9° -> 5.2°   (GT-fitted upper bound 5.3°)
    PG2 119.5° -> 8.5°   (GT-fitted upper bound 6.4°)

Translation is NOT corrected here and cannot be: the lifted `trans` is virtual-
camera depth from a 5000 px focal, not metres. See scripts/fit_camera_offset.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

CALIB_DIR = Path(__file__).resolve().parent.parent / "data/Calib"

# World-convention relabel (x, y, z) -> (-y, x, z): +90° about Z.
WORLD_FLIP = np.array([[0.0, -1.0, 0.0],
                       [1.0,  0.0, 0.0],
                       [0.0,  0.0, 1.0]])


@lru_cache(maxsize=4)
def camera_to_world(camera: str, calib_dir: str | None = None) -> R:
    """
    Rotation taking a camera-frame root orientation into the AMASS world frame.

    camera : "PG1" / "pg1" / "PG2" / "pg2"
    """
    d = Path(calib_dir) if calib_dir else CALIB_DIR
    ext = np.load(d / f"Extrinsics_{camera.upper()}.npz")["rotationMatrix"]
    # .T for MATLAB's row-vector convention; .inv() to go camera -> world.
    return R.from_matrix(WORLD_FLIP).inv() * R.from_matrix(ext.T).inv()


def correct_root(poses: np.ndarray, camera: str, calib_dir: str | None = None) -> np.ndarray:
    """
    Rotate the root joint of a lifted clip into the world frame.

    poses : (T, 52, 3) axis-angle, joint 0 = root
    returns a copy with poses[:, 0] corrected; joints 1-51 are untouched
    because they are parent-relative and carry no camera offset.
    """
    poses = np.asarray(poses, dtype=np.float32).copy()
    root = R.from_rotvec(poses[:, 0, :])
    poses[:, 0, :] = (camera_to_world(camera, calib_dir) * root).as_rotvec()
    return poses


def uncorrect_root(poses: np.ndarray, camera: str, calib_dir: str | None = None) -> np.ndarray:
    """
    Inverse of `correct_root`: world-frame root back into the camera frame.

    A reprojection reward needs this. The dataset stores world-frame poses (the
    file carries `root_corrected`), but projecting into the image requires the
    camera frame — and the *corrected* pose coming out of a policy cannot use the
    stored `root_cam`, which is the untouched lifted root rather than the
    policy's output.

    poses : (T, 52, 3) axis-angle with a world-frame root
    """
    poses = np.asarray(poses, dtype=np.float32).copy()
    root = R.from_rotvec(poses[:, 0, :])
    poses[:, 0, :] = (camera_to_world(camera, calib_dir).inv() * root).as_rotvec()
    return poses
