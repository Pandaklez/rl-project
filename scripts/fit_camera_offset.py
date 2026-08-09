"""
fit_camera_offset.py
────────────────────
Estimate the rotation offset between world-frame GT root orientation and the
camera-frame root orientation produced by the lifter, directly from the data —
no camera calibration files required.

Hypothesis under test (root joint only; joints 1-51 are parent-relative and
therefore frame-independent):

    R_lifted  ≈  R_off · R_gt

If a single constant R_off per camera explains the data, the lifted poses are a
rigidly rotated view of the GT and the offset can simply be undone. The residual
angle after removing the best-fit R_off says how well that holds.

R_off is the chordal L2 mean of the per-frame products R_lifted · R_gtᵀ,
computed via SVD (proper rotation averaging, not an elementwise matrix mean).

GT runs at 120 fps and the lifted data at 30 fps, so GT is *subsampled* onto the
lifted timeline. Rotations are never interpolated.

Usage:
    python scripts/fit_camera_offset.py
    python scripts/fit_camera_offset.py --gt_h5 data/movi_smplx.h5 --n_clips 60
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


def align_indices(t_gt: int, t_lift: int) -> np.ndarray:
    """Map each lifted frame onto the nearest GT frame (no rotation interpolation)."""
    return np.round(np.linspace(0, t_gt - 1, t_lift)).astype(int)


def mean_rotation(mats: np.ndarray) -> R:
    """Chordal L2 mean of a stack of rotation matrices, (N,3,3) -> Rotation."""
    u, _, vt = np.linalg.svd(mats.mean(axis=0))
    d = np.sign(np.linalg.det(u @ vt))
    return R.from_matrix(u @ np.diag([1.0, 1.0, d]) @ vt)


def collect(gt_h5: str, lifted_h5: str, camera: str, joint: int,
            n_clips: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame (R_lifted, R_gt) matrix stacks for one camera."""
    lifted_rots, gt_rots = [], []
    with h5py.File(gt_h5, "r") as gf, h5py.File(lifted_h5, "r") as lf:
        used = 0
        for split in gf:
            if split not in lf:
                continue
            for clip in gf[split]:
                if n_clips and used >= n_clips:
                    break
                if clip not in lf[split] or camera not in lf[split][clip]:
                    continue
                gt_poses = gf[split][clip]["poses"]
                lift_poses = lf[split][clip][camera]["poses"]
                idx = align_indices(gt_poses.shape[0], lift_poses.shape[0])

                gt_rots.append(R.from_rotvec(gt_poses[:, joint, :][idx]).as_matrix())
                lifted_rots.append(R.from_rotvec(lift_poses[:, joint, :]).as_matrix())
                used += 1
    return np.concatenate(lifted_rots), np.concatenate(gt_rots)


def analyse(gt_h5: str, lifted_h5: str, camera: str, joint: int,
            n_clips: int | None) -> None:
    lifted, gt = collect(gt_h5, lifted_h5, camera, joint, n_clips)
    n = len(lifted)

    # Per-frame offset, then its rotation-average.
    per_frame = lifted @ gt.transpose(0, 2, 1)
    r_off = mean_rotation(per_frame)

    # Spread of the per-frame offsets about the mean = how constant the offset is.
    spread = (R.from_matrix(per_frame) * r_off.inv()).magnitude()

    # Residual if we simply apply the fitted offset to GT.
    resid = (R.from_matrix(lifted) * (r_off * R.from_matrix(gt)).inv()).magnitude()

    # Baseline: how far apart are they with no correction at all?
    raw = (R.from_matrix(lifted) * R.from_matrix(gt).inv()).magnitude()

    vec = r_off.as_rotvec()
    ang = np.linalg.norm(vec)
    axis = vec / ang if ang > 1e-9 else np.zeros(3)

    print(f"\n── {camera}  joint {joint}  ({n} frames) ──")
    print(f"  best-fit offset : angle {np.degrees(ang):7.2f}°  "
          f"axis [{axis[0]:+.3f} {axis[1]:+.3f} {axis[2]:+.3f}]")
    print(f"  euler xyz (deg) : {np.round(r_off.as_euler('xyz', degrees=True), 2)}")
    print(f"  uncorrected err : mean {np.degrees(raw.mean()):7.2f}°  "
          f"median {np.degrees(np.median(raw)):7.2f}°")
    print(f"  residual after  : mean {np.degrees(resid.mean()):7.2f}°  "
          f"median {np.degrees(np.median(resid)):7.2f}°  "
          f"p90 {np.degrees(np.percentile(resid, 90)):7.2f}°")
    print(f"  offset spread   : std {np.degrees(spread.std()):7.2f}°  "
          f"(0° would mean a perfectly constant offset)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_h5", default="data/movi.h5")
    parser.add_argument("--lifted_h5", default="lifted_movi_part1_upd2.h5")
    parser.add_argument("--cameras", nargs="+", default=["PG1", "PG2"])
    parser.add_argument("--joints", nargs="+", type=int, default=[0, 1, 2],
                        help="0 is the root; others are parent-relative controls")
    parser.add_argument("--n_clips", type=int, default=40)
    args = parser.parse_args()

    print(f"GT     : {args.gt_h5}")
    print(f"lifted : {args.lifted_h5}   (GT subsampled onto the lifted timeline)")
    for camera in args.cameras:
        for joint in args.joints:
            analyse(args.gt_h5, args.lifted_h5, camera, joint, args.n_clips)


if __name__ == "__main__":
    main()
