"""
check_extrinsics.py
───────────────────
Measure how well MoVi's calibration places GT in the image, per camera.

This exists because the number it produces was once wrong in the docstrings for
months. §5 of `follow-up-on-gustaf-questions.md` claimed the PG2 extrinsics were
"measurably less accurate" than PG1's — 65-75 px against 9.7 px — and that claim
propagated into `src/reproject.py`, `src/viz_pose.py` and the summary. It does
not reproduce: with the correct convention both cameras land within ~15 px. The
measurement was never committed, which is exactly why it outlived its own
correctness, so it is committed now.

What is measured:

    GT SMPL-X poses (world frame, metres)
      -> SMPL-X forward kinematics                       (src/smplx_fk.py)
      -> + GT trans                                      model origin, not pelvis
      -> world -> camera through Extrinsics_PG*.npz
      -> project with IntrinsicMatrix + RadialDistortion (src/reproject.py)
      -> pixel distance to the ViTPose COCO-17 detections, on the 12 joints
         with an unambiguous SMPL-X counterpart

Two conventions are load-bearing and neither is documented in the MoVi release:

  * `rotationMatrix` is MATLAB's row-vector convention (`X_cam = X_world·R + t`),
    so the column-vector map is `R.T`, and it composes with the same `WORLD_FLIP`
    relabel `src/camera_frame.py` needs: `X_cam = R.T · F · X_world + t`.
  * `translationVector` is in **millimetres**; GT is in metres.

Get either wrong and the error is 90-270 px — on *both* cameras, which is the
tell that a large asymmetry between PG1 and PG2 is not a convention error.

`--sweep` prints every combination so that stays visible.

Usage:
    python scripts/check_extrinsics.py                  # per-clip, 60 test clips
    python scripts/check_extrinsics.py --sweep --clips 10
    python scripts/check_extrinsics.py --split val --clips 0   # 0 = all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.camera_frame import WORLD_FLIP  # noqa: E402
from src.data.datasets import resample_to  # noqa: E402
from src.reproject import COCO_IDX, SMPLX_IDX, project  # noqa: E402
from src.smplx_fk import joints_from_poses  # noqa: E402

CAMERAS = ("pg1", "pg2")
# Below this the detection is too uncertain to be a target worth scoring against.
MIN_CONF = 0.5


def unnormalize(stats, key, arr):
    mu = np.array(stats[key]["mu"], dtype=np.float64)
    sigma = np.array(stats[key]["sigma"], dtype=np.float64)
    return arr * np.where(sigma == 0, 1.0, sigma) + mu


def calibration(cam: str, calib_dir: Path):
    """(K, radial, R_extrinsic, t_millimetres) as stored."""
    params = np.load(calib_dir / f"cameraParams_{cam.upper()}.npz")
    ext = np.load(calib_dir / f"Extrinsics_{cam.upper()}.npz")
    return (params["IntrinsicMatrix"], params["RadialDistortion"],
            ext["rotationMatrix"], ext["translationVector"])


def conventions(R_ext, t_mm, sweep: bool):
    """(name, world->camera matrix, translation) for each convention to test."""
    correct = ("R^T·F + t/1000", R_ext.T @ WORLD_FLIP, t_mm / 1000.0)
    if not sweep:
        yield correct
        return
    for t_name, t in (("t/1000", t_mm / 1000.0), ("t(m)", t_mm), ("t=0", np.zeros(3))):
        for r_name, R in (("R^T", R_ext.T), ("R", R_ext)):
            for f_name, F in (("F", WORLD_FLIP), ("I", np.eye(3))):
                yield f"{r_name}·{f_name} + {t_name}", R @ F, t


def gt_joints_world(clip, stats):
    """(T, 22, 3) GT joints in the world frame, metres, at the model origin."""
    gt = clip["gt"] if "gt" in clip else clip
    joints = joints_from_poses(
        unnormalize(stats, "poses", gt["poses"][:]),
        unnormalize(stats, "betas", gt["betas"][:]),
        n_joints=22,
        gender=str(clip.attrs.get("gender", "neutral")),
    ).numpy().astype(np.float64)
    # + trans, NOT pelvis-recentred — same trap as src/reproject.place_in_camera.
    return joints + unnormalize(stats, "trans", gt["trans"][:])[:, None, :]


def targets(target_grp, T):
    """(uv (T,12,2), usable (T,12) bool) resampled onto the clip's timeline."""
    kp = resample_to(target_grp["kp2d"][:], T)
    valid = resample_to(target_grp["valid"][:].astype(np.float32), T) > 0.5
    return kp[:, COCO_IDX, :2], valid[:, None] & (kp[:, COCO_IDX, 2] > MIN_CONF)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=str(REPO / "data/processed_movi.h5"))
    ap.add_argument("--reproj", default=str(REPO / "data/reproj_targets.h5"))
    ap.add_argument("--calib_dir", default=str(REPO / "data/Calib"))
    ap.add_argument("--norm_stats", default=str(REPO / "data/normalization.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", type=int, default=60, help="0 for all")
    ap.add_argument("--sweep", action="store_true",
                    help="try every rotation/translation convention, not just the correct one")
    args = ap.parse_args()

    calib_dir = Path(args.calib_dir)
    with open(args.norm_stats) as f:
        stats = json.load(f)

    with h5py.File(args.h5, "r") as h5, h5py.File(args.reproj, "r") as tg:
        split, tsplit = h5[args.split], tg[args.split]
        names = [c for c in split if c in tsplit]
        if args.clips:
            names = names[:args.clips]
        print(f"{len(names)} {args.split} clips, {calib_dir}\n")

        # {cam: {convention: [per-frame errors]}} plus per-clip medians.
        errors = {cam: {} for cam in CAMERAS}
        per_clip = {cam: [] for cam in CAMERAS}

        for name in names:
            clip = split[name]
            joints = gt_joints_world(clip, stats)
            T = joints.shape[0]
            for cam in CAMERAS:
                if cam not in tsplit[name] or not tsplit[name][cam].attrs["aligned"]:
                    continue
                K, radial, R_ext, t_mm = calibration(cam, calib_dir)
                uv_target, usable = targets(tsplit[name][cam], T)
                if not usable.any():
                    continue
                for conv, M, t in conventions(R_ext, t_mm, args.sweep):
                    uv = project(joints @ M.T + t, K, radial)[:, SMPLX_IDX, :]
                    err = np.linalg.norm(uv - uv_target, axis=-1)[usable]
                    errors[cam].setdefault(conv, []).append(err)
                    if not args.sweep:
                        per_clip[cam].append((float(np.median(err)), name))

    for cam in CAMERAS:
        if not errors[cam]:
            continue
        print(f"── {cam.upper()} ──")
        rows = sorted((float(np.median(e)), conv, float(e.mean()))
                      for conv, chunks in errors[cam].items()
                      for e in [np.concatenate(chunks)])
        for median, conv, mean in rows:
            print(f"   {conv:22s} median {median:9.1f} px   mean {mean:9.1f} px")
        if not args.sweep:
            medians = np.array([m for m, _ in per_clip[cam]])
            print(f"   per-clip median: mean {medians.mean():.1f}  "
                  f"p90 {np.percentile(medians, 90):.1f}  max {medians.max():.1f}")
            worst = ", ".join(f"{n}={v:.0f}px" for v, n in sorted(per_clip[cam])[-3:])
            print(f"   worst clips: {worst}")
        print()


if __name__ == "__main__":
    main()
