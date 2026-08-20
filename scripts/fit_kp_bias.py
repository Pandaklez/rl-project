"""
Estimate the systematic COCO <-> SMPL-X keypoint offset, per joint and camera.

**What this corrects.** SMPL-X skeleton joints are internal kinematic centres
produced by the joint regressor; COCO keypoints are human-annotated surface
landmarks, and ViTPose reproduces the convention it was trained on. The two are
not the same anatomical point, so projecting a *correct* SMPL-X pose lands a
systematic distance away from a *correct* detection. Measured here at ~0.02 bbox
heights (about 10 px), down and slightly right, on every clip and both cameras.

**Why it is not a pose error.** Ground truth projected through the extrinsics
path — which shares no code with the lifted/`cam_trans` path — carries the same
offset with the same per-joint shape: global (+0.0147, +0.0235) on PG1 against
the lifted path's (+0.0112, +0.0208), and hips roughly twice everything else in
both. Two independent 3D sources agreeing with each other and disagreeing with
the detector the same way is a correspondence problem, not an accuracy problem.

**Why bbox-height units.** Regressing the offset in pixels on bbox height gives
`v: +3.62 + 0.01324*h`, i.e. mostly proportional to apparent body size rather
than fixed in the image — an offset in body units, as a joint-definition offset
should be. It is also the unit the reward already normalises in, so the
correction composes with `err_norm` directly. (The discrimination is a lean, not
a proof: const-px and const-bbox-height models differ by 0.07 px RMS against a
~4 px frame-to-frame spread.)

**No ground truth.** Fitted from the *lifted* poses on the *train* split only,
so a reward that subtracts it stays GT-free and free of test leakage.

    python -m scripts.fit_kp_bias --out data/kp_bias.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=str(REPO / "data/processed_movi.h5"))
    ap.add_argument("--reproj", default=str(REPO / "data/reproj_targets.h5"))
    ap.add_argument("--norm_stats", default=str(REPO / "data/normalization.json"))
    ap.add_argument("--split", default="train",
                    help="fit on train only; anything else leaks into evaluation")
    ap.add_argument("--clips", type=int, default=250)
    ap.add_argument("--stride", type=int, default=29, help="frame stride within a clip")
    ap.add_argument("--max_frames", type=int, default=30, help="frames per clip")
    ap.add_argument("--out", default=str(REPO / "data/kp_bias.json"))
    args = ap.parse_args()

    from src.data.datasets import MoViDataset
    from src.rewards import N_KP, ReprojectionReward, load_calib

    if args.split != "train":
        print(f"WARNING: fitting on '{args.split}', not 'train'. Anything the "
              f"reward subtracts must be estimated on train only.")

    ds = MoViDataset(args.h5, args.norm_stats, split=args.split, verbose=False,
                     reproj_path=args.reproj)
    rr = ReprojectionReward(load_calib(args.h5), correct_translation=False,
                            bias=None)          # fit the raw residual, uncorrected

    rng = np.random.default_rng(0)
    order = rng.permutation(len(ds))[:args.clips]
    per_cam: dict[str, list] = {}
    for i in order:
        sample = ds[int(i)]
        if not rr.reset(sample):
            continue
        cam = str(sample["meta"]["camera"]).upper()
        frames = np.flatnonzero(rr._clip["valid"])[::args.stride][:args.max_frames]
        for t in frames:
            t = int(t)
            _, info = rr.step({"poses": sample["x"]["poses"][t],
                               "trans": sample["x"]["trans"][t]}, t)
            if not info["valid"] or info["conf"].sum() <= 0:
                continue
            # Joints below the confidence threshold carry a zeroed residual, which
            # is not a measurement of zero — mask them out rather than averaging
            # them in.
            r = np.where(info["conf"][:, None] > 0, info["resid"], np.nan)
            per_cam.setdefault(cam, []).append(r)

    if not per_cam:
        raise SystemExit("no usable frames — check --h5 / --reproj paths")

    out = {"_units": "bbox heights, (du, dv) = projected - observed",
           "_split": args.split, "_note": "subtract from the residual before scoring"}
    print(f"{'cam':5s} {'frames':>7s} {'joint':>3s}  per-joint median offset (du, dv)")
    for cam, chunks in sorted(per_cam.items()):
        A = np.array(chunks)                                     # (F, 12, 2)
        # Median, not mean: a handful of badly-detected frames would otherwise
        # drag a per-joint offset that is supposed to describe the typical case.
        med = np.nanmedian(A, axis=0)
        n = int(np.isfinite(A[..., 0]).sum())
        out[cam] = med.round(6).tolist()
        print(f"{cam:5s} {len(A):7d}  global ({med[:, 0].mean():+.4f}, {med[:, 1].mean():+.4f})  "
              f"n_obs={n}")
        for j in range(N_KP):
            print(f"{'':5s} {'':7s} {j:3d}  ({med[j, 0]:+.4f}, {med[j, 1]:+.4f})")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
