"""
Does the ground-truth pose reproject *worse* than the lifted pose?

This is the argument for condition (C): if maximising agreement with ViTPose
moves the pose away from GT, then a reward built on reprojection alone has its
optimum displaced from the thing we actually want, and an adversarial term is
needed to pull it back.

The claim used to be quoted in `report.md` from a **validation**-split
measurement, while every other reprojection figure in the report had been moved
to `test`. That made the one number carrying the (C) argument the only one
describing different clips from the rest. This script measures it on whichever
split is asked for, defaulting to `test`.

It scores the **same clips** as `scripts/heldout_eval.py`: the clip set comes
from `ImagePoseVizLogger`'s own selection at the same `--clip_seed`, and the
frames are picked by the same rule (`_rollout`'s evenly-spaced draw from frames
that carry 2D evidence). No policy is involved -- neither side of this
comparison is a rollout, so no actor is loaded.

Errors are **raw** pixels, confidence-weighted over the 17 COCO joints, with no
keypoint-bias subtraction -- the same convention as `heldout_eval.py`, and the
reason the levels here read ~12 px rather than the ~8 px of the bias-corrected
training-rollout rows.

Betas: the lifted side is rendered with the lifted betas and the GT side with GT
betas, which is what each pose is actually paired with. Because body *shape*
also moves the projection, the GT pose is scored a second time with the lifted
betas -- `gt_pose_lifted_betas` -- which isolates the effect of the pose alone.
Compare that column, not the first one, before attributing the gap to pose.

Usage:
    python scripts/gt_reproj_check.py                       # test, 400 clips
    python scripts/gt_reproj_check.py --split val --n_clips 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.datasets import MoViDataset          # noqa: E402
from src.reproject import COCO_IDX, SMPLX_IDX      # noqa: E402
from src.rewards import _load_bias, load_calib, load_lifted_stats, unscale  # noqa: E402
from src.viz_pose import KP_MIN_CONFIDENCE, ImagePoseVizLogger   # noqa: E402


def stats_tuple(path: str) -> tuple:
    """`data/normalization.json` -> the (mu, sigma) tuple `unscale` expects.

    `load_lifted_stats` does this for the per-camera lifted files but refuses
    anything without `_root_corrected`, and the GT file does not carry the flag
    -- correctly, since GT is natively in the world frame and was never rotated.
    """
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key in ("poses", "trans", "betas"):
        mu = np.asarray(raw[key]["mu"], dtype=np.float32)
        sigma = np.asarray(raw[key]["sigma"], dtype=np.float32)
        out[key] = (mu, np.where(sigma == 0, 1.0, sigma).astype(np.float32))
    return tuple(out.items())


def weighted_px_err(proj: np.ndarray, kp: np.ndarray, bias=None,
                    bbox_h: float = 1.0) -> float | None:
    """Confidence-weighted mean |projected - ViTPose| over the COCO joints.

    With `bias`, subtracts the per-joint COCO<->SMPL-X offset first, exactly as
    `ReprojectionReward.step` does -- offset in bbox-height units, scaled by the
    frame's bbox height. Note the offset in `data/kp_bias.json` is fitted on the
    **lifted** poses, so applying it to a GT projection is a deliberate
    mismatch: it is what the older bias-corrected comparison did, and measuring
    it is the point of the `--bias` flag.
    """
    w = np.where(kp[:, 2] >= KP_MIN_CONFIDENCE, kp[:, 2], 0.0)
    if w.sum() <= 0:
        return None
    delta = proj[SMPLX_IDX] - kp[:, :2]
    if bias is not None:
        delta = delta - bias * (bbox_h or 1.0)
    d = np.linalg.norm(delta, axis=-1)
    return float(np.sum(w * d) / np.sum(w))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5_path", default="data/processed_movi.h5")
    p.add_argument("--norm_stats_path", default="data/normalization.json")
    p.add_argument("--reproj_path", default="data/reproj_targets.h5")
    p.add_argument("--video_root", default="demo/videos")
    p.add_argument("--split", default="test")
    p.add_argument("--n_clips", type=int, default=400)
    p.add_argument("--n_frames", type=int, default=12)
    p.add_argument("--clip_seed", type=int, default=42,
                   help="must match heldout_eval.py to score identical clips")
    p.add_argument("--bias", default=None,
                   help="path to data/kp_bias.json to subtract the fitted "
                        "COCO<->SMPL-X offset, reproducing the bias-corrected "
                        "convention of the training-rollout rows. The offset is "
                        "fitted on LIFTED poses, so it under-corrects GT -- "
                        "which is what this flag exists to quantify.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dump", default=None)
    args = p.parse_args()

    gt_stats = stats_tuple(args.norm_stats_path)
    dataset = MoViDataset(args.h5_path, args.norm_stats_path, split=args.split,
                          verbose=False, reproj_path=args.reproj_path)
    calib = load_calib(args.h5_path)
    bias = _load_bias(args.bias) if args.bias else None

    # Actor is None on purpose: __init__ only discovers clips, and nothing below
    # calls _rollout. Reusing the class is what guarantees the clip set matches.
    logger = ImagePoseVizLogger(dataset, None, gt_stats, calib, args.reproj_path,
                                video_root=args.video_root, device=args.device,
                                n_clips=args.n_clips, n_frames=args.n_frames,
                                seed=args.clip_seed)

    acc: dict[str, list[float]] = {"lifted": [], "gt": [], "gt_lifted_betas": []}
    n_frames_scored = 0

    for entry in logger.clips:
        sample = dataset[entry["idx"]]
        x, y, reproj = sample["x"], sample["y"], sample["reproj"]
        valid = reproj["valid"].cpu().numpy().astype(bool)
        T = y["poses"].shape[0]
        steps = min(logger.max_steps, T) - 1
        if steps < 1 or not valid[:steps].any():
            continue

        usable = np.flatnonzero(valid[:steps])
        picks = usable[np.linspace(0, len(usable) - 1,
                                   args.n_frames).round().astype(int)]

        cam = entry["camera"]
        lifted_stats = load_lifted_stats(cam)
        betas_lift = unscale(x["betas"].cpu().numpy(), "betas", lifted_stats)
        betas_gt = unscale(y["betas"].cpu().numpy(), "betas", gt_stats)
        kp2d = reproj["kp2d"].cpu().numpy()
        tmet = reproj["trans_metric"].cpu().numpy()
        bbox = reproj["bbox"].cpu().numpy()
        b = bias.get(cam.upper()) if bias else None

        for t in picks:
            kp = kp2d[t][COCO_IDX]
            proj = {
                "lifted": logger._project(x["poses"][t].cpu().numpy(), betas_lift,
                                          lifted_stats, cam, tmet[t]),
                "gt": logger._project(y["poses"][t].cpu().numpy(), betas_gt,
                                      gt_stats, cam, tmet[t]),
                "gt_lifted_betas": logger._project(y["poses"][t].cpu().numpy(),
                                                   betas_lift, gt_stats, cam, tmet[t]),
            }
            errs = {k: weighted_px_err(v, kp, b, float(bbox[t][3]))
                    for k, v in proj.items()}
            if any(e is None for e in errs.values()):
                continue
            for k, e in errs.items():
                acc[k].append(e)
            n_frames_scored += 1

    if not n_frames_scored:
        raise SystemExit(f"no scorable frames in split '{args.split}'")

    means = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"\nsplit={args.split}  {len(logger.clips)} clip-cameras  "
          f"{n_frames_scored} frames scored  "
          f"bias={'lifted-fitted (' + args.bias + ')' if bias else 'none (raw)'}\n")
    print("| pose | betas | reprojection error (px) |")
    print("|---|---|---|")
    print(f"| lifted (SMPLer-X) | lifted | {means['lifted']:.2f} |")
    print(f"| ground truth | GT | {means['gt']:.2f} |")
    print(f"| ground truth | lifted | {means['gt_lifted_betas']:.2f} |")
    gap = means["gt_lifted_betas"] - means["lifted"]
    print(f"\nGT minus lifted, betas held fixed: {gap:+.2f} px "
          f"({'GT reprojects WORSE' if gap > 0 else 'GT reprojects BETTER'})")

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump({"split": args.split, "n_frames": n_frames_scored,
                       "n_clips": len(logger.clips), "means": means}, f, indent=2)
        print(f"\nraw results -> {args.dump}")


if __name__ == "__main__":
    main()
