"""
Held-out reprojection improvement, recovered from saved checkpoints.

`pose/img_improvement_px` is normally written to TensorBoard during training by
`ImagePoseVizLogger`, but only when `--viz_interval > 0`. The nine runs in
`checkpoints/sweep/` were launched with `viz_interval = 0`, so the scalar was
never written.

It does not need to be. The metric is a pure function of the final actor weights
and the scored clips — `ImagePoseVizLogger._rollout` just calls `rollout_policy`
with the actor — so it can be recovered afterwards by rolling each saved
`actor_final.pt` over the split. Only the endpoint comes back this way, not the
curve, which is what `report.md` reads anyway.

**Scored on `test` by default**, the same split as the PA-MPJPE table, so the two
rows of report.md describe the same clips. It used to default to `val`, which was
inherited from `ImagePoseVizLogger`'s role as a training-time logger rather than
chosen for this metric; nothing was ever selected on val, so the separation bought
nothing and cost comparability. `--split val` reproduces the older numbers.

Every checkpoint is scored on the **same** clips: `--clip_seed` is fixed here
rather than taken from each run's own `cfg["seed"]`, so the lifted error is
identical across all runs and the variant/seed comparison is exact. Check that in
the output — if the lifted column is not constant, the clips differed and the
comparison is void.

Note the errors are **raw** pixels: `correction_magnitude()` computes distances
directly and applies no keypoint-bias subtraction, unlike the training-rollout
reprojection rows. The bias is common to the lifted and corrected sides, so the
improvement column is comparable; the levels are not.

Usage:
    python scripts/heldout_eval.py                       # all runs, markdown table
    python scripts/heldout_eval.py --runs frozen_s42     # just one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Same idiom as scripts/build_reproj_targets.py: invoked as a path, not a
# module, so the repo root is not on sys.path.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.datasets import MoViDataset  # noqa: E402
from src.models.policy import PoseActor, trans_mode_from_width
from src.rewards import ReprojectionReward, load_calib
from src.viz_pose import ImagePoseVizLogger

VARIANTS = {"frozen": "(B1) frozen", "uv": "(B2) du,dv", "notrans": "(B3) pose-only",
            # (C) feet ablation, under checkpoints/gail_c; absent from a (B)
            # run set, where the table loop simply skips them.
            "feet_in": "(C) GAIL, feet in", "feet_out": "(C) GAIL, feet out",
            # (D) supervised-MSE arms, under checkpoints/exp_d.
            "mse1": "(D) MSE, w=1", "mse10": "(D) MSE, w=10"}


def evaluate_one(ckpt_path: Path, h5: str, norm: str, n_clips: int,
                 n_frames: int, clip_seed: int, device: str,
                 split: str = "test") -> dict:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt["config"]

    # Width and observation layout are properties of the checkpoint, not of the
    # current defaults — same reasoning as src/evaluate.py.
    trans_mode = trans_mode_from_width(ckpt["actor"]["log_std"].shape[0])
    use_evidence = bool(cfg.get("use_evidence", False)) and cfg.get("reward_mode") == "reproj"
    state_trans = bool(cfg.get("state_trans", True))

    actor = PoseActor(hidden_dims=tuple(cfg["hidden_dims"]), trans_mode=trans_mode,
                      use_evidence=use_evidence, state_trans=state_trans).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    reproj_path = cfg.get("reproj_path", "data/reproj_targets.h5")
    dataset = MoViDataset(h5, norm, split=split, verbose=False, reproj_path=reproj_path)
    with open(norm) as f:
        gt_stats = json.load(f)

    logger = ImagePoseVizLogger(
        dataset, actor, gt_stats, load_calib(h5), reproj_path,
        video_root=cfg.get("viz_video_root", "demo/videos"), device=device,
        n_clips=n_clips, n_frames=n_frames,
        seed=clip_seed,   # fixed across runs, deliberately NOT cfg["seed"]
        reproj_reward=ReprojectionReward(
            load_calib(h5), sigma=cfg.get("reproj_sigma", 0.0225),
            bias=cfg.get("kp_bias_path") or None, correct_translation=False),
        use_evidence=use_evidence, trans_mode=trans_mode, state_trans=state_trans)

    out = logger.correction_magnitude()
    out["n_clips_used"] = len(logger.clips)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_dir", default="checkpoints/sweep")
    p.add_argument("--runs", nargs="+", default=None,
                   help="run names; default is every <variant>_s<seed> found")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--h5_path", default="data/processed_movi.h5")
    p.add_argument("--norm_stats_path", default="data/normalization.json")
    p.add_argument("--split", default="test",
                   help="split to score on; 'test' matches the PA-MPJPE table, "
                        "'val' reproduces the older held-out numbers")
    p.add_argument("--n_clips", type=int, default=200,
                   help="clips sampled from the split; the default covers "
                        "all 187 test clips")
    p.add_argument("--n_frames", type=int, default=12)
    p.add_argument("--clip_seed", type=int, default=42,
                   help="fixed clip selection, so all runs see identical clips")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dump", default=None, help="write raw results to this JSON")
    args = p.parse_args()

    sweep = Path(args.sweep_dir)
    runs = args.runs or [f"{v}_s{s}" for v in VARIANTS for s in args.seeds
                         if (sweep / f"{v}_s{s}" / "actor_final.pt").exists()]

    res = {}
    for r in runs:
        res[r] = evaluate_one(sweep / r / "actor_final.pt", args.h5_path,
                              args.norm_stats_path, args.n_clips, args.n_frames,
                              args.clip_seed, args.device, split=args.split)
        print(f"  {r:<14} improvement {res[r]['pose/img_improvement_px']:+.4f} px "
              f"(lifted {res[r]['pose/img_err_lifted_px']:.4f}, "
              f"{res[r]['n_clips_used']} clips)", flush=True)

    lifted = {round(v["pose/img_err_lifted_px"], 6) for v in res.values()}
    print(f"\nLifted error across runs: {sorted(lifted)}")
    if len(lifted) != 1:
        print("  WARNING: runs did not see identical clips — comparison is void.")

    print(f"\n| variant | " + " | ".join(f"s{s}" for s in args.seeds)
          + " | mean ± sd | sign |")
    print("|---" * (len(args.seeds) + 3) + "|")
    for v, label in VARIANTS.items():
        got = [res[f"{v}_s{s}"]["pose/img_improvement_px"] if f"{v}_s{s}" in res
               else None for s in args.seeds]
        have = [g for g in got if g is not None]
        if not have:
            continue
        cells = " | ".join("—" if g is None else f"{g:+.3f}" for g in got)
        sign = " ".join("+" if g > 0 else "-" for g in have)
        # ddof=1 needs two samples; a single run has no spread to report.
        sd = f"{np.std(have, ddof=1):.3f}" if len(have) > 1 else "n/a"
        print(f"| {label} | {cells} | **{np.mean(have):+.3f} ± {sd}** | {sign} |")

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nraw results -> {args.dump}")


if __name__ == "__main__":
    main()
