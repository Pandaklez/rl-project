"""
PA-MPJPE evaluation on the test split.

GT is always loaded from processed_movi.h5 and unnormalized with GT norm stats
(it was upsampled to full frame rate during preprocessing).

Two optional comparisons:
  1. Raw lifted baseline  — poses from lifted_movi_part1_upd1.h5, upsampled to GT length.
     Enable with: --lifted_h5
  2. Model output         — actor rolled out on processed_movi.h5, unnormalized.
     Enable with: --checkpoint

At least one of the two must be requested.

Usage examples:
    # Lifted baseline only
    python -m src.evaluate \\
        --lifted_h5 lifted_movi_part1_upd1.h5

    # Both
    python -m src.evaluate \\
        --lifted_h5 lifted_movi_part1_upd1.h5 \\
        --checkpoint checkpoints/ckpt_03000.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import h5py
import numpy as np
import torch

from src.data.datasets import MoViDataset, gt_group
from src.env import rollout_policy
from src.models.policy import PoseActor, trans_mode_from_width
from src.smplx_fk import joints_from_poses


# ─── Checkpoints ──────────────────────────────────────────────────────────────

def load_checkpoint(path: str, device) -> dict:
    """
    Load a training checkpoint, with `config` normalised to a plain dict.

    `src/train.py` now stores `asdict(cfg)`. Checkpoints written before that
    pickled the `Config` dataclass itself, and because `python -m src.train` runs
    that module as `__main__`, the pickle names the class `__main__.Config` — so
    any other process loading it dies with

        AttributeError: Can't get attribute 'Config' on <module '__main__'>

    Re-exporting the class under `__main__` before the load is what lets those
    older checkpoints still open. Unpickling only needs the name to resolve; the
    dataclass it resolves to is the same one.
    """
    import __main__

    if not hasattr(__main__, "Config"):
        try:
            from src.train import Config
            __main__.Config = Config
        except ImportError:      # training deps absent — fine unless the ckpt is old
            pass

    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt.get("config")
    if cfg is not None and not isinstance(cfg, dict):
        cfg = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
        ckpt["config"] = cfg
    return ckpt


# ─── Procrustes ───────────────────────────────────────────────────────────────

def procrustes_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Full Procrustes alignment (rotation + uniform scale + translation) per frame.
    pred, gt : (T, J, 3)
    Returns  : (T, J, 3) — pred aligned to gt
    """
    mu_p   = pred.mean(dim=1, keepdim=True)
    mu_g   = gt.mean(dim=1, keepdim=True)
    pred_c = pred - mu_p
    gt_c   = gt   - mu_g

    s_p    = pred_c.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)
    s_g    = gt_c.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)
    pred_n = pred_c / s_p
    gt_n   = gt_c   / s_g

    H        = pred_n.transpose(1, 2) @ gt_n          # (T, 3, 3)
    U, _, Vh = torch.linalg.svd(H)

    d    = torch.linalg.det(Vh.transpose(1, 2) @ U.transpose(1, 2))
    ones = torch.ones_like(d)
    D    = torch.diag_embed(torch.stack([ones, ones, d], dim=-1))
    R    = Vh.transpose(1, 2) @ D @ U.transpose(1, 2)

    return s_g * (pred_n @ R.transpose(1, 2)) + mu_g


def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Mean PA-MPJPE in metres over all frames and joints.

    pred/gt must be 3D joint POSITIONS (T, J, 3) from the body model — see
    src/smplx_fk.joints_from_poses. Passing axis-angle pose vectors here yields
    a number with no geometric meaning.
    """
    return (procrustes_align(pred, gt) - gt).pow(2).sum(dim=-1).sqrt().mean().item()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def unnormalize_np(arr: np.ndarray, key: str, stats: dict) -> np.ndarray:
    mu    = np.array(stats[key]["mu"],    dtype=np.float32)
    sigma = np.array(stats[key]["sigma"], dtype=np.float32)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return arr.astype(np.float32) * sigma + mu


def unnormalize_t(t: torch.Tensor, key: str, stats: dict) -> torch.Tensor:
    mu    = torch.tensor(stats[key]["mu"],    dtype=t.dtype, device=t.device)
    sigma = torch.tensor(stats[key]["sigma"], dtype=t.dtype, device=t.device).clamp(min=1e-8)
    return t * sigma + mu


def upsample_to(arr: np.ndarray, T: int) -> np.ndarray:
    """Linearly interpolate arr (shape T0, ...) along axis 0 to length T."""
    T0 = arr.shape[0]
    if T0 == T:
        return arr
    rest   = arr.shape[1:]
    flat   = arr.reshape(T0, -1).astype(np.float64)
    t_src  = np.linspace(0, T - 1, T0)
    t_dst  = np.arange(T)
    out    = np.stack([np.interp(t_dst, t_src, flat[:, d]) for d in range(flat.shape[1])], axis=1)
    return out.reshape(T, *rest).astype(np.float32)


# ─── Mode 1: raw lifted baseline ──────────────────────────────────────────────

def eval_lifted_baseline(
    lifted_h5_path: str,
    processed_h5_path: str,
    norm_stats: dict,
    cameras: tuple[str, ...],
    split: str,
    device: str = "cpu",
) -> dict:
    scores: list[float] = []

    with h5py.File(processed_h5_path, "r") as proc_f, \
         h5py.File(lifted_h5_path,    "r") as lift_f:

        clip_names = list(proc_f[split].keys())
        for i, clip_name in enumerate(clip_names):
            # GT: unnormalize from processed_movi.h5
            gt_grp  = gt_group(proc_f[split][clip_name])
            gt_pose = unnormalize_np(gt_grp["poses"][:].astype(np.float32), "poses", norm_stats)
            betas   = unnormalize_np(gt_grp["betas"][:].astype(np.float32), "betas", norm_stats)
            T       = gt_pose.shape[0]

            if clip_name not in lift_f[split]:
                continue

            # GT betas are used for both sides so the metric reflects pose error
            # alone; the lifted shape estimate is near-chance and would only add
            # noise to a pose benchmark.
            gt_joints = joints_from_poses(gt_pose, betas, device=device)

            clip_scores: list[float] = []
            for cam in cameras:
                if cam not in lift_f[split][clip_name]:
                    continue
                lift_raw = lift_f[split][clip_name][cam]["poses"][:].astype(np.float32)
                lift_up  = upsample_to(lift_raw, T)                   # (T, J, 3)
                clip_scores.append(pa_mpjpe(
                    joints_from_poses(lift_up, betas, device=device),
                    gt_joints,
                ))

            if clip_scores:
                scores.append(float(np.mean(clip_scores)))

            if (i + 1) % 20 == 0:
                print(f"  [baseline {i+1:4d}/{len(clip_names)}]  PA-MPJPE={np.mean(scores):.5f}")

    return {"pa_mpjpe_lifted_raw": float(np.mean(scores)), "n_clips_baseline": len(scores)}


# ─── Mode 2: model checkpoint ─────────────────────────────────────────────────

@torch.no_grad()
def eval_model(
    actor:         PoseActor,
    dataset:       MoViDataset,
    norm_stats:    dict,
    device:        torch.device,
    reproj_reward = None,
    use_evidence:  bool = False,
    trans_mode:    str  = "none",
) -> dict:
    """
    PA-MPJPE of the corrected pose against GT.

    `use_evidence` must match how the policy was trained — see
    `src.env.rollout_policy`. A policy trained on the 362-d observation and
    evaluated on the 318-d one does not raise; it silently scores badly, which
    reads as a negative result rather than as the bug it is. `main` reads the
    flag out of the checkpoint's config for exactly that reason.
    """
    actor.eval()

    corrected_scores: list[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        y      = sample["y"]

        corr_frames = rollout_policy(
            sample, actor, device=str(device), reproj_reward=reproj_reward,
            use_evidence=use_evidence, deterministic=True, trans_mode=trans_mode,
        )
        if not corr_frames:
            continue
        gt_frames = [y["poses"][t].cpu() for t in range(len(corr_frames))]

        corr_n = torch.stack(corr_frames)   # (T-1, J, 3), normalized
        gt_n   = torch.stack(gt_frames)     # (T-1, J, 3), normalized

        corr = unnormalize_t(corr_n, "poses", norm_stats)
        gt   = unnormalize_t(gt_n,   "poses", norm_stats)

        betas = unnormalize_t(y["betas"].cpu(), "betas", norm_stats)
        corrected_scores.append(pa_mpjpe(
            joints_from_poses(corr, betas, device=str(device)),
            joints_from_poses(gt,   betas, device=str(device)),
        ))

        if (idx + 1) % 20 == 0:
            print(f"  [model {idx+1:4d}/{len(dataset)}]  PA-MPJPE={np.mean(corrected_scores):.5f}")

    return {"pa_mpjpe_corrected": float(np.mean(corrected_scores)), "n_clips_model": len(dataset)}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--processed_h5",    default="processed_movi.h5",
                        help="Normalized+upsampled HDF5 (always used for GT)")
    parser.add_argument("--norm_stats_path", default="data/normalization.json",
                        help="GT normalization stats (mu/sigma per key)")
    parser.add_argument("--lifted_h5",  default=None,
                        help="Raw lifted HDF5 for baseline (lifted_movi_part1_upd1.h5)")
    parser.add_argument("--cameras",    nargs="+", default=["PG1", "PG2"])
    parser.add_argument("--checkpoint", default=None,
                        help="Actor checkpoint .pt for model evaluation (optional)")
    parser.add_argument("--split",  default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.lifted_h5 is None and args.checkpoint is None:
        parser.error("Provide --lifted_h5 for baseline, --checkpoint for model eval, or both.")

    with open(args.norm_stats_path) as f:
        norm_stats = json.load(f)

    all_results: dict = {}

    # ── Lifted baseline ───────────────────────────────────────────────────────
    if args.lifted_h5:
        print(f"\n── Raw lifted baseline (GT from {args.processed_h5}, unnormalized) ──")
        res = eval_lifted_baseline(
            args.lifted_h5, args.processed_h5,
            norm_stats, tuple(args.cameras), args.split, args.device,
        )
        all_results.update(res)
        print(f"  PA-MPJPE lifted (raw)  : {res['pa_mpjpe_lifted_raw']:.5f}  ({res['n_clips_baseline']} clips)")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.checkpoint:
        device = torch.device(args.device)
        ckpt   = load_checkpoint(args.checkpoint, device)
        cfg    = ckpt["config"]

        # The action width is a property of the checkpoint, not of the current
        # default: 156 for a pose-only policy, 159 for one that also corrected
        # `trans`. Reading it back means checkpoints from either regime load.
        act_dim  = ckpt["actor"]["log_std"].shape[0]
        trans_mode = trans_mode_from_width(act_dim)
        # Likewise the observation width. A policy trained with the 2D evidence
        # block cannot be rolled out without it, and the failure is silent — it
        # just scores badly. The checkpoint records which it was.
        use_evidence = bool(cfg.get("use_evidence", False)) and cfg.get("reward_mode") == "reproj"
        actor = PoseActor(hidden_dims=tuple(cfg["hidden_dims"]),
                          trans_mode=trans_mode,
                          use_evidence=use_evidence).to(device)
        actor.load_state_dict(ckpt["actor"])
        print(f"\n── Model: {args.checkpoint} ──")
        print(f"  Action: {act_dim}-d (trans_mode={trans_mode})")
        print(f"  Observation: {actor.net[0].in_features}-d "
              f"({'pose + 2D evidence' if use_evidence else 'pose only'})")

        dataset = MoViDataset(
            args.processed_h5, args.norm_stats_path,
            split=args.split, verbose=True,
            reproj_path=cfg.get("reproj_path") if use_evidence else None,
        )
        print(f"  Clips: {len(dataset)}")

        reproj_reward = None
        if use_evidence:
            from src.rewards import ReprojectionReward, load_calib
            reproj_reward = ReprojectionReward(
                load_calib(args.processed_h5), sigma=cfg.get("reproj_sigma", 0.04),
                correct_translation=False,
            )

        res = eval_model(actor, dataset, norm_stats, device,
                         reproj_reward=reproj_reward, use_evidence=use_evidence,
                         trans_mode=trans_mode)
        all_results.update(res)
        print(f"  PA-MPJPE corrected     : {res['pa_mpjpe_corrected']:.5f}  ({res['n_clips_model']} clips)")

    print("\n=== Summary ===")
    for k, v in all_results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()



# PA-MPJPE lifted (raw)  : 0.605  (147 clips)
# original VS. lifted Vs. ppo updated Vs. GAIL + PPO