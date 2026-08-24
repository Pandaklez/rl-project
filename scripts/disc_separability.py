"""
How separable are the GAIL discriminator's real and fake samples, before it runs?

The discriminator's real samples are `gt/poses` and its fake samples are the
policy's corrected poses, which live in the lifted per-camera space. Both are
stored **already normalised** in `data/processed_movi.h5` — GT with the GT stats
(`data/normalization.json`), lifted with the per-camera stats
(`data/normalization_lifted_{pg1,pg2}.json`) — so this script reads them as they
are. Do not re-apply the stats: that would normalise twice and manufacture a
mismatch that the discriminator never sees.

Two questions, because they have different answers:

1. Can a trivial rule label the pair? Reported as per-dimension
   `d' = |mu_r - mu_f| / sqrt((sd_r^2 + sd_f^2)/2)`, plus the best *linear*
   rule (LDA), which bounds the whole family rather than one dimension at a
   time. On the 52-joint data the answer was yes and it was fatal: MoVi has no
   finger mocap, so the 90 hand dimensions normalised to exactly +/-1 in GT
   where SMPLer-X's predicted fingers never did.

2. Do the two spaces *mean* the same thing? They do not, and standardisation
   hides it: both sides have mean 0 and sd 1 in their own space, so the
   marginals match while a given coordinate denotes a different physical pose
   on each side. `--physical` reports the systematic pose error the reward
   would drive toward if the policy satisfied the discriminator perfectly.

3. **`--common_space`: separability in the space the discriminator (C) actually
   sees.** Questions 1 and 2 both describe the code as it stood *before*
   `src.models.discriminator.PoseSpace`, which fixed (2) by mapping the fake
   side into GT-normalised space. That fix necessarily undoes (1)'s answer: the
   ~50% of question 1 was a consequence of each side being standardised in its
   own space, and once both sides share one space the classes separate widely.
   This mode applies `PoseSpace` and reports the same two statistics on the
   result, so the number is comparable with the discriminator accuracy the (C)
   runs actually log. `--exclude_joints` matches `gail_train --disc_exclude_joints`
   (default `0` = global_orient; `0 10 11` also drops the feet, the (C) ablation).

    python scripts/disc_separability.py
    python scripts/disc_separability.py --physical
    python scripts/disc_separability.py --common_space --exclude_joints 0
    python scripts/disc_separability.py --common_space --exclude_joints 0 10 11
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _report(real: np.ndarray, fake: np.ndarray) -> None:
    """The two statistics, on whatever space the caller put the samples in.

    Shared by `separability` (each side in its own normalisation, i.e. what the
    discriminator saw before `PoseSpace`) and `separability_common` (both sides
    in GT space, i.e. what it sees now), so the two are measured identically and
    the numbers are directly comparable.
    """
    n_dim = real.shape[1]
    print(f"real {real.shape}  fake {fake.shape}")

    d = np.abs(real.mean(0) - fake.mean(0)) / np.sqrt((real.var(0) + fake.var(0)) / 2)
    print(f"per-dim d'          : mean {d.mean():.3f}  median {np.median(d):.3f}  "
          f"max {d.max():.3f} (dim {d.argmax()})")
    print(f"dims with d' > 0.5  : {(d > 0.5).sum()} / {d.size}")

    # The best linear rule, which no single-dimension statistic bounds.
    mu_r, mu_f = real.mean(0), fake.mean(0)
    S = (np.cov(real, rowvar=False) + np.cov(fake, rowvar=False)) / 2
    w = np.linalg.solve(S + 1e-6 * np.eye(n_dim), mu_r - mu_f)
    sep = float((mu_r - mu_f) @ w / np.sqrt(w @ S @ w))
    acc = 0.5 * (1 + math.erf(sep / (2 * math.sqrt(2))))
    print(f"best linear (LDA)   : d' = {sep:.2f}  -> ~{100*acc:.1f}% accuracy")

    # The 52-joint leak: a dimension pinned to exactly +/-1 on the real side was
    # a perfect label. Report the worst dimension rather than assuming which.
    sat = (np.abs(np.abs(real) - 1.0) < 1e-4).mean(0)
    print(f"worst dim's fraction of real frames at exactly +/-1: {sat.max():.4f} "
          f"(dim {sat.argmax()})")


def _load(h5_path: str, split: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """GT poses, and the lifted poses per camera (kept separate: `PoseSpace.fake`
    needs to know which camera's statistics a sample carries)."""
    real, per_cam = [], {c: [] for c in ("pg1", "pg2")}
    with h5py.File(h5_path, "r") as f:
        for clip in f[split].keys():
            grp = f[split][clip]
            real.append(grp["gt"]["poses"][:].reshape(len(grp["gt"]["poses"]), -1))
            for cam in ("pg1", "pg2"):
                if cam in grp:
                    per_cam[cam].append(
                        grp[cam]["poses"][:].reshape(len(grp[cam]["poses"]), -1))
    return (np.concatenate(real),
            [np.concatenate(per_cam[c]) for c in ("pg1", "pg2") if per_cam[c]])


def separability(h5_path: str, split: str) -> None:
    """Each side in its own normalisation — the pre-`PoseSpace` comparison."""
    real, per_cam = _load(h5_path, split)
    _report(real.astype(np.float64), np.concatenate(per_cam).astype(np.float64))


def separability_common(h5_path: str, split: str, exclude_joints: tuple[int, ...],
                        gt_stats: str, lifted_stats: str) -> None:
    """Both sides in GT-normalised space, via the same `PoseSpace` the (C) runs
    use — so this is the separability the trained discriminator is up against."""
    import torch

    from src.models.discriminator import PoseSpace

    space = PoseSpace(gt_stats, lifted_stats, exclude_joints=tuple(exclude_joints))
    print(f"PoseSpace: {len(space.keep_joints)} joints / {space.dim} dims, "
          f"excluding {list(space.exclude_joints)}")

    real, per_cam = _load(h5_path, split)
    real_t = space.real(torch.from_numpy(real.astype(np.float32))).numpy()
    fake_t = np.concatenate([
        space.fake(torch.from_numpy(p.astype(np.float32)), cam).numpy()
        for cam, p in zip(space.CAMERAS, per_cam)])
    _report(real_t.astype(np.float64), fake_t.astype(np.float64))


def joint_sd(h5_path: str, split: str, gt_stats: str, lifted_stats: str) -> None:
    """Per-joint articulation, GT against lifted, in **physical** axis-angle units.

    Motivates the (C) feet ablation. Both sides are stored normalised, each with
    its own statistics, so a standard deviation read off the stored values is 1
    by construction and says nothing; the stats have to be undone first. A joint
    SMPLer-X regresses toward its conditional mean has a much smaller physical
    spread than GT, and the discriminator can read that ratio without learning
    anything about pose plausibility.
    """
    g = json.load(open(gt_stats))["poses"]
    mu_g = np.array(g["mu"]).ravel()
    sd_g = np.array(g["sigma"]).ravel()

    real, per_cam = _load(h5_path, split)
    phys_r = real * sd_g + mu_g
    parts = []
    for cam, p in zip(("pg1", "pg2"), per_cam):
        l = json.load(open(lifted_stats.format(cam=cam)))["poses"]
        parts.append(p * np.array(l["sigma"]).ravel() + np.array(l["mu"]).ravel())
    phys_f = np.concatenate(parts)

    n_joints = mu_g.size // 3
    print(f"physical axis-angle sd (rad), {split} split, pooled over both cameras\n")
    print(f"{'joint':>6}{'GT':>9}{'lifted':>9}{'GT/lifted':>11}")
    ratios = {}
    for j in range(n_joints):
        d = slice(j * 3, j * 3 + 3)
        r = phys_r[:, d].std(0).mean()
        f = phys_f[:, d].std(0).mean()
        ratios[j] = r / f
        note = "  <- foot" if j in (10, 11) else ""
        print(f"{j:>6}{r:>9.4f}{f:>9.4f}{r / f:>11.2f}{note}")
    rest = [j for j in range(1, n_joints) if j not in (10, 11)]
    print(f"\nfeet (10, 11): {ratios[10]:.2f}x / {ratios[11]:.2f}x")
    print(f"mean over the other {len(rest)} body joints (excl. global_orient): "
          f"{np.mean([ratios[j] for j in rest]):.2f}x")


def physical(gt_stats: str, lifted_stats: str) -> None:
    """The pose error implied by scoring lifted-space poses with a GT-space critic.

    A latent z that is realistic in GT space denotes physical pose
    `mu_gt + sd_gt * z`. The policy emits that same z in lifted space, which
    denotes `mu_lifted + sd_lifted * z`. The gap is the bias the reward drives
    toward; over the real distribution z ~ (0, 1) its RMS per dimension is
    `sqrt((mu_l - mu_g)^2 + (sd_l - sd_g)^2)`.
    """
    g = json.load(open(gt_stats))["poses"]
    mu_g, sd_g = np.array(g["mu"]).ravel(), np.array(g["sigma"]).ravel()
    print(f"\nGT axis-angle sd, for scale: {sd_g.mean():.4f} rad "
          f"({np.degrees(sd_g.mean()):.1f} deg)")
    for cam in ("pg1", "pg2"):
        l = json.load(open(lifted_stats.format(cam=cam)))["poses"]
        mu_l, sd_l = np.array(l["mu"]).ravel(), np.array(l["sigma"]).ravel()
        rms = np.sqrt((mu_l - mu_g) ** 2 + (sd_l - sd_g) ** 2)
        ratio = sd_l / np.maximum(sd_g, 1e-12)
        print(f"{cam}: sd ratio min {ratio.min():.2f} / median {np.median(ratio):.2f} "
              f"/ max {ratio.max():.2f}")
        print(f"      implied pose error RMS {rms.mean():.4f} rad "
              f"({np.degrees(rms.mean()):.2f} deg); worst dim {rms.max():.4f} rad "
              f"({np.degrees(rms.max()):.2f} deg, dim {rms.argmax()})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_path", default=str(REPO / "data/processed_movi.h5"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--gt_stats", default=str(REPO / "data/normalization.json"))
    ap.add_argument("--lifted_stats",
                    default=str(REPO / "data/normalization_lifted_{cam}.json"))
    ap.add_argument("--common_space", action="store_true",
                    help="measure in the GT-normalised space PoseSpace maps both "
                         "sides into, i.e. what the (C) discriminator sees")
    ap.add_argument("--exclude_joints", type=int, nargs="*", default=[0],
                    help="joints hidden from the discriminator, matching "
                         "gail_train --disc_exclude_joints (default: 0)")
    ap.add_argument("--joint_sd", action="store_true",
                    help="per-joint GT-vs-lifted articulation in physical rad, "
                         "the motivation for the feet ablation")
    ap.add_argument("--physical", action="store_true",
                    help="also report the pose error the space mismatch implies")
    args = ap.parse_args()

    if args.joint_sd:
        joint_sd(args.h5_path, args.split, args.gt_stats, args.lifted_stats)
        return
    if args.common_space:
        separability_common(args.h5_path, args.split, tuple(args.exclude_joints),
                            args.gt_stats, args.lifted_stats)
    else:
        separability(args.h5_path, args.split)
    if args.physical:
        physical(args.gt_stats, args.lifted_stats)


if __name__ == "__main__":
    main()
