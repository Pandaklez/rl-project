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

    python scripts/disc_separability.py
    python scripts/disc_separability.py --physical
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def separability(h5_path: str, split: str) -> None:
    real, fake = [], []
    with h5py.File(h5_path, "r") as f:
        for clip in f[split].keys():
            grp = f[split][clip]
            real.append(grp["gt"]["poses"][:].reshape(len(grp["gt"]["poses"]), -1))
            for cam in ("pg1", "pg2"):
                if cam in grp:
                    fake.append(grp[cam]["poses"][:].reshape(len(grp[cam]["poses"]), -1))
    real = np.concatenate(real).astype(np.float64)
    fake = np.concatenate(fake).astype(np.float64)
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
    ap.add_argument("--physical", action="store_true",
                    help="also report the pose error the space mismatch implies")
    args = ap.parse_args()

    separability(args.h5_path, args.split)
    if args.physical:
        physical(args.gt_stats, args.lifted_stats)


if __name__ == "__main__":
    main()
