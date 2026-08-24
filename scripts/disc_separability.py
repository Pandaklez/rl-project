"""
How separable are the GAIL discriminator's real and fake samples, before it runs?

The discriminator sees real samples as `gt/poses` standardised with the **GT**
statistics and fake samples as the policy's output in **lifted per-camera**
space. Those are two different normalisations of the same physical quantity, so
a trivial rule can label the pair without learning anything about pose realism —
which would saturate the discriminator and pin `r_gail` to a constant.

That is exactly what happened on the 52-joint data: MoVi GT has no finger
articulation, so the 90 hand dimensions had sigma ~1e-7 and normalised to
exactly +/-1 where SMPLer-X's predicted fingers never did. This script is what
says whether the finger removal closed it.

Separation is reported per dimension as

    d' = |mu_real - mu_fake| / sqrt((sd_real^2 + sd_fake^2) / 2)

the effect size a single-dimension threshold rule would exploit. It bounds the
*trivial* rules only: an MLP can still combine many weak dimensions, so a small
d' everywhere is necessary for a healthy discriminator, not sufficient.

    python scripts/disc_separability.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_path", default=str(REPO / "data/processed_movi.h5"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--gt_stats", default=str(REPO / "data/normalization.json"))
    ap.add_argument("--lifted_stats", default=str(REPO / "data/normalization_lifted_{cam}.json"))
    args = ap.parse_args()

    gt = json.load(open(args.gt_stats))["poses"]
    mu_g = np.array(gt["mu"]).ravel()
    sd_g = np.array(gt["sigma"]).ravel()
    lifted = {c: json.load(open(args.lifted_stats.format(cam=c)))["poses"]
              for c in ("pg1", "pg2")}

    real, fake = [], []
    with h5py.File(args.h5_path, "r") as f:
        for clip in f[args.split].keys():
            grp = f[args.split][clip]
            real.append(grp["gt"]["poses"][:].reshape(-1, mu_g.size))
            for cam in ("pg1", "pg2"):
                if cam not in grp:
                    continue
                mu_l = np.array(lifted[cam]["mu"]).ravel()
                sd_l = np.array(lifted[cam]["sigma"]).ravel()
                p = grp[cam]["poses"][:].reshape(-1, mu_g.size)
                fake.append((p - mu_l) / np.maximum(sd_l, 1e-8))
    real = (np.concatenate(real) - mu_g) / np.maximum(sd_g, 1e-8)
    fake = np.concatenate(fake)

    d = np.abs(real.mean(0) - fake.mean(0)) / np.sqrt((real.var(0) + fake.var(0)) / 2)
    print(f"real {real.shape}  fake {fake.shape}")
    print(f"per-dim d'          : mean {d.mean():.3f}  median {np.median(d):.3f}  "
          f"max {d.max():.3f} (dim {d.argmax()})")
    print(f"  non-root dims 3..  : mean {d[3:].mean():.3f}  max {d[3:].max():.3f}")
    print(f"  root dims 0..2     : {np.round(d[:3], 3).tolist()}")
    print(f"dims with d' > 1.0  : {(d > 1.0).sum()} / {d.size}")

    # The 52-joint leak: a dimension pinned to exactly +/-1 on the real side is a
    # perfect label. Report the worst dimension rather than assuming which it is.
    sat = (np.abs(np.abs(real) - 1.0) < 1e-4).mean(0)
    print(f"worst dim's fraction of real frames at exactly +/-1: {sat.max():.4f} "
          f"(dim {sat.argmax()})")


if __name__ == "__main__":
    main()
