"""
compute_norm_stats.py
─────────────────────
Compute per-joint normalization statistics (mean, std) separately for PG1 and
PG2 from a lifted HDF5 file. Stats are computed on the train split only.

Output JSON structure matches normalization.json:
  {
    "poses": {"shape": [...], "mu": [[52 x 3]], "sigma": [[52 x 3]]},
    "trans": {"shape": [...], "mu": [3],         "sigma": [3]},
    "betas": {"shape": [...], "mu": [16],         "sigma": [16]}
  }

Usage:
  python compute_norm_stats.py --h5 lifted_movi_part1_upd1.h5
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


class OnlineStats:
    """Welford online mean/variance, supports arbitrary trailing shape."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2   = np.zeros(shape, dtype=np.float64)

    def update_batch(self, batch: np.ndarray):
        """batch shape: (T, *trailing) — each row is one sample."""
        for row in batch:
            self.n += 1
            delta  = row.astype(np.float64) - self.mean
            self.mean += delta / self.n
            delta2 = row.astype(np.float64) - self.mean
            self.M2 += delta * delta2

    def update_single(self, x: np.ndarray):
        """x shape == trailing shape — one sample."""
        self.n += 1
        delta  = x.astype(np.float64) - self.mean
        self.mean += delta / self.n
        delta2 = x.astype(np.float64) - self.mean
        self.M2 += delta * delta2

    def sigma(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1))


def compute_for_pg(h5f: h5py.File, pg: str) -> dict:
    stats = {
        "poses": OnlineStats((52, 3)),
        "trans": OnlineStats((3,)),
        "betas": OnlineStats((16,)),
    }
    frame_counts = {"poses": 0, "trans": 0, "betas": 0}

    for clip_name, clip_grp in h5f["train"].items():
        if pg not in clip_grp:
            continue
        g = clip_grp[pg]

        poses = g["poses"][:]   # (T, 52, 3)
        trans = g["trans"][:]   # (T, 3)
        betas = g["betas"][:]   # (16,)

        stats["poses"].update_batch(poses)
        stats["trans"].update_batch(trans)
        stats["betas"].update_single(betas)

        frame_counts["poses"] += poses.shape[0]
        frame_counts["trans"] += trans.shape[0]
        frame_counts["betas"] += 1

    return {
        "poses": {
            "shape": [frame_counts["poses"], 52, 3],
            "mu":    stats["poses"].mean.tolist(),
            "sigma": stats["poses"].sigma().tolist(),
        },
        "trans": {
            "shape": [frame_counts["trans"], 3],
            "mu":    stats["trans"].mean.tolist(),
            "sigma": stats["trans"].sigma().tolist(),
        },
        "betas": {
            "shape": [frame_counts["betas"], 16],
            "mu":    stats["betas"].mean.tolist(),
            "sigma": stats["betas"].sigma().tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="lifted_movi_part1_upd1.h5",
                        help="Path to the lifted HDF5 file")
    parser.add_argument("--out_dir", default=".",
                        help="Directory to write the JSON files into")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    with h5py.File(args.h5, "r") as h5f:
        for pg, out_name in (("PG1", "normalization_lifted_pg1.json"),
                             ("PG2", "normalization_lifted_pg2.json")):
            print(f"Computing stats for {pg} ...")
            result = compute_for_pg(h5f, pg)
            out_path = out_dir / out_name
            with open(out_path, "w") as f:
                json.dump(result, f)
            print(f"  poses  : {result['poses']['shape']} frames")
            print(f"  trans  : {result['trans']['shape']} frames")
            print(f"  betas  : {result['betas']['shape'][0]} clips")
            print(f"  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
