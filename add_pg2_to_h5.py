"""
add_pg2_to_h5.py
─────────────────
Append PG2 SMPLer-X results for subjects 8, 9, 73-90 into an existing
lifted_movi_part1_upd1.h5 that already has PG1 for those subjects.

Opens the H5 in r+ mode and writes PG2 subgroups alongside existing PG1 data.

Usage:
    python add_pg2_to_h5.py \
        --h5       lifted_movi_part1_upd1.h5 \
        --results  demo/results \
        --v3d_dir  F_Subjects_meta \
        --subjects 8 9 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90
"""

from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FRAMERATE = 30


# ── MAT helpers ───────────────────────────────────────────────────────────────

def _unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x

def _scalar(x):
    return x.flat[0].item() if isinstance(x, np.ndarray) else x

def _str(x) -> str:
    return str(x.flat[0]) if isinstance(x, np.ndarray) else str(x)


def load_v3d(v3d_path: Path):
    """Returns (meta, action_names, action_inds) or (None, None, None)."""
    try:
        mat = sio.loadmat(str(v3d_path), struct_as_record=False, squeeze_me=False)
    except Exception as e:
        log.warning(f"Cannot load {v3d_path}: {e}")
        return None, None, None
    top_key = next(k for k in mat if not k.startswith("__"))
    subj = _unwrap(mat[top_key])
    s    = _unwrap(subj.subject)
    meta = {
        "id":     _str(s.id),
        "gender": _str(s.sex),
        "height": int(_scalar(s.height)),
        "mass":   int(_scalar(s.mass)),
        "age":    int(_scalar(s.age)),
    }
    move_arr = _unwrap(subj.move)
    action_names = [
        str(_unwrap(a[0])).lstrip("['").rstrip("']").replace("/", "_").replace(" ", "_")
        for a in move_arr.motions_list
    ]
    action_inds = np.array([[int(t[0]), int(t[1])] for t in move_arr.flags30])
    return meta, action_names, action_inds


# ── SMPLer-X npz loader ───────────────────────────────────────────────────────

def load_frame_map(smplx_dir: Path) -> dict[int, Path]:
    """Returns {0-based frame index: npz path} for person-0 files."""
    frame_map = {}
    for p in smplx_dir.glob("*_0.npz"):
        frame_1based = int(p.stem.split("_")[0])
        frame_map[frame_1based - 1] = p
    return frame_map


def build_clip(frame_map: dict[int, Path], start: int, end: int):
    """Build (poses, trans, betas, poses_vel, trans_vel) or None if too sparse."""
    poses_list, trans_list, betas_list = [], [], []
    missing = 0
    for fi in range(start, end):
        if fi not in frame_map:
            missing += 1
            continue
        d = np.load(frame_map[fi])
        poses = np.concatenate([
            d["global_orient"].reshape(-1, 3),
            d["body_pose"].reshape(-1, 3),
            d["left_hand_pose"].reshape(-1, 3),
            d["right_hand_pose"].reshape(-1, 3),
        ], axis=0)
        poses_list.append(poses)
        trans_list.append(d["transl"].reshape(3))
        betas_list.append(d["betas"].reshape(-1))

    if not poses_list:
        return None
    if missing / (end - start) > 0.20:
        return None

    poses = np.stack(poses_list).astype(np.float32)
    trans = np.stack(trans_list).astype(np.float32)
    betas = np.zeros(16, dtype=np.float32)
    betas_raw = np.mean(np.stack(betas_list), axis=0)
    betas[:len(betas_raw)] = betas_raw

    poses_vel = np.concatenate([
        np.diff(poses, axis=0) * FRAMERATE,
        np.diff(poses, axis=0)[-1:] * FRAMERATE,
    ], axis=0).astype(np.float32)
    trans_vel = np.concatenate([
        np.diff(trans, axis=0) * FRAMERATE,
        np.diff(trans, axis=0)[-1:] * FRAMERATE,
    ], axis=0).astype(np.float32)

    return poses, trans, betas, poses_vel, trans_vel


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",       required=True, help="Path to existing lifted H5")
    parser.add_argument("--results",  required=True, help="demo/results root dir")
    parser.add_argument("--v3d_dir",  required=True, help="Dir with F_v3d_Subject_*.mat")
    parser.add_argument("--subjects", nargs="+", type=int,
                        default=[8, 9, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                                 82, 83, 84, 85, 86, 87, 88, 89, 90])
    args = parser.parse_args()

    results_root = Path(args.results)
    v3d_root     = Path(args.v3d_dir)

    written  = 0
    skipped  = 0

    with h5py.File(args.h5, "r+") as h5f:
        for subj_id in args.subjects:
            pg2_dir = results_root / f"F_PG2_Subject_{subj_id}_L" / "smplx"
            if not pg2_dir.exists():
                log.warning(f"Subject {subj_id}: no smplx dir at {pg2_dir}, skipping")
                skipped += 1
                continue

            frame_map = load_frame_map(pg2_dir)
            if not frame_map:
                log.warning(f"Subject {subj_id}: empty smplx dir, skipping")
                skipped += 1
                continue

            v3d_path = v3d_root / f"F_v3d_Subject_{subj_id}.mat"
            meta, action_names, action_inds = load_v3d(v3d_path)
            if meta is None:
                skipped += 1
                continue

            log.info(f"Subject {subj_id}: {len(frame_map)} frames, {len(action_names)} actions")

            for action, (start, end) in zip(action_names, action_inds):
                clip_name = f"Subject_{subj_id}__{action}"

                # find which split this clip is in
                clip_grp = None
                for split in ("train", "val", "test"):
                    if clip_name in h5f[split]:
                        clip_grp = h5f[split][clip_name]
                        break

                if clip_grp is None:
                    log.debug(f"  {clip_name}: not found in H5, skipping")
                    skipped += 1
                    continue

                if "PG2" in clip_grp:
                    log.debug(f"  {clip_name}/PG2 already exists, skipping")
                    skipped += 1
                    continue

                arrays = build_clip(frame_map, int(start), int(end))
                if arrays is None:
                    log.debug(f"  {clip_name}: too many missing frames [{start},{end})")
                    skipped += 1
                    continue

                poses, trans, betas, poses_vel, trans_vel = arrays

                g = clip_grp.create_group("PG2")
                for key, data in (("poses", poses), ("trans", trans), ("betas", betas),
                                  ("poses_vel", poses_vel), ("trans_vel", trans_vel)):
                    g.create_dataset(key, data=data, compression="gzip",
                                     compression_opts=4, chunks=True)
                g.attrs["action"]    = action
                g.attrs["subject"]   = meta["id"]
                g.attrs["gender"]    = meta["gender"]
                g.attrs["height"]    = meta["height"]
                g.attrs["mass"]      = meta["mass"]
                g.attrs["age"]       = meta["age"]
                g.attrs["framerate"] = FRAMERATE
                g.attrs["n_frames"]  = poses.shape[0]

                written += 1
                log.debug(f"  wrote {clip_name}/PG2  T={poses.shape[0]}")

    log.info(f"Done. Wrote {written} PG2 clips, skipped {skipped}.")


if __name__ == "__main__":
    main()
