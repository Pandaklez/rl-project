"""
smplx_to_h5.py
──────────────
Convert per-frame SMPLer-X .npz outputs into the same HDF5 layout used by
movi_raw_processing.py, driven by the pre-computed split_index.json.

HDF5 layout:
    movi_smplx.h5
    ├── train/
    │   ├── Subject_1__walking/
    │   │   ├──PG1 
    │   │   │   ├──poses  (T, 52, 3)  float32  axis-angle [root|body|lhand|rhand]
    │   │   │   ├──trans  (T,  3)     float32  root translation
    │   │   │   ├──betas  (16,)       float32  mean shape, zero-padded to 16
    │   │   │   ├──attrs: gender, action, subject, height, mass, age,
    │   │           framerate, n_frames, split
    |   │   ├──PG2 ...
    |   |   │   ├──poses  (T, 52, 3)  float32  axis-angle [root|body|lhand|rhand]
    │   │   │   ├──trans  (T,  3)     float32  root translation
    │   │   │   ├──betas  (16,)      float32  mean shape, zero-padded to 16
    │   │   │   ├──attrs: gender, action, subject, height, mass, age,
    │   │           framerate, n_frames, split
    │   └── ...
    ├── val/  ...
    └── test/ ...

Inputs:
  --smplx_results_dir  root dir containing per-video subdirs
                       (e.g.  demo/results/F_PG1_Subject_1_L/smplx/)
  --v3d_dir            dir with F_v3d_Subject_X.mat files (from F_Subjects_meta.zip)
  --split_index        path to split_index.json
  --out                output .h5 path
  --min_frames         skip action clips shorter than this (default 16)

Usage:
  python smplx_to_h5.py \\
      --smplx_results_dir demo/results \\
      --v3d_dir           F_Subjects_meta \\
      --split_index       split_index.json \\
      --out               movi_smplx.h5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def load_norm(norm_path: str) -> dict:
    """Load normalization.json → {field: (mu_np, sigma_np)}."""
    with open(norm_path) as f:
        raw = json.load(f)
    norms = {}
    for field in ("poses", "trans", "betas"):
        mu    = np.array(raw[field]["mu"],    dtype=np.float32)
        sigma = np.array(raw[field]["sigma"], dtype=np.float32)
        norms[field] = (mu, sigma)
    return norms


def normalize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z-score: (x - mu) / sigma, guarded against near-zero sigma."""
    return ((x - mu) / np.where(sigma < 1e-8, 1.0, sigma)).astype(np.float32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FRAMERATE = 30  # Hz — SMPLer-X inference was run at 30 fps

# `trans` here is SMPLer-X's `cam_trans`, stored under the name `transl` at
# inference.py:152. It is NOT metres: get_camera_trans() derives depth from a
# virtual focal length of 5000 px in the cropped-bbox camera, so a subject 4.5 m
# from the lens comes out at z ~= 42. Recovering metres needs the per-frame bbox
# and the real intrinsics — see src/reproject.py (crop_intrinsics,
# metric_translation). It also positions the model origin, not the pelvis;
# place_in_camera() handles that.
TRANS_UNITS = (
    "SMPLer-X cam_trans in the cropped-bbox VIRTUAL camera (focal 5000 px) — "
    "NOT metres. Convert with src/reproject.py:metric_translation, which needs "
    "the per-frame bbox and the real intrinsics. Positions the model origin, "
    "not the pelvis (see src/reproject.py:place_in_camera)."
)


# ─────────────────────────────────────────────────────────────────────────────
# MAT helpers (reused from movi_raw_processing.py)
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x

def _scalar(x):
    if isinstance(x, np.ndarray):
        return x.flat[0].item()
    return x

def _str(x) -> str:
    if isinstance(x, np.ndarray):
        return str(x.flat[0])
    return str(x)


def load_v3d_mat(v3d_path: Path):
    """
    Returns (meta, action_names, action_inds) where
      meta         — dict: id, gender, height, mass, age
      action_names — list[str]
      action_inds  — np.ndarray (N, 2), 0-indexed frame intervals at 30 fps
    """
    try:
        mat = sio.loadmat(str(v3d_path), struct_as_record=False, squeeze_me=False)
    except Exception as e:
        log.warning(f"Cannot load {v3d_path.name}: {e}")
        return None, None, None

    top_key = next(k for k in mat if not k.startswith("__"))
    subj     = _unwrap(mat[top_key])
    s        = _unwrap(subj.subject)

    meta = {
        "id":     _str(s.id),
        "gender": _str(s.sex),
        "height": int(_scalar(s.height)),
        "mass":   int(_scalar(s.mass)),
        "age":    int(_scalar(s.age)),
    }

    move_arr     = _unwrap(subj.move)
    action_names = [
        str(_unwrap(action[0])).lstrip("['").rstrip("']")
        for action in move_arr.motions_list
    ]
    action_inds = np.array([
        [int(tup[0]), int(tup[1])]
        for tup in move_arr.flags30
    ])
    return meta, action_names, action_inds


# ─────────────────────────────────────────────────────────────────────────────
# SMPLer-X npz loader
# ─────────────────────────────────────────────────────────────────────────────

def load_smplx_frames(smplx_dir: Path) -> dict[int, Path]:
    """
    Return {0-based-frame-index: npz_path} for all person-0 files found.
    npz filenames are  {1-based-frame:05d}_0.npz  so we subtract 1.
    """
    frame_map: dict[int, Path] = {}
    for p in smplx_dir.glob("*_0.npz"):
        frame_1based = int(p.stem.split("_")[0])
        frame_map[frame_1based - 1] = p
    return frame_map


def build_clip_arrays(frame_map: dict[int, Path], start: int, end: int):
    """
    Stack SMPLer-X outputs for frames [start, end) into arrays.

    Returns (poses, trans, betas) or None if there are too many missing frames.
      poses  (T, 52, 3)  — [global_orient | body | lhand | rhand]
      trans  (T,  3)
      betas  (16,)       — mean, zero-padded
    """
    poses_list, trans_list, betas_list = [], [], []
    missing = 0

    for fi in range(start, end):
        if fi not in frame_map:
            missing += 1
            continue
        d = np.load(frame_map[fi])
        # TODO: double check that shapes split is correct
        poses = np.concatenate([
            d["global_orient"].reshape(-1, 3),   # (1,  3)
            d["body_pose"].reshape(-1, 3),        # (21, 3)
            d["left_hand_pose"].reshape(-1, 3),   # (15, 3)
            d["right_hand_pose"].reshape(-1, 3),  # (15, 3)
        ], axis=0)  # (52, 3)
        poses_list.append(poses)
        trans_list.append(d["transl"].reshape(3))
        betas_list.append(d["betas"].reshape(-1))

    if not poses_list:
        return None

    # allow up to 20 % missing frames
    total = end - start
    if missing / total > 0.20:
        log.debug(f"  {missing}/{total} frames missing (>{20}%) — skipping")
        return None

    poses = np.stack(poses_list, axis=0).astype(np.float32)   # (T, 52, 3)
    trans = np.stack(trans_list, axis=0).astype(np.float32)   # (T, 3)
    betas_mean = np.mean(np.stack(betas_list, axis=0), axis=0)
    betas = np.zeros(16, dtype=np.float32)
    betas[:len(betas_mean)] = betas_mean

    # finite-difference velocities (rad/s and m/s), last frame repeats T-2
    poses_vel = np.concatenate([
        np.diff(poses, axis=0) * FRAMERATE,
        np.diff(poses, axis=0)[-1:] * FRAMERATE,
    ], axis=0).astype(np.float32)  # (T, 52, 3)
    trans_vel = np.concatenate([
        np.diff(trans, axis=0) * FRAMERATE,
        np.diff(trans, axis=0)[-1:] * FRAMERATE,
    ], axis=0).astype(np.float32)  # (T, 3)

    return poses, trans, betas, poses_vel, trans_vel


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────

def write_clip(split_grp: h5py.Group, name: str, pg_group: str, poses, trans, betas,
               poses_vel, trans_vel, meta: dict, action: str, split: str,
               norms: dict | None = None) -> None:
    clip_grp = split_grp.require_group(name)
    if pg_group in clip_grp:
        log.warning(f"  {name}/{pg_group} already exists, skipping duplicate")
        return
    g = clip_grp.create_group(pg_group)

    # if norms is not None:
    #     poses = normalize(poses, *norms["poses"])
    #     trans = normalize(trans, *norms["trans"])
    #     betas = normalize(betas, *norms["betas"])

    for key, data in (("poses", poses), ("trans", trans), ("betas", betas),
                      ("poses_vel", poses_vel), ("trans_vel", trans_vel)):
        g.create_dataset(key, data=data, compression="gzip", compression_opts=4, chunks=True)

    g.attrs["action"]    = action
    g.attrs["subject"]   = meta["id"]
    g.attrs["gender"]    = meta["gender"]
    g.attrs["height"]    = meta["height"]
    g.attrs["mass"]      = meta["mass"]
    g.attrs["age"]       = meta["age"]
    g.attrs["framerate"] = FRAMERATE
    g.attrs["n_frames"]  = poses.shape[0]
    g.attrs["split"]     = split


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_results_dir", required=True,
                        help="Root dir with per-video subdirs (each containing smplx/)")
    parser.add_argument("--v3d_dir",           required=True,
                        help="Dir with F_v3d_Subject_X.mat files")
    parser.add_argument("--split_index",       required=True,
                        help="Path to split_index.json")
    parser.add_argument("--out",               required=True,
                        help="Output .h5 path")
    parser.add_argument("--norm_json",         default=None,
                        help="Path to normalization.json; if given, poses/trans/betas are z-scored")
    parser.add_argument("--min_frames",        type=int, default=16,
                        help="Skip clips shorter than this many frames")
    args = parser.parse_args()

    norms = load_norm(args.norm_json) if args.norm_json else None
    if norms:
        log.info(f"dozation loaded from {args.norm_json}")

    with open(args.split_index) as f:
        split_index: dict[str, list[str]] = json.load(f)

    # build a fast lookup: clip_name → split
    clip_to_split: dict[str, str] = {}
    for split, clips in split_index.items():
        for c in clips:
            clip_to_split[c] = split

    results_root = Path(args.smplx_results_dir)
    v3d_root     = Path(args.v3d_dir)

    # find all per-video smplx dirs: results_root/<VIDEO_NAME>/smplx/
    video_dirs = sorted(
        d for d in results_root.iterdir()
        if (d / "smplx").is_dir()
    )
    if not video_dirs:
        raise FileNotFoundError(f"No <VIDEO>/smplx/ subdirectories found in {results_root}")
    log.info(f"Found {len(video_dirs)} video result dirs")

    counts  = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    with h5py.File(args.out, "w") as h5f:
        h5f.attrs["description"]  = "MoVi SMPLer-X — per-action SMPL-X parameters"
        h5f.attrs["framerate"]    = FRAMERATE
        h5f.attrs["n_joints"]     = 52
        h5f.attrs["pose_shape"]   = "T x 52 x 3  (axis-angle, radians)"
        h5f.attrs["trans_units"]  = TRANS_UNITS
        h5f.attrs["joint_layout"] = (
            "joint[0]=root  joint[1-21]=body  "
            "joint[22-36]=left_hand  joint[37-51]=right_hand"
        )
        for split in ("train", "val", "test"):
            h5f.require_group(split)

        for vid_dir in video_dirs:
            video_name = vid_dir.name  # e.g. F_PG1_Subject_66_L

            # parse PG group
            m_pg = re.search(r"(PG\d+)", video_name)
            pg_group = m_pg.group(1) if m_pg else "PG_unknown"

            # parse subject id
            m = re.search(r"Subject_(\d+)", video_name)
            if not m:
                log.warning(f"Cannot parse subject id from {video_name}, skipping")
                skipped += 1
                continue
            subj_id = int(m.group(1))

            # load v3d timing + metadata
            v3d_path = v3d_root / f"F_v3d_Subject_{subj_id}.mat"
            if not v3d_path.exists():
                log.warning(f"v3d file not found: {v3d_path}, skipping {video_name}")
                skipped += 1
                continue

            meta, action_names, action_inds = load_v3d_mat(v3d_path)
            if meta is None:
                skipped += 1
                continue

            # load per-frame smplx npz files
            smplx_dir = vid_dir / "smplx"
            frame_map = load_smplx_frames(smplx_dir)
            if not frame_map:
                log.warning(f"No smplx frames found in {smplx_dir}, skipping")
                skipped += 1
                continue

            log.info(f"Processing {video_name}  subject={meta['id']}  "
                     f"actions={len(action_names)}  frames={len(frame_map)}")

            for action, (start, end) in zip(action_names, action_inds):
                action_safe = action.replace("/", "_").replace(" ", "_")
                clip_name   = f"Subject_{subj_id}__{action_safe}"

                split = clip_to_split.get(clip_name)
                if split is None:
                    log.debug(f"  {clip_name} not in split_index, skipping")
                    skipped += 1
                    continue

                arrays = build_clip_arrays(frame_map, int(start), int(end))
                if arrays is None:
                    log.debug(f"  {clip_name}: no usable frames [{start},{end})")
                    skipped += 1
                    continue

                poses, trans, betas, poses_vel, trans_vel = arrays
                if poses.shape[0] < args.min_frames:
                    log.debug(f"  {clip_name}: only {poses.shape[0]} frames, skipping")
                    skipped += 1
                    continue

                write_clip(h5f[split], clip_name, pg_group, poses, trans, betas,
                           poses_vel, trans_vel, meta, action, split, norms)
                counts[split] += 1
                log.debug(f"  {split:5s}  {clip_name}/{pg_group}  T={poses.shape[0]}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total    = sum(counts.values())
    size_mb  = Path(args.out).stat().st_size / 1e6
    log.info("=== Done ===")
    log.info(f"  train : {counts['train']:4d} clips")
    log.info(f"  val   : {counts['val']:4d} clips")
    log.info(f"  test  : {counts['test']:4d} clips")
    log.info(f"  total : {total:4d} clips  ({skipped} skipped)")
    log.info(f"  file  : {args.out}  ({size_mb:.1f} MB)")

    with h5py.File(args.out, "r") as h5f:
        for split in ("train", "val", "test"):
            clip_keys = list(h5f[split].keys())
            log.info(f"\n  /{split}/  ({len(clip_keys)} clips)")
            if clip_keys:
                clip = h5f[split][clip_keys[0]]
                pg_keys = list(clip.keys())
                log.info(f"    PG groups: {pg_keys}")
                ex = clip[pg_keys[0]]
                for ds in ("poses", "trans", "betas", "poses_vel", "trans_vel"):
                    log.info(f"    {ds:10s}: shape={ex[ds].shape}  dtype={ex[ds].dtype}")
                log.info(f"    attrs: {dict(ex.attrs)}")


if __name__ == "__main__":
    main()
