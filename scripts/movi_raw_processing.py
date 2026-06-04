"""
pack_movi_hdf5.py
─────────────────
Read all MoVi F_amass_Subject_X.mat files and pack them into a single HDF5.

Each .mat file contains one subject with 21 action clips.
We store the three SMPL-X-compatible parameter arrays per clip:
  • poses  (T, 52, 3)  float32  — axis-angle per joint (radians)
  • trans  (T,  3)     float32  — root translation (metres)
  • betas  (16,)       float32  — shape coefficients (static per subject)

The train/val/test split is done at the SUBJECT level so no subject
appears in more than one split.

HDF5 layout:
    movi.h5
    ├── train/
    │   ├── Subject_1__walking/
    │   │   ├── poses   (T, 52, 3)
    │   │   ├── trans   (T,  3)
    │   │   ├── betas   (16,)
    |   |   ├── PG1     (T/4,H,W,3)
    |   |   ├── PG2     (T/4,H,W,3)
    │   │   └── attrs:  gender, action, subject, height, mass, age,
    │   │               framerate, n_frames, split
    │   └── ...
    ├── val/  ...
    └── test/ ...

Usage:
    python pack_movi_hdf5.py \\
        --mat_dir    data/F_amass/ \\
        --out_hdf5   movi.h5 \\
        --train_frac 0.80 \\
        --val_frac   0.10 \\
        --seed       42
"""

import argparse
import logging
import random
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from decord import VideoReader, cpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FRAMERATE = 120  # Hz — fixed for MoVi
NO_DATA_SUBJECTS = [7, 10, 26, 49]


# ─────────────────────────────────────────────────────────────────────────────
# MAT reader
# ─────────────────────────────────────────────────────────────────────────────

def file_name(file_type, id):
    if file_type == 'mat':
        return f"F_amass_Subject_{id}.mat"
    elif file_type == 'pg1':
        return f"F_PG1_Subject_{id}_L.avi"
    elif file_type == 'pg2':
        return f"F_PG2_Subject_{id}_L.avi"
    elif file_type == 'v3d':
        return f"F_v3d_Subject_{id}.mat"
    else:
        raise Exception("Unrecognized file type")

def _unwrap(x):
    """
    Unwrap nested numpy object arrays from scipy.io.loadmat (squeeze_me=False).
    Only peels single-element (size==1) object arrays, so multi-element arrays
    like move (21,1) are left intact. Stops when we reach a mat_struct,
    a numeric ndarray, or a scalar -- regardless of how many layers deep.
    """
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x


def _scalar(x) -> int | float:
    """Extract a Python scalar from any numpy array shape."""
    if isinstance(x, np.ndarray):
        return x.flat[0].item()
    return x


def _str(x) -> str:
    """Extract a plain Python str from a numpy string scalar or 1-element array."""
    if isinstance(x, np.ndarray):
        return str(x.flat[0])
    return str(x)

def load_v3d_mat(v3d_path):
    try:
        mat = sio.loadmat(str(v3d_path), struct_as_record=False, squeeze_me=False)
    except Exception as e:
        log.warning(f"Cannot load {v3d_path.name}: {e}")
        return None, None
    top_key = next(k for k in mat if not k.startswith("__"))
    subj    = _unwrap(mat[top_key])   # mat_struct with fields: id, subject, move
    move_arr = _unwrap(subj.move)

    action_names = [str(_unwrap(action[0])).lstrip("['").rstrip("']") for action in move_arr.motions_list]
    action_inds = np.array([[int(tup[0]),int(tup[1])] for tup in move_arr.flags30])

    return action_names, action_inds

def load_amass_mat(path: Path) -> tuple[dict, list[dict]] | tuple[None, None]:
    """
    Load one F_amass_Subject_X.mat.

    Uses squeeze_me=False and explicit unwrapping to be robust across
    all scipy versions and all 90 subject files.

    Returns
    -------
    meta  : dict        — subject metadata
    clips : list[dict]  — one dict per action clip
    """
    try:
        mat = sio.loadmat(str(path), struct_as_record=False, squeeze_me=False)
    except Exception as e:
        log.warning(f"Cannot load {path.name}: {e}")
        return None, None

    top_key = next(k for k in mat if not k.startswith("__"))
    subj    = _unwrap(mat[top_key])   # mat_struct with fields: id, subject, move
    s       = _unwrap(subj.subject)   # mat_struct with fields: id, sex, height, ...

    meta = {
        "id":     _str(s.id),
        "gender": _str(s.sex),
        "height": int(_scalar(s.height)),
        "mass":   int(_scalar(s.mass)),
        "age":    int(_scalar(s.age)),
    }

    # move_arr shape varies across subjects: (21,1), (1,21), etc.
    # Iterate .flat so we never assume a particular axis layout.
    move_arr = subj.move
    clips = []
    for cell in move_arr.flat:
        action = _unwrap(cell)

        # Some subjects have extra nesting layers — keep peeling
        # until we reach a mat_struct or give up
        max_depth = 10
        depth = 0
        while not hasattr(action, "_fieldnames") and depth < max_depth:
            if isinstance(action, np.ndarray) and action.size > 0:
                action = action.flat[0]
            else:
                break
            depth += 1

        if not hasattr(action, "_fieldnames"):
            log.warning(f"  Could not unwrap action cell in {meta['id']}, skipping. "
                        f"Final type: {type(action).__name__}")
            continue

        poses = action.jointsExpMaps_amass.astype(np.float32)    # (T, 52, 3)
        trans = action.RootTranslation_amass.astype(np.float32)  # (T, 3)
        betas = action.jointsBetas_amass.astype(np.float32).reshape(16)  # (16,)

        clips.append({
            "action": _str(action.description),
            "poses":  poses,
            "trans":  trans,
            "betas":  betas,
            "T":      poses.shape[0],
        })

    return meta, clips

def load_clip_video(video_path, action_names, action_inds, fps=30):
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        log.warning(f"Cannot load {video_path.name}: {e}")
        return None, None
    num_frames = len(vr)
    video_clips = []
    for name, interval in zip(action_names,action_inds):
        start_frame, end_frame = interval
        # TODO: figure out if end+1 or something, check against synching with T in clips, something is 1 off from right... 
        try:
            video_clip = vr[start_frame:end_frame].asnumpy()
            video_clips.append(video_clip)
        except Exception as e:
            log.warning(f"action inds out of range for {video_path.name}, action {name}: {e}")
            return None, None
    # NOTE: should video_clips be stacked here or later when writing to h5? 
    return video_clips, num_frames

# ─────────────────────────────────────────────────────────────────────────────
# Subject-level split
# ─────────────────────────────────────────────────────────────────────────────

def subject_split(
    subjects:  list[int],
    train_frac: float,
    val_frac:   float,
    seed:       int,
) -> dict[str, list[Path]]:
    """
    Shuffle subjects then assign to train / val / test.
    Returns {'train': [...], 'val': [...], 'test': [...]}
    """
    subjects = sorted(subjects)
    rng   = random.Random(seed)
    rng.shuffle(subjects)

    n       = len(subjects)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    splits = {
        "train": subjects[:n_train],
        "val":   subjects[n_train : n_train + n_val],
        "test":  subjects[n_train + n_val :],
    }
    for s, fl in splits.items():
        log.info(f"  {s:5s}: {len(fl):3d} subjects")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────

def write_clip(
    grp:   h5py.Group,
    name:  str,
    clip:  dict,
    meta:  dict,
    split: str,
) -> None:
    """Write one action clip into an HDF5 group."""
    # Handle duplicate action names (some subjects repeat an action)
    # TODO: Currently writing PG1 and PG2 in the same observation, but will treat as different during training as 
    # lifted from different angles. Worth considering to adress here or in the data loader? 
    unique_name = name
    counter = 1
    while unique_name in grp:
        unique_name = f"{name}__{counter}"
        counter += 1
    g = grp.create_group(unique_name)

    for key in ("poses", "trans", "betas","pg1","pg2"):
        g.create_dataset(
            key,
            data             = clip[key], 
            compression      = "gzip",
            compression_opts = 4,
            chunks           = True,
        )

    # Attach everything the DataLoader might want without loading arrays
    g.attrs["action"]    = clip["action"]
    g.attrs["subject"]   = meta["id"]
    g.attrs["gender"]    = meta["gender"]
    g.attrs["height"]    = meta["height"]
    g.attrs["mass"]      = meta["mass"]
    g.attrs["age"]       = meta["age"]
    g.attrs["framerate"] = FRAMERATE
    g.attrs["n_frames"]  = clip["T"]
    g.attrs["split"]     = split


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_dir",    required=True,
                        help="Directory containing F_amass_Subject_*.mat files")
    parser.add_argument("--pg1_dir",    required=True,
                        help="Directory containing F_PG1_Subject_*_L.avi files")
    parser.add_argument("--pg2_dir",    required=True,
                        help="Directory containing F_PG2_Subject_*_L.avi files")
    parser.add_argument("--v3d_dir",    required=True,
                        help="Directory containing F_v3d_Subject_*.mat files")
    parser.add_argument("--out_hdf5",   required=True,
                        help="Output path, e.g. movi.h5")
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac",   type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--min_frames", type=int,   default=16,
                        help="Skip clips shorter than this many frames")
    args = parser.parse_args()

    subjects = [id for id in range(1,100) if id not in NO_DATA_SUBJECTS]

    # mat_dir = Path(args.mat_dir)
    # mat_files = sorted(mat_dir.glob("F_amass_Subject_*.mat"))
    # if not mat_files:
    #     raise FileNotFoundError(f"No F_amass_Subject_*.mat files in {mat_dir}")
    # log.info(f"Found {len(mat_files)} .mat files")

    splits = subject_split(subjects, args.train_frac, args.val_frac, args.seed)
    counts = {"train": 0, "val": 0, "test": 0}
    skipped = []

    with h5py.File(args.out_hdf5, "w") as h5f:

        # Dataset-level metadata — useful for anyone opening the file cold
        h5f.attrs["description"]  = "MoVi AMASS — SMPL-H/X parameters per action clip"
        h5f.attrs["framerate"]    = FRAMERATE
        h5f.attrs["n_joints"]     = 52
        h5f.attrs["pose_shape"]   = "T x 52 x 3  (axis-angle, radians)"
        h5f.attrs["trans_units"]  = "metres"
        h5f.attrs["joint_layout"] = (
            "joint[0]=root  "
            "joint[1-21]=body  "
            "joint[22-36]=left_hand  "
            "joint[37-51]=right_hand"
        )
        h5f.attrs["smplx_mapping"] = (
            "global_orient=poses[:,0,:]  "
            "body_pose=poses[:,1:22,:].reshape(-1,63)  "
            "left_hand_pose=poses[:,22:37,:].reshape(-1,45)  "
            "right_hand_pose=poses[:,37:52,:].reshape(-1,45)  "
            "betas=betas[:10]"
        )

        for split, subjs in splits.items():
            grp = h5f.require_group(split)

            for id in subjs:
                # mat_path = Path(f'{args.mat_dir}/{file_name("mat",id)}')
                # v3d_path = Path(f'{args.v3d_dir}/{file_name("v3d",id)}')
                # pg1_path = Path(f'{args.pg1_dir}/{file_name("pg1",id)}')
                # pg2_path = Path(f'{args.pg2_dir}/{file_name("pg2",id)}')
                mat_path = Path(args.mat_dir)/Path(file_name("mat",id))
                v3d_path = Path(args.v3d_dir)/Path(file_name("v3d",id))
                pg1_path = Path(args.pg1_dir)/Path(file_name("pg1",id))
                pg2_path = Path(args.pg2_dir)/Path(file_name("pg2",id))

                

                meta, clips = load_amass_mat(mat_path)
                action_names, action_inds = load_v3d_mat(v3d_path)

                if meta is None:
                    skipped.append(id)
                    log.info(
                        f"Skipping Subject {id}, empty meta"
                    )
                    continue
                if action_names is None:
                    skipped.append(id)
                    log.info(
                        f"Skipping Subject {id}, could not load v3d"
                    )
                    continue
                if len(action_names) != len(clips):
                    skipped.append(id)
                    log.info(
                        f"Skipping Subject {id}, len of clips and action names not matching"
                    )
                    continue
                if len(action_inds) != len(clips):
                    skipped.append(id)
                    log.info(
                        f"Skipping Subject {id}, len of clips and action inds not matching"
                    )
                    continue

                # TODO read videos.
                pg1_clips, pg1_n_frames = load_clip_video(pg1_path,action_names,action_inds)
                pg2_clips, pg2_n_frames = load_clip_video(pg2_path,action_names,action_inds)
                if (pg1_clips is None) or (pg2_clips is None):
                    skipped.append(id)
                    log.info(
                        f"Skipping Subject {id}, issue reading video"
                    )
                    continue
                if pg1_n_frames != pg2_n_frames:
                    skipped.append(id)
                    log.info(
                        f"Skipping Subject {id}, trimmed videos of different length"
                    )
                    continue
                # Success
                log.info(
                    f"  {split:5s}  {meta['id']:12s}  "
                    f"gender={meta['gender']:6s}  {len(clips)} clips"
                )



                for i,clip in enumerate(clips):
                    if clip["T"] < args.min_frames:
                        log.debug(f"    skip {clip['action']}: only {clip['T']} frames")
                        skipped.append(id)
                        continue
                    if clip["action"] != action_names[i]:
                        log.info(f'Actions are not in the right order, subject {id} has issues')
                        skipped.append(id)
                    
                    clip["pg1"] = pg1_clips[i]
                    clip["pg2"] = pg2_clips[i]


                    # Double-underscore separator avoids collisions with
                    # action names that contain single underscores
                    action_safe = clip["action"].replace("/", "_").replace(" ", "_")
                    name = f"{meta['id']}__{action_safe}"
                    

                    write_clip(grp, name, clip, meta, split)
                    counts[split] += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    total = counts["train"] + counts["val"] + counts["test"]
    size_mb = Path(args.out_hdf5).stat().st_size / 1e6
    log.info("\n=== Done ===")
    log.info(f"  train : {counts['train']:4d} clips")
    log.info(f"  val   : {counts['val']:4d} clips")
    log.info(f"  test  : {counts['test']:4d} clips")
    log.info(f"  total : {total:4d} clips  ({len(skipped)} skipped)")
    log.info(f"  file  : {args.out_hdf5}  ({size_mb:.1f} MB)")

    # ── Sanity-print one example entry ───────────────────────────────────────
    with h5py.File(args.out_hdf5, "r") as h5f:
        for split in ("train", "val", "test"):
            keys = list(h5f[split].keys())
            log.info(f"\n  /{split}/  ({len(keys)} clips)")
            if keys:
                ex = h5f[split][keys[0]]
                for ds in ("poses", "trans", "betas"):
                    log.info(f"    {ds:6s}: shape={ex[ds].shape}  dtype={ex[ds].dtype}")
                log.info(f"    attrs: { dict(ex.attrs) }")


if __name__ == "__main__":
    main()