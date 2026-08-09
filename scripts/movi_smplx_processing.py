
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FRAMERATE = 120  # Hz — fixed for MoVi
SUBJECTS = list(range(1, 91))  # 90 subjects in total

def subject_split(
    subjects:  list[int],
    train_frac: float,
    val_frac:   float,
    seed:       int,
) -> dict[str, list[int]]:
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
    for s, subs in splits.items():
        log.info(f"  {s:5s}: {len(subs):3d} subjects")
    return splits

def get_existing_subject_split(split_path: Path) -> dict[str, list[int]]:
    """
    Load an existing subject split from a JSON file.
    Returns {'train': [...], 'val': [...], 'test': [...]}
    """
    splits = {}
    with open(split_path, "r") as f:
        raw = json.load(f)
    for split in ["train", "val", "test"]:
        subjs = []
        for safe_action in raw[split]:
            subj_id = int(safe_action.split("__")[0].split("_")[-1])
            if subj_id not in subjs:
                subjs.append(subj_id)
        splits[split] = subjs
    return splits

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

def get_v3d(meta_path):
    mat = sio.loadmat(str(meta_path), struct_as_record=False, squeeze_me=False)
    top_key = next(k for k in mat if not k.startswith("__"))
    s    = _unwrap(mat[top_key])   # mat_struct with fields: id, subject, move
    subject = _unwrap(s.subject)
    subject_id = _str(subject.id)
    
    move_arr = _unwrap(s.move)
    safe_action_names = [subject_id + "__" + str(_unwrap(action[0])).lstrip("['").rstrip("']").replace("/","_").replace(" ","_") for action in move_arr.motions_list]
    flags30 = np.array([[int(tup[0]),int(tup[1])] for tup in move_arr.flags30])
    flags120 = np.array([[int(tup[0]),int(tup[1])] for tup in move_arr.flags120])

    Ts = [int(tup[1]) - int(tup[0]) + 1 for tup in move_arr.flags120]
    
    meta = {
        'id': subject_id,
        'gender': _str(subject.sex),
        'handedness': _str(subject.handedness),
        'height': _scalar(subject.height),
        'mass': _scalar(subject.mass),
        'age': _scalar(subject.age),
        'safe_action_names': safe_action_names,
        'flags30': flags30,
        'flags120': flags120,
        'Ts': Ts
    }
    return meta

def load_move_npz(move_path, meta, move_idx):
    try:
        move = np.load(move_path, allow_pickle=True)
    except:
        log.error(f"Cannot load npz file: {move_path}")
        return None

    T_smplx = move['poses'].shape[0]
    T_meta = meta['Ts'][move_idx]

    if T_smplx != T_meta:
        log.warning(f"Mismatch in T for move {move_idx+1}: T_smplx={T_smplx}, T_meta={T_meta}")
        return None
    
    safe_action_name = meta['safe_action_names'][move_idx]

    poses = np.concatenate(
            [
            move["root_orient"].reshape(T_smplx,1, 3), # (T, 1, 3)
            move["pose_body"].reshape(T_smplx,-1, 3), # (T, 21, 3)
            move["pose_hand"].reshape(T_smplx,-1, 3)   # (T, 30, 3)
        ],
        axis = 1
    ) # (T, 52, 3)

    return {
        'poses': poses,  # (T, 52, 3)
        'trans': move['trans'],  # (T, 3)
        'betas': move['betas'],  # (16,)
        'safe_action_name': safe_action_name,
        "T": T_smplx
    }
    
def load_clips_for_subject(subj_id, v3d_path, npz_path):
    # NOTE: ignoring the meta npz file for now since all info is contained in v3d, including action names in order. 
    # NOTE: hard-coding some paths now
    v3d_path = f"{v3d_path}/F_v3d_Subject_{subj_id}.mat"
    subj_npz_path = f"{npz_path}/Subject_{subj_id}_F_MoSh"
    
    meta = get_v3d(v3d_path)
    clips = []
    for move_idx in range(len(meta['safe_action_names'])):
        move_path = f"{subj_npz_path}/Subject_{subj_id}_F_{move_idx+1}_stageii.npz"
        move = load_move_npz(move_path, meta, move_idx)
        if move is not None:
            clips.append(move)

    return meta, clips

def write_clip(
    grp:   h5py.Group,
    clip:  dict,
    meta:  dict,
    split: str,
) -> None:
    """Write one action clip into an HDF5 group."""
    # Handle duplicate action names (some subjects repeat an action)
    unique_name = clip["safe_action_name"]
    counter = 1
    while unique_name in grp:
        unique_name = f"{unique_name}__{counter}"
        counter += 1
    g = grp.create_group(unique_name)

    for key in ("poses", "trans", "betas"):
        g.create_dataset(
            key,
            data             = clip[key],
            compression      = "gzip",
            compression_opts = 4,
            chunks           = True,
        )

    # Attach everything the DataLoader might want without loading arrays
    g.attrs["action"]    = unique_name,#clip["action"]
    g.attrs["subject"]   = meta["id"]
    g.attrs["gender"]    = meta["gender"]
    g.attrs["handedness"] = meta["handedness"]
    g.attrs["height"]    = meta["height"]
    g.attrs["mass"]      = meta["mass"]
    g.attrs["age"]       = meta["age"]
    g.attrs["framerate"] = FRAMERATE
    g.attrs["n_frames"]  = clip["T"]
    g.attrs["split"]     = split

def validate_old_new(old_h5,new_h5):

    splits = ["train", "val", "test"]

    val_check = {}

    for split in splits:
        old_keys = list(old_h5[split].keys())
        new_keys = list(new_h5[split].keys())
    
        in_old_not_new = []
        in_new_not_old = []
        in_wrong_split = []
        
        for ok in [x for x in old_keys if x not in new_keys]:
            in_old_not_new.append(ok)
        for nk in [x for x in new_keys if x not in old_keys]:
            in_new_not_old.append(nk)

        for other_split in [s for s in splits if s != split]:
            other_old_keys = list(old_h5[other_split].keys())
            for wrong_nk in [x for x in new_keys if x in other_old_keys]:
                in_wrong_split.append(wrong_nk)

        val_check[split] = {
            "in_old_not_new" : in_old_not_new,
            "in_new_not_old" : in_new_not_old,
            "in_wrong_split" : in_wrong_split
        }
    for split in splits:
        log.info(f"=== {split} ===")
        for k,v in val_check[split].items():
            log.info(f"{k}: {len(v)}")
            if len(v) > 0:
                log.info(f"  {v}")
        
        


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--v3d_path",    type = str, default = "data/F_Subjects_meta",
                        help="Directory containing v3d mat files")
    parser.add_argument("--npz_path",    type = str, default = "data/MoVi_SMPLX/BMLmovi",
                        help="Directory containing npz files")
    parser.add_argument("--out_hdf5",   type = str, default = "data/movi_smplx.h5",
                        help="Output path, e.g. movi.h5")
    parser.add_argument("--old_h5_from_mat",   type = str, default = "data/_archive/Gmovi.h5",
                        help="Old file built from .mat to compare to, e.g. movi.h5")
    parser.add_argument("--split_path", type=str, default="data/split_index.json")
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac",   type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--min_frames", type=int,   default=16,
                        help="Skip clips shorter than this many frames")
    args = parser.parse_args()

    split_path = Path(args.split_path)
    if split_path.exists():
        splits = get_existing_subject_split(split_path)
    else:
        # NOTE: Not making new here for now, since slightly different use (only subject level)
        splits = subject_split(subjects=SUBJECTS, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    # for subj_id in range(1,91): # TODO: set up split: [ids] and loop through them in stead. 
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
        for split, subj_ids in splits.items():
            grp = h5f.require_group(split)
            # for subj_id in range(1,2):
            for subj_id in subj_ids:
                meta, clips = load_clips_for_subject(subj_id, args.v3d_path, args.npz_path)

                if meta is None:
                    counts["skipped"] += 1
                    log.warning(f"Skipping subject {subj_id} due to missing meta.")
                    continue

                log.info(
                    f"  {split:5s}  {meta['id']:12s}  "
                    f"gender={meta['gender']:6s}  {len(clips)} clips"
                )

                for clip in clips:
                    if clip["T"] < args.min_frames:
                        counts["skipped"] += 1
                        log.warning(
                            f"Skipping clip {clip['safe_action_name']} "
                            f"for subject {subj_id} due to insufficient frames: {clip['T']}"
                        )
                        continue
                    write_clip(grp, clip, meta, split)
                    counts[split] += 1

    total = counts["train"] + counts["val"] + counts["test"]
    size_mb = Path(args.out_hdf5).stat().st_size / 1e6
    log.info("\n=== Done ===")
    log.info(f"  train : {counts['train']:4d} clips")
    log.info(f"  val   : {counts['val']:4d} clips")
    log.info(f"  test  : {counts['test']:4d} clips")
    log.info(f"  total : {total:4d} clips  ({counts['skipped']} skipped)")
    log.info(f"  file  : {args.out_hdf5}  ({size_mb:.1f} MB)")


    # ----- Checking against the old h5 built from mat files

    old_path = Path(args.old_h5_from_mat)
    if old_path.exists():
        old_h5 = h5py.File(old_path,"r")
        new_h5 = h5py.File(args.out_hdf5,"r")
        validate_old_new(old_h5,new_h5)

if __name__ == "__main__":
    main()