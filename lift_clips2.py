import os

import numpy as np
import h5py
import argparse
import json
import scipy.io as sio
from decord import VideoReader, cpu

FRAMERATE = 120  # Hz — fixed for MoVi
NO_DATA_SUBJECTS = [7, 10, 26, 49]


"""
GT data already processed in the following format:

movi.h5
    ├── train/
    │   ├── Subject_1__walking/
    │   │   ├── poses   (T, 52, 3)
    │   │   ├── trans   (T,  3)
    │   │   ├── betas   (16,)
    │   │   └── attrs:  gender, action, subject, height, mass, age,
    │   │               framerate, n_frames, split
    │   └── ...
    ├── val/  ...
    └── test/ ...

Retrieve metadata from v3d files to get frames corresponding to each Subject and action, 
using the same (randomized) order, documented in split_index.json
Read video files for the angles pg1 and pg2 and lift them to align with GT data, 
resulting in the following format: 

lifted.h5
    ├── train/
    │   ├── Subject_1__walking/
    |   |   ├── PG1/     
    |   |   |   ├── poses    (T/4,52,3)
    |   |   |   ├── trans    (T/4,3)
    |   |   |   ├── betas    (16,)
    |   |   ├── PG2/     
    |   |   |   ├── poses    (T/4,52,3)
    |   |   |   ├── trans    (T/4,3)
    |   |   |   ├── betas    (16,)
    │   └── ...
    ├── val/  ...
    └── test/ ...    
"""

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
        print(f"Cannot load {v3d_path}: {e}")
        return None, None
    top_key = next(k for k in mat if not k.startswith("__"))
    subj    = _unwrap(mat[top_key])   # mat_struct with fields: id, subject, move
    move_arr = _unwrap(subj.move)

    action_names = [str(_unwrap(action[0])).lstrip("['").rstrip("']").replace("/", "_").replace(" ", "_") for action in move_arr.motions_list]
    action_inds = np.array([[int(tup[0]),int(tup[1])] for tup in move_arr.flags30])

    return action_names, action_inds

def load_clip_video(video_path, action_names, action_inds, fps=30):
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        print(f"Cannot load {video_path}: {e}")
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
            print(f"action inds out of range for {video_path}, action {name}: {e}")
            return None, None
    # NOTE: should video_clips be stacked here or later when writing to h5? 
    return video_clips, num_frames

def load_video_action(video_path, interval, action_name, fps = 30):

    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        start_frame, end_frame = interval
    except Exception as e:
        print(f"Cannot load {video_path}: {e}")
        return None
    # num_frames = len(vr)
    try:
        video_clip = vr[start_frame:end_frame].asnumpy()
    except Exception as e:
        print(f"action inds out of range for {video_path}, action {action_name}: {e}")
        return None
    return video_clip

# NOTE: this is where lifting should be called,
# currently giving gt_data which is also returned as this is a mockup-script for now. 
def lift_clip(video_clip, gt_data):
    return {key : gt_data[key] for key in ("trans","poses","betas")}

def lift_and_write_clips(out_file, action_flags_data, gt_data, split_indices, pg1_dir, pg2_dir):
    for split in ("train", "val", "test"):
        grp_movi = gt_data[split]
        grp_out = out_file.require_group(split)
        index_names = split_indices[split]
        for index_name in index_names:
            action_grp = grp_out.require_group(index_name)
            action_gt_group = grp_movi[index_name]

            subject_id = int(index_name.split("_")[1])
            action_name = index_name.split("__")[1]

            # if subject_id in [16]:
            #     print(index_name)
            #     print(action_name)
            #     print(action_flags_data[subject_id][action_name])

            if subject_id in NO_DATA_SUBJECTS:
                print(f"Skipping {index_name} due to missing data")
                continue
            action_flags = action_flags_data.get(subject_id, {}).get(action_name, None)

            pg1_clip = load_video_action(os.path.join(os.getcwd(), pg1_dir, file_name('pg1', subject_id)), action_flags, action_name)
            pg2_clip = load_video_action(os.path.join(os.getcwd(), pg2_dir, file_name('pg2', subject_id)), action_flags, action_name)

            pg1_lifted = lift_clip(pg1_clip, action_gt_group)
            pg2_lifted = lift_clip(pg2_clip, action_gt_group)

            writing = False

            if pg1_lifted is not None:
                writing = True
                pg1_grp = action_grp.require_group("PG1")
                for dimension in ("trans","poses","betas"):
                    pg1_grp.create_dataset(
                        dimension,
                        data = pg1_lifted[dimension][:],
                        compression="gzip",
                        compression_opts=4,
                        chunks=True
                    )
 
            if pg2_lifted is not None:
                writing = True
                pg2_grp = action_grp.require_group("PG2")
                for dimension in ("trans","poses","betas"):
                    pg2_grp.create_dataset(
                        dimension,
                        data = pg2_lifted[dimension][:],
                        compression="gzip",
                        compression_opts=4,
                        chunks=True
                    )
            if writing:
                # TODO: add metadata? 
                for dimension in ("trans","poses","betas"):
                    action_grp.create_dataset(
                        dimension,
                        data = action_gt_group[dimension][:],
                        compression="gzip",
                        compression_opts=4,
                        chunks=True
                    )
                    # Attach everything the DataLoader might want without loading arrays
                    action_grp.attrs["action"]    = action_gt_group.attrs["action"]
                    action_grp.attrs["subject"]   = action_gt_group.attrs["subject"]
                    action_grp.attrs["gender"]    = action_gt_group.attrs["gender"]
                    action_grp.attrs["height"]    = action_gt_group.attrs["height"]
                    action_grp.attrs["mass"]      = action_gt_group.attrs["mass"]
                    action_grp.attrs["age"]       = action_gt_group.attrs["age"]
                    action_grp.attrs["framerate"] = FRAMERATE
                    action_grp.attrs["n_frames"]  = action_gt_group.attrs["n_frames"]
                    action_grp.attrs["split"]     = split
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movi_path",    default = "Gmovi.h5",
                        help="Directory containing GT poses, trans, betas and attrs")
    parser.add_argument("--pg1_dir",    default = "videos/PG1_avi",
                        help="Directory containing F_PG1_Subject_*_L.avi files")
    parser.add_argument("--pg2_dir",    default = "videos/PG2_avi",
                        help="Directory containing F_PG2_Subject_*_L.avi files")
    parser.add_argument("--v3d_path",   default = "F_Subjects_meta",
                        help="Directory containing F_v3d_Subject_*.mat files for each subject with frame flags")
    parser.add_argument("--split_index_path",   default = "split_index.json",
                        help="Path to json file containing split indices for subjects and actions")
    parser.add_argument("--out_hdf5",   default = "lifted_test2.h5",
                        help="Output path, e.g. lifted.h5")
    
    args = parser.parse_args()

    movi_h5 = h5py.File(args.movi_path, "r")
    pg1_dir = args.pg1_dir
    pg2_dir = args.pg2_dir
    v3d_path = args.v3d_path

    split_indices = json.load(open(args.split_index_path, "r"))

    out_file = h5py.File(args.out_hdf5, "w")

    # Build metadata per subject 
    action_flags_data = {}
    # for id in range(1,12):
    # for id in range(1,91):
    for id in [15,16]:
        if id in NO_DATA_SUBJECTS:
            print(f"Skipping Subject {id} due to missing data")
            continue

        subject_name = f"Subject_{id}"
        print(f"Processing {subject_name}...")
        subject_meta = {}
        action_names, action_inds = load_v3d_mat(os.path.join(os.getcwd(), v3d_path, file_name('v3d', id)))
        if action_names is None or action_inds is None or len(action_names) == 0 or len(action_inds) == 0:
            print(f"Skipping {subject_name} due to v3d loading error")
            continue
        for i, action_name in enumerate(action_names):

            subject_meta[action_name] = action_inds[i]
        action_flags_data[id] = subject_meta
    # print(action_flags_data[15])
    # raise Exception()

    lift_and_write_clips(out_file, action_flags_data, movi_h5, split_indices, pg1_dir, pg2_dir)
    
        
if __name__ == "__main__":
    main()


