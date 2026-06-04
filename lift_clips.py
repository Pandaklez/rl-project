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

    action_names = [str(_unwrap(action[0])).lstrip("['").rstrip("']") for action in move_arr.motions_list]
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

def lift_and_write_clips(out_file, pg1_data, pg2_data, gt_data, split_indices):
    for split in ("train", "val", "test"):
        grp_movi = gt_data[split]
        grp_out = out_file.require_group(split)
        index_names = split_indices[split]
        for index_name in index_names:
            subject_id = int(index_name.split("_")[1])
            action_name = index_name.split("__")[1]
            action_grp = grp_out.require_group(index_name)
            action_gt_group = grp_movi[index_name]

            if subject_id in NO_DATA_SUBJECTS:
                print(f"Skipping {index_name} due to missing data")
                continue
            subj_video_pg1 = pg1_data.get(subject_id, {}).get(action_name, None)
            subj_video_pg2 = pg2_data.get(subject_id, {}).get(action_name, None)
            
            writing = False
            if subj_video_pg1 is not None:
                writing = True
                pg1_grp = action_grp.require_group("PG1")
                for dimension in ("trans","poses","betas"):
                    pg1_grp.create_dataset(
                        dimension,
                        data = action_gt_group[dimension][:],
                        compression="gzip",
                        compression_opts=4,
                        chunks=True
                    )
 
            if subj_video_pg2 is not None:
                writing = True
                pg2_grp = action_grp.require_group("PG2")
                for dimension in ("trans","poses","betas"):
                    pg2_grp.create_dataset(
                        dimension,
                        data = action_gt_group[dimension][:],
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
    parser.add_argument("--out_hdf5",   default = "lifted_test.h5",
                        help="Output path, e.g. lifted.h5")
    
    args = parser.parse_args()

    movi_h5 = h5py.File(args.movi_path, "r")
    pg1_dir = args.pg1_dir
    pg2_dir = args.pg2_dir
    v3d_path = args.v3d_path

    split_indices = json.load(open(args.split_index_path, "r"))

    out_file = h5py.File(args.out_hdf5, "w")

    # Build metadata per subject 
    pg1_data = {}
    pg2_data = {}
    # for id in range(1,91):
    for id in range(1,12):
        pg1_subj = {}
        pg2_subj = {}
        if id in NO_DATA_SUBJECTS:
            print(f"Skipping Subject {id} due to missing data")
            continue

        subject_name = f"Subject_{id}"
        print(f"Processing {subject_name}...")

        # Get action names and frame indices for this subject from v3d mat file
        # action_names, action_inds = load_v3d_mat(v3d_path / file_name('v3d', id))
        action_names, action_inds = load_v3d_mat(os.path.join(os.getcwd(), v3d_path, file_name('v3d', id)))
        if action_names is None or action_inds is None:
            print(f"Skipping {subject_name} due to v3d loading error")
            continue

        # Load video clips for pg1 and pg2
        # pg1_clips, pg1_num_frames = load_clip_video(pg1_dir / file_name('pg1', id), action_names, action_inds)
        # pg2_clips, pg2_num_frames = load_clip_video(pg2_dir / file_name('pg2', id), action_names, action_inds)
        pg1_clips, pg1_num_frames = load_clip_video(os.path.join(os.getcwd(), pg1_dir, file_name('pg1', id)), action_names, action_inds)
        pg2_clips, pg2_num_frames = load_clip_video(os.path.join(os.getcwd(), pg2_dir, file_name('pg2', id)), action_names, action_inds)

        if pg1_clips is not None:
            for i in range(len(action_names)):
                pg1_subj[action_names[i]] = pg1_clips[i]
            pg1_data[id] = pg1_subj
            
        if pg2_clips is not None:
            for i in range(len(action_names)):
                pg2_subj[action_names[i]] = pg2_clips[i]
            pg2_data[id] = pg2_subj

    lift_and_write_clips(out_file, pg1_data, pg2_data, movi_h5, split_indices)
    
        
if __name__ == "__main__":
    main()


