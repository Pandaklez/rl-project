import numpy as np
import h5py
import argparse
import json

FRAMERATE = 120  # Hz — fixed for MoVi

def normalize_clip(clip,clip_name,norm_stats):
    normalized = {}
    for key in ("poses", "trans", "betas"):
        if key not in clip:
            raise ValueError(f"Missing key {key} in clip {clip_name}")
        mu = np.array(norm_stats[key]["mu"])
        sigma = np.array(norm_stats[key]["sigma"])
        normalized[key] = (clip[key][:] - mu) / sigma
    return normalized



def upsample(poses, trans, T):
    """
    Upsample `poses` and `trans` along time to length T.

    Args:
        poses: array-like, shape (t0, J, C)  (e.g., (T/4, 52, 3))
        trans: array-like, shape (t0, D) (e.g., (T/4, 3)). Must have same time dimension as poses.
        T: int target time length. 

    Returns:
        poses_up: ndarray shape (T, J, C)
        trans_up: ndarray shape (T, D) or None if trans was None
    """
    poses = np.asarray(poses)
    trans = np.asarray(trans)
    if poses.ndim != 3:
        raise ValueError("poses must have shape (t, J, C)")
    if trans.ndim != 3:
        raise ValueError("trans must have shape (t, C)")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError("poses and trans must have the same time dimension")

    t0 = poses.shape[0]

    if T == t0:
        poses_up = poses.copy()
        trans_up = trans.copy()
    else:
        t_target = np.arange(T)
        t_src = np.linspace(0, T - 1, t0)

        D = poses.shape[1] * poses.shape[2]
        flat = poses.reshape(t0, D).astype(float)
        up_flat = np.empty((T, D), dtype=flat.dtype)
        for d in range(D):
            up_flat[:, d] = np.interp(t_target, t_src, flat[:, d])
        poses_up = up_flat.reshape(T, poses.shape[1], poses.shape[2])

        trans_up = np.empty((T, trans.shape[1]), dtype=trans.dtype)
        for c in range(trans.shape[1]):
            trans_up[:, c] = np.interp(t_target, t_src, trans[:, c])

    return poses_up, trans_up

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movi_path",     default = "Gmovi.h5",
                        help="Directory containing GT poses, trans, betas and attrs")
    parser.add_argument("--lifted_path",   default = "lifted_movi_part1_upd1.h5",
                        help="Directory containing lifted poses from pg1 and pg2")
    parser.add_argument("--gt_norm_path",   default = "normalization.json",
                        help="Path to json file containing normalization stats for gt (e.g., mean and std of pose and trans)")
    parser.add_argument("--pg1_norm_path",   default = "normalization_lifted_pg1.json",
                        help="Path to json file containing normalization stats for pg1 (e.g., mean and std of pose and trans)")
    parser.add_argument("--pg2_norm_path",   default = "normalization_lifted_pg2.json",
                        help="Path to json file containing normalization stats for pg2 (e.g., mean and std of pose and trans)")
    parser.add_argument("--out_hdf5",   default = "processed_movi.h5",
                        help="Output path, e.g. processed_data.h5")
                        
    
    args = parser.parse_args()

    movi_h5 = h5py.File(args.movi_path, "r")
    lifted_h5 = h5py.File(args.lifted_path, "r")
    gt_norm_stats = json.load(open(args.gt_norm_path, "r"))
    pg1_norm_stats = json.load(open(args.pg1_norm_path, "r"))
    pg2_norm_stats = json.load(open(args.pg2_norm_path, "r"))

    out_file = h5py.File(args.out_hdf5, "w")

    # Dataset-level metadata — useful for anyone opening the file cold
    out_file.attrs["description"]  = "MoVi AMASS — SMPL-H/X parameters per action clip"
    out_file.attrs["framerate"]    = FRAMERATE
    out_file.attrs["n_joints"]     = 52
    out_file.attrs["pose_shape"]   = "T x 52 x 3  (axis-angle, radians)"
    out_file.attrs["trans_units"]  = "metres"
    out_file.attrs["joint_layout"] = (
        "joint[0]=root  "
        "joint[1-21]=body  "
        "joint[22-36]=left_hand  "
        "joint[37-51]=right_hand"
    )
    out_file.attrs["smplx_mapping"] = (
        "global_orient=poses[:,0,:]  "
        "body_pose=poses[:,1:22,:].reshape(-1,63)  "
        "left_hand_pose=poses[:,22:37,:].reshape(-1,45)  "
        "right_hand_pose=poses[:,37:52,:].reshape(-1,45)  "
        "betas=betas[:10]"
    )

    for split in ("train", "val", "test"):
        print(f"Processing split {split}...")
        grp_movi = movi_h5[split]
        grp_lifted = lifted_h5[split]
        
        grp_out = out_file.require_group(split)

        for clip_name in grp_movi.keys():
            movi_processed = normalize_clip(grp_movi[clip_name], clip_name, gt_norm_stats)
            
            lifted_processed = {}

            for camera in ("pg1", "pg2"):
                if camera not in grp_lifted or clip_name not in grp_lifted[camera]:
                    print(f"Skipping missing lifted data for {split}/{camera}/{clip_name}")
                    continue

                if camera == "pg1":
                    norm_stats = pg1_norm_stats
                else:
                    norm_stats = pg2_norm_stats

                lifted_normalized = normalize_clip(grp_lifted[camera][clip_name], clip_name, norm_stats)
                poses_up, trans_up = upsample(
                    lifted_normalized["poses"],
                    lifted_normalized["trans"],
                    T=grp_movi[clip_name].attrs.get("n_frames")
                )
                lifted_processed[camera] = {
                    "poses": poses_up,
                    "trans": trans_up,
                    "betas": lifted_normalized["betas"]
                }

            if not lifted_processed:
                print(f"Skipping: {split}/{clip_name}, no lifted data found")
                continue
            g = grp_out.create_group(clip_name)
            for key, value in grp_movi[clip_name].attrs.items():
                g.attrs[key] = value
            for key in ("poses", "trans", "betas"):
                g.create_dataset(key, data=movi_processed[key], compression="gzip", compression_opts=4, chunks=True)
            for camera, camera_data in lifted_processed.items():
                cam_grp = g.create_group(camera)
                for key in ("poses", "trans", "betas"):
                    cam_grp.create_dataset(key, data=camera_data[key], compression="gzip", compression_opts=4, chunks=True)



if __name__ == "__main__":
    main()