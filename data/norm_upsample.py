import numpy as np
import h5py
import argparse
import json
import os

FRAMERATE = 120  # Hz — fixed for MoVi


def normalize_clip(clip, clip_name, norm_stats):
    normalized = {}
    for key in ("poses", "trans", "betas"):
        if key not in clip:
            raise ValueError(f"Missing key '{key}' in clip {clip_name}")
        mu = np.array(norm_stats[key]["mu"])
        sigma = np.array(norm_stats[key]["sigma"])
        sigma = np.where(sigma == 0, 1.0, sigma)  # avoid division by zero for constant dims
        normalized[key] = (clip[key][:] - mu) / sigma
    return normalized


def upsample(poses, trans, T):
    """
    Upsample `poses` and `trans` along time axis to length T using linear interpolation.

    poses : (t0, J, C)
    trans : (t0, D)
    T     : int, target length
    """
    poses = np.asarray(poses)
    trans = np.asarray(trans)
    if poses.ndim != 3:
        raise ValueError(f"poses must have shape (t, J, C), got {poses.shape}")
    if trans.ndim != 2:
        raise ValueError(f"trans must have shape (t, D), got {trans.shape}")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError("poses and trans must have the same time dimension")

    t0 = poses.shape[0]
    if T == t0:
        return poses.copy(), trans.copy()

    t_target = np.arange(T)
    t_src = np.linspace(0, T - 1, t0)

    J, C = poses.shape[1], poses.shape[2]
    flat = poses.reshape(t0, J * C).astype(float)
    up_flat = np.empty((T, J * C), dtype=np.float32)
    for d in range(J * C):
        up_flat[:, d] = np.interp(t_target, t_src, flat[:, d])
    poses_up = up_flat.reshape(T, J, C)

    D = trans.shape[1]
    trans_up = np.empty((T, D), dtype=np.float32)
    for c in range(D):
        trans_up[:, c] = np.interp(t_target, t_src, trans[:, c].astype(float))

    return poses_up, trans_up

def create_split_index(data_file, index_out_file):
    split_index = {}
    for split in ["train", "val", "test"]:
        split_index[split] = list(data_file[split].keys())

    with open(index_out_file, "w") as file:
        json.dump(split_index, file)

def create_normalization(data_file, norm_out_file,camera = None):
    print(f"Normalizing {norm_out_file}...")
    stats = {}
    for ds in ["poses","trans", "betas"]:
        raw_data = []
        for key in list(data_file["train"].keys()):
            if camera:
                try:
                    ex = data_file["train"][key][camera]
                except:
                    continue # Just move on with loop if camera lacking for clip
            else: 
                ex = data_file["train"][key]
            raw_data.append(ex[ds])
        if ds == "betas":
            np_data = np.stack(raw_data,axis = 0)
        else:
            np_data = np.concat(raw_data,axis = 0)
        stats[ds] = {
            "shape" : np_data.shape,
            "mu" : np_data.mean(axis=0).tolist(),
            "sigma" : np_data.std(axis=0).tolist()
        }
    with open(norm_out_file,"w") as dump_file:
        json.dump(stats,dump_file)
    

def main():
    parser = argparse.ArgumentParser(
        description="Normalize GT and lifted poses and write to a single HDF5 "
                    "matching the MoViDataset layout."
    )
    parser.add_argument("--movi_path",    default="data/Gmovi.h5",
                        help="HDF5 with raw GT poses/trans/betas (from movi_raw_processing.py)")
    parser.add_argument("--lifted_path",  default="data/lifted_movi_part1_upd2.h5",
                        help="HDF5 with lifted poses from PG1/PG2 cameras")
    parser.add_argument("--split_index",  default="data/split_index.json",
                        help="JSON with {'train': [...], 'val': [...], 'test': [...]}")
    parser.add_argument("--gt_norm_path", default="data/normalization.json",
                        help="Normalization stats for GT (mu/sigma per key)")
    parser.add_argument("--pg1_norm_path", default="data/normalization_lifted_pg1.json",
                        help="Normalization stats for PG1 lifted data")
    parser.add_argument("--pg2_norm_path", default="data/normalization_lifted_pg2.json",
                        help="Normalization stats for PG2 lifted data")
    parser.add_argument("--out_hdf5",    default="data/processed_movi.h5",
                        help="Output path for the normalized, merged HDF5")
    args = parser.parse_args()

    movi_h5   = h5py.File(args.movi_path,   "r")
    lifted_h5 = h5py.File(args.lifted_path, "r")
    out_file  = h5py.File(args.out_hdf5,    "w")


    index_file = os.path.join(os.getcwd(),args.split_index)
    if not os.path.exists(index_file):
        create_split_index(movi_h5,index_file)

    with open(args.split_index) as f:
        split_index = json.load(f)
    

    gt_norm_path = os.path.join(os.getcwd(),args.gt_norm_path)
    pg1_norm_path = os.path.join(os.getcwd(),args.pg1_norm_path)
    pg2_norm_path = os.path.join(os.getcwd(),args.pg2_norm_path)

    if not os.path.exists(gt_norm_path):
        create_normalization(movi_h5,gt_norm_path)
    if not os.path.exists(pg1_norm_path):
        create_normalization(lifted_h5,pg1_norm_path,"PG1")
    if not os.path.exists(pg2_norm_path):
        create_normalization(lifted_h5,pg2_norm_path,"PG2")

    gt_norm_stats  = json.load(open(gt_norm_path))
    pg1_norm_stats = json.load(open(pg1_norm_path))
    pg2_norm_stats = json.load(open(pg2_norm_path))


    out_file.attrs["description"] = "MoVi — normalized GT + lifted poses per action clip"
    out_file.attrs["framerate"]   = FRAMERATE
    out_file.attrs["n_joints"]    = 52

    cam_norm = {"pg1": pg1_norm_stats, "pg2": pg2_norm_stats}

    counts  = {"train": 0, "val": 0, "test": 0}
    skipped = []

    for split, clip_names in split_index.items():
        grp_movi   = movi_h5[split]
        grp_lifted = lifted_h5[split]
        grp_out    = out_file.require_group(split)

        for clip_name in clip_names:
            if clip_name not in grp_movi:
                print(f"  skip {split}/{clip_name}: not in GT h5")
                skipped.append(clip_name)
                continue

            gt_clip = grp_movi[clip_name]
            T_gt    = gt_clip.attrs.get("n_frames", gt_clip["poses"].shape[0])

            try:
                gt_norm = normalize_clip(gt_clip, clip_name, gt_norm_stats)
            except Exception as e:
                print(f"  skip {split}/{clip_name}: GT normalisation failed — {e}")
                skipped.append(clip_name)
                continue

            # Normalise and upsample each camera
            lifted_norm = {}
            for cam_out, cam_src in (("pg1", "PG1"), ("pg2", "PG2")):
                if clip_name not in grp_lifted or cam_src not in grp_lifted[clip_name]:
                    print(f"  warn  {split}/{clip_name}: missing {cam_src} in lifted h5, skipping camera")
                    continue
                cam_grp = grp_lifted[clip_name][cam_src]
                try:
                    cam_norm_data = normalize_clip(cam_grp, clip_name, cam_norm[cam_out])
                except Exception as e:
                    print(f"  warn  {split}/{clip_name}/{cam_src}: normalisation failed — {e}")
                    continue
                poses_up, trans_up = upsample(cam_norm_data["poses"], cam_norm_data["trans"], T_gt)
                lifted_norm[cam_out] = {
                    "poses": poses_up,
                    "trans": trans_up,
                    "betas": cam_norm_data["betas"],
                }

            if not lifted_norm:
                print(f"  skip {split}/{clip_name}: no lifted cameras available")
                skipped.append(clip_name)
                continue

            g = grp_out.create_group(clip_name)
            for key in ("poses", "trans", "betas"):
                g.create_dataset(key, data=gt_norm[key], compression="gzip", compression_opts=4, chunks=True)
            for cam_out, cam_data in lifted_norm.items():
                cam_grp = g.create_group(cam_out)
                for key in ("poses", "trans", "betas"):
                    cam_grp.create_dataset(key, data=cam_data[key], compression="gzip", compression_opts=4, chunks=True)

            for attr_key, attr_val in gt_clip.attrs.items():
                g.attrs[attr_key] = attr_val

            counts[split] += 1

    movi_h5.close()
    lifted_h5.close()
    out_file.close()

    total = sum(counts.values())
    print(f"\n=== Done ===")
    print(f"  train : {counts['train']:4d} clips")
    print(f"  val   : {counts['val']:4d}   clips")
    print(f"  test  : {counts['test']:4d}   clips")
    print(f"  total : {total:4d} clips  ({len(skipped)} skipped)")
    print(f"  output: {args.out_hdf5}")


if __name__ == "__main__":
    main()
