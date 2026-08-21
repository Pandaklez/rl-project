import numpy as np
import h5py
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.camera_frame import correct_root  # noqa: E402

FRAMERATE = 120  # Hz — fixed for MoVi

# SMPL-X stores 52 joints: 0 global orient, 1-21 body, 22-51 the two hands.
# Only the first 22 are kept.
#
# MoVi has no finger mocap, so GT joints 22-51 hold one constant canonical hand
# pose in every frame of every clip. Normalising a constant dimension divides
# float noise by its own standard deviation (~1e-12 here, which the `sigma == 0`
# guard below does not catch) and maps it to exactly +/-1 — while SMPLer-X does
# predict fingers, so the lifted values are genuine. That turns 90 of 156 pose
# dimensions into a perfect "GT or lifted" label, which a motion discriminator
# reads instead of learning anything about pose.
#
# Nothing downstream wants them either: SMPL-X forward kinematics returns 22 body
# joints and they are unaffected by finger pose (fingers are leaves of the
# kinematic tree), ViTPose is COCO-17 and has no finger keypoints, and PA-MPJPE
# is measured over 22 joints or fewer. Their only other effect was on the
# smoothness reward, which averages squared acceleration over every dimension it
# is given -- and 91.6% of that energy was finger jitter.
N_JOINTS = 22


def _crop_joints(arr, n_joints=N_JOINTS):
    """Keep the first `n_joints` of a (T, J, 3) pose array; pass anything else through."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[1] > n_joints:
        return a[:, :n_joints, :]
    return a


def normalize_clip(clip, clip_name, norm_stats):
    normalized = {}
    for key in ("poses", "trans", "betas"):
        if key not in clip:
            raise ValueError(f"Missing key '{key}' in clip {clip_name}")
        mu = np.array(norm_stats[key]["mu"])
        sigma = np.array(norm_stats[key]["sigma"])
        sigma = np.where(sigma == 0, 1.0, sigma)  # avoid division by zero for constant dims
        value = clip[key][:]
        if key == "poses":
            value = _crop_joints(value)
        normalized[key] = (value - mu) / sigma
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

def create_normalization(data_file, norm_out_file, camera=None, correct=False, calib_dir=None):
    """
    Compute mu/sigma over the train split.

    When `correct` is set the lifted root is rotated into the world frame first,
    so the stats describe the same data the clips are normalized against. The
    flag is recorded in the file so a stale mismatch can be detected on load.
    """
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
            arr = ex[ds][:]
            if ds == "poses":
                # Crop before the stats, not after: mu/sigma must describe the
                # same vector the clips are normalised against.
                arr = _crop_joints(arr)
            if correct and camera and ds == "poses":
                arr = correct_root(arr, camera, calib_dir)
            raw_data.append(arr)
        if ds == "betas":
            np_data = np.stack(raw_data,axis = 0)
        else:
            np_data = np.concatenate(raw_data,axis = 0)
        stats[ds] = {
            "shape" : np_data.shape,
            "mu" : np_data.mean(axis=0).tolist(),
            "sigma" : np_data.std(axis=0).tolist()
        }
    stats["_root_corrected"] = bool(correct and camera)
    # Guard against a stale 52-joint stats file being used against 22-joint
    # poses: the shapes broadcast in some directions and would corrupt silently.
    stats["_n_joints"] = N_JOINTS
    with open(norm_out_file,"w") as dump_file:
        json.dump(stats,dump_file)


def norm_is_stale(path, want_corrected, want_joints=N_JOINTS):
    """
    True if an existing norm file describes a different vector than the one we
    are about to normalise against.

    Two ways it can disagree. The root correction changes joint 0's *values*, and
    a mismatch there is silent -- the shapes still line up and the pose comes out
    plausible but wrong. The joint count changes the vector's *length*; that one
    raises on the broadcast, but catching it here regenerates the stats instead
    of failing a training run an hour in.

    Files written before `_n_joints` existed are 52-joint by definition.
    """
    if not os.path.exists(path):
        return False
    with open(path) as f:
        stats = json.load(f)
    return (bool(stats.get("_root_corrected", False)) != bool(want_corrected)
            or int(stats.get("_n_joints", 52)) != int(want_joints))


def main():
    parser = argparse.ArgumentParser(
        description="Normalize GT and lifted poses and write to a single HDF5 "
                    "matching the MoViDataset layout."
    )
    parser.add_argument("--movi_path",    default="data/movi_smplx.h5",
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
    parser.add_argument("--calib_dir",   default="data/Calib",
                        help="MoVi camera calibration (Extrinsics_PGX.npz, cameraParams_PGX.npz)")
    parser.add_argument("--no_correct_root", action="store_true",
                        help="Skip rotating the lifted root into the world frame. The raw "
                             "SMPLer-X root is camera-relative and sits ~95 deg (PG1) / "
                             "~120 deg (PG2) from GT — see src/camera_frame.py")
    args = parser.parse_args()

    correct = not args.no_correct_root

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

    # GT is never root-corrected (it is already world-frame), but it is subject
    # to the joint-count check, so it goes through the same staleness gate.
    if norm_is_stale(gt_norm_path, False):
        print(f"  {os.path.basename(gt_norm_path)} describes a different pose "
              f"vector, regenerating")
        os.remove(gt_norm_path)
    if not os.path.exists(gt_norm_path):
        create_normalization(movi_h5,gt_norm_path)
    for path, cam in ((pg1_norm_path, "PG1"), (pg2_norm_path, "PG2")):
        if norm_is_stale(path, correct):
            print(f"  {os.path.basename(path)} describes a different pose vector "
                  f"(root_corrected or joint count), regenerating")
            os.remove(path)
        if not os.path.exists(path):
            create_normalization(lifted_h5, path, cam, correct, args.calib_dir)

    gt_norm_stats  = json.load(open(gt_norm_path))
    pg1_norm_stats = json.load(open(pg1_norm_path))
    pg2_norm_stats = json.load(open(pg2_norm_path))


    out_file.attrs["description"] = "MoVi — normalized GT + lifted poses per action clip"
    out_file.attrs["framerate"]   = FRAMERATE
    # Body joints only; the 30 finger joints are cropped above. Was hard-coded 52.
    out_file.attrs["n_joints"]    = N_JOINTS
    out_file.attrs["root_corrected"] = correct
    out_file.attrs["root_note"] = (
        "lifted poses[:,0] is world-frame when root_corrected; the untouched "
        "camera-frame root is kept per camera as 'root_cam' (unnormalized). "
        "lifted 'trans' is virtual-camera depth from SMPLer-X, NOT metres."
    )

    # Embed the calibration so downstream reprojection does not need data/Calib.
    calib_out = out_file.create_group("calib")
    for cam_out, cam_src in (("pg1", "PG1"), ("pg2", "PG2")):
        cg = calib_out.create_group(cam_out)
        for fname, keys in (("cameraParams", ("IntrinsicMatrix", "RadialDistortion")),
                            ("Extrinsics",   ("rotationMatrix", "translationVector"))):
            src = np.load(os.path.join(args.calib_dir, f"{fname}_{cam_src}.npz"))
            for k in keys:
                cg.create_dataset(k, data=src[k])
        cg.attrs["note"] = "MATLAB row-vector convention: X_cam = X_world @ R + t"

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
                raw_poses = cam_grp["poses"][:]
                # Keep the untouched camera-frame root: it is what a reprojection
                # reward needs to get back into the image, and it is the only part
                # of the lifted pose the correction changes.
                root_cam = raw_poses[:, 0, :].copy()
                src_clip = {
                    "poses": correct_root(raw_poses, cam_src, args.calib_dir) if correct else raw_poses,
                    "trans": cam_grp["trans"][:],
                    "betas": cam_grp["betas"][:],
                }
                try:
                    cam_norm_data = normalize_clip(src_clip, clip_name, cam_norm[cam_out])
                except Exception as e:
                    print(f"  warn  {split}/{clip_name}/{cam_src}: normalisation failed — {e}")
                    continue
                poses_up, trans_up = upsample(cam_norm_data["poses"], cam_norm_data["trans"], T_gt)
                root_cam_up, _ = upsample(root_cam[:, None, :], cam_norm_data["trans"], T_gt)
                lifted_norm[cam_out] = {
                    "poses": poses_up,
                    "trans": trans_up,
                    "betas": cam_norm_data["betas"],
                    "root_cam": root_cam_up[:, 0, :],
                }

            if not lifted_norm:
                print(f"  skip {split}/{clip_name}: no lifted cameras available")
                skipped.append(clip_name)
                continue

            g = grp_out.create_group(clip_name)
            gt_grp = g.create_group("gt")
            for key in ("poses", "trans", "betas"):
                gt_grp.create_dataset(key, data=gt_norm[key], compression="gzip", compression_opts=4, chunks=True)
            for cam_out, cam_data in lifted_norm.items():
                cam_grp = g.create_group(cam_out)
                for key in ("poses", "trans", "betas", "root_cam"):
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