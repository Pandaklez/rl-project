import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np

"""
Expected HDF5 layout (produced by data/norm_upsample.py):

processed_movi.h5
    ├── train/
    │   ├── Subject_1__walking/
    │   │   ├── gt/
    │   │   │   ├── poses  (T, 52, 3)  — normalized GT axis-angle
    │   │   │   ├── trans  (T,  3)     — normalized GT root translation
    │   │   │   └── betas  (16,)       — normalized GT shape coefficients
    │   │   ├── pg1/
    │   │   │   ├── poses  (T, 52, 3)  — normalized + upsampled lifted (PG1 camera)
    │   │   │   ├── trans  (T,  3)
    │   │   │   └── betas  (16,)
    │   │   ├── pg2/
    │   │   │   ├── poses  (T, 52, 3)
    │   │   │   ├── trans  (T,  3)
    │   │   │   └── betas  (16,)
    │   │   └── attrs: gender, action, subject, height, mass, age,
    │   │               framerate, n_frames, split
    │   └── ...
    ├── val/  ...
    └── test/ ...

Each sample returned by __getitem__ is a (clip_name, camera) pair:
    {
        "x": {"poses": (T,52,3), "trans": (T,3), "betas": (16,)}  — lifted input
        "y": {"poses": (T,52,3), "trans": (T,3), "betas": (16,)}  — GT target
    }

If `reproj_path` is given (data/reproj_targets.h5, from
scripts/build_reproj_targets.py) each sample also carries the 2D evidence a
reprojection reward needs, resampled onto the same T-frame timeline:

    "reproj": {
        "kp2d":         (T, 17, 3)  ViTPose COCO-17, image px + confidence
        "trans_metric": (T, 3)      root translation in metres, real camera frame
        "bbox":         (T, 4)      xywh after process_bbox
        "valid":        (T,)        per-frame usability
    }

Clips with no usable targets still return the key, filled with zeros and
`valid` all-False, so batching never has to special-case them.
"""

GT_GROUP = "gt"
N_COCO = 17


def gt_group(clip_grp):
    """
    GT lives in its own 'gt' subgroup. Files written before that change stored it
    flat at the clip root, so fall back to the clip group itself for those.
    """
    return clip_grp[GT_GROUP] if GT_GROUP in clip_grp else clip_grp


def resample_to(arr, T):
    """
    Lift a native-30 Hz target array (t0, ...) onto the clip's T-frame timeline.

    Uses exactly the mapping data/norm_upsample.py:47-52 applied to the poses —
    `t_src = linspace(0, T-1, t0)` — so target frame i lines up with the pose the
    policy sees at frame i. Anything else would put the 2D evidence a few frames
    away from the pose it is supposed to score.
    """
    arr = np.asarray(arr)
    t0 = arr.shape[0]
    if t0 == T:
        return arr.copy()
    if t0 == 0:
        return np.zeros((T,) + arr.shape[1:], dtype=arr.dtype)
    t_src = np.linspace(0, T - 1, t0)
    flat = arr.reshape(t0, -1).astype(np.float64)
    out = np.empty((T, flat.shape[1]), dtype=np.float64)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(np.arange(T), t_src, flat[:, d])
    return out.reshape((T,) + arr.shape[1:]).astype(arr.dtype)


class MoViDataset(Dataset):
    def __init__(self, h5_path, norm_stats_path, split="train", device="cpu",
                 cameras=("pg1", "pg2"), keys=("poses", "trans", "betas"), verbose=False,
                 reproj_path=None):
        super().__init__()
        self.h5_path = h5_path
        self.norm_stats_path = norm_stats_path
        self.split = split
        self.device = torch.device(device)
        self.cameras = cameras
        self.keys = keys
        self.reproj_path = reproj_path

        self.samples = []
        self.verbose = verbose

        with open(norm_stats_path, "r") as f:
            self.norm_stats = json.load(f)

        with h5py.File(h5_path, "r") as f:
            for clip_name in f[split].keys():
                clip_grp = f[split][clip_name]
                for camera in self.cameras:
                    if camera in clip_grp:
                        self.samples.append((clip_name, camera))
                    elif self.verbose:
                        print(f"Warning: missing lifted data for {split}/{clip_name}/{camera}, skipping")

        self._len = len(self.samples)

    def __len__(self):
        return self._len

    def unscale(self, data_dict):
        """Invert normalization using stored mu/sigma stats."""
        unscaled = {}
        for key, stats in self.norm_stats.items():
            if key in data_dict:
                mu    = np.array(stats["mu"])
                sigma = np.array(stats["sigma"])
                sigma = np.where(sigma == 0, 1.0, sigma)
                unscaled[key] = data_dict[key] * sigma + mu
            else:
                raise ValueError(f"Key '{key}' not found in data_dict for unscaling")
        return unscaled

    def _load_reproj(self, clip_name, camera, T):
        """
        2D targets for one clip, resampled from the native 30 Hz onto T frames.

        Returns zeros with valid=False when the clip has no usable targets — the
        20 cam-clips whose frame alignment was lost (see
        scripts/build_reproj_targets.py), a camera with no video, or a missing
        sidecar entry. Shapes are identical either way so batching is uniform.
        """
        empty = {
            "kp2d": torch.zeros((T, N_COCO, 3), dtype=torch.float32, device=self.device),
            "trans_metric": torch.zeros((T, 3), dtype=torch.float32, device=self.device),
            "bbox": torch.zeros((T, 4), dtype=torch.float32, device=self.device),
            "valid": torch.zeros((T,), dtype=torch.bool, device=self.device),
        }
        with h5py.File(self.reproj_path, "r") as rf:
            grp = rf.get(f"{self.split}/{clip_name}/{camera}")
            if grp is None or not grp.attrs.get("aligned", False):
                return empty

            out = {}
            for key in ("kp2d", "trans_metric", "bbox"):
                out[key] = torch.from_numpy(
                    resample_to(grp[key][:], T).astype(np.float32)).to(self.device)
            # A resampled frame counts as valid only if both source frames it is
            # interpolated between are valid, which linear interp on a 0/1 mask
            # gives for free: anything blended with an invalid frame drops below 1.
            v = resample_to(grp["valid"][:].astype(np.float32), T)
            out["valid"] = torch.from_numpy(v >= 1.0).to(self.device)
        return out

    def __getitem__(self, idx):
        clip_name, camera = self.samples[idx]
        data = {"x": {}, "y": {}}
        with h5py.File(self.h5_path, "r") as f:
            clip_grp = f[self.split][clip_name]
            gt_grp   = gt_group(clip_grp)
            for key in self.keys:
                data["x"][key] = torch.from_numpy(clip_grp[camera][key][:].astype(np.float32)).to(self.device)
                data["y"][key] = torch.from_numpy(gt_grp[key][:].astype(np.float32)).to(self.device)
            T = gt_grp["poses"].shape[0]

        # A reprojection reward has to know which camera's calibration and
        # normalisation stats apply, and the sample is otherwise anonymous.
        data["meta"] = {"clip": clip_name, "camera": camera, "split": self.split}

        if self.reproj_path is not None:
            data["reproj"] = self._load_reproj(clip_name, camera, T)
        return data
    
# TODO: Anya: check length and norm stats, unnormalization - round trip, visualize before & after renders
