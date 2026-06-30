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
    │   │   ├── poses   (T, 52, 3)   — normalized GT axis-angle
    │   │   ├── trans   (T,  3)      — normalized GT root translation
    │   │   ├── betas   (16,)        — normalized GT shape coefficients
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
"""


class MoViDataset(Dataset):
    def __init__(self, h5_path, norm_stats_path, split="train", device="cpu",
                 cameras=("pg1", "pg2"), keys=("poses", "trans", "betas"), verbose=False):
        super().__init__()
        self.h5_path = h5_path
        self.norm_stats_path = norm_stats_path
        self.split = split
        self.device = torch.device(device)
        self.cameras = cameras
        self.keys = keys

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

    def __getitem__(self, idx):
        clip_name, camera = self.samples[idx]
        data = {"x": {}, "y": {}}
        with h5py.File(self.h5_path, "r") as f:
            clip_grp = f[self.split][clip_name]
            for key in self.keys:
                data["x"][key] = torch.from_numpy(clip_grp[camera][key][:].astype(np.float32)).to(self.device)
                data["y"][key] = torch.from_numpy(clip_grp[key][:].astype(np.float32)).to(self.device)
        return data
    
# TODO: Anya: check length and norm stats, unnormalization - round trip, visualize before & after renders
