import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np

"""
Assuming data in h5_path is on the following format and already normalized:

final_dataset.h5
    ├── train/
    │   ├── Subject_1__walking/
    │   │   ├── norm_poses   (T, 52, 3)
    │   │   ├── norm_trans   (T,  3)
    │   │   ├── norm_betas   (16,)
    |   |   ├── norm_PG1/     
    |   |   |   ├── norm_poses    (T,52,3)
    |   |   |   ├── norm_trans    (T,3)
    |   |   |   ├── norm_betas    (16,)
    |   |   ├── norm_PG2/     
    |   |   |   ├── norm_poses    (T,52,3)
    |   |   |   ├── norm_trans    (T,3)
    |   |   |   ├── norm_betas    (16,)
    │   │   └── attrs:  gender, action, subject, height, mass, age,
    │   │               framerate, n_frames, split
    │   └── ...
    ├── val/  ...
    └── test/ ...
"""

# NOTE: if decided that h_t i.e. recurrent component is needed, data set needs to be windowed in time 
class MoViDataset(Dataset):
    def __init__(self, h5_path, norm_stats_path, split="train", device="cpu", 
                 cameras = ("pg1", "pg2"),keys = ("poses", "trans", "betas")):
        super().__init__()
        self.h5_path = h5_path
        self.norm_stats_path = norm_stats_path
        self.split = split
        self.device = torch.device(device)
        self.cameras = cameras
        self.keys = keys

        self.samples = []

        with open(norm_stats_path, "r") as f:
            self.norm_stats = json.load(f)

        with h5py.File(h5_path, "r") as f:
            for clip_name in f[split].keys():
                for camera in self.cameras:
                    if camera in f[split] and clip_name in f[split][camera]:
                        self.samples.append((clip_name, camera))
                    else:
                        print(f"Warning: Missing lifted data for {split}/{camera}/{clip_name}, skipping")

        self._len = len(self.samples)

    def __len__(self):
        return self._len # NOTE: len is now returning the number of samples (clip-camera pairs) rather than just clips, i.e. not number of frames.
    
    def unscale(self, data_dict):
        """
        Unscale normalized data using the stored mean and std from norm_stats.
        Expects data_dict to have the same keys as norm_stats ("poses", "trans", "betas").
        """
        unscaled = {}
        for key, stats in self.norm_stats.items():
            if key in data_dict:
                unscaled[key] = data_dict[key] * stats["std"] + stats["mean"]
            else:
                raise ValueError(f"Key {key} not found in data_dict for unscaling")
        return unscaled

    def __getitem__(self, idx):
        clip_name, camera = self.samples[idx]
        # NOTE: Flatten and replace with poses_x, trans_x etc...
        data = {
            "x": {},
            "y": {}
        }
        with h5py.File(self.h5_path, "r") as f:
            clip_data = f[self.split][clip_name]
            for key in self.keys:
                data["x"][key] = torch.from_numpy(clip_data[camera][key]).to(self.device)
                data["y"][key] = torch.from_numpy(clip_data[key]).to(self.device)
        return data 