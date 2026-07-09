
import argparse
import logging
import random
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

class DataManager:

    def __init__(self, 
                 processed_data_path, 
                 gt_norm_path, 
                 pg1_norm_path, 
                 pg2_norm_path, 
                 camera_param_path, 
                 split = "train", 
                 cameras = ["pg1", "pg2"]):
        
        self.cameras = cameras
        self.split = split
        self.file = None
        
        self.processed_data_path = processed_data_path
        self.norm_stats = self._get_norm_stats(gt_norm_path,pg1_norm_path,pg2_norm_path)
        self.cam_params = self._get_camera_params(camera_param_path)


    def _get_camera_params(self, camera_param_path):
        cam_params = {}
        for cam in self.cameras:
            params = np.load(f"{camera_param_path}/cameraParams_{cam.upper()}.npz")
            extrinsics = np.load(f"{camera_param_path}/Extrinsics_{cam.upper()}.npz")
            cam_params[cam] = {pk: params[pk] for pk in params.keys()} | {ek: extrinsics[ek] for ek in extrinsics.keys()}

        return cam_params



    def _get_norm_stats(self, gt_norm_path, pg1_norm_path, pg2_norm_path):
        gt_norm = json.load(open(gt_norm_path))
        pg1_norm = json.load(open(pg1_norm_path))
        pg2_norm = json.load(open(pg2_norm_path))
        return  {
            "gt" : gt_norm,
            "pg1" : pg1_norm,
            "pg2" : pg2_norm
        }

    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.processed_data_path,"r")
        return self.file
    
    def unnormalize(self, clip, camera):
        """Invert normalization using stored mu/sigma stats."""
        unscaled = {}
        for key, stats in self.norm_stats[camera].items():
            if key in clip:
                mu    = np.array(stats["mu"])
                sigma = np.array(stats["sigma"])
                sigma = np.where(sigma == 0, 1.0, sigma)
                unscaled[key] = clip[key] * sigma + mu
            else:
                raise ValueError(f"Key '{key}' not found in data_dict for unscaling")
        return unscaled
    
    def check_param(self, camera,n_clips = None, joint = 0, comparison = "R_ext*Rt", permutation = None):
        """
        Some terminology I'm using here:
    
        Rc = describes orientation as seen in 'camera space'
        Rt = describes orientation as seen in gt space
        Rext = Rotation matrix provided from data/calib/Extrinsics_PGX.npz

        From the MoVi paper I understand it as Rext is for moving from the global space to the camera space
        hence 
        Rc_hat = R_ext*Rt should be like 'take this global orientation into camera space' i.e. should be close to Rc.

        The goal should be to figure out what might be the difference between gt and lifted, and then apply the inverse
        in our processing script, i.e.(R_ideal^-1)clip['pg1']['poses'][:,0,:], but then we need to understand what this true 
        rotation and its inverse is. 
        NOTE: there could be another issue on top of this, i.e. if the coordinate systems are simply defined differently between 
        gt and cameras (e.g., right, front, up or front, left, up). In this case, maybe we should try permuting x and y? in combination 
        with using the camera extrinsics? like take [-y, x, z] or [y, -x, z]? 
        NOTE: this is now projecting gt into camera, but when we know what seems like the proper combo of permutation and rotation, 
        we need to incorporate the INVERSE of that into the processing script. 
        """
        
        offsets = []

        data = self._get_file()[self.split]
        R_ext = R.from_matrix(self.cam_params[camera]['rotationMatrix'])
        trans_ext = self.cam_params[camera]['translationVector'] # NOTE: Currently not checked, but could be used with the translation data. 
        clip_names = list(data.keys())
        if n_clips:
            clip_names = clip_names[:n_clips]
        for clip_name in clip_names:
            clip = data[clip_name]
            # gt_clip = {key: clip[key] for key in ["poses","trans", "betas"]} # NOTE: commented out as added gt as a key. 
            gt_clip = clip["gt"]
            camera_clip = clip.get(camera)
            if camera_clip:
                camera_scaled = self.unnormalize(camera_clip, camera)
                gt_scaled = self.unnormalize(gt_clip, "gt")

                camera_joint = camera_scaled["poses"][:, joint, :].reshape(-1,3) # (T, 1, 3) -> (T, 3)
                gt_joint = gt_scaled["poses"][:, joint, :].reshape(-1,3) # (T, 1, 3) -> (T, 3)

                if permutation == "y,-x,z":
                    gt_joint_copy = gt_joint
                    gt_joint[:,0] = gt_joint_copy[:,1]
                    gt_joint[:,1] = -1*gt_joint_copy[:,0]
                elif permutation == "-y,x,z":
                    gt_joint_copy = gt_joint
                    gt_joint[:,0] = -1*gt_joint_copy[:,1]
                    gt_joint[:,1] = gt_joint_copy[:,0]
                    

                T = camera_joint.shape[0]
                for i in range(T):
                    
                    Rc = R.from_rotvec(camera_joint[i])
                    Rt = R.from_rotvec(gt_joint[i])
                    if comparison == "R_ext*Rt":
                        Rc_hat = R_ext*Rt
                    elif comparison == "Rt":
                        Rc_hat = Rt
                    elif comparison == "R_ext.inv()*Rt":
                        Rc_hat = R_ext.inv()*Rt

                    R_err = Rc * Rc_hat.inv()

                    angle = R_err.magnitude()

                    offsets.append(angle)

        offsets = np.array(offsets)

        mu = offsets.mean(axis=0)
        sigma = offsets.std(axis = 0)

        return mu, sigma

                

    
    def get_joint_comp(self, camera, joint = 0): # Root orientation by default
        """
        NOTE: initially written func for checking orientations, before camera extrinsics were incorporated
        pretty much not in use. 
        """

        offsets = []

        data = self._get_file()[self.split]

        for clip_name in data.keys():
            clip = data[clip_name]
            # gt_clip = {key: clip[key] for key in ["poses","trans", "betas"]}
            gt_clip = clip["gt"]
            camera_clip = clip.get(camera)

            if camera_clip:
                camera_scaled = self.unnormalize(camera_clip, camera)
                gt_scaled = self.unnormalize(gt_clip, "gt")

                camera_joint = camera_scaled["poses"][:, joint, :].reshape(-1,3) # (T, 1, 3) -> (T, 3)
                gt_joint = gt_scaled["poses"][:, joint, :].reshape(-1,3) # (T, 1, 3) -> (T, 3)

                T = camera_joint.shape[0]
                for i in range(T):
                    Rc_inv = R.from_rotvec(camera_joint[i]).inv()
                    Rt = R.from_rotvec(gt_joint[i])
                    offset = Rt*Rc_inv
                    offsets.append(offset.as_matrix())

        offsets = np.array(offsets)

        mu = offsets.mean(axis=0)
        sigma = offsets.std(axis = 0)

        return mu, sigma



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path",    type = str, default = "data/processed_movi.h5",
                        help="Directory containing full processed data (upsampled and normed)")
    parser.add_argument("--gt_norm_path",    type = str, default = "data/normalization.json",
                        help="Path to the ground truth normalization file of gt")
    parser.add_argument("--pg1_norm_path",    type = str, default = "data/normalization_lifted_pg1.json",
                        help="Path to the pg1 normalization file")
    parser.add_argument("--pg2_norm_path",    type = str, default = "data/normalization_lifted_pg2.json",
                        help="Path to the pg2 normalization file")
    parser.add_argument("--camera_param_path",    type = str, default = "data/Calib",
                        help="Path to the camera calibration files")
    
    args = parser.parse_args()

    dm = DataManager(args.processed_data_path, args.gt_norm_path, args.pg1_norm_path, args.pg2_norm_path, args.camera_param_path, split = "train")
    
    for camera in ("pg1","pg2"):
        joint = 0
        for comparison in ["R_ext*Rt","Rt","R_ext.inv()*Rt"]:
            for permutation in [None,"y,-x,z","-y,x,z"]:

                print(f"{camera}, {joint}, Rc against {comparison}, permutation: {permutation}")
                # mu, sigma = dm.get_joint_comp(camera, joint)
                # mu, sigma = dm.check_param(camera, joint, comparison)
                mu, sigma = dm.check_param(camera=camera,n_clips = 10, joint = 0, comparison = comparison, permutation=permutation)

                print("mu")
                print(mu) # Around 2.9
                print("sigma")
                print(sigma) # Around 0.2

                print()
                print()

        





if __name__ == "__main__":
    main()