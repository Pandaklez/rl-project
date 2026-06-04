import random
from pathlib import Path
# import logging

import h5py
import numpy as np
import scipy.io as sio
from decord import VideoReader, cpu
import matplotlib.pyplot as plt


from pack_movi_hdf5 import _scalar, _str, _unwrap

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)s  %(message)s",
#     datefmt="%H:%M:%S",
#     filename="check_times.log"
# )
# log = logging.getLogger(__name__)

def frame_to_time(frame, fps):
    return frame/fps

def time_to_frame(time,fps):
    return int(round(time*fps,0))

def seconds_to_minutes(seconds):
    mins = seconds // 60
    secs = seconds % 60
    return f'{int(mins)}:{secs:.1f}'

def get_action_meta(meta_path):
    mat = sio.loadmat(str(meta_path), struct_as_record=False, squeeze_me=False)
    top_key = next(k for k in mat if not k.startswith("__"))
    subj    = _unwrap(mat[top_key])   # mat_struct with fields: id, subject, move
    move_arr = _unwrap(subj.move)

    action_names = [str(_unwrap(action[0])).lstrip("['").rstrip("']") for action in move_arr.motions_list]
    action_inds = np.array([[int(tup[0]),int(tup[1])] for tup in move_arr.flags30])

    return action_names, action_inds

def check_move_times(video_path, action_names, action_inds,n_checks = 3, fps=30):
        
    vr = VideoReader(video_path, ctx=cpu(0))

    num_frames = len(vr)
    avg_fps = vr.get_avg_fps()
    duration = num_frames / fps
    mins = duration // 60
    secs = duration % 60
    print(f"Analyzing file: {video_path}")
    print(f"Total frames: {num_frames}")
    print(f"FPS: {avg_fps:.2f}")
    print(f"Duration (s): {duration:.2f}")
    print(f'Duration (m:s): {int(mins)}:{secs:.2f}\n')

    for name, interval in zip(action_names[:n_checks],action_inds[:n_checks]):
        start, end = interval
        start_time = frame_to_time(start, avg_fps)
        end_time = frame_to_time(end,avg_fps)
        duration = end_time-start_time

        start_pretty = seconds_to_minutes(start_time)
        end_pretty = seconds_to_minutes(end_time)
        duration_pretty = seconds_to_minutes(duration)

        print(f"{name}: {start_pretty} - {end_pretty} ({duration_pretty})\n\n")





if __name__ == '__main__':
    subject = 1
    
    meta_path = f"F_Subjects_1_45/F_v3d_Subject_{subject}.mat"

    action_names, action_inds = get_action_meta(meta_path)

    video_paths = [
        f"videos/CP1_mp4/F_CP1_Subject_{subject}.mp4",
        f"videos/CP2_mp4/F_CP2_Subject_{subject}.mp4",
        f"videos/PG1_avi/F_PG1_Subject_{subject}_L.avi",
        f"videos/PG2_avi/F_PG2_Subject_{subject}_L.avi",
    ]

    for video_path in video_paths:
        check_move_times(video_path, action_names, action_inds)

