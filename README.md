# rl-project

## Repository structure

```
.
├── data
│   ├── __init__.py
    ├── Calib
│   ├── Gmovi.h5
│   ├── lifted_movi_part1_upd1.h5
│   ├── normalization.json
│   ├── normalization_lifted_pg1.json
│   ├── normalization_lifted_pg2.json
│   ├── norm_upsample.py
│   ├── pack_movi_hdf5.py
│   ├── processed_movi.h5
│   ├── split_index.json
│   ├── F_AMASS
│   │   ├── F_amass_Subject_1.mat
│   │   ├── ...
│   │   └── F_amass_Subject_90.mat
│   └── F_Subjects_meta
│       ├── F_v3d_Subject_1.mat
│       ├── ...
│       └── F_v3d_Subject_60.mat
├── scripts
│   ├── __init__.py
│   ├── check_times.py
│   ├── lift_clips.py
│   ├── lift_clips2.py
│   └── movi_raw_processing.py
├── src
│   ├── __init__.py
│   ├── env.py
│   ├── data
│   └── models
├── videos -- videos should move to data/videos, do it locally. 
│   ├── PG1_avi
│   │   ├── F_PG1_Subject_1_L.avi
│   │   ├── ...
│   │   └── F_PG1_Subject_90_L.avi
│   └── PG2_avi
├── .gitignore
├── README.md
```

> Note: Large folders such as `data/F_AMASS`, `data/F_Subjects_meta`, and `videos/PG1_avi` are abbreviated for readability.

