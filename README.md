# rl-project

IRL-guided refinement of 3D human pose estimation on MoVi. A PPO policy learns to
correct the output of a monocular lifter (SMPLer-X), scored against image evidence
rather than ground truth.

Three experiments:

| | condition |
|---|---|
| **A** | SMPLer-X baseline, no correction |
| **B** | PPO with reprojection + smoothness rewards |
| **C** | PPO with the full AMP reward, including the GAIL discriminator |

See `follow-up-on-gustaf-questions.md` for the full record of what was verified,
what was fixed, and the measurements behind each decision.

## Environment

Two interpreters are in play, and the difference matters:

- **`smplerx` conda env (Python 3.8)** — needed for anything touching `smplx` or
  `torch`: training, evaluation, forward kinematics, `scripts/extract_2d.py`.
- **system Python 3.12** — fine for the data/geometry scripts and the test suite.

Code that has to run under both carries `from __future__ import annotations`;
without it, `list[int]` annotations are evaluated at def-time and crash on 3.8.

## Pipeline

```
BMLmovi.tar.bz2 ──► scripts/movi_smplx_processing.py ──► data/movi_smplx.h5   (SMPL-X GT)
                                                              │
SMPLer-X ──► lifted_movi_part1_upd2.h5 ──► data/norm_upsample.py
                                                              ▼
                                                   data/processed_movi.h5
                                             (normalised GT + lifted, per camera,
                                              root-corrected, calibration embedded)

demo/videos ──► scripts/extract_2d.py ──► data/keypoints2d.h5      (bbox + ViTPose 2D)
                                                │
                                                ▼
                            scripts/build_reproj_targets.py ──► data/reproj_targets.h5
                                             (metric translation + 2D targets)
```

## Layout

```
.
├── src
│   ├── camera_frame.py      camera <-> world root rotation, from calibration only
│   ├── reproject.py         virtual-camera transl -> metres; projection
│   ├── rewards.py           GT-free reprojection + smoothness reward
│   ├── smplx_fk.py          SMPL-X forward kinematics -> joint positions
│   ├── env.py               gymnasium env; reward_mode "gt" | "reproj"
│   ├── train.py             PPO via skrl, with TensorBoard pose overlays
│   ├── evaluate.py          PA-MPJPE on real joint positions
│   ├── viz_pose.py          lifted / corrected / GT skeleton overlays
│   ├── data/datasets.py     MoViDataset, optionally with 2D targets
│   └── models/policy.py     Gaussian actor + critic
├── scripts
│   ├── movi_smplx_processing.py   build the SMPL-X GT h5   (canonical copy)
│   ├── extract_2d.py              bbox + ViTPose keypoints from video
│   ├── build_reproj_targets.py    metric translation + 2D targets
│   ├── fit_camera_offset.py       diagnostic: per-camera root offset
│   ├── raw_data_val.py            RMSE-style validation
│   └── migrate_gt_layout.py       one-off flat -> gt/ subgroup migration
├── data
│   ├── norm_upsample.py     normalise + upsample lifted onto the GT timeline
│   ├── Calib/               MoVi camera calibration (intrinsics + extrinsics)
│   ├── normalization.json                 GT stats
│   ├── normalization_lifted_pg{1,2}.json  lifted stats, per camera  ← current
│   └── *.h5                 generated artifacts (gitignored)
└── tests                    test_reproject.py, test_rewards.py
```

> The `normalization_lifted_pg{1,2}.json` at the **repo root** predate the root
> correction and are kept only for provenance. Always read the copies in `data/`;
> `src/rewards.py` enforces this via the `_root_corrected` flag.

## Running

```bash
# rebuild GT + processed  (paths must be passed explicitly, see §1 of the notes)
python scripts/movi_smplx_processing.py --v3d_path F_Subjects_meta \
    --npz_path data/MoVi_SMPLX/BMLmovi --out_hdf5 data/movi_smplx.h5 \
    --split_path data/split_index.json
python data/norm_upsample.py --movi_path data/movi_smplx.h5 \
    --lifted_path lifted_movi_part1_upd2.h5

# 2D evidence + reprojection targets
python scripts/extract_2d.py --out data/keypoints2d.h5     # resumable; re-runs as a no-op
python scripts/build_reproj_targets.py

# train  (needs the smplerx env)
python -m src.train --h5_path data/processed_movi.h5 --reward_mode reproj

# evaluate
python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --device cpu

# tests  (pyproject sets body_visualizer coverage flags, so override addopts)
python -m pytest tests/test_reproject.py tests/test_rewards.py -o addopts=""
```

## Notes

- **Rewards must not use ground truth.** `reward_mode="gt"` exists for ablations
  only; it is a supervised objective. Experiments (B) and (C) use
  `reward_mode="reproj"`, which reads only image evidence.
- **Splits are subject-disjoint** — 68 / 9 / 9 subjects across train / val / test.
- The repo root also vendors `body_visualizer/`, `smplx-main/` and `smpler-x-main/`,
  none of which are part of this project's source.
