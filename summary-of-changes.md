# Summary of changes

Condensed from `follow-up-on-gustaf-questions.md`, which follows Gustaf's handover
notes point by point and keeps the full reasoning and measurements. This document
lists what changed and how to reproduce it.

---

## Main changes

### Data and ground truth

| change | detail |
|---|---|
| GT rebuilt on SMPL-X | `data/movi_smplx.h5`, 1801 clips. Norm stats regenerated and **bit-identical** to the committed ones |
| Processed dataset | `data/processed_movi.h5`, 1779 clips (1403 / 189 / 187). The 22 skipped lack both camera angles |
| `gt/` subgroup layout | Kept. Restructuring moves links only — array values verified byte-identical |
| Splits | Subject-disjoint: 68 / 9 / 9 subjects, no overlap |

### Camera frame — the root rotation fix

The lifted root was in the **camera** frame, the GT in the world frame: a 93.9°
(PG1) / 119.5° (PG2) constant offset that the policy would otherwise have spent
its capacity undoing.

- `src/camera_frame.py` rotates the root to world using **calibration only**, no GT:
  `R_world = F⁻¹ · (R_extᵀ)⁻¹ · R_camera`, with `F = Rz(+90°)`.
- Applied in `data/norm_upsample.py` before normalising. Result: **93.9° → 5.2°**
  (PG1), **119.5° → 8.5°** (PG2), matching the GT-fitted upper bound.
- Only joint 0 is affected — joints 1-51 are parent-relative.
- The untouched camera-frame root is preserved per camera as `root_cam`, and
  calibration is embedded at `/calib/`, so the file is self-describing.
- Fitting the offset *against GT* was rejected as leakage; those numbers are
  diagnostics only.

### Metrics

- **PA-MPJPE was not PA-MPJPE.** `src/evaluate.py` had no forward kinematics — it
  Procrustes-aligned *axis-angle vectors*, which is meaningless. `src/smplx_fk.py`
  now runs SMPL-X forward first. Baseline on test: **30.0 mm**.
- Note PA-MPJPE is blind to a constant root rotation (Procrustes absorbs it), so
  it is not a substitute for the camera-frame fix.

### The GT-free reward (experiments B and C)

- `scripts/extract_2d.py` recovered the missing 2D evidence: `data/keypoints2d.h5`,
  all 175 videos, 803,324 frames, 99.95% detection within clip ranges.
- `src/reproject.py` converts SMPLer-X's virtual-camera `transl` to metres.
  Validated end to end at **11.6 px (PG1) / 14.5 px (PG2)** with no GT involved.
- `scripts/build_reproj_targets.py` writes `data/reproj_targets.h5` (591,013 target
  frames). 20 of 3532 cam-clips are unrecoverable and marked `aligned=False`.
- `src/rewards.py` implements reprojection + smoothness: `r = exp(-err²/σ²)`,
  σ = 0.04 in bbox-height units, acceleration-based smoothness.

### Training setup

- **Pose-only action.** 159 → 156 dims; the lifted `trans` is virtual-camera depth,
  not metres, so the policy no longer predicts it and it passes through untouched.
  `--predict_trans` restores the old width.
- **Policy init.** `log_std` starts at `log(0.05)` instead of `0`. At std 1.0 both
  reward terms saturate near zero, leaving PPO no gradient.
- **Time-limit bootstrapping.** A clip running out of frames is truncation, not
  termination. It was reported as `terminated`, zeroing the bootstrap value at
  every episode boundary; `time_limit_bootstrap` also defaults to `False` and was
  never set. Both fixed — either alone is inert.
- **Checkpoints** store `asdict(cfg)`, not the `Config` dataclass, which pickled as
  `__main__.Config` and made every checkpoint unloadable outside the trainer.

### Logging

- `src/viz_pose.py` `ImagePoseVizLogger` projects skeletons through the real camera
  and draws them **on the video frame**, beside the ViTPose detections — the same
  path the reward uses. GT is not drawn: it is 65-75 px off on PG2 through the
  calibration.
- Fixed a bug where the figure unnormalised poses with **GT** stats rather than the
  per-camera lifted stats — 31° mean per-joint error.
- TensorBoard: reward components, reprojection error in pixels, and
  **improvement over lifted**, which is the curve that matters.

---

## Results worth knowing

**The reprojection reward is correctly implemented and its signal is weak.**

```
lifted pose     reward 0.704    13.2 px
lifted + noise  reward 0.343    26.8 px    penalised on 98.5% of frames
GT pose         reward 0.711    14.0 px
```

Noise is punished sharply, so the implementation is sound — but GT scores
essentially the same as the lifted pose. That is monocular depth ambiguity, not a
bug: SMPLer-X was trained to fit the 2D evidence, and its residual error lives in
depth, where reprojection is blind.

Consequences: **(B) has little headroom on its own** — expect stabilisation, not
large 3D gains. **(C) is the stronger experiment**: the GAIL/AMP discriminator
supplies the 3D plausibility prior that 2D evidence cannot, while the reprojection
term keeps the policy anchored to the image.

---

## Open items

- **Experiment (B) has not produced a usable run yet.** The first attempt degraded
  monotonically (reward 0.71 → 0.26, improvement 0 → −37 px) and was stopped at
  14%. The truncation/bootstrap fix above is the leading explanation, not a proven
  one; the current run is the test.
- **(C) is not built.** Expert demonstrations are in `gt/`; `poses_vel` / `trans_vel`
  exist in the raw lifted h5 but `norm_upsample` drops them (derivable by finite
  differences).
- **`scripts/raw_data_val.py` betas comparison is broken.** It compares each clip
  against `clips[i-1]`, which is the *same subject* 95.2% of the time, and GT betas
  are identical within a subject — so 95.2% of cases are exact ties scored as
  failures by a strict `<`. The reported 2.7% is an artifact; against a different
  subject it is **53.3%**, i.e. near chance rather than worse than chance. Fix is
  two lines. Only betas is affected.
- **Re-rendering validation** still to do, now that the metric is trustworthy.
- **Do not** build reprojection targets by projecting GT 3D joints — it turns the
  reward into a supervised GT loss in disguise.

---

## Reproducibility

### 1. Build the environment

`environment.yaml` pins the `smplerx` conda environment, which runs training,
evaluation and every SMPL-X forward pass.

```bash
conda env create -f environment.yaml
conda activate smplerx
```

The conda half is verified to solve with `conda env create --dry-run`. Python 3.8 /
torch 1.13.1 / CUDA 11.7 are load-bearing: `mmcv-full` and `mmdet` are compiled
against that combination.

**Three things the file cannot capture:**

1. **`mmcv-full` builds from source on PyPI.** Install from the OpenMMLab wheel
   index instead:
   ```bash
   pip install mmcv-full==1.7.1 \
     -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
   ```
2. **`mmpose` is an editable install** pointing at `smpler-x-main/transformer_utils`,
   and `conda env export` drops editable installs silently. Nothing under `src/`
   imports it — only `scripts/extract_2d.py` uses the mm* stack, via `mmdet`. Needed
   only for the SMPLer-X inference path:
   ```bash
   pip install -e smpler-x-main/transformer_utils
   ```
3. **Model and data files are not packages**: SMPL-X body models
   (`SMPLX_NEUTRAL.pkl` etc.), `vitpose_base.pth`, the mmdet checkpoint, and
   everything under `data/`.

`pytest` is absent from `smplerx`, matching the working environment. Most tests run
in the base env; the ones needing `smplx` or `gymnasium` skip there. To run the full
suite inside `smplerx`: `pip install pytest`.

### 2. Rebuild the dataset

Paths must be passed explicitly — the defaults do not match the layout.

```bash
python scripts/movi_smplx_processing.py \
    --v3d_path   F_Subjects_meta \
    --npz_path   data/MoVi_SMPLX/BMLmovi \
    --out_hdf5   data/movi_smplx.h5 \
    --split_path data/split_index.json \
    --old_h5_from_mat data/movi.h5

python data/norm_upsample.py \
    --movi_path   data/movi_smplx.h5 \
    --lifted_path lifted_movi_part1_upd2.h5
```

`data/MoVi_SMPLX/` (9.8 GB of extracted npz) was deleted after the rebuild —
re-extract from `BMLmovi.tar.bz2` if needed. Add `--no_correct_root` to disable the
camera-frame correction.

### 3. Rebuild the 2D evidence and reprojection targets

```bash
python scripts/extract_2d.py --out data/keypoints2d.h5   # resumable; re-runs as a no-op
python scripts/build_reproj_targets.py                   # -> data/reproj_targets.h5
```

### 4. Validate

```bash
python scripts/raw_data_val.py --processed_h5 data/processed_movi.h5
python scripts/fit_camera_offset.py --gt_h5 data/movi_smplx.h5 --n_clips 40

# tests — base env: 32 passed, 9 skipped (the 9 need smplx)
python -m pytest tests/test_reproject.py tests/test_rewards.py -o addopts=""

# the other two files import skrl / gymnasium, so they only run in smplerx,
# which has no pytest by default:
#     conda activate smplerx && pip install pytest
python -m pytest tests/test_action_layout.py tests/test_env_episode.py -o addopts=""
```

Running the last command in the base env fails at collection with
`ModuleNotFoundError: No module named 'skrl'` — that is the environment, not a
broken test.

### 5. Train and evaluate

```bash
# experiment (B): GT-free reward, pose-only action, image-plane pose logging
python -m src.train --h5_path data/processed_movi.h5 \
    --reward_mode reproj --reproj_path data/reproj_targets.h5 --reproj_sigma 0.04 \
    --viz_mode image --viz_interval 25 --viz_clips 3

# PA-MPJPE
python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --device cpu
```

Always pass `--viz_mode image` so pose figures are logged, not just scalars — a
rising reward curve alone proves little, since the policy can bank a score the
lifted pose had already earned. Watch **`Reprojection / improvement over lifted
(px)`**.

View with the **base** environment's TensorBoard — `smplerx`'s 2.14.0 has a protobuf
incompatibility that makes the dashboard render empty:

```bash
/home/annkle/miniconda/bin/tensorboard --logdir checkpoints --port 6006 --bind_all
```
