# Follow-up on Gustaf's handover notes

Structured to follow his notes point by point. Each section states what was checked,
the answer, and which script produces it.

**Scripts written for this follow-up**

| script | what it does |
|---|---|
| `scripts/fit_camera_offset.py` | Recovers the per-camera root rotation offset directly from data, no calibration files needed |
| `src/camera_frame.py` | **The fix.** Rotates a lifted root into the world frame using calibration only — no GT involved |
| `src/smplx_fk.py` | SMPL-X forward kinematics: axis-angle pose params -> 3D joint positions |
| `src/viz_pose.py` | TensorBoard 2D skeleton overlays (lifted / corrected / GT) on held-out clips |
| `scripts/extract_2d.py` | Re-derives per-frame bbox + ViTPose 2D keypoints, unblocking the reprojection reward |
| `scripts/migrate_gt_layout.py` | One-off in-place migration of an old flat `processed_movi.h5` to the `gt/` subgroup layout |

**Files changed**: `src/evaluate.py`, `src/data/datasets.py`, `scripts/raw_data_val.py`,
`scripts/movi_smplx_processing.py`.

---

## 1. SMPL-H / SMPL-X rebuild

### Reproduced, end to end

| artifact | result |
|---|---|
| `data/movi_smplx.h5` | 1801 clips (1424 train / 190 val / 187 test), 0 skipped, 648 MB |
| `data/processed_movi.h5` | 1779 clips (1403 / 189 / 187), **22 skipped** |
| `data/normalization.json` | regenerated from the SMPL-X GT |

The 22 skipped clips are exactly the ones lacking both camera angles, as you predicted.

Three checks passed:

- Your `validate_old_new` reports `in_old_not_new: 0`, `in_new_not_old: 0`, `in_wrong_split: 0` on all three splits.
- **The regenerated norm stats are bit-identical to the ones you committed** — 967,016 frames, 1424 clips, `max|dmu| = 0.00e+00`, `max|dsigma| = 0.00e+00`. The rebuild matches yours exactly.
- `scripts/raw_data_val.py` reproduces your ~90%: poses at **96.1 / 96.3 / 97.3%** (PG1, train/val/test). Current numbers are recorded in the comment block at the bottom of that script.

### The `.gitignore` path warning was justified

Two things had to be passed explicitly rather than relying on defaults:

```bash
python scripts/movi_smplx_processing.py \
    --v3d_path   F_Subjects_meta \            # repo root, not data/
    --npz_path   data/MoVi_SMPLX/BMLmovi \
    --out_hdf5   data/movi_smplx.h5 \
    --split_path data/split_index.json \
    --old_h5_from_mat data/movi.h5            # enables your old-vs-new comparison

python data/norm_upsample.py \
    --movi_path   data/movi_smplx.h5 \
    --lifted_path lifted_movi_part1_upd2.h5   # repo root, not data/
```

`scripts/movi_smplx_processing.py` also needed `from __future__ import annotations` — its
`list[int]` annotations are evaluated at def-time and crashed on the `smplerx` env
(Python 3.8). Added; it now runs on both 3.8 and 3.12. Same fix already applied to
`pack_movi_hdf5.py`.

### "Could the val_rmse percentage differences be a rounding error?"

**No.** Restructuring the HDF5 moves links, it does not touch float values — verified when
migrating the old file with `scripts/migrate_gt_layout.py`, where a full old-vs-new
comparison across 1680 clips found byte-identical arrays and preserved attributes. The
percentage shifts came from the **GT data changing** (SMPL-H -> SMPL-X), not from the layout.

### "Feel free to revert the `gt` level if you don't like it"

Keeping it. It is the cleaner structure, and it is what made the two bugs below separable.
`src/` had to be updated to match — see §4.

### "Final validation of re-rendering, I leave this to you"

Deferred, and there is a reason to do it *after* §3: the metric it would have been validated
against was broken. See §4.

---

## 2. Do we actually care about global translation?

**Your instinct is right, and the reason is stronger than the one in your notes.**

It is not just that the two cameras have different translational offsets. The lifted `trans`
is not a translation in metres at all:

| | root axis-angle mean | trans mean |
|---|---|---|
| GT | `[1.512, 0.220, -0.092]` | `[0.007, -0.199, **0.862**]` m |
| Lifted PG1 | `[-3.069, 0.007, -0.183]` | `[-0.064, 0.242, **41.78**]` |

`smpler-x-main/inference.py:152` stores `cam_trans` **under the name `transl`**. That value
comes from `get_camera_trans()` (`SMPLer_X.py:68-76`), which derives depth from a **virtual
focal length of 5000 px** and a fixed `camera_3d_size`, in the *cropped-bbox* camera
(`config_smpler_x_b32.py:86-98`). A depth of 41.8 against a GT height of 0.86 m is that
virtual camera, not the room.

Consequences:

- No rigid transform recovers it. Converting to the real camera needs the per-frame bbox and
  real intrinsics (`f_crop = 5000/192 * bbox_w`, `inference.py:159`), i.e. the `meta/*.json`
  files written next to each npz. If those are gone, absolute translation is unrecoverable.
- This is why `trans` sits at ~57% in `val_rmse` and will not improve from rotation work.
- `smplx_to_h5.py:303` sets `trans_units = "metres"`. **That attribute is wrong for the
  lifted groups** and worth correcting before it misleads someone.

Recommendation: drop absolute translation, train on pose. PA-MPJPE Procrustes-aligns anyway,
so the metric never depended on it — `src/smplx_fk.py` zeroes `transl` explicitly.

---

## 3. Camera angles and rotations

### "If the camera params are used by the lifter, this is already taken care of"

**They are not used. Action is needed.** Traced through the whole path:

- `smpler-x-main/inference.py:143` stores `smplx_root_pose` — camera-frame, untouched
- `smpler-x-main/inference.py:152` stores `cam_trans` as `transl`
- Real `focal`/`princpt` are computed at `inference.py:159-160` but used **only for rendering**, then discarded
- `smplx_to_h5.py:177-184` (and `add_pg{1,2}_to_h5.py`) pass both straight through — no rotation, no rescale
- `data/norm_upsample.py` only z-scores and interpolates

Training-side proof that the output is camera-relative:
`smpler-x-main/common/utils/preprocessing.py:282-287` rotates the *world* root into the
camera with `cv2.Rodrigues(np.dot(R, root_pose))` to build the supervision target. So the
network is trained to emit camera-frame root by construction. Body joints are
parent-relative and therefore frame-independent — **only joint 0 carries the camera offset.**

### Your ~3 rad / ~1.6 rad observation: two unrelated problems

`scripts/fit_camera_offset.py` fits the best-fit constant rotation `R_lifted ≈ R_off · R_gt`
per joint by proper SVD rotation averaging, subsampling GT onto the lifted timeline so no
rotations are interpolated. Run against both GTs:

```
                OLD GT (SMPL-H)              NEW GT (SMPL-X)
joint   PG1 off  PG2 off |diff|      PG1 off  PG2 off |diff|   PG1 raw  resid
    0    101.8°   129.4°  83.4°       95.3°   122.1°  83.8°     93.4°   5.8°
    1     61.3°    59.6°   5.4°        2.8°     5.7°   3.0°      7.1°   6.3°
    2     27.6°    29.7°   2.4°        3.2°     2.5°   2.9°      7.8°   6.2°
   12     51.5°    47.3°   4.8°        8.4°    11.6°   4.3°      9.3°   6.4°
   16     35.5°    37.0°   1.7°        4.5°     5.5°   1.8°      8.3°   8.4°
```

Read the `|diff|` column — it is the diagnostic:

- **Joint 0**: PG1 and PG2 differ by **83°**, unchanged across the GT swap (83.4° -> 83.8°).
  That is camera geometry. Your ~3 rad.
- **Joints 1-21**: PG1 and PG2 agree to within 1-5°. Camera-invariant, exactly as
  parent-relative joints should be. Your ~1.6 rad was **not** a camera issue — it was the
  SMPL-H/SMPL-X mismatch, and it **collapsed from 20-61° to 2.5-11.6°** when the new GT
  landed. Your rebuild fixed it.

So the coordinate-convention hypothesis in your notes was half right: there is a convention
issue, but it lives only at the root. PG1's offset axis is `[+0.982, -0.033, -0.184]`,
essentially a pure ~95° X-rotation, consistent with AMASS z-up world -> OpenCV
y-down/z-forward camera. PG2 adds an azimuth on top.

### The offset is stable enough to correct

```
PG1: global offset  93.7°   per-clip deviation: median 3.3°  p90 6.6°  max 13.2°
PG2: global offset 119.2°   per-clip deviation: median 4.7°  p90 9.4°  max 23.7°
```

A single constant rotation per camera is a valid model.

### "Risks making the lifted data even closer to GT, but feels proper"

**Trust that unease — it is a real problem.** Fitting `R_off` against GT leaks GT into the
model input. The numbers above are *diagnostics*, not a preprocessing step. Do not ship them
into the pipeline.

The clean fix uses `data/Calib/` extrinsics, which are GT-independent. That is the actual
reason those files matter — not that the offset is unmeasurable without them, but that
measuring it *from GT* is not methodologically sound.

### Resolved — the calibration arrived and it works

`Camera Parameters.tar` extracted to `data/Calib/`. It has exactly what was needed:
`cameraParams_PG{1,2}.npz` (IntrinsicMatrix, RadialDistortion) and `Extrinsics_PG{1,2}.npz`
(rotationMatrix, translationVector) — the filenames `check_globals.py:40-47` already expects.

First check is flip-invariant, so it needs no convention assumptions: the *relative* rotation
between the two cameras must match the measured separation between the two offsets.

```
relative camera rotation from calibration  = 85.86°
measured PG1-vs-PG2 offset separation      = 83.8°
```

Then a sweep over pairing × convention × side, scoring how consistent the implied common
flip `F` is between the two cameras (lower is better):

```
pairing    convention     side      |F_PG1 - F_PG2|
direct     R_ext          right             100.18°
direct     R_ext.T        left               11.35°
direct     R_ext.T        right               4.19°   <- winner
SWAPPED    R_ext.T        right             169.21°
```

**`R_ext.T` confirmed** — the MATLAB row-vector transpose. Using `R_ext` directly gives
100°+, which is why every hypothesis in `check_globals.py` scored the same ~166°. Pairing is
direct, not swapped.

The residual common flip `F` is **88.38° about [0, 0, 1]**, and the nearest exact axis
relabel — `(x, y, z) -> (-y, x, z)`, i.e. +90° about Z — is only **1.72°** away. That is
your `"-y,x,z"` permutation. **Your intuition about a coordinate-convention difference was
correct**; the aliasing bug at `check_globals.py:122` is why you never saw it work.

So the correction needs no GT at all:

```
R_world = F⁻¹ · (R_extᵀ)⁻¹ · R_camera        F = Rz(+90°), exact
```

Implemented in `src/camera_frame.py` (`correct_root(poses, camera)`), applied to
`poses[:, 0, :]` only — joints 1-51 are parent-relative and need nothing.

**Measured on the test split, using calibration only:**

| camera | before | after | GT-fitted upper bound |
|---|---|---|---|
| PG1 | 93.9° | **5.2°** | 5.3° |
| PG2 | 119.5° | **8.5°** | 6.4° |

The calibration-based correction reaches the GT-fitted upper bound on PG1 and comes within
2° on PG2 — so the clean, leak-free fix recovers essentially all of the benefit. **Use this,
not the fitted offsets.**

### Wired into the pipeline

`data/norm_upsample.py` now applies `correct_root` to the lifted poses *before* normalizing,
as you proposed. On the rebuilt `data/processed_movi.h5` (1779 clips, unchanged):

```
root err  corrected-vs-GT :   2.9°
root err  root_cam -vs-GT :  93.7°   (raw camera frame, preserved)
body joint 1 err          :   2.3°   (untouched by the correction)
```

Nothing is lost. Three additions keep the file self-describing:

- **`<clip>/<cam>/root_cam` (T, 3), unnormalized** — the untouched camera-frame root. A
  reprojection reward needs this to get back into the image, and it lets you evaluate a
  genuinely uncorrected baseline.
- **`/calib/{pg1,pg2}/`** — IntrinsicMatrix, RadialDistortion, rotationMatrix,
  translationVector embedded in the h5, so downstream code needs no `data/Calib`.
- **`root_corrected` root attr**, plus `_root_corrected` in the lifted norm JSONs.
  `norm_is_stale()` regenerates the stats automatically if that flag disagrees, so
  correcting and not-correcting can never silently mix.

Disable with `--no_correct_root`.

Note PA-MPJPE is **unchanged** at 30.0 mm before and after. That is expected and is a useful
check on §4: Procrustes aligns per frame, so the metric is blind to a constant root rotation.
It confirms the root fix matters for *training input*, not for that metric.

### Two bugs in `scripts/check_globals.py`

1. **`check_globals.py:122`** — `gt_joint_copy = gt_joint` is a numpy *reference*, not a
   copy. Line 123 overwrites column 0, then line 124 reads the already-overwritten column.
   `"y,-x,z"` actually computes `[y, -y, z]`, and `"-y,x,z"` computes `[-y, -y, z]`.
   **Neither permutation was ever really tested**, so the "around 2.9 for everything"
   readings are uninformative. Needs `gt_joint = gt_joint.copy()` first.
2. **Permuting axis-angle components is not a change of basis.** Rotating a frame requires
   conjugation `R_new = P · R_old · Pᵀ`, not `P · r`.

`get_joint_comp` (`:159-194`) also averages rotation *matrices* elementwise, which does not
produce a rotation. `fit_camera_offset.py` uses SVD projection instead.

### "Correcting from a noisy lifter, not from data turned 90 degrees"

Direct answer: **right now you are training on data turned 95° (PG1) / 122° (PG2) at the
root.** Remove that and the true lifter noise is **5.8°**. That residual is what the policy
should be learning. As it stands it would spend its capacity undoing a constant rotation.

---

## 4. A separate problem found on the way: PA-MPJPE was not PA-MPJPE

Independent of everything above, and worth knowing before any re-rendering validation.

`src/evaluate.py` had **no forward kinematics anywhere**. It Procrustes-aligned and measured
Euclidean distances on **axis-angle rotation vectors**, and called the result PA-MPJPE.
MPJPE is a per-joint *position* error in metres; axis-angle vectors do not live in a space
where distances or rigid alignment mean anything. The reported `0.5096` was unitless and
uninterpretable.

It also explains something confusing: the headline number barely moved across the GT rebuild
(0.532 -> 0.510) even though the same data improved from ~50% to ~96% by RMSE. The metric
was not measuring pose.

**Fixed.** `src/smplx_fk.py` runs SMPL-X forward to joint positions; `src/evaluate.py` now
converts before scoring, at both call sites (`eval_lifted_baseline` and `eval_model`).
GT betas are used for both sides so the metric isolates pose — the lifted shape estimate is
near-chance (§5) and would only add noise. Translation is zeroed, since Procrustes removes
it anyway.

Baseline on the test split is now **30.0 mm**, a plausible PA-MPJPE for a 3D lifter.
Controls:

```
GT vs itself             :   0.000 mm   (must be 0)
GT vs lifted PG1         :  32.51 mm
GT vs a DIFFERENT clip   : 181.56 mm   (must be much larger)
GT vs T-pose (zero pose) : 120.57 mm
```

Note this 30 mm is measured on data that still carries the uncorrected root rotation from §3.
Procrustes absorbs a global rotation per frame, which is why the number is reasonable despite
that — but it also means **PA-MPJPE is blind to the root offset**. It is not a substitute for
fixing it, and a non-aligned MPJPE would look far worse.

---

## 5. Preparing for the three experiments

The dataset now has to support three conditions on the held-out test split:
**(A)** SMPLer-X baseline, no correction; **(B)** PPO with reprojection + smoothness
rewards; **(C)** PPO with the full AMP reward including the GAIL discriminator.

### The split is already suitable

Subject-disjoint, which is stronger than "sequences never seen during training":

```
train: 68 subjects, 1424 clips     val: 9 subjects, 190 clips
test:   9 subjects,  187 clips     train∩test = {}   train∩val = {}
```

### Root correction applies to all three conditions equally

It is data preprocessing, not policy behaviour. If (A) alone ran uncorrected it would
carry a 95-120° root error that says nothing about whether correction policies help, and
(B)/(C) would look artificially good by comparison. `root_cam` is preserved either way,
so a deliberately uncorrected baseline is still available.

### What was missing, and why

The reprojection reward in (B) and (C) needs 2D image evidence. Two things blocked it:

1. **No 2D targets.** Every `output/*/result/` directory is empty — 0 npz and 0 json
   across all 245 of them. The `meta/*.json` files holding bbox / focal / princpt
   (written at `inference.py:169-179`) are gone.
2. **No metric translation.** Lifted `trans` is virtual-camera depth (z ≈ 41.8 against a
   GT height of 0.86 m), and §3's rotation fix does not touch it.

Both are recoverable from material already on disk — 87 PG1 + 88 PG2 videos in
`demo/videos/`, plus `vitpose_base.pth` — and neither needs GT.

### `scripts/extract_2d.py`

Recovers both. Detection replicates the original pipeline exactly: same mmdet Faster
R-CNN checkpoint, same largest-confidence selection, same `bbox_thr` filters, same
`process_bbox` (aspect 384/512, ratio 1.25). Videos are natively 30 fps, matching the
original `ffmpeg -vf fps=30/1`, so frame indices line up 1:1 and the recovered bbox is the
one that produced the lifted poses.

2D keypoints come from **ViTPose-B (COCO-17), implemented directly in PyTorch** — the
installed `mmpose` is SMPLer-X's fork and does not import
(`transformer_utils/mmpose/models/detectors/poseur.py:13` does `from config import cfg`).

Validated on 300 frames: mean keypoint confidence **0.933**, **100%** of keypoints inside
the bbox, **100%** correct anatomical vertical ordering, and the rendered skeleton lands
on the subject. Worth stressing — the checkpoint loaded without error while two real bugs
were still present, so "it loads" proved nothing.

Once it has run, metric translation follows with no GT:

```
f_crop = 5000/192 · bbox_w                 (inference.py:159)
Z_real = tz · f_real / f_crop              f_real from IntrinsicMatrix
X, Y   = (u_img − c) · Z_real / f
```

### Optimisation: mostly a dead end, and here is the evidence

Profiled per frame: **decode 0.95 ms, detection 25.2 ms, ViTPose ~1.0 ms**. Detection is
~93% of the cost, so only detection matters. (An early reading of 22 ms for ViTPose was
cudnn autotuning warmup, not steady state.)

| idea | result | verdict |
|---|---|---|
| Batch the detector | 25.2 -> 35.3 ms/frame | **slower** — mmdet pads to a common size |
| fp16 detector | 1.06x, bboxes shift up to 7.8 px | rejected — negligible gain, breaks fidelity |
| Parallel workers | 40 -> 45 fps aggregate | GPU already saturated; ~12% for real complexity |
| Lower `img_scale` | 1.5-1.9x, bboxes shift 9-23 px | rejected — ~5% depth error |
| Skip frames no clip uses | ~15% fewer frames | **kept** (`flags30` ranges) |
| GPU-side preprocessing | 0.35 -> 0.03 ms/frame | kept |
| cudnn.benchmark + ViT batch 128 | 1.04 -> 0.72 ms/frame | kept |

The three rejected ideas all trade bbox fidelity for speed, and bbox fidelity is exactly
what makes the recovered translation consistent with the existing lifted poses. **The job
cannot be made dramatically faster without invalidating its own output.**

### What is still open for the experiments

- **(C) GAIL/AMP is ready** — GT expert demonstrations are in `gt/`, and AMP state
  transitions come from the sequences. Note `poses_vel` / `trans_vel` exist in the raw
  lifted h5 but `norm_upsample` drops them; they are derivable by finite differences.
- Do not build reprojection targets by projecting GT 3D joints. It is tempting now that
  intrinsics are available, but it turns the reward into a supervised GT loss in disguise —
  the same leakage trap as the fitted offsets in §3.

## 6. TensorBoard pose visualisation

Scalar reward curves do not show whether the policy actually changes the pose.
`src/viz_pose.py` logs a skeleton overlay so that is visible directly.

Each figure is a grid — one row per held-out validation clip, one column per sampled
frame — overlaying **lifted (red, dashed)**, **corrected (blue)** and **GT (green)**. If
blue sits on top of red the policy is doing nothing; if it moves toward green it is
learning. Clips and frame indices are fixed at construction, so figures are directly
comparable across epochs, and the rollout uses the policy **mean** rather than a sample,
so differences reflect learning and not exploration noise.

Wired in via `PPOWithPoseViz` in `src/train.py`, which mirrors PPO's own update trigger
and logs every `--viz_interval` updates (**default 3**), not every step. Alongside the
figure it logs three cheap scalars that answer the same question numerically:

```
pose/delta_from_lifted     how far the policy moved the pose
pose/err_corrected_vs_gt   error after correction
pose/err_lifted_vs_gt      error before correction  (the baseline to beat)
```

Joints come from a real SMPL-X forward pass and are projected orthographically in the
**x-z plane** — AMASS world frame is z-up, verified empirically (head sits 0.62 m above
the pelvis in z; the z spread matches human height). A fixed 1.1 m window centred on the
GT pelvis keeps every panel at the same scale. True perspective reprojection needs the
metric translation above; swapping it in later does not change this file's interface.

Failures in the visualiser are caught and disable it rather than killing a training run.

```bash
python -m src.train --h5_path data/processed_movi.h5 --viz_interval 3 --viz_clips 3
```

## 7. Open items

- **Betas score 2.7%**, worse than chance, i.e. a lifted shape estimate is usually closer to
  some other clip's GT than to its own. Independent of the camera issue and unexamined.
- **`movi_smplx_processing.py` is committed twice**, byte-identical in `data/` and `scripts/`.
  Your note explains why; `scripts/` is canonical, so the `data/` copy should go.
- **`smplx_to_h5.py:303`** sets `trans_units = "metres"`, wrong for the lifted groups (§2).
- **Re-rendering validation** still to do, now that there is a metric worth validating against.
- **`README.md`** is currently the vendored body_visualizer README, overwritten by mistake;
  the committed version at `git show HEAD:README.md` is still the rl-project one.

## Reproducing

```bash
# rebuild GT + processed (see §1 for the path flags)
python scripts/movi_smplx_processing.py --v3d_path F_Subjects_meta ...
python data/norm_upsample.py --movi_path data/movi_smplx.h5 --lifted_path lifted_movi_part1_upd2.h5

# validation
python scripts/raw_data_val.py --processed_h5 data/processed_movi.h5

# camera offsets (§3) — needs no calibration files
python scripts/fit_camera_offset.py --gt_h5 data/movi_smplx.h5 --n_clips 40

# PA-MPJPE (§4) — needs the smplerx env for smplx + torch
python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --device cpu
```

Note `data/MoVi_SMPLX/` (9.8 GB of extracted npz) was deleted after the rebuild to reclaim
disk. Re-extract from `BMLmovi.tar.bz2` if `movi_smplx.h5` ever needs rebuilding.
