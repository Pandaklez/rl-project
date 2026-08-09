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
| `src/reproject.py` | Virtual-camera `transl` -> metres in the real camera, plus projection |
| `scripts/build_reproj_targets.py` | Builds `data/reproj_targets.h5`, the reprojection targets |
| `src/rewards.py` | **The GT-free reward.** Reprojection + smoothness for experiments (B)/(C) |
| `tests/test_reproject.py`, `tests/test_rewards.py` | 41 tests pinning the conversion, the resampling rule and the reward |
| `scripts/migrate_gt_layout.py` | One-off in-place migration of an old flat `processed_movi.h5` to the `gt/` subgroup layout |

**Files changed**: `src/evaluate.py`, `src/data/datasets.py`, `src/env.py`, `src/train.py`,
`src/camera_frame.py`, `scripts/raw_data_val.py`, `scripts/movi_smplx_processing.py`.

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

### Done — `data/keypoints2d.h5`

The full run has completed: **all 175 videos** (87 PG1 + 88 PG2), 803,324 frames, 94 MB.

| | |
|---|---|
| Detection rate *within clip ranges* | **99.95%** mean, 98.1% worst clip |
| Mean keypoint confidence | 0.889 |
| Malformed or truncated groups | none |

One number needs reading carefully: detection across *all* frames is 78%, which is not a
failure rate — it is the `flags30` skip working, since the wanted-frame fraction is 0.783
and frames outside clip ranges are deliberately left zero-filled.

Coverage against the dataset is complete. The 1779 clips span 85 subjects; all have PG2 2D
and all have PG1 2D except `Subject_6`, which has no PG1 video at all and correspondingly
carries only a `pg2` group. The three subjects with 2D but no clips (10, 26, 49) are among
the 22 skipped for lacking both angles.

### Metric translation — `src/reproject.py`

Recovered with no GT, as anticipated:

```
f_crop = 5000/192 · bbox_w                 (inference.py:159)
Z_real = tz · f_real / f_crop              f_real from IntrinsicMatrix
X, Y   = (u_img − c) · Z_real / f
```

`process_bbox` fixes the aspect at 384/512, so `5000/192·bbox_w` and `5000/256·bbox_h` are
equal; `crop_intrinsics` asserts this rather than assuming it. Recovered depth over the test
split is **median 4.5 m, range 2.5–6.1 m** — a mocap lab, not a 42 m virtual one.

**Two traps worth knowing about**, both found by measurement rather than reading:

1. **`cam_trans` positions the model origin, not the pelvis.** SMPLer-X composes its mesh as
   `vertices + cam_trans`, and the SMPL-X pelvis sits ~0.35 m below that origin. The
   natural-looking `J − J[:,0] + trans` therefore shifts the whole body by 0.35 m, which at
   4.5 m depth is ~76 px — bigger than the error being measured. `place_in_camera()` exists
   to make this hard to get wrong.

   | placement | PG1 | PG2 |
   |---|---|---|
   | raw + `cam_trans` (correct) | **11.6 px** | **14.5 px** |
   | pelvis at `cam_trans` | 83.4 px | 95.4 px |

2. **Stay in the camera frame; the PG2 extrinsics are not good enough.** Projecting GT
   through the calibration lands 9.7 px off for PG1 but **65–75 px for PG2 under every
   rotation/translation convention tried** — so this is PG2 calibration accuracy, not a
   convention error. It does not matter, because the whole reward path (lifted camera-frame
   pose → FK → `place_in_camera` → `project`) needs only the intrinsics and the bbox. Both
   cameras validate at 11.6 / 14.5 px that way, with no GT and no extrinsics involved.

The 11.6 / 14.5 px figures are the honest end-to-end check: lifted poses projected against
the ViTPose detections on the test split, on an 800×600 image.

### Reprojection targets — `scripts/build_reproj_targets.py`

Writes `data/reproj_targets.h5` (143 MB), a **sidecar** rather than more groups inside
`processed_movi.h5`, so that rebuilding the processed file does not destroy it.

```
data/reproj_targets.h5
└── <split>/<clip>/<cam>/
    ├── trans_metric (t0, 3)     metres, real camera frame
    ├── kp2d         (t0, 17, 3) ViTPose COCO-17, image px + confidence
    ├── bbox         (t0, 4)     xywh after process_bbox
    ├── valid        (t0,)       per-frame usability
    └── attrs: start, end, t0, n_frames_gt, aligned, detected
```

**Frame alignment is the delicate part, and 20 cam-clips are not recoverable.**
`add_pg{1,2}_to_h5.py:82-103` builds a lifted clip by walking the `flags30` range and
*silently skipping* frames the detector missed, without recording which. So
`lifted index j → video frame start + j` is exact only when nothing was dropped, i.e.
`t0 == end − start`. That holds for **3512 of 3532 cam-clips (99.4%)**. The other 20 —
mostly `crawling` and `cross_legged_sitting`, where the detector struggles — are written
with `aligned = False` and no targets.

Reconstructing the dropped frames from the re-run detector was tried and **rejected**: it
disagrees with the original on 21 cam-clips (it even misses 17 frames on a clip where
nothing was originally dropped), so it is not a reliable witness to what SMPLer-X kept.
Guessing there would silently misalign the reward on exactly the hardest clips.

Targets are stored at the **native 30 Hz** length `t0`, not the 120 Hz GT length —
upsampling 17 keypoints fourfold would quadruple the file for no information.

Coverage: 591,013 target frames, per-clip valid fraction **1.000 median, 0.932 worst**,
no clip below 0.9.

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

### Folded into the loader

`MoViDataset(..., reproj_path="data/reproj_targets.h5")` adds a `reproj` key to each
sample, resampled onto the same T-frame timeline as the poses:

```python
sample["reproj"]["kp2d"]          # (T, 17, 3)
sample["reproj"]["trans_metric"]  # (T, 3)
sample["reproj"]["bbox"]          # (T, 4)
sample["reproj"]["valid"]         # (T,) bool
```

Resampling reuses the exact mapping `norm_upsample.py:47-52` applied to the poses
(`t_src = linspace(0, T-1, t0)`), so a target frame lines up with the pose it scores rather
than sitting a few frames away. Validity is resampled as a 0/1 mask and thresholded at
`>= 1.0`, so a frame interpolated between a valid and an invalid source frame does *not*
inherit validity. Clips with no usable targets — the 20 unaligned ones — still return the
key, zero-filled with `valid` all-False, so batching never special-cases them. Omitting
`reproj_path` leaves the loader's behaviour exactly as before.

`src/reproject.py` also carries `COCO17_TO_SMPLX`, the 12 COCO joints with an unambiguous
SMPL-X counterpart. The five face keypoints (nose, eyes, ears) are deliberately excluded:
the 52-joint skeleton has only `head` in that region, and pairing it with the nose would
bake in a systematic offset.

`tests/test_reproject.py` (21 tests) pins the conversion. The central one is that it is
projection-preserving — the recovered metric point must land on the same pixel under the
real camera as `transl` did under the virtual one — plus regression guards for the pelvis
re-centring trap and the validity-mask rule.

## 6. The reprojection reward — `src/rewards.py`

Written, wired in, and measured. `--reward_mode reproj` on `src/train.py` switches from the
old GT similarity term to reprojection + smoothness. **The GT branch is not read at all on
that path**; `reward_mode="gt"` is kept for ablations, but it is a supervised objective and
is not a valid reward for (B) or (C).

Per frame:

```
corrected pose (normalised, world-frame root, from the policy)
  -> unnormalise with the LIFTED per-camera stats
  -> uncorrect_root: world -> camera frame              src/camera_frame.py
  -> SMPL-X forward kinematics                          src/smplx_fk.py
  -> place_in_camera at the metric translation          src/reproject.py
  -> project with real intrinsics + radial distortion
  -> weighted distance to the ViTPose keypoints
```

`r = exp(-err²/σ²)`, bounded to (0, 1]. Errors are divided by bbox height, so a clip filmed
close up is not scored more harshly than one across the room — verified exact: doubling the
subject doubles the pixel error and leaves the normalised error unchanged.

Design points worth keeping:

- **σ = 0.04 is a conditioning choice, not a taste one.** `exp(-e²/σ²)` is monotone, so σ
  does *not* change which of two poses scores higher — frame ordering was identical at every
  σ from 0.01 to 0.08. It only moves where the reward is steep. `d/de` peaks at `e = σ/√2`,
  and the measured operating point is `e ≈ 0.028`, giving σ ≈ 0.04.
- **Acceleration, not velocity, for smoothness.** A velocity penalty is minimised by a
  subject who stops moving, which is wrong for a dataset of people walking and crawling.
- **Frames with no 2D evidence return `nan`, not 0.** Scoring them zero would teach the
  policy that poorly-detected clips are intrinsically bad — a property of the detector, not
  of the pose. `combine()` falls back instead.
- **Lifted betas, not GT betas** — GT shape would leak and is unavailable at inference.
- **Degenerate depth scores 0, not `nan`.** `t_z` is a sigmoid output in `(0, 56.38)`; a
  policy pushing it outside that has produced a meaningless pose, which is a real miss.

Cost is **3.9 ms/step (255 steps/s)**, dominated by the batch-1 SMPL-X forward pass
(~2 ms). A 3200-step rollout therefore spends ~13 s in the reward.

`tests/test_rewards.py` covers it (32 pass in the default env; the 9 FK-dependent ones need
the `smplerx` env, where they also pass). One guard is worth calling out: **the stale
`normalization_lifted_pg{1,2}.json` at the repo root** predate the root correction and lack
the `_root_corrected` flag. Unnormalising with them puts the pose **138 px** off instead of
13. `load_lifted_stats` now reads from `data/` and raises on the flag rather than failing
silently — this cost me a debugging cycle and would cost anyone else one too.

### The reward works. The signal in it is weak, and that is a real result.

Measured on the test split, lifted baseline vs the reward:

```
lifted pose        reward 0.704    13.2 px
lifted + noise     reward 0.343    26.8 px     penalised on 98.5% of frames
GT pose            reward 0.711    14.0 px
```

Noise is penalised sharply, so the reward is correctly oriented and the implementation is
sound. But **GT scores essentially the same as the lifted pose** — 0.0275 vs 0.0281 of bbox
height. Interpolating lifted → GT moves the reward by only ~1.8% end to end.

That is not a bug, it is monocular depth ambiguity. SMPLer-X was *trained* to fit the 2D
evidence, so it already fits it about as well as GT does; its residual error lives in depth,
which reprojection is blind to. §4 made the same point from the other direction — Procrustes
alignment hides a 95° root error.

Two consequences for the experiments:

- **(B) reprojection + smoothness has little headroom on its own.** Expect it to stabilise
  and smooth the lifted output rather than substantially improve 3D accuracy. Worth running,
  but the hypothesis to state up front is modest.
- **This strengthens the case for (C).** The GAIL/AMP discriminator supplies exactly the
  prior that 2D evidence cannot: which 3D poses are plausible at all. The reprojection term
  keeps the policy anchored to the image; the discriminator is what can move it in depth.

One caveat on the interpolation probe: linearly blending two poses in axis-angle space is
not a valid path between them, which is why the mid-points score *worse* in 2D than either
end. The endpoint comparison (lifted vs GT) is unaffected and is the number that matters.

### Logged to TensorBoard

The components come back through the env's `info` dict, which skrl's gymnasium wrapper
passes untouched, and go through `track_data` so skrl averages them and writes at
`write_interval` rather than once per step.

```
Reward / reprojection                          the r_reproj term
Reward / smoothness                            the r_smooth term
Reprojection / error corrected (px)            interpretable units, not reward units
Reprojection / error lifted (px)               the same metric on the untouched lifted frame
Reprojection / improvement over lifted (px)    >0 means the policy is actually helping
Reprojection / frames with 2D evidence         detection coverage
Reprojection / joints scored                   how many of the 12 passed the confidence gate
```

Frames with no 2D evidence carry `nan`, and those are skipped rather than tracked — a single
`nan` would poison the mean and the curve would silently disappear from TensorBoard.

**"Improvement over lifted" is the one to watch.** Given that the 2D evidence barely
separates the lifted pose from GT, a rising reward curve on its own proves very little; the
policy can bank a high reward that the lifted pose had already earned. Scoring the untouched
lifted frame alongside costs a second forward pass, so it runs every `--baseline_every`
steps (default 10, ~10% overhead; 0 disables).

### The default policy init cannot train against this reward

Logging the components made this visible immediately. `SkrlPoseActor` initialises
`log_std = 0`, i.e. an action standard deviation of **1.0 in normalised pose units** — a
full standard deviation of the pose distribution, per joint, per frame. Measured:

| action std | err_px | r_reproj | r_smooth | combined |
|---|---|---|---|---|
| 0.00 (identity) | 15.5 | 0.584 | 0.936 | **0.678** |
| 0.05 | 15.7 | 0.578 | 0.927 | 0.671 |
| 0.20 | 18.2 | 0.479 | 0.799 | 0.559 |
| **1.00 (current default)** | **49.8** | **0.034** | **0.016** | **0.036** |

At std 1.0 both terms are saturated near zero, where `exp(-x²)` is flat — so PPO starts with
almost no gradient and the first thing it must learn is to undo its own initialisation. The
identity correction already scores 0.678.

Fix before any (B) run: initialise `log_std` near `log(0.05) ≈ -3` and zero-init the actor's
final layer so the mean correction starts at identity. Both are standard for residual
policies and neither is a change to the reward.

### What is still open for the experiments

- **(C) GAIL/AMP is ready** — GT expert demonstrations are in `gt/`, and AMP state
  transitions come from the sequences. Note `poses_vel` / `trans_vel` exist in the raw
  lifted h5 but `norm_upsample` drops them; they are derivable by finite differences.
- Do not build reprojection targets by projecting GT 3D joints. It is tempting now that
  intrinsics are available, but it turns the reward into a supervised GT loss in disguise —
  the same leakage trap as the fitted offsets in §3.

## 7. TensorBoard pose visualisation

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

## 8. Open items

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

# 2D evidence + reprojection targets (§5)
python scripts/extract_2d.py --out data/keypoints2d.h5        # done; resumable, re-runs as a no-op
python scripts/build_reproj_targets.py                        # -> data/reproj_targets.h5
python -m pytest tests/test_reproject.py tests/test_rewards.py -o addopts=""

# training with the GT-free reward (§6) — needs the smplerx env for smplx
python -m src.train --h5_path data/processed_movi.h5 --reward_mode reproj \
    --reproj_path data/reproj_targets.h5 --reproj_sigma 0.04

# PA-MPJPE (§4) — needs the smplerx env for smplx + torch
python -m src.evaluate --processed_h5 data/processed_movi.h5 \
    --lifted_h5 lifted_movi_part1_upd2.h5 --device cpu
```

Note `data/MoVi_SMPLX/` (9.8 GB of extracted npz) was deleted after the rebuild to reclaim
disk. Re-extract from `BMLmovi.tar.bz2` if `movi_smplx.h5` ever needs rebuilding.
