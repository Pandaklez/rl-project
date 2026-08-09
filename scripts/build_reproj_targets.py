"""
build_reproj_targets.py
───────────────────────
Combine the recovered 2D evidence (`data/keypoints2d.h5`, from
`scripts/extract_2d.py`) with the raw lifted `transl` to produce the targets a
reprojection reward needs, as a sidecar HDF5.

Two conversions happen here:

  * `src/reproject.metric_translation` turns SMPLer-X's virtual-camera depth
    into metres in the real camera (see that module for the derivation).
  * The per-frame bbox and 2D keypoints are carried through unchanged, indexed
    onto each clip's own timeline.

Neither uses ground truth.

**Frame alignment** is the delicate part. `add_pg{1,2}_to_h5.py:82-103` builds a
lifted clip by walking `range(start, end)` over the v3d `flags30` range and
*silently skipping* frames the detector missed, without recording which. So the
mapping `lifted index j -> video frame start + j` is exact only when no frame
was dropped, i.e. when `t0 == end - start`. That holds for 3514 of 3532
cam-clips (99.5%); the other 18 are written with `aligned = False` and no
targets, because their true alignment is unrecoverable. Reconstructing the
dropped frames from the re-run detector was tried and rejected — it disagrees
with the original on 21 cam-clips, so it is not a reliable witness.

Targets are stored at the **native 30 Hz** clip length `t0`, not at the 120 Hz
GT length. Upsampling 17 keypoints to 120 Hz would quadruple the file for no
information; `src/data/datasets.py` resamples on read using the same rule
`data/norm_upsample.py:47-52` applied to the poses.

Output:
    data/reproj_targets.h5
    └── <split>/<clip>/<cam>/
        ├── trans_metric (t0, 3)     float32  metres, real camera frame
        ├── kp2d         (t0, 17, 3) float32  x, y in original image px + conf
        ├── bbox         (t0, 4)     float32  xywh after process_bbox
        ├── valid        (t0,)       bool     per-frame usability
        └── attrs: start, end, t0, n_frames_gt, aligned, detected

Usage:
    python scripts/build_reproj_targets.py
    python scripts/build_reproj_targets.py --splits test --limit 20
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.reproject import COCO17_NAMES, metric_translation  # noqa: E402

CAMERAS = (("pg1", "PG1"), ("pg2", "PG2"))


@lru_cache(maxsize=None)
def action_ranges(subject: str, v3d_dir: str):
    """{action_name: (start, end)} from the v3d flags30 ranges."""
    import scipy.io as sio

    mat = sio.loadmat(str(Path(v3d_dir) / f"F_v3d_{subject}.mat"),
                      struct_as_record=False, squeeze_me=False)
    top = next(k for k in mat if not k.startswith("__"))
    s = mat[top]
    while isinstance(s, np.ndarray) and s.size == 1:
        s = s.flat[0]
    move = s.move
    while isinstance(move, np.ndarray) and move.size == 1:
        move = move.flat[0]
    names = [str(a[0]).lstrip("['").rstrip("']").replace("/", "_").replace(" ", "_")
             for a in move.motions_list]
    inds = [(int(t[0]), int(t[1])) for t in move.flags30]
    return dict(zip(names, inds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_h5", default="data/processed_movi.h5")
    ap.add_argument("--lifted_h5", default="lifted_movi_part1_upd2.h5")
    ap.add_argument("--kp2d_h5", default="data/keypoints2d.h5")
    ap.add_argument("--v3d_dir", default="F_Subjects_meta")
    ap.add_argument("--out", default="data/reproj_targets.h5")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="clips per split, for smoke tests")
    args = ap.parse_args()

    stats = dict(written=0, no_cam=0, no_video=0, unaligned=0, no_meta=0, len_mismatch=0)

    with h5py.File(args.processed_h5, "r") as proc, \
            h5py.File(args.lifted_h5, "r") as lif, \
            h5py.File(args.kp2d_h5, "r") as kp, \
            h5py.File(args.out, "w") as out:

        calib = {c: {k: proc["calib"][c][k][:] for k in proc["calib"][c]}
                 for c, _ in CAMERAS}

        out.attrs["description"] = ("reprojection targets: metric translation + ViTPose "
                                    "COCO-17 2D, at the native 30 Hz clip length")
        out.attrs["fps"] = 30
        out.attrs["coco17_names"] = list(COCO17_NAMES)
        out.attrs["note"] = ("trans_metric is metres in the REAL camera frame, recovered "
                             "from the virtual-camera transl via the bbox (src/reproject.py). "
                             "kp2d is raw detected pixels, still radially distorted — project "
                             "with src.reproject.project(..., radial=...) to compare. "
                             "Index j here is video frame start + j.")

        for split in args.splits:
            g_split = out.require_group(split)
            clips = list(proc[split].keys())
            if args.limit:
                clips = clips[:args.limit]

            for clip in clips:
                m = re.match(r"(Subject_\d+)__(.*)", clip)
                if m is None:
                    stats["no_meta"] += 1
                    continue
                subject, action = m.group(1), m.group(2)
                try:
                    ranges = action_ranges(subject, args.v3d_dir)
                except Exception:
                    stats["no_meta"] += 1
                    continue
                if action not in ranges:
                    stats["no_meta"] += 1
                    continue
                start, end = ranges[action]
                n_gt = int(proc[split][clip].attrs.get(
                    "n_frames", proc[split][clip]["gt"]["poses"].shape[0]))

                for cam_lo, cam_up in CAMERAS:
                    if clip not in lif[split] or cam_up not in lif[split][clip]:
                        stats["no_cam"] += 1
                        continue
                    if subject not in kp[cam_up]:
                        # e.g. Subject_6 has no PG1 video at all
                        stats["no_video"] += 1
                        continue

                    src = lif[split][clip][cam_up]
                    t0 = src["poses"].shape[0]
                    g_cam = g_split.require_group(clip).require_group(cam_lo)
                    g_cam.attrs.update(start=int(start), end=int(end), t0=int(t0),
                                       n_frames_gt=n_gt)

                    if t0 != end - start:
                        # frames were dropped and not recorded — alignment is lost
                        g_cam.attrs["aligned"] = False
                        stats["unaligned"] += 1
                        continue

                    sub = kp[cam_up][subject]
                    bbox = sub["bbox"][start:end].astype(np.float32)
                    kp2d = sub["kp2d"][start:end].astype(np.float32)
                    score = sub["score"][start:end]
                    if len(bbox) != t0:
                        # video shorter than the annotated range
                        g_cam.attrs["aligned"] = False
                        stats["len_mismatch"] += 1
                        continue

                    trans_metric, _, ok = metric_translation(
                        src["trans"][:], bbox, calib[cam_lo]["IntrinsicMatrix"])
                    valid = ok & (score > 0)

                    g_cam.attrs["aligned"] = True
                    g_cam.attrs["detected"] = float(valid.mean())
                    for name, arr in (("trans_metric", trans_metric), ("kp2d", kp2d),
                                      ("bbox", bbox), ("valid", valid)):
                        g_cam.create_dataset(name, data=arr, compression="gzip",
                                             compression_opts=4)
                    stats["written"] += 1

            out.flush()
            print(f"  {split}: {stats['written']} cam-clips written so far", flush=True)

    print("\n".join(f"{k:14s} {v}" for k, v in stats.items()))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
