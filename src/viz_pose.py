"""
viz_pose.py
───────────
TensorBoard 2D skeleton overlays, so you can see whether the policy is making
meaningful changes to the pose rather than only reading scalar curves.

Each logged figure is a grid: one row per validation clip, one column per
sampled frame. Every cell overlays three skeletons —

    lifted (red, dashed)   what the policy is given
    corrected (blue)       what the policy produced
    GT (green)             the target

If the blue skeleton sits on top of the red one, the policy is doing nothing.
If it moves toward the green one, it is learning something useful.

Two loggers, same interface (`__call__` -> Figure, `correction_magnitude` ->
scalars), so `PPOWithPoseViz` does not care which it holds:

`ImagePoseVizLogger` — **the real 2D one, and the default.** Skeletons are
projected through the actual camera (real intrinsics + radial distortion, metric
translation from `data/reproj_targets.h5`) and drawn *on the video frame the
pose came from*, next to the ViTPose detections the reward scores against. This
is the picture that answers "is the policy putting the body on the person".
GT is not drawn here. The original reason — that GT through the PG2 calibration
was 65-75 px off — turned out to be wrong (`scripts/check_extrinsics.py` puts it
at 12.0 px PG1 / 14.5 px PG2), so drawing GT is now a viable option rather than
a misleading one. The ViTPose keypoints remain the reference the reward actually
scores against, which is what this figure is for.

`PoseVizLogger` — the original orthographic world-frame view (x-right, z-up),
kept for runs without reprojection targets. No image, no camera, so it shows
pose change but not image alignment.

Clips and frame indices are fixed at construction, so successive figures are
directly comparable across epochs. The rollout uses the policy *mean*, not a
sample, so differences between figures are learning and not exploration noise.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — must be set before pyplot

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.camera_frame import uncorrect_root
from src.env import rollout_policy
from src.reproject import COCO_IDX, place_in_camera, project
from src.rewards import load_lifted_stats, unscale
from src.smplx_fk import joints_from_poses

# SMPL-X body joints 0-21
SKELETON = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12), (12, 15), (12, 13), (12, 14),
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

# Half-width of the fixed view window, in metres (a person is ~1.6 m tall).
HALF_RANGE = 1.1

STYLES = {                      # name -> (colour, linestyle, z-order)
    "lifted":    ("#d62728", "--", 1),
    "corrected": ("#1f77b4", "-",  3),
    "gt":        ("#2ca02c", "-",  2),
}


def unnormalize(arr: torch.Tensor, key: str, stats: dict) -> torch.Tensor:
    mu = torch.as_tensor(np.array(stats[key]["mu"]), dtype=torch.float32)
    sigma = torch.as_tensor(np.array(stats[key]["sigma"]), dtype=torch.float32)
    sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
    return arr * sigma + mu


def _unscale_seq(seq: torch.Tensor, stats) -> torch.Tensor:
    """Unnormalise a (T, 22, 3) pose sequence with per-camera lifted stats."""
    # The stats are stored per joint as (22, 3), which broadcasts over T as-is.
    return torch.from_numpy(unscale(seq.numpy(), "poses", stats))


def _draw(ax, joints2d, style):
    colour, ls, z = style
    for a, b in SKELETON:
        ax.plot(joints2d[[a, b], 0], joints2d[[a, b], 1],
                color=colour, linestyle=ls, linewidth=1.4, zorder=z, alpha=0.9)
    ax.scatter(joints2d[:, 0], joints2d[:, 1], s=5, color=colour, zorder=z + 3)


class PoseVizLogger:
    """
    Builds the comparison figure. Call it to get a matplotlib Figure.

    The rollout is deterministic — the policy mean is used, not a sample — so
    frame-to-frame differences between epochs reflect learning, not exploration
    noise.
    """

    def __init__(self, dataset, actor, norm_stats, device="cpu",
                 n_clips=3, n_frames=4, max_steps=120, seed=0):
        self.dataset = dataset
        self.actor = actor
        self.norm_stats = norm_stats
        self.device = torch.device(device)
        self.n_frames = n_frames
        self.max_steps = max_steps
        rng = np.random.default_rng(seed)
        n = min(n_clips, len(dataset))
        self.sample_ids = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())

    @torch.no_grad()
    def _rollout(self, idx):
        """Return (lifted, corrected, gt) unnormalized pose sequences, (T, 22, 3)."""
        sample = self.dataset[idx]
        x, y = sample["x"], sample["y"]
        keys = ("poses", "trans")
        steps = min(self.max_steps, y["poses"].shape[0]) - 1
        if steps < 1:
            return None

        corr = rollout_policy(sample, self.actor, keys=keys,
                              device=str(self.device), max_steps=self.max_steps)
        lift = [x["poses"][t].cpu() for t in range(steps)]
        gt   = [y["poses"][t].cpu() for t in range(steps)]

        # The lifted cameras and the GT are normalised with *different* stats
        # (data/norm_upsample.py: normalization_lifted_pg{1,2}.json vs
        # data/normalization.json). Unnormalising the policy's output with the
        # GT stats produces a plausible-looking pose that is simply the wrong
        # one — measured at 31 deg mean per-joint error, 86 deg worst.
        lifted_stats = load_lifted_stats(sample["meta"]["camera"])
        out = [_unscale_seq(torch.stack(lift), lifted_stats),
               _unscale_seq(torch.stack(corr), lifted_stats),
               unnormalize(torch.stack(gt), "poses", self.norm_stats)]
        return out

    @torch.no_grad()
    def __call__(self):
        was_training = self.actor.training
        self.actor.eval()
        try:
            rows = []
            for idx in self.sample_ids:
                r = self._rollout(idx)
                if r is not None:
                    rows.append(r)
            if not rows:
                return None

            n_rows, n_cols = len(rows), self.n_frames
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(2.6 * n_cols, 3.0 * n_rows), squeeze=False)

            for r, (lift, corr, gt) in enumerate(rows):
                T = lift.shape[0]
                frames = np.linspace(0, T - 1, n_cols).round().astype(int)
                joints = {
                    name: joints_from_poses(seq[frames], device=str(self.device)).numpy()
                    for name, seq in (("lifted", lift), ("corrected", corr), ("gt", gt))
                }
                for c in range(n_cols):
                    ax = axes[r][c]
                    # AMASS world frame is z-up (verified: head sits ~0.62 m above
                    # the pelvis in z, and the z spread matches human height),
                    # so the frontal view is x-right / z-up.
                    for name in ("lifted", "gt", "corrected"):
                        j = joints[name][c]
                        _draw(ax, np.stack([j[:, 0], j[:, 2]], axis=1), STYLES[name])
                    # Fixed window centred on the GT pelvis keeps every panel the
                    # same scale, so motion between epochs is comparable by eye.
                    px, pz = joints["gt"][c][0, 0], joints["gt"][c][0, 2]
                    ax.set_xlim(px - HALF_RANGE, px + HALF_RANGE)
                    ax.set_ylim(pz - HALF_RANGE, pz + HALF_RANGE)
                    ax.set_aspect("equal")
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_alpha(0.2)
                    if r == 0:
                        ax.set_title(f"t={frames[c]}", fontsize=8)
                    if c == 0:
                        ax.set_ylabel(f"clip {self.sample_ids[r]}", fontsize=8)

            handles = [plt.Line2D([], [], color=c, linestyle=ls, label=n)
                       for n, (c, ls, _) in STYLES.items()]
            fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            return fig
        finally:
            if was_training:
                self.actor.train()

    @torch.no_grad()
    def correction_magnitude(self):
        """
        Mean |corrected - lifted| and |corrected - gt| in radians, as scalars.
        Cheap companion to the figure: says whether the policy moves the pose at
        all, and whether the movement is toward GT.
        """
        was_training = self.actor.training
        self.actor.eval()
        try:
            d_lift, d_gt, d_base = [], [], []
            for idx in self.sample_ids:
                r = self._rollout(idx)
                if r is None:
                    continue
                lift, corr, gt = r
                d_lift.append((corr - lift).abs().mean().item())
                d_gt.append((corr - gt).abs().mean().item())
                d_base.append((lift - gt).abs().mean().item())
            if not d_lift:
                return {}
            return {
                "pose/delta_from_lifted": float(np.mean(d_lift)),
                "pose/err_corrected_vs_gt": float(np.mean(d_gt)),
                "pose/err_lifted_vs_gt": float(np.mean(d_base)),
            }
        finally:
            if was_training:
                self.actor.train()


class SupervisedPoseVizLogger:
    """
    `PoseVizLogger`'s sibling for `src.models.supervised.SupervisedPoseRegressor`
    (experiment (E)) — same three-skeleton grid, same `__call__` /
    `correction_magnitude` interface, so `src.train_supervised` can log it the
    same way `PPOWithPoseViz` logs `PoseVizLogger`.

    The reason this cannot just *be* `PoseVizLogger` with a different actor:
    `PoseVizLogger._rollout` goes through `src.env.rollout_policy`, which steps
    an env and expects a `PoseActor`-shaped policy (`.act()`/`.distribution()`,
    `corrected_{t-1}` fed back as part of the observation). The supervised
    regressor is a plain stateless `forward(lifted_pose) -> corrected_pose` —
    no env, no action, no recurrence — so there is nothing for `rollout_policy`
    to step through. `_rollout` here is the direct replacement: call the model
    once on every frame of the clip.
    """

    def __init__(self, dataset, model, norm_stats, device="cpu",
                 n_clips=3, n_frames=4, max_steps=120, seed=0):
        self.dataset = dataset
        self.model = model
        self.norm_stats = norm_stats
        self.device = torch.device(device)
        self.n_frames = n_frames
        self.max_steps = max_steps
        rng = np.random.default_rng(seed)
        n = min(n_clips, len(dataset))
        self.sample_ids = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())

    @torch.no_grad()
    def _rollout(self, idx):
        """Return (lifted, corrected, gt) unnormalized pose sequences, (T, 22, 3)."""
        sample = self.dataset[idx]
        x, y = sample["x"], sample["y"]
        T = min(self.max_steps, y["poses"].shape[0])
        if T < 1:
            return None

        lifted_n = x["poses"][:T].to(self.device)     # (T, 22, 3), normalised
        corr_n   = self.model(lifted_n).cpu()          # (T, 22, 3), normalised
        lift_n   = lifted_n.cpu()
        gt_n     = y["poses"][:T].cpu()

        # Same unnormalisation split as PoseVizLogger, for the same reason: the
        # lifted cameras and GT are normalised with different stats, and the
        # model's output lives in the *lifted* space (see
        # `SupervisedPoseRegressor`'s docstring), so it is unnormalised with the
        # lifted stats, not the GT ones.
        lifted_stats = load_lifted_stats(sample["meta"]["camera"])
        out = [_unscale_seq(lift_n, lifted_stats),
               _unscale_seq(corr_n, lifted_stats),
               unnormalize(gt_n, "poses", self.norm_stats)]
        return out

    @torch.no_grad()
    def __call__(self):
        was_training = self.model.training
        self.model.eval()
        try:
            rows = []
            for idx in self.sample_ids:
                r = self._rollout(idx)
                if r is not None:
                    rows.append(r)
            if not rows:
                return None

            n_rows, n_cols = len(rows), self.n_frames
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(2.6 * n_cols, 3.0 * n_rows), squeeze=False)

            for r, (lift, corr, gt) in enumerate(rows):
                T = lift.shape[0]
                frames = np.linspace(0, T - 1, n_cols).round().astype(int)
                joints = {
                    name: joints_from_poses(seq[frames], device=str(self.device)).numpy()
                    for name, seq in (("lifted", lift), ("corrected", corr), ("gt", gt))
                }
                for c in range(n_cols):
                    ax = axes[r][c]
                    for name in ("lifted", "gt", "corrected"):
                        j = joints[name][c]
                        _draw(ax, np.stack([j[:, 0], j[:, 2]], axis=1), STYLES[name])
                    px, pz = joints["gt"][c][0, 0], joints["gt"][c][0, 2]
                    ax.set_xlim(px - HALF_RANGE, px + HALF_RANGE)
                    ax.set_ylim(pz - HALF_RANGE, pz + HALF_RANGE)
                    ax.set_aspect("equal")
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_alpha(0.2)
                    if r == 0:
                        ax.set_title(f"t={frames[c]}", fontsize=8)
                    if c == 0:
                        ax.set_ylabel(f"clip {self.sample_ids[r]}", fontsize=8)

            handles = [plt.Line2D([], [], color=c, linestyle=ls, label=n)
                       for n, (c, ls, _) in STYLES.items()]
            fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            return fig
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def correction_magnitude(self):
        """Mean |corrected - lifted| and |corrected - gt| in radians — see
        `PoseVizLogger.correction_magnitude`, identical definition."""
        was_training = self.model.training
        self.model.eval()
        try:
            d_lift, d_gt, d_base = [], [], []
            for idx in self.sample_ids:
                r = self._rollout(idx)
                if r is None:
                    continue
                lift, corr, gt = r
                d_lift.append((corr - lift).abs().mean().item())
                d_gt.append((corr - gt).abs().mean().item())
                d_base.append((lift - gt).abs().mean().item())
            if not d_lift:
                return {}
            return {
                "pose/delta_from_lifted": float(np.mean(d_lift)),
                "pose/err_corrected_vs_gt": float(np.mean(d_gt)),
                "pose/err_lifted_vs_gt": float(np.mean(d_base)),
            }
        finally:
            if was_training:
                self.model.train()


# ─── Image-plane visualisation ────────────────────────────────────────────────

# Confidence below this and a ViTPose keypoint is not drawn — matches the gate
# the reward applies, so the figure shows the same evidence the reward scored.
KP_MIN_CONFIDENCE = 0.3

# Fraction of bbox size added around the crop, so limbs that leave the box are
# still visible rather than clipped at the edge.
CROP_MARGIN = 0.25

IMAGE_STYLES = {                 # name -> (colour, linestyle, z-order)
    "lifted":    ("#ff4d4d", "--", 2),
    "corrected": ("#4da6ff", "-",  4),
}


def _video_path(video_root: Path, camera: str, clip: str) -> Path | None:
    """`Subject_13__checking_watch` + `pg1` -> demo/videos/PG1_avi/F_PG1_Subject_13_L.avi"""
    subject = clip.split("__")[0]                     # Subject_13
    cam = camera.upper()                              # PG1
    p = video_root / f"{cam}_avi" / f"F_{cam}_{subject}_L.avi"
    return p if p.exists() else None


class ImagePoseVizLogger:
    """
    Skeletons projected through the real camera, drawn on the video frame.

    Each cell is one frame of one held-out validation clip:

        the video frame itself, cropped to the subject
        lifted    (red, dashed)   what SMPLer-X produced
        corrected (blue)          what the policy produced
        ViTPose   (green dots)    the 2D evidence the reward scores against

    Blue sitting on red means the policy is doing nothing; blue moving onto the
    green dots means it is improving the image fit. Because this is the same
    projection path the reward uses (`uncorrect_root` -> FK -> `place_in_camera`
    -> `project`), what you see is what is being optimised.
    """

    def __init__(self, dataset, actor, gt_stats, calib, reproj_path,
                 video_root="demo/videos", device="cpu", n_clips=3, n_frames=4,
                 max_steps=120, seed=0, reproj_reward=None, use_evidence=False,
                 trans_mode="none", state_trans=True):
        import h5py

        self.dataset = dataset
        self.actor = actor
        self.gt_stats = gt_stats
        self.calib = calib
        self.device = torch.device(device)
        self.n_frames = n_frames
        self.max_steps = max_steps
        # The policy's observation contains the reward's 2D residual, so the
        # logger needs its own reward instance to reproduce it. Its `reset` binds
        # to one clip, which is why this cannot be the trainer's — that one is
        # bound to whichever clip the rollout is on.
        self.reproj_reward = reproj_reward
        self.use_evidence = bool(use_evidence) and reproj_reward is not None
        self.trans_mode = trans_mode
        self.state_trans = bool(state_trans)
        video_root = Path(video_root)

        # Pick clips once, at construction, so every figure shows the same ones.
        # A clip qualifies only if it has aligned targets and a video on disk —
        # the 20 unaligned cam-clips and Subject_6/pg1 (no video) are skipped.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(dataset)).tolist()
        self.clips = []
        rejected = {"no_targets": 0, "unaligned": 0, "no_video": 0}
        with h5py.File(reproj_path, "r") as rf:
            for idx in order:
                if len(self.clips) >= n_clips:
                    break
                clip, camera = dataset.samples[idx]
                grp = rf.get(f"{dataset.split}/{clip}/{camera}")
                if grp is None:
                    rejected["no_targets"] += 1
                    continue
                if not bool(grp.attrs.get("aligned", False)):
                    rejected["unaligned"] += 1
                    continue
                path = _video_path(video_root, camera, clip)
                if path is None:
                    rejected["no_video"] += 1
                    continue
                self.clips.append({
                    "idx": idx, "clip": clip, "camera": camera, "video": path,
                    "start": int(grp.attrs["start"]), "t0": int(grp.attrs["t0"]),
                })

        # Refuse to construct rather than return None from every call. A logger
        # with no clips produces no figures for the whole run and says nothing
        # about why — the failure would only surface as an empty TensorBoard
        # image tab hours later. Every cause here is a misconfiguration
        # (wrong video_root, wrong reproj_path, wrong split) worth failing on.
        if not self.clips:
            raise RuntimeError(
                f"ImagePoseVizLogger found no usable clips in split "
                f"'{dataset.split}' among {len(order)} candidates "
                f"(no targets: {rejected['no_targets']}, unaligned: "
                f"{rejected['unaligned']}, no video file: {rejected['no_video']}). "
                f"Checked videos under {video_root!r} and targets in "
                f"{reproj_path!r}.")

    # ── projection ───────────────────────────────────────────────────────────
    def _project(self, poses_norm, betas, stats, camera, trans_metric):
        """One normalised frame -> (22, 2) image-plane joints."""
        poses = unscale(poses_norm.reshape(-1, 3), "poses", stats)
        poses_cam = uncorrect_root(poses[None], camera)
        joints = joints_from_poses(poses_cam, betas, device=str(self.device)).numpy()
        placed = place_in_camera(joints, trans_metric[None])
        return project(placed, self.calib[camera]["IntrinsicMatrix"],
                       self.calib[camera]["RadialDistortion"])[0]

    @torch.no_grad()
    def _rollout(self, entry):
        """Roll the policy out and return everything needed to draw one row."""
        sample = self.dataset[entry["idx"]]
        x, reproj = sample["x"], sample.get("reproj")
        if reproj is None:
            return None
        valid = reproj["valid"].cpu().numpy().astype(bool)
        T = sample["y"]["poses"].shape[0]
        steps = min(self.max_steps, T) - 1
        if steps < 1 or not valid[:steps].any():
            return None

        keys = ("poses", "trans")
        # Same observation the trainer builds, including the 2D evidence block —
        # a figure drawn from a differently-shaped observation would be measuring
        # a policy that never existed.
        corr = [c.numpy() for c in rollout_policy(
            sample, self.actor, keys=keys, device=str(self.device),
            reproj_reward=self.reproj_reward, use_evidence=self.use_evidence,
            max_steps=self.max_steps, trans_mode=self.trans_mode,
            state_trans=self.state_trans)]

        # Sample frames evenly, but only from frames that carry 2D evidence —
        # a panel with no detection would have nothing to compare against.
        usable = np.flatnonzero(valid[:steps])
        picks = usable[np.linspace(0, len(usable) - 1, self.n_frames).round().astype(int)]

        stats = load_lifted_stats(entry["camera"])
        betas = unscale(x["betas"].cpu().numpy(), "betas", stats)
        kp2d = reproj["kp2d"].cpu().numpy()
        bbox = reproj["bbox"].cpu().numpy()
        tmet = reproj["trans_metric"].cpu().numpy()

        # Pose index t (120 Hz GT timeline) -> target index j (native 30 Hz) ->
        # video frame. This inverts the resampling norm_upsample.py applied.
        panels = []
        for t in picks:
            j = int(round(t * (entry["t0"] - 1) / max(T - 1, 1)))
            panels.append({
                "t": int(t),
                "frame": entry["start"] + j,
                "bbox": bbox[t],
                "kp2d": kp2d[t],
                "lifted": self._project(x["poses"][t].cpu().numpy(), betas, stats,
                                        entry["camera"], tmet[t]),
                "corrected": self._project(corr[t], betas, stats,
                                           entry["camera"], tmet[t]),
            })
        return panels

    def _read_frames(self, entry, panels):
        """Decode the needed video frames in one pass, ascending."""
        import cv2

        cap = cv2.VideoCapture(str(entry["video"]))
        try:
            for p in sorted(panels, key=lambda q: q["frame"]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
                ok, img = cap.read()
                p["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None
        finally:
            cap.release()

    # ── figure ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def __call__(self):
        was_training = self.actor.training
        self.actor.eval()
        try:
            rows = []
            for entry in self.clips:
                panels = self._rollout(entry)
                if panels:
                    self._read_frames(entry, panels)
                    rows.append((entry, panels))
            if not rows:
                return None

            n_rows, n_cols = len(rows), self.n_frames
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)

            for r, (entry, panels) in enumerate(rows):
                for c, p in enumerate(panels):
                    ax = axes[r][c]
                    x0, y0, w, h = p["bbox"]
                    mx, my = CROP_MARGIN * w, CROP_MARGIN * h
                    left, right = x0 - mx, x0 + w + mx
                    top, bottom = y0 - my, y0 + h + my

                    if p["image"] is not None:
                        ax.imshow(p["image"])
                        # Clamp to the frame, otherwise a bbox near an edge pads
                        # the panel with blank canvas instead of picture.
                        ih, iw = p["image"].shape[:2]
                        left, right = max(left, 0), min(right, iw - 1)
                        top, bottom = max(top, 0), min(bottom, ih - 1)
                    else:
                        ax.set_facecolor("#111111")

                    for name in ("lifted", "corrected"):
                        _draw(ax, p[name], IMAGE_STYLES[name])

                    kp = p["kp2d"][COCO_IDX]
                    vis = kp[:, 2] >= KP_MIN_CONFIDENCE
                    ax.scatter(kp[vis, 0], kp[vis, 1], s=22, marker="o",
                               facecolors="none", edgecolors="#2ca02c",
                               linewidths=1.4, zorder=6)

                    # Image coordinates: y grows downward, so the limits invert.
                    ax.set_xlim(left, right)
                    ax.set_ylim(bottom, top)
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0:
                        ax.set_title(f"frame {p['frame']}", fontsize=8)
                    if c == 0:
                        ax.set_ylabel(f"{entry['clip']}\n{entry['camera']}", fontsize=7)

            handles = [plt.Line2D([], [], color=IMAGE_STYLES["lifted"][0],
                                  linestyle="--", label="lifted"),
                       plt.Line2D([], [], color=IMAGE_STYLES["corrected"][0],
                                  linestyle="-", label="corrected"),
                       plt.Line2D([], [], color="#2ca02c", linestyle="none",
                                  marker="o", markerfacecolor="none", label="ViTPose 2D")]
            fig.legend(handles=handles, loc="lower center", ncol=3,
                       frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            return fig
        finally:
            if was_training:
                self.actor.train()

    @torch.no_grad()
    def correction_magnitude(self):
        """
        Reprojection error in pixels, before and after the policy — the image-plane
        counterpart of the orthographic logger's radian scalars.
        """
        was_training = self.actor.training
        self.actor.eval()
        try:
            e_lift, e_corr = [], []
            for entry in self.clips:
                panels = self._rollout(entry)
                if not panels:
                    continue
                for p in panels:
                    kp = p["kp2d"][COCO_IDX]
                    w = np.where(kp[:, 2] >= KP_MIN_CONFIDENCE, kp[:, 2], 0.0)
                    if w.sum() <= 0:
                        continue
                    from src.reproject import SMPLX_IDX
                    for name, acc in (("lifted", e_lift), ("corrected", e_corr)):
                        d = np.linalg.norm(p[name][SMPLX_IDX] - kp[:, :2], axis=-1)
                        acc.append(float(np.sum(w * d) / np.sum(w)))
            if not e_lift:
                return {}
            return {
                "pose/img_err_lifted_px": float(np.mean(e_lift)),
                "pose/img_err_corrected_px": float(np.mean(e_corr)),
                "pose/img_improvement_px": float(np.mean(e_lift) - np.mean(e_corr)),
            }
        finally:
            if was_training:
                self.actor.train()
