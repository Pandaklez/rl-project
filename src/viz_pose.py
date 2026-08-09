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

Projection: joints come from a real SMPL-X forward pass (src/smplx_fk.py), then
are projected orthographically along the camera axis. True perspective
reprojection needs metric translation, which is not recoverable until the
per-frame bboxes are re-extracted (scripts/extract_2d.py). Orthographic is
enough to see whether corrections are meaningful, and swapping in a real
projection later does not change this file's interface.

Clips and frame indices are fixed at construction, so successive figures are
directly comparable across epochs.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # headless — must be set before pyplot

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env import MoviEnv
from src.models.policy import flatten_state, unflatten_action
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
        """Return (lifted, corrected, gt) unnormalized pose sequences, (T, 52, 3)."""
        sample = self.dataset[idx]
        x, y = sample["x"], sample["y"]
        keys = ("poses", "trans")
        steps = min(self.max_steps, y["poses"].shape[0]) - 1
        if steps < 1:
            return None

        env = MoviEnv(device=str(self.device))
        state = env.reset({k: x[k][0] for k in keys})

        corr, lift, gt = [], [], []
        for t in range(steps):
            flat = flatten_state(state).to(self.device)
            # policy mean, not a sample: deterministic across epochs
            action = self.actor.net(flat.unsqueeze(0)).squeeze(0).cpu()
            state = env.step(unflatten_action(action), {k: x[k][t + 1] for k in keys})
            corr.append(state["corrected_state"]["poses"].cpu())
            lift.append(x["poses"][t].cpu())
            gt.append(y["poses"][t].cpu())

        out = []
        for seq in (lift, corr, gt):
            out.append(unnormalize(torch.stack(seq), "poses", self.norm_stats))
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
