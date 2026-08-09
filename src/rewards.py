"""
rewards.py
──────────
The reprojection and smoothness rewards for experiment (B), and the AMP-style
combination used by (C).

**Neither term uses ground truth.** That is the point of them. The GT-based
`compute_reward` in `src/env.py` measures the thing the experiments are supposed
to be evaluated on, so training against it makes (B) and (C) supervised
regression wearing an RL costume. These rewards use only image evidence the
lifter itself had access to: the ViTPose 2D keypoints and the per-frame bbox
recovered by `scripts/extract_2d.py`.

The path for one frame:

    corrected pose (normalised, world-frame root, from the policy)
      -> unnormalise with the *lifted* per-camera stats
      -> uncorrect_root: world -> camera frame           (src/camera_frame.py)
      -> SMPL-X forward kinematics                       (src/smplx_fk.py)
      -> place_in_camera at the metric translation       (src/reproject.py)
      -> project with the real intrinsics + distortion
      -> compare against the ViTPose keypoints

Three details that are easy to get wrong, all of them load-bearing:

* **Lifted stats, not GT stats.** `data/norm_upsample.py` normalises the lifted
  cameras with `normalization_lifted_pg{1,2}.json` and GT with
  `data/normalization.json`. Unnormalising a policy output with the GT stats
  yields a plausible-looking pose that is simply wrong.
* **Lifted betas, not GT betas.** Using GT shape would leak, and it is not
  available at inference. The lifted betas score poorly against GT (§7) but they
  are what SMPLer-X regressed `cam_trans` jointly with, so they are the
  self-consistent choice — and the 11.6 px validation was measured with them.
* **Scale normalisation.** Raw pixel error depends on how large the subject
  happens to appear. Errors are divided by the bbox height so a clip filmed
  close up is not scored more harshly than one across the room.

Reward shape is the DeepMimic/AMP convention `exp(-err² / σ²)`, bounded to
(0, 1]. A bounded term keeps the value function well conditioned and stops one
badly-detected frame from dominating a rollout, which an unbounded negative
distance does.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.camera_frame import uncorrect_root
from src.reproject import (COCO_IDX, SMPLX_IDX, T_Z_MAX, metric_translation,
                           place_in_camera, project)

REPO = Path(__file__).resolve().parent.parent

# Depth below this is treated as the policy having pushed the body behind the
# camera; the frame is scored as a miss rather than producing a wild projection.
MIN_DEPTH_VIRTUAL = 0.5


@lru_cache(maxsize=4)
def load_lifted_stats(camera: str, path: str | None = None) -> tuple:
    """
    (mu, sigma) for poses / trans / betas from normalization_lifted_<cam>.json.

    Reads from `data/`, which is where `data/norm_upsample.py` writes the current
    stats. There are older copies of the same filenames at the repo root, left
    from before the root correction — they lack the `_root_corrected` flag and
    unnormalising with them puts the pose ~130 px off in the image. The flag is
    checked below precisely so that mix-up cannot pass silently.

    Returned as a tuple of arrays so the result stays hashable for the cache.
    """
    p = Path(path) if path else REPO / "data" / f"normalization_lifted_{camera.lower()}.json"
    with open(p) as f:
        stats = json.load(f)
    if not stats.get("_root_corrected", False):
        raise ValueError(
            f"{p} predates the root correction (no _root_corrected flag). The "
            f"dataset stores world-frame roots, so these stats do not match it. "
            f"Regenerate with data/norm_upsample.py.")
    out = {}
    for key in ("poses", "trans", "betas"):
        mu = np.asarray(stats[key]["mu"], dtype=np.float32)
        sigma = np.asarray(stats[key]["sigma"], dtype=np.float32)
        out[key] = (mu, np.where(sigma == 0, 1.0, sigma).astype(np.float32))
    return tuple(out.items())


def unscale(value: np.ndarray, key: str, stats: tuple) -> np.ndarray:
    mu, sigma = dict(stats)[key]
    return np.asarray(value, dtype=np.float32) * sigma + mu


class ReprojectionReward:
    """
    Per-frame reprojection reward for one clip.

    Construct once, then call `reset(sample)` at the start of each episode and
    `step(corrected, t)` per frame. Frames with no usable 2D evidence return
    `nan` as the reward and `valid=False` in the info dict, so the caller can
    fall back rather than silently scoring them as zero.

    `sigma` is expressed in *bbox-height units*: 0.04 means a 4%-of-body-height
    reprojection error costs one e-fold. The measured operating point is
    err ~= 0.028 (about 13 px against a ~470 px bbox height).

    The default follows from conditioning rather than taste. `exp(-e²/σ²)` is
    monotone in `e`, so σ does **not** change which of two poses scores higher —
    frame ordering is identical at every σ. What it changes is where the reward
    is steep: d/de is maximised at `e = σ/√2`, so putting the steepest response
    on the operating point gives σ = 0.028·√2 ≈ 0.04. Much smaller and the
    reward saturates near 0, much larger and it saturates near 1; both flatten
    the gradient PPO learns from.
    """

    def __init__(
        self,
        calib: dict,
        sigma: float = 0.04,
        min_confidence: float = 0.3,
        correct_translation: bool = True,
        gender: str = "neutral",
        device: str = "cpu",
        lifted_stats_paths: dict | None = None,
    ):
        self.calib = calib
        self.sigma = float(sigma)
        self.min_confidence = float(min_confidence)
        self.correct_translation = bool(correct_translation)
        self.gender = gender
        self.device = device
        self._paths = lifted_stats_paths or {}
        self._clip = None

    # ── episode setup ────────────────────────────────────────────────────────
    def reset(self, sample: dict) -> bool:
        """
        Bind the reward to one clip. Returns False when the clip carries no
        usable targets (the 20 unaligned cam-clips, or a missing sidecar), in
        which case `step` will report every frame invalid.
        """
        self._clip = None
        reproj = sample.get("reproj")
        meta = sample.get("meta")
        if reproj is None or meta is None:
            return False

        valid = np.asarray(_np(reproj["valid"]), dtype=bool)
        if not valid.any():
            return False

        camera = meta["camera"]
        stats = load_lifted_stats(camera, self._paths.get(camera))
        self._clip = {
            "camera": camera,
            "stats": stats,
            "betas": unscale(_np(sample["x"]["betas"]), "betas", stats),
            "kp2d": _np(reproj["kp2d"]),
            "bbox": _np(reproj["bbox"]),
            "trans_metric": _np(reproj["trans_metric"]),
            "valid": valid,
            "K": self.calib[camera]["IntrinsicMatrix"],
            "radial": self.calib[camera]["RadialDistortion"],
        }
        return True

    # ── per-frame reward ─────────────────────────────────────────────────────
    def step(self, corrected: dict, t: int) -> tuple[float, dict]:
        """
        corrected : {"poses": (52,3), "trans": (3,)} normalised, as the policy
                    emits them (world-frame root)
        t         : frame index into the clip

        returns (reward in (0, 1], info dict)
        """
        info = {"valid": False, "err_px": float("nan"), "err_norm": float("nan"),
                "n_joints": 0}
        c = self._clip
        if c is None or t >= len(c["valid"]) or not c["valid"][t]:
            return float("nan"), info

        obs = c["kp2d"][t][COCO_IDX]                    # (12, 3) x, y, confidence
        w = obs[:, 2].astype(np.float64)
        w = np.where(w >= self.min_confidence, w, 0.0)
        if w.sum() <= 0:
            return float("nan"), info

        uv = self._project_frame(corrected, t)
        if uv is None:
            # body pushed behind the camera — a real miss, not a missing target
            info["valid"] = True
            return 0.0, info

        d = np.linalg.norm(uv - obs[:, :2], axis=-1)
        err_px = float(np.sum(w * d) / np.sum(w))

        # Scale-normalise so depth does not change how harshly a frame is judged.
        scale = float(c["bbox"][t][3]) or 1.0
        err = err_px / scale
        reward = float(np.exp(-(err * err) / (self.sigma * self.sigma)))

        info.update(valid=True, err_px=err_px, err_norm=err,
                    n_joints=int((w > 0).sum()))
        return reward, info

    def _project_frame(self, corrected: dict, t: int):
        """Corrected normalised frame -> (12, 2) projected keypoints, or None."""
        from src.smplx_fk import joints_from_poses

        c = self._clip
        poses = unscale(_np(corrected["poses"]).reshape(52, 3), "poses", c["stats"])
        poses_cam = uncorrect_root(poses[None], c["camera"])           # (1, 52, 3)

        if self.correct_translation:
            transl = unscale(_np(corrected["trans"]).reshape(1, 3), "trans", c["stats"])
            # The policy is free to move the root, but t_z is a sigmoid output in
            # (0, T_Z_MAX) by construction; outside that the conversion is
            # meaningless rather than merely wrong.
            if not (MIN_DEPTH_VIRTUAL < transl[0, 2] < T_Z_MAX):
                return None
            trans_metric, _, ok = metric_translation(
                transl, c["bbox"][t][None], c["K"])
            if not ok[0]:
                return None
        else:
            trans_metric = c["trans_metric"][t][None]

        joints = joints_from_poses(poses_cam, c["betas"], gender=self.gender,
                                   device=self.device).numpy()          # (1, 22, 3)
        placed = place_in_camera(joints, trans_metric)
        uv = project(placed, c["K"], c["radial"])                       # (1, 22, 2)
        return uv[0][SMPLX_IDX]


def smoothness_reward(corrected: np.ndarray, prev: np.ndarray, prev2: np.ndarray,
                      sigma: float = 0.5) -> float:
    """
    Penalise *acceleration* in the corrected pose, not velocity.

    Velocity penalties punish motion itself, so the optimum is a subject who
    stops moving — wrong for a dataset of people walking and crawling.
    Acceleration penalises jitter while leaving smooth motion free, which is what
    "smoothness" is supposed to mean here.

    All three frames are normalised poses, so the units are already comparable
    across joints.
    """
    accel = np.asarray(corrected) - 2.0 * np.asarray(prev) + np.asarray(prev2)
    a = float(np.mean(accel * accel))
    return float(np.exp(-a / (sigma * sigma)))


def combine(reproj: float, smooth: float, w_reproj: float = 1.0,
            w_smooth: float = 0.1, fallback: float = 0.0) -> float:
    """
    Weighted sum, with frames that have no 2D evidence falling back rather than
    being scored zero.

    Scoring an unobserved frame as zero would teach the policy that clips with
    poor detection are intrinsically bad, which is a property of the detector
    and not of the pose.
    """
    if reproj != reproj:                      # nan
        return w_smooth * smooth + w_reproj * fallback
    return w_reproj * reproj + w_smooth * smooth


def _np(x):
    """Accept torch tensors or arrays without importing torch at module level."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_calib(h5_path: str = "data/processed_movi.h5") -> dict:
    """Calibration as embedded in the processed HDF5 by data/norm_upsample.py."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return {cam: {k: f["calib"][cam][k][:] for k in f["calib"][cam]}
                for cam in f["calib"]}
