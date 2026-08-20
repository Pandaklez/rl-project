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
                           place_in_camera, project, real_intrinsics)

REPO = Path(__file__).resolve().parent.parent

# Depth below this is treated as the policy having pushed the body behind the
# camera; the frame is scored as a miss rather than producing a wild projection.
MIN_DEPTH_VIRTUAL = 0.5

# Metric depth floor, in metres, for the reparameterised translation action. The
# subjects are 3-6 m from the cameras, so anything under half a metre is the
# policy having pushed the body into the lens.
MIN_DEPTH_METRIC = 0.5

# Number of COCO joints that survive the SMPL-X correspondence (see COCO17_TO_SMPLX).
N_KP = len(COCO_IDX)

# Per-joint 2D residuals go into the *observation*, so one badly detected frame
# must not be allowed to dominate the running state normaliser. Half a bbox
# height is already far outside anything a plausible pose produces — the
# operating point is 0.028 — so clipping there loses no usable signal and bounds
# the state.
RESID_CLIP = 0.5

# What a frame scores when the body is pushed behind the camera: no projection
# exists, so there is no residual direction to report, only "this is very wrong".
ERR_NORM_MISS = 2.0 * RESID_CLIP

# Layout of the 2D-evidence block appended to the observation (§4 / fix 06).
# Kept here rather than in models/policy.py because every number in it comes out
# of this module, and a layout that lives apart from the code that fills it is a
# layout that drifts.
#
#   [0:24)   resid    (12, 2)  projected - observed, in bbox-height units
#   [24:36)  conf     (12,)    ViTPose confidence, zeroed below the threshold
#   [36]     err_norm          the scalar the reward exponentiates
#   [37]     valid             1.0 when the frame carried usable 2D evidence
#   [38]     t_frac            t / (T-1), so the critic can see episode progress
#   [39:44)  context           camera one-hot (2) + bbox bearing/size (3)
CONTEXT_DIM  = 5
EVIDENCE_DIM = N_KP * 2 + N_KP + 3 + CONTEXT_DIM        # 24 + 12 + 3 + 5 = 44


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
    def step(self, corrected: dict, t: int, trans_delta=None) -> tuple[float, dict]:
        """
        corrected : {"poses": (52,3), "trans": (3,)} normalised, as the policy
                    emits them (world-frame root)
        t         : frame index into the clip

        returns (reward in (0, 1], info dict)

        The info dict carries the **per-joint residual** alongside the scalar
        error, because the observation needs it (§4: "the policy cannot see what
        it is scored on"). It comes out of the same projection the reward is
        computed from, so feeding the policy its own error costs nothing beyond
        the forward pass already being paid for here.

        `resid` is `(projected - observed) / bbox_height`, i.e. the per-joint
        decomposition of exactly the quantity `err_norm` averages. Joints below
        the confidence threshold are zeroed, matching their zero weight in the
        reward.
        """
        info = empty_info()
        c = self._clip
        if c is None or t >= len(c["valid"]) or not c["valid"][t]:
            return float("nan"), info

        obs = c["kp2d"][t][COCO_IDX]                    # (12, 3) x, y, confidence
        w = obs[:, 2].astype(np.float64)
        w = np.where(w >= self.min_confidence, w, 0.0)
        info["conf"] = w.astype(np.float32)
        if w.sum() <= 0:
            return float("nan"), info

        uv = self._project_frame(corrected, t, trans_delta)
        if uv is None:
            # Body pushed behind the camera — a real miss, not a missing target.
            # There is no projection, so there is no residual *direction* to
            # report; the observation gets a zero vector and a saturated
            # err_norm, which is a state the policy can tell apart from "fits
            # perfectly" (zero residual, zero error).
            info.update(valid=True, err_norm=ERR_NORM_MISS)
            return 0.0, info

        delta = uv - obs[:, :2]                         # (12, 2) signed, pixels
        d = np.linalg.norm(delta, axis=-1)
        err_px = float(np.sum(w * d) / np.sum(w))

        # Scale-normalise so depth does not change how harshly a frame is judged.
        scale = float(c["bbox"][t][3]) or 1.0
        err = err_px / scale
        reward = float(np.exp(-(err * err) / (self.sigma * self.sigma)))

        resid = np.clip(delta / scale, -RESID_CLIP, RESID_CLIP)
        resid = resid * (w[:, None] > 0)                # unscored joints read zero
        info.update(valid=True, err_px=err_px, err_norm=err,
                    n_joints=int((w > 0).sum()),
                    resid=resid.astype(np.float32))
        return reward, info

    def context(self, t: int) -> np.ndarray:
        """
        Camera identity and where in the image the subject sits, as 5 numbers.

        The residual tells the policy *that* a joint is 8 px low. Turning that
        into a joint-angle correction needs the projection Jacobian, which
        depends on the camera and on where in the frame the body is — neither of
        which is recoverable from 318 pose numbers. These are the cheapest
        sufficient statistics for it.

        Lengths are divided by the focal length rather than an image size, so the
        units are bearing angles (radians, small-angle) and no image-dimension
        constant has to be kept in sync with the calibration.
        """
        out = np.zeros(CONTEXT_DIM, dtype=np.float32)
        c = self._clip
        if c is None or t >= len(c["bbox"]):
            return out
        fx, fy, px, py = real_intrinsics(c["K"])
        x, y, w, h = (float(v) for v in c["bbox"][t])
        out[0 if str(c["camera"]).upper() == "PG1" else 1] = 1.0
        out[2] = (x + 0.5 * w - px) / fx
        out[3] = (y + 0.5 * h - py) / fy
        out[4] = h / fy
        return out

    def _project_frame(self, corrected: dict, t: int, trans_delta=None):
        """Corrected normalised frame -> (12, 2) projected keypoints, or None."""
        from src.smplx_fk import joints_from_poses

        c = self._clip
        poses = unscale(_np(corrected["poses"]).reshape(52, 3), "poses", c["stats"])
        # A diverged policy reaches here with inf or nan and `Rotation.from_rotvec`
        # raises "Found zero norm quaternions", killing the run hours in. Report
        # it as a miss and let the caller end the episode instead.
        if not np.isfinite(poses).all():
            return None
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
            trans_metric = np.array(c["trans_metric"][t][None], dtype=np.float64)

        trans_metric = self._apply_trans_delta(trans_metric, trans_delta, t)
        if trans_metric is None:
            return None

        joints = joints_from_poses(poses_cam, c["betas"], gender=self.gender,
                                   device=self.device).numpy()          # (1, 22, 3)
        placed = place_in_camera(joints, trans_metric)
        uv = project(placed, c["K"], c["radial"])                       # (1, 22, 2)
        return uv[0][SMPLX_IDX]

    def _apply_trans_delta(self, trans_metric: np.ndarray, trans_delta, t: int):
        """
        Apply `(du, dv, dlog_tz)` to a metric translation. Returns None if the
        result puts the body implausibly close to the camera.

        The point of this parameterisation is that the two components are
        *separated by how much the reward can see them*:

        * `dlog_tz` scales the whole translation, so the body slides along its
          own viewing ray: `u, v` are unchanged to first order and only apparent
          size moves. This is the direction reprojection is nearly blind in.
        * `du, dv` then shift the body in the image plane. At depth Z a shift of
          one bbox height `h` on the sensor is `h·Z/f` metres, so the conversion
          is exact for the pinhole term — the residual the policy observes and
          the correction it writes are in the same units, one for one.

        Order matters: depth first, then the image shift computed at the new
        depth, so the two are independent rather than the shift being scaled by
        a depth change the policy did not intend.
        """
        if trans_delta is None:
            return trans_metric
        du, dv, dlog = (float(v) for v in np.asarray(trans_delta).reshape(3))
        if du == 0.0 and dv == 0.0 and dlog == 0.0:
            return trans_metric
        if not np.isfinite([du, dv, dlog]).all():
            return None

        out = np.array(trans_metric, dtype=np.float64, copy=True)
        if dlog:
            out *= float(np.exp(dlog))
        z = float(out[0, 2])
        if not z > MIN_DEPTH_METRIC:
            return None

        c = self._clip
        fx, fy, _, _ = real_intrinsics(c["K"])
        h = float(c["bbox"][t][3]) or 1.0
        out[0, 0] += du * h * z / fx
        out[0, 1] += dv * h * z / fy
        return out


def empty_info() -> dict:
    """
    The info dict for a frame with no usable 2D evidence.

    One definition, used by `ReprojectionReward.step` and by callers that have no
    reward bound at all, so "no measurement" is the same shape everywhere and
    `pack_evidence` needs no special case.
    """
    return {"valid": False, "err_px": float("nan"), "err_norm": float("nan"),
            "n_joints": 0,
            "resid": np.zeros((N_KP, 2), dtype=np.float32),
            "conf": np.zeros(N_KP, dtype=np.float32)}


def pack_evidence(info: dict, context: np.ndarray, t_frac: float) -> np.ndarray:
    """
    Assemble the EVIDENCE_DIM block the policy observes, from one `step()` info
    dict plus the clip context.

    This is the whole of fix 06. The observation gains the residual the reward is
    computed from, so the mapping the policy has to learn goes from
    "reduce a 2D error you cannot measure" to "your left knee projects 0.06 bbox
    heights low, move it" — which is close to invertible.

    Note what is *not* here: ground truth. Every number is derived from the
    lifted pose, the ViTPose detections and the calibration, all of which are
    inputs at inference time. The GT-free property of experiment (B) is intact.
    """
    err_norm = info.get("err_norm", float("nan"))
    if err_norm != err_norm:                       # nan -> no measurement
        err_norm = 0.0
    return np.concatenate([
        np.asarray(info["resid"], dtype=np.float32).reshape(-1),
        np.asarray(info["conf"], dtype=np.float32).reshape(-1),
        np.array([float(err_norm),
                  1.0 if info.get("valid") else 0.0,
                  float(t_frac)], dtype=np.float32),
        np.asarray(context, dtype=np.float32).reshape(-1),
    ]).astype(np.float32)


def empty_evidence(t_frac: float = 0.0) -> np.ndarray:
    """
    The evidence block for a frame with no 2D targets at all: no residual, no
    confidence, `valid = 0`.

    Used by the clips carrying no usable sidecar (20 of 3532 cam-clips) and by
    rollout paths that run without a reward attached. Episode progress is still
    filled in, because that is known regardless of whether the detector fired.
    """
    return pack_evidence(empty_info(), np.zeros(CONTEXT_DIM, dtype=np.float32), t_frac)


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
