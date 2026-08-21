"""
PPO + GAIL-style discriminator training using skrl (experiment C).

Structurally this is `src/train.py` plus one addition: a `MotionDiscriminator`
(`src/models/discriminator.py`) trained concurrently on GT motion vs. the
policy's own corrected output, whose score feeds back into the env as an
extra reward term. See `discriminator.py`'s module docstring for why that is
a plain torch net driven by a custom skrl hook rather than skrl's own GAIL
utility, and `GAILRewardProvider` for how a reward that changes every rollout
gets into the env without the env needing to change.

Usage:
    python -m src.gail_train [args]
"""
from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer

from src.data.datasets import MoViDataset
from src.gail_env import GymMoviEnv
from src.models.discriminator import (GAILRewardProvider, MotionDiscriminator,
                                      discriminator_step, load_gt_transitions)
from src.models.policy import SkrlPoseActor, SkrlPoseCritic, extract_corrected_pose
from src.viz_pose import PoseVizLogger


class PPOWithPoseViz(PPO):
    """
    PPO plus a periodic 2D skeleton overlay in TensorBoard, and per-step logging
    of the reward components.

    The figure is expensive relative to a scalar (it rolls the policy out over a
    few validation clips and runs SMPL-X forward), so it is logged every
    `viz_interval` updates rather than every step.

    The reward components come through the env's `info` dict, which skrl's
    gymnasium wrapper passes along untouched. They go through `track_data`, so
    skrl averages them and writes at the configured `write_interval` rather than
    once per step.
    """

    def __init__(self, *args, viz_logger=None, viz_interval: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._viz_logger = viz_logger
        self._viz_interval = viz_interval
        self._update_count = 0

    def record_transition(self, states, actions, rewards, next_states,
                          terminated, truncated, infos, timestep, timesteps) -> None:
        super().record_transition(states, actions, rewards, next_states,
                                  terminated, truncated, infos, timestep, timesteps)
        if isinstance(infos, dict) and infos:
            self._track_reward_components(infos)

    def _track_reward_components(self, info: dict) -> None:
        """
        Break the scalar reward into the parts that explain it.

        `r_reproj` and `err_px` are nan on frames with no 2D evidence — those are
        skipped rather than tracked, since a single nan would poison the mean and
        the curve would silently vanish from TensorBoard.
        """
        def ok(x):
            return isinstance(x, (int, float)) and x == x        # not nan

        if ok(info.get("r_reproj")):
            self.track_data("Reward / reprojection", float(info["r_reproj"]))
        if ok(info.get("r_smooth")):
            self.track_data("Reward / smoothness", float(info["r_smooth"]))
        if ok(info.get("r_gail")):
            self.track_data("Reward / GAIL (discriminator)", float(info["r_gail"]))
        if ok(info.get("err_px")):
            self.track_data("Reprojection / error corrected (px)", float(info["err_px"]))
            self.track_data("Reprojection / joints scored", float(info.get("n_joints", 0)))

        # The comparison that matters: is the policy beating the lifted input, or
        # just banking the reward the lifted pose already earned? (§6 — the 2D
        # evidence barely separates the lifted pose from GT, so a rising reward
        # curve on its own proves very little.)
        if ok(info.get("err_px_lifted")):
            self.track_data("Reprojection / error lifted (px)", float(info["err_px_lifted"]))
            if ok(info.get("err_px")):
                self.track_data("Reprojection / improvement over lifted (px)",
                                float(info["err_px_lifted"]) - float(info["err_px"]))

        # The translation action, so the two trans modes can be compared directly
        # in TensorBoard: du/dv say how far the policy is sliding the body in the
        # image, dlog_tz whether it is drifting in the direction the reward
        # cannot see (it is structurally 0 under --trans_mode uv).
        for key, tag in (("trans_du",    "Translation / du (bbox heights)"),
                         ("trans_dv",    "Translation / dv (bbox heights)"),
                         ("trans_dlogz", "Translation / dlog_tz")):
            if ok(info.get(key)):
                self.track_data(tag, float(info[key]))
                if key != "trans_dlogz":
                    self.track_data(tag.replace(" (bbox", " |abs| (bbox"),
                                    abs(float(info[key])))

        if "valid" in info:
            self.track_data("Reprojection / frames with 2D evidence",
                            1.0 if info["valid"] else 0.0)

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        # Mirror PPO's own trigger so we know an update is about to happen.
        updating = (not (self._rollout + 1) % self._rollouts
                    and timestep >= self._learning_starts)
        super().post_interaction(timestep, timesteps)

        if not (updating and self._viz_logger):
            return
        self._update_count += 1
        if self._update_count % self._viz_interval:
            return

        for tag, value in self._viz_logger.correction_magnitude().items():
            self.writer.add_scalar(tag, value, timestep)
        fig = self._viz_logger()
        if fig is not None:
            self.writer.add_figure("pose/lifted_vs_corrected_vs_gt", fig, timestep)
            import matplotlib.pyplot as plt
            plt.close(fig)


class PPOWithGAIL(PPOWithPoseViz):
    """
    PPOWithPoseViz plus a discriminator trained on skrl's own rollout buffer.

    The discriminator update runs once per rollout, on exactly the batch PPO
    is about to consume for its policy update — no separate data collection
    pass, since the "fake" samples it needs (the policy's corrected-pose
    transitions) are already sitting in `self.memory`.

    skrl's on-policy memory stores one tensor per step, `states`, not a
    parallel `next_states` (`next_states` exists only transiently, as the
    single extra observation needed for the value bootstrap — it is never
    written per-step, precisely because it would just be `states` shifted by
    one row). That shift is exactly what this needs though: `flatten_state`
    packs the *previous* step's correction into the *current* step's
    observation, so row t of `states` carries corrected_{t-1} and row t+1
    carries corrected_t (`extract_corrected_pose`, src/models/policy.py). The
    transition pair is therefore `states[t], states[t+1]` — with one
    exception: at an episode boundary (`terminated[t]` or `truncated[t]`),
    `states[t+1]` is the *next* episode's reset observation, not a
    continuation, and pairing across that seam would hand the discriminator a
    fake "transition" that never happened in either clip. Those rows are
    masked out below rather than scored.

    Runs **before** `super().post_interaction()` on an update boundary, so the
    discriminator is trained on this rollout before PPO's own `update()`
    touches the same memory (it only overwrites `values`/`returns`/
    `advantages`, never `states`, but ordering it this way keeps the two
    updates from having to reason about each other at all).

    One rollout of lag is therefore built in: rollout N's reward used the
    discriminator as of rollout N-1's update. That is the standard scheme for
    a concurrently-trained reward (AMP, vanilla GAIL) and is intentional, not
    a bug — see `GAILRewardProvider`.
    """

    def __init__(self, *args, discriminator: MotionDiscriminator,
                 disc_optimizer: torch.optim.Optimizer, real_bank: torch.Tensor,
                 state_trans: bool = True, disc_updates: int = 4,
                 disc_batch_size: int = 1024, disc_grad_penalty: float = 5.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator    = discriminator
        self.disc_optimizer   = disc_optimizer
        # Kept on whatever device it was built on (see gail_train.train —
        # typically CPU, to leave GPU memory for the PPO batch); sampled
        # batches are moved to the discriminator's device per update.
        self.real_bank        = real_bank
        self.state_trans      = state_trans
        self.disc_updates     = int(disc_updates)
        self.disc_batch_size  = int(disc_batch_size)
        self.disc_grad_penalty = float(disc_grad_penalty)

    def _update_discriminator(self) -> None:
        device = next(self.discriminator.parameters()).device

        states     = self.memory.get_tensor_by_name("states")      # (T, num_envs, obs_dim)
        terminated = self.memory.get_tensor_by_name("terminated")  # (T, num_envs, 1)
        truncated  = self.memory.get_tensor_by_name("truncated")   # (T, num_envs, 1)
        done = (terminated | truncated).squeeze(-1)                # (T, num_envs)

        # states[t] -> states[t+1], dropping pairs that straddle an episode
        # boundary (see class docstring).
        prev, curr = states[:-1], states[1:]
        valid = ~done[:-1]
        obs_dim = states.shape[-1]
        prev = prev.reshape(-1, obs_dim)[valid.reshape(-1)]
        curr = curr.reshape(-1, obs_dim)[valid.reshape(-1)]

        fake_prev = extract_corrected_pose(prev, self.state_trans)
        fake_curr = extract_corrected_pose(curr, self.state_trans)
        # (n_valid, 2*POSE_DIM), oldest frame first — matches `make_window`'s
        # ordering and `load_gt_transitions`'s layout.
        fake_windows = torch.cat([fake_prev, fake_curr], dim=-1).to(device).detach()

        n_fake = fake_windows.shape[0]
        n_real = self.real_bank.shape[0]
        if n_fake == 0:      # every step this rollout ended an episode — degenerate but possible
            return

        self.discriminator.train()
        totals = {"loss": 0.0, "acc_real": 0.0, "acc_fake": 0.0, "grad_penalty": 0.0}
        for _ in range(self.disc_updates):
            fi = torch.randint(0, n_fake, (self.disc_batch_size,), device=device)
            ri = torch.randint(0, n_real, (self.disc_batch_size,))
            real_batch = self.real_bank[ri].to(device)
            fake_batch = fake_windows[fi]

            loss, logs = discriminator_step(
                self.discriminator, real_batch, fake_batch,
                grad_penalty_weight=self.disc_grad_penalty,
            )
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
            for k in totals:
                totals[k] += logs[k]
        self.discriminator.eval()

        n = max(self.disc_updates, 1)
        self.track_data("GAIL / Discriminator loss", totals["loss"] / n)
        self.track_data("GAIL / Discriminator accuracy (real)", totals["acc_real"] / n)
        self.track_data("GAIL / Discriminator accuracy (fake)", totals["acc_fake"] / n)
        self.track_data("GAIL / Discriminator gradient penalty", totals["grad_penalty"] / n)

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        updating = (not (self._rollout + 1) % self._rollouts
                    and timestep >= self._learning_starts)
        if updating:
            self._update_discriminator()
        super().post_interaction(timestep, timesteps)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Data
    h5_path:         str   = "processed_movi.h5"
    norm_stats_path: str   = "data/normalization.json"

    # Rollout / PPO (mapped directly to skrl PPO_DEFAULT_CONFIG)
    rollouts:           int   = 3200    # transitions collected before each update
    learning_epochs:    int   = 4       # gradient passes over the rollout buffer
    mini_batches:       int   = 6       # buffer split into this many minibatches (~512 each)
    learning_rate:      float = 1e-4
    discount_factor:    float = 0.99
    gae_lambda:         float = 0.95
    ratio_clip:         float = 0.2
    value_clip:         float = 0.2
    # Fix 04: zero. The action is a residual around an already-good pose, so
    # there is nothing to explore; meanwhile the bonus's gradient w.r.t. log_std
    # is a constant +1 per dimension regardless of reward, which was one of the
    # two forces inflating sigma to divergence.
    entropy_loss_scale: float = 0.0
    value_loss_scale:   float = 1.0
    grad_norm_clip:     float = 0.5
    # Fix 07: a KL-adaptive learning rate, so a divergence throttles itself
    # instead of running unattended for eight hours.
    #
    # **Not skrl's default 0.008.** For two Gaussians with a shared sigma the KL
    # is `(D/2)·(dmu/sigma)²`, so the trust region a threshold buys depends on
    # the action dimension and on sigma — and this policy is extreme in both: 156
    # dims and sigma = 0.05, twenty times smaller than a typical continuous-
    # control policy because the action is a residual around an already-good
    # pose. At 0.008 one update may move the mean by 1.0% of sigma, which is
    # 5e-4 in pose units against a mean scale of 8e-3. Measured: the scheduler
    # floors the learning rate at skrl's min_lr of 1e-6 within three updates and
    # learning stops.
    #
    # 0.05 corresponds to 2.5% of sigma per update — still a tight trust region,
    # but one an Adam step at 1e-4 can actually stay inside.
    kl_threshold:       float = 0.05

    # Reward weights
    w_similarity: float = 1.0
    w_smoothness: float = 0.1
    reward_scale: float = 10.0

    # Reward mode. "gt" is the original supervised similarity term — valid for
    # ablations, not for experiments (B)/(C). "reproj" is the GT-free
    # reprojection + smoothness reward and requires --reproj_path.
    #
    # Defaults to "reproj" here (unlike src/train.py's "gt"): experiment (C)
    # is "add a discriminator on top of (B)", not on top of the GT-similarity
    # ablation, and the GAIL term below stacks on whichever base reward this
    # selects.
    reward_mode:  str   = "reproj"
    reproj_path:  str   = "data/reproj_targets.h5"
    w_reproj:     float = 1.0
    # Retuned after the keypoint-bias correction. `exp(-e^2/s^2)` is steepest at
    # e = s/sqrt(2), so sigma should sit at operating_point * sqrt(2). Removing
    # the systematic COCO<->SMPL-X offset drops the held-out operating point from
    # 0.0306 to 0.0159 bbox heights (14.2 -> 7.4 px), so sigma follows it down
    # from 0.043 to 0.0225. Leaving it at 0.04 would put the reward in its
    # saturated corner, where there is almost no gradient.
    reproj_sigma: float = 0.0225
    # Per-joint COCO<->SMPL-X offset, fitted by scripts/fit_kp_bias.py on train.
    # "" disables the correction (ablation).
    kp_bias_path: str = "data/kp_bias.json"
    # How often the lifted-vs-corrected comparison is *logged*. The projection
    # itself now happens every frame regardless, because the observation needs it
    # (fix 06), so this no longer buys any compute back.
    baseline_every: int = 1
    # Fix 05: reward = r_corrected - r_lifted, per step.
    reward_baseline: bool = True
    # Fix 06: put the reward's own 2D residual into the observation.
    use_evidence: bool = True
    # Whether `trans` is in the observation. The lifted `trans` is virtual-camera
    # depth, not metres, and is never corrected — so with this off the state is
    # poses only and the translation is kept inside the env for the reprojection
    # reward alone. This is the third variant of experiment (B), reported
    # alongside trans_mode none/uv.
    state_trans: bool = True

    # ── GAIL discriminator (experiment C) ──────────────────────────────────
    # Weight of the discriminator's "trick me" reward, additive on top of the
    # reward `reward_mode` already computes (src/gail_env.py). 0.0 runs
    # identically to src/train.py (modulo the "reproj" default above) and is
    # the ablation to compare (C) against (B).
    w_gail:            float = 0.5
    disc_hidden_dims:  tuple = (256, 128)
    disc_lr:           float = 3e-4
    # Discriminator gradient steps per PPO rollout. More steps track the
    # policy's (fast-moving, early on) output distribution more closely but
    # risk exactly the over-confident-discriminator failure mode
    # `discriminator_step`'s gradient penalty is there to slow down.
    disc_updates:      int   = 4
    disc_batch_size:   int   = 1024
    # R1 gradient penalty weight on real samples (src/models/discriminator.py).
    # 0 disables it — useful as an ablation to see the accuracy-saturation
    # failure mode directly, not a setting to train with.
    disc_grad_penalty: float = 5.0

    # Translation (§2). The lifted `trans` is virtual-camera depth, not metres,
    # so it is never corrected as a normalised delta — it is passed through, and
    # the policy's translation action is reparameterised into image-plane
    # coordinates instead.
    # Translation action (src/models/policy.py): "none" freezes it entirely,
    # "uv" gives the policy an image-plane shift only, "uvz" also unfreezes
    # log-depth as an ablation.
    trans_mode: str = "none"

    # Policy architecture. Dropout is gone, not defaulted to 0 — see
    # src/models/policy.py::_mlp for why it was the cause of the (B) divergence.
    hidden_dims: tuple = (512, 256)

    # Training duration
    total_updates: int = 3000   # total_updates * rollouts = total timesteps

    # Experiment / logging
    out_dir:             str = "checkpoints"
    log_interval:        int = 200   # TensorBoard write interval (timesteps)
    checkpoint_interval: int = 0     # skrl checkpoint interval (0 = end only)
    viz_interval:        int = 3     # log the 2D skeleton figure every N updates (0 = off)
    viz_clips:           int = 3     # held-out val clips shown per figure
    viz_frames:          int = 4     # frames sampled per clip
    # "image" projects through the real camera and draws on the video frame the
    # pose came from; "ortho" is the older world-frame stick figure, for runs
    # with no reprojection targets.
    viz_mode:            str = "image"
    viz_video_root:      str = "demo/videos"

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False

    device = torch.device(cfg.device)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # ── Dataset & env ────────────────────────────────────────────────────────
    use_reproj = cfg.reward_mode == "reproj"
    dataset = MoViDataset(cfg.h5_path, cfg.norm_stats_path, split="train", verbose=False,
                          reproj_path=cfg.reproj_path if use_reproj else None)
    print(f"Train samples: {len(dataset)}")

    reproj_reward = None
    if use_reproj:
        from src.rewards import ReprojectionReward, load_calib
        reproj_reward = ReprojectionReward(
            load_calib(cfg.h5_path), sigma=cfg.reproj_sigma,
            bias=cfg.kp_bias_path or None,
            # Under pose-only training the policy never moves `trans`, so there
            # is nothing to re-derive: use the metric translation already in the
            # sidecar, which is the path the 11.6 / 14.5 px validation measured.
            # The reparameterised translation action is applied to the metric
            # translation directly, so the virtual-camera path stays off.
            correct_translation=False,
        )
        print(f"Reward: reprojection (sigma={cfg.reproj_sigma}) + smoothness — no GT"
              + (f", keypoint bias from {cfg.kp_bias_path}" if cfg.kp_bias_path
                 else ", NO keypoint-bias correction"))
    else:
        print("Reward: GT similarity + smoothness (supervised; ablation only)")

    # ── Discriminator (experiment C) ────────────────────────────────────────
    # Built before the env because the env holds a *reference* to the
    # provider wrapping it (`GAILRewardProvider`) — see src/gail_env.py and
    # src/models/discriminator.py for why that reference is what lets the
    # reward track training without the env changing.
    discriminator = MotionDiscriminator(hidden_dims=cfg.disc_hidden_dims).to(device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.disc_lr)
    real_bank = load_gt_transitions(cfg.h5_path, split="train", device="cpu")
    print(f"GAIL: discriminator {sum(p.numel() for p in discriminator.parameters()):,} params, "
          f"real bank {real_bank.shape[0]:,} GT transitions, w_gail={cfg.w_gail}")
    disc_reward = GAILRewardProvider(discriminator, device=cfg.device)

    gym_env = GymMoviEnv(
        dataset,
        device=cfg.device,
        w_similarity=cfg.w_similarity,
        w_smoothness=cfg.w_smoothness,
        reward_scale=cfg.reward_scale,
        reward_mode=cfg.reward_mode,
        reproj_reward=reproj_reward,
        w_reproj=cfg.w_reproj,
        baseline_every=cfg.baseline_every,
        trans_mode=cfg.trans_mode,
        reward_baseline=cfg.reward_baseline,
        use_evidence=cfg.use_evidence,
        state_trans=cfg.state_trans,
        disc_reward=disc_reward,
        w_gail=cfg.w_gail,
    )
    _TRANS_DESC = {"none": "poses only, translation frozen",
                   "uv":   "poses + du,dv image shift (log-depth frozen)",
                   "uvz":  "poses + du,dv,dlog_tz (depth unfrozen — ablation)"}
    print(f"Action: {gym_env.action_space.shape[0]}-d ({_TRANS_DESC[cfg.trans_mode]}), "
          f"pose delta bounded to +-{gym_env.action_space.high[0]:g}")
    print(f"Observation: {gym_env.observation_space.shape[0]}-d "
          + ("poses" if not gym_env.state_trans else "poses + trans")
          + (" + 2D evidence (residual, confidence, error, progress, camera)"
             if gym_env.use_evidence else "  — the reward is NOT observable"))
    if gym_env.reward_mode == "reproj":
        print("Reward: " + ("improvement over the lifted pose (baselined)"
                            if gym_env.reward_baseline else "absolute level (no baseline)"))
    gym_env.reset(seed=cfg.seed)
    env = wrap_env(gym_env)

    obs_space = gym_env.observation_space
    act_space = gym_env.action_space

    # ── Models ───────────────────────────────────────────────────────────────
    actor  = SkrlPoseActor(obs_space, act_space, device, hidden_dims=cfg.hidden_dims)
    critic = SkrlPoseCritic(obs_space, act_space, device, hidden_dims=cfg.hidden_dims)

    n_actor  = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"Actor params: {n_actor:,}  |  Critic params: {n_critic:,}")

    # ── Memory ───────────────────────────────────────────────────────────────
    # number of rollouts should equal the memory size for PPO always
    memory = RandomMemory(memory_size=cfg.rollouts, num_envs=1, device=device)

    # ── PPO config ───────────────────────────────────────────────────────────
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["rollouts"]           = cfg.rollouts
    ppo_cfg["learning_epochs"]    = cfg.learning_epochs
    ppo_cfg["mini_batches"]       = cfg.mini_batches
    ppo_cfg["learning_rate"]      = cfg.learning_rate
    ppo_cfg["discount_factor"]    = cfg.discount_factor
    ppo_cfg["gae_lambda"]         = cfg.gae_lambda
    ppo_cfg["ratio_clip"]         = cfg.ratio_clip
    ppo_cfg["value_clip"]         = cfg.value_clip
    ppo_cfg["entropy_loss_scale"] = cfg.entropy_loss_scale
    ppo_cfg["value_loss_scale"]   = cfg.value_loss_scale
    ppo_cfg["grad_norm_clip"]     = cfg.grad_norm_clip
    # Episodes end because the clip runs out of frames, which src/env.py reports
    # as truncation. skrl only adds the bootstrap `gamma * V(s)` to the reward on
    # truncated steps when this is on, and it defaults to False — without it the
    # value target at every episode boundary is short by the entire remaining
    # return (~60 here), which is what the first (B) attempt was training against.
    ppo_cfg["time_limit_bootstrap"] = True

    # Fix 02. Both were left at None. A per-step reward near 0.7 at gamma = 0.99
    # gives returns around 70 and episode returns of 883 were logged, so the value
    # loss started at 99 and took ~500k steps to come down — for that entire
    # period the GAE advantage was dominated by value error rather than by which
    # action earned reward. The state scaler matters just as much now that the
    # observation mixes normalised pose units with pixel residuals, confidences
    # and a one-hot, which are on entirely different scales.
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": obs_space, "device": device}
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Fix 07. Without this a divergence just runs.
    if cfg.kl_threshold > 0:
        ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
        ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": cfg.kl_threshold}
    ppo_cfg["experiment"]["directory"]           = cfg.out_dir
    ppo_cfg["experiment"]["write_interval"]      = cfg.log_interval
    ppo_cfg["experiment"]["checkpoint_interval"] = cfg.checkpoint_interval

    # ── Pose visualisation on held-out clips ─────────────────────────────────
    viz_logger = None
    if cfg.viz_interval > 0:
        try:
            import json
            with open(cfg.norm_stats_path) as f:
                viz_stats = json.load(f)

            # The image-plane figure needs the same 2D evidence the reward does,
            # so the val dataset is built with the targets attached.
            want_image = cfg.viz_mode == "image" and use_reproj
            if cfg.viz_mode == "image" and not use_reproj:
                # Silently dropping to the world-frame view would mean noticing
                # hours later that the run has no image logging.
                print("WARNING: --viz_mode image needs --reward_mode reproj "
                      "(the figure reuses the reward's 2D targets). Falling back "
                      "to the orthographic view; pass --viz_mode ortho to silence "
                      "this.")
            viz_dataset = MoViDataset(
                cfg.h5_path, cfg.norm_stats_path, split="val",
                reproj_path=cfg.reproj_path if want_image else None,
            )
            if want_image:
                from src.rewards import load_calib
                from src.viz_pose import ImagePoseVizLogger
                from src.rewards import ReprojectionReward
                viz_logger = ImagePoseVizLogger(
                    viz_dataset, actor, viz_stats, load_calib(cfg.h5_path),
                    cfg.reproj_path, video_root=cfg.viz_video_root,
                    device=cfg.device, n_clips=cfg.viz_clips,
                    n_frames=cfg.viz_frames, seed=cfg.seed,
                    # Its own instance: `reset` binds a reward to one clip, and
                    # the trainer's is bound to whatever the rollout is on.
                    reproj_reward=ReprojectionReward(
                        load_calib(cfg.h5_path), sigma=cfg.reproj_sigma,
            bias=cfg.kp_bias_path or None,
                        correct_translation=False,
                    ),
                    use_evidence=gym_env.use_evidence,
                    trans_mode=cfg.trans_mode,
                    state_trans=cfg.state_trans,
                )
                kind = f"image-plane on video ({len(viz_logger.clips)} clips)"
            else:
                viz_logger = PoseVizLogger(
                    viz_dataset, actor, viz_stats, device=cfg.device,
                    n_clips=cfg.viz_clips, n_frames=cfg.viz_frames, seed=cfg.seed,
                )
                kind = "orthographic world frame"
            print(f"Pose viz: {kind}, every {cfg.viz_interval} updates")
        except Exception as e:                       # never let logging kill a run
            # Loud on purpose. A run that quietly trains for eight hours with no
            # pose figures is worse than one that fails at startup, because the
            # figures are how the policy's behaviour is checked at all.
            print("=" * 72)
            print(f"WARNING: POSE VIZ DISABLED — {type(e).__name__}: {e}")
            print("This run will log scalars but NO pose images.")
            print("=" * 72)

    # TODO: train.py: Do we need to use PPO or PPO_RNN?
    agent = PPOWithGAIL(
        models={"policy": actor, "value": critic},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        viz_logger=viz_logger,
        viz_interval=cfg.viz_interval,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        real_bank=real_bank,
        state_trans=cfg.state_trans,
        disc_updates=cfg.disc_updates,
        disc_batch_size=cfg.disc_batch_size,
        disc_grad_penalty=cfg.disc_grad_penalty,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    timesteps = cfg.total_updates * cfg.rollouts
    trainer   = SequentialTrainer(
        cfg={"timesteps": timesteps, "headless": True},
        env=env,
        agents=agent,
    )

    print(f"Training for {timesteps:,} timesteps ({cfg.total_updates} updates × {cfg.rollouts} rollout steps)")
    trainer.train()

    # ── Save actor in evaluate.py-compatible format ───────────────────────────
    # The config goes in as a plain dict, not the dataclass. `python -m src.train`
    # runs this module as `__main__`, so pickling `Config` itself records the class
    # as `__main__.Config` and every reader that is not the training script fails
    # with `AttributeError: Can't get attribute 'Config' on <module '__main__'>`.
    ckpt_path = Path(cfg.out_dir) / "actor_final.pt"
    torch.save({"actor": actor.state_dict(), "config": asdict(cfg)}, str(ckpt_path))
    # ASCII "->", not "→" — see src/train.py's identical checkpoint print for why
    # (Windows console cp1252 has no glyph for U+2192; discovered because this
    # crashed the discriminator save right below on a real Windows run).
    print(f"Actor weights saved -> {ckpt_path}")

    # The discriminator is not needed by evaluate.py/viz_pose.py, but keeping
    # it is what makes it possible to resume GAIL training, or to inspect
    # afterward what it learned to key on (e.g. real/fake accuracy per joint).
    disc_ckpt_path = Path(cfg.out_dir) / "discriminator_final.pt"
    torch.save({"discriminator": discriminator.state_dict(), "config": asdict(cfg)},
              str(disc_ckpt_path))
    print(f"Discriminator weights saved -> {disc_ckpt_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5_path",             default="processed_movi.h5")
    parser.add_argument("--norm_stats_path",     default="data/normalization.json")
    parser.add_argument("--rollouts",            type=int,   default=3200)
    parser.add_argument("--learning_epochs",     type=int,   default=4)
    parser.add_argument("--mini_batches",        type=int,   default=6)
    parser.add_argument("--learning_rate",       type=float, default=1e-4)
    parser.add_argument("--discount_factor",     type=float, default=0.99)
    parser.add_argument("--gae_lambda",          type=float, default=0.95)
    parser.add_argument("--ratio_clip",          type=float, default=0.2)
    parser.add_argument("--entropy_loss_scale",  type=float, default=0.0,
                        help="0 by default: the action is a residual around a good "
                             "initial guess, so there is nothing to explore and the "
                             "bonus only inflates the policy standard deviation")
    parser.add_argument("--value_loss_scale",    type=float, default=1.0)
    parser.add_argument("--grad_norm_clip",      type=float, default=0.5)
    parser.add_argument("--kl_threshold",        type=float, default=0.05,
                        help="target KL for the adaptive learning-rate scheduler "
                             "(0 disables it). Not skrl's 0.008: at 156 action "
                             "dims and sigma=0.05 that allows a mean step of 1%% "
                             "of sigma and floors the learning rate in three "
                             "updates. See Config.kl_threshold.")
    parser.add_argument("--w_similarity",        type=float, default=1.0)
    parser.add_argument("--w_smoothness",        type=float, default=0.1)
    parser.add_argument("--reward_scale",        type=float, default=10.0)
    parser.add_argument("--reward_mode",         choices=("gt", "reproj"), default="reproj",
                        help="base reward the GAIL term stacks on top of; "
                             "'reproj' is experiment (C)'s intended setting")
    parser.add_argument("--reproj_path",         default="data/reproj_targets.h5")
    parser.add_argument("--w_reproj",            type=float, default=1.0)
    parser.add_argument("--reproj_sigma",        type=float, default=0.0225,
                        help="reward width in bbox-height units; tuned to the "
                             "bias-corrected operating point (see Config)")
    parser.add_argument("--kp_bias_path",        default="data/kp_bias.json",
                        help="per-joint COCO<->SMPL-X offset from "
                             "scripts.fit_kp_bias; empty string disables it")
    parser.add_argument("--trans_mode", choices=("none", "uv", "uvz"), default="none",
                        help="translation action. 'none' (156-d) freezes it. 'uv' "
                             "(158-d) lets the policy shift the body in the image "
                             "plane in bbox-height units — the coordinates the "
                             "reprojection reward is actually sensitive in — with "
                             "log-depth structurally frozen. 'uvz' (159-d) "
                             "unfreezes log-depth as an ablation.")
    parser.add_argument("--baseline_every",      type=int,   default=10,
                        help="score the raw lifted frame every N steps for comparison (0 disables)")
    parser.add_argument("--no_reward_baseline",  dest="reward_baseline",
                        action="store_false",
                        help="score the absolute reprojection reward instead of the "
                             "improvement over the lifted pose. Off by default "
                             "because the absolute level is ~0.68 of constant the "
                             "policy cannot influence.")
    parser.add_argument("--no_state_trans",      dest="state_trans",
                        action="store_false",
                        help="drop `trans` from the observation, leaving a "
                             "poses-only state (312-d, or 356-d with evidence). "
                             "The translation stays inside the env and is still "
                             "used by the reprojection reward.")
    parser.add_argument("--no_evidence",         dest="use_evidence",
                        action="store_false",
                        help="drop the 2D residual block from the observation, "
                             "leaving the 318-d pose-only state. Reproduces the "
                             "setting in which the reward is unobservable — for "
                             "the ablation, not for a real run.")
    parser.add_argument("--w_gail",              type=float, default=0.5,
                        help="weight of the discriminator's reward term, "
                             "additive on top of --reward_mode's reward")
    # disc_hidden_dims is Config-only, like the actor/critic's hidden_dims —
    # architecture, not a run-to-run sweep knob.
    parser.add_argument("--disc_lr",             type=float, default=3e-4)
    parser.add_argument("--disc_updates",        type=int,   default=4,
                        help="discriminator gradient steps per PPO rollout")
    parser.add_argument("--disc_batch_size",     type=int,   default=1024)
    parser.add_argument("--disc_grad_penalty",   type=float, default=5.0,
                        help="R1 gradient penalty weight on real samples; 0 disables "
                             "it (see src/models/discriminator.py)")
    parser.add_argument("--total_updates",       type=int,   default=3000)
    parser.add_argument("--out_dir",             default="checkpoints")
    parser.add_argument("--log_interval",        type=int,   default=200)
    parser.add_argument("--checkpoint_interval", type=int,   default=0)
    parser.add_argument("--viz_interval",        type=int,   default=3,
                        help="log the 2D skeleton overlay every N PPO updates (0 disables)")
    parser.add_argument("--viz_mode",            choices=("image", "ortho"), default="image",
                        help="'image' projects through the real camera onto the video "
                             "frame (needs --reward_mode reproj); 'ortho' is the "
                             "world-frame stick figure")
    parser.add_argument("--viz_video_root",      default="demo/videos")
    parser.add_argument("--viz_clips",           type=int,   default=3)
    parser.add_argument("--viz_frames",          type=int,   default=4)
    parser.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    cfg = Config(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
