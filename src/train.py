"""
PPO training using skrl.

Usage:
    python -m src.train [args]
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from src.data.datasets import MoViDataset
from src.env import GymMoviEnv
from src.models.policy import SkrlPoseActor, SkrlPoseCritic
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
    entropy_loss_scale: float = 0.01
    value_loss_scale:   float = 1.0
    grad_norm_clip:     float = 0.5

    # Reward weights
    w_similarity: float = 1.0
    w_smoothness: float = 0.1
    reward_scale: float = 10.0

    # Reward mode. "gt" is the original supervised similarity term — valid for
    # ablations, not for experiments (B)/(C). "reproj" is the GT-free
    # reprojection + smoothness reward and requires --reproj_path.
    reward_mode:  str   = "gt"
    reproj_path:  str   = "data/reproj_targets.h5"
    w_reproj:     float = 1.0
    reproj_sigma: float = 0.04
    baseline_every: int = 10    # score the raw lifted frame every N steps (0 = off)

    # Policy architecture
    hidden_dims: tuple = (512, 256)
    dropout:     float = 0.1

    # Training duration
    total_updates: int = 3000   # total_updates * rollouts = total timesteps

    # Experiment / logging
    out_dir:             str = "checkpoints"
    log_interval:        int = 200   # TensorBoard write interval (timesteps)
    checkpoint_interval: int = 0     # skrl checkpoint interval (0 = end only)
    viz_interval:        int = 3     # log the 2D skeleton figure every N updates (0 = off)
    viz_clips:           int = 3     # held-out val clips shown per figure
    viz_frames:          int = 4     # frames sampled per clip

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
        )
        print(f"Reward: reprojection (sigma={cfg.reproj_sigma}) + smoothness — no GT")
    else:
        print("Reward: GT similarity + smoothness (supervised; ablation only)")

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
    )
    gym_env.reset(seed=cfg.seed)
    env = wrap_env(gym_env)

    obs_space = gym_env.observation_space
    act_space = gym_env.action_space

    # ── Models ───────────────────────────────────────────────────────────────
    actor = SkrlPoseActor(
        obs_space, act_space, device,
        hidden_dims=cfg.hidden_dims, dropout=cfg.dropout,
    )
    critic = SkrlPoseCritic(
        obs_space, act_space, device,
        hidden_dims=cfg.hidden_dims, dropout=cfg.dropout,
    )

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
    ppo_cfg["experiment"]["directory"]           = cfg.out_dir
    ppo_cfg["experiment"]["write_interval"]      = cfg.log_interval
    ppo_cfg["experiment"]["checkpoint_interval"] = cfg.checkpoint_interval

    # ── Pose visualisation on held-out clips ─────────────────────────────────
    viz_logger = None
    if cfg.viz_interval > 0:
        try:
            viz_dataset = MoViDataset(cfg.h5_path, cfg.norm_stats_path, split="val")
            import json
            with open(cfg.norm_stats_path) as f:
                viz_stats = json.load(f)
            viz_logger = PoseVizLogger(
                viz_dataset, actor, viz_stats, device=cfg.device,
                n_clips=cfg.viz_clips, n_frames=cfg.viz_frames, seed=cfg.seed,
            )
            print(f"Pose viz: {cfg.viz_clips} val clips every {cfg.viz_interval} updates")
        except Exception as e:                       # never let logging kill a run
            print(f"Pose viz disabled: {type(e).__name__}: {e}")

    # TODO: train.py: Do we need to use PPO or PPO_RNN?
    agent = PPOWithPoseViz(
        models={"policy": actor, "value": critic},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        viz_logger=viz_logger,
        viz_interval=cfg.viz_interval,
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
    ckpt_path = Path(cfg.out_dir) / "actor_final.pt"
    torch.save({"actor": actor.state_dict(), "config": cfg}, str(ckpt_path))
    print(f"Actor weights saved → {ckpt_path}")


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
    parser.add_argument("--entropy_loss_scale",  type=float, default=0.01)
    parser.add_argument("--value_loss_scale",    type=float, default=1.0)
    parser.add_argument("--grad_norm_clip",      type=float, default=0.5)
    parser.add_argument("--w_similarity",        type=float, default=1.0)
    parser.add_argument("--w_smoothness",        type=float, default=0.1)
    parser.add_argument("--reward_scale",        type=float, default=10.0)
    parser.add_argument("--reward_mode",         choices=("gt", "reproj"), default="gt",
                        help="'reproj' is the GT-free reward for experiments (B)/(C)")
    parser.add_argument("--reproj_path",         default="data/reproj_targets.h5")
    parser.add_argument("--w_reproj",            type=float, default=1.0)
    parser.add_argument("--reproj_sigma",        type=float, default=0.04)
    parser.add_argument("--baseline_every",      type=int,   default=10,
                        help="score the raw lifted frame every N steps for comparison (0 disables)")
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--total_updates",       type=int,   default=3000)
    parser.add_argument("--out_dir",             default="checkpoints")
    parser.add_argument("--log_interval",        type=int,   default=200)
    parser.add_argument("--checkpoint_interval", type=int,   default=0)
    parser.add_argument("--viz_interval",        type=int,   default=3,
                        help="log the 2D skeleton overlay every N PPO updates (0 disables)")
    parser.add_argument("--viz_clips",           type=int,   default=3)
    parser.add_argument("--viz_frames",          type=int,   default=4)
    parser.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    cfg = Config(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
