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

# NOTE: the PPO_DEFAULT_CONFIG is hallucinated, need to re-write training to suit latest version of skrl. 
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from src.data.datasets import MoViDataset
from src.env import GymMoviEnv
from src.models.policy import SkrlPoseActor, SkrlPoseCritic


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

    # Policy architecture
    hidden_dims: tuple = (512, 256)
    dropout:     float = 0.1

    # Training duration
    total_updates: int = 3000   # total_updates * rollouts = total timesteps

    # Experiment / logging
    out_dir:             str = "checkpoints"
    log_interval:        int = 200   # TensorBoard write interval (timesteps)
    checkpoint_interval: int = 0     # skrl checkpoint interval (0 = end only)

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
    dataset = MoViDataset(cfg.h5_path, cfg.norm_stats_path, split="train", verbose=False)
    print(f"Train samples: {len(dataset)}")

    gym_env = GymMoviEnv(
        dataset,
        device=cfg.device,
        w_similarity=cfg.w_similarity,
        w_smoothness=cfg.w_smoothness,
        reward_scale=cfg.reward_scale,
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

    agent = PPO(
        models={"policy": actor, "value": critic},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
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
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--total_updates",       type=int,   default=3000)
    parser.add_argument("--out_dir",             default="checkpoints")
    parser.add_argument("--log_interval",        type=int,   default=200)
    parser.add_argument("--checkpoint_interval", type=int,   default=0)
    parser.add_argument("--device",              default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    cfg = Config(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
