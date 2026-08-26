"""
Supervised MSE-to-GT training: the plain-regression benchmark.

Direct-regression counterpart to experiment (D) (`scripts/run_exp_d.sh`),
which adds an MSE-to-GT term *inside a PPO reward*. This trains the same
quantity by gradient descent instead, with no RL, no discriminator, no
reprojection reward — see `src/models/supervised.py`'s module docstring for
why that separation is the point of this experiment.

Trains on i.i.d. shuffled frames rather than on-policy rollouts: nothing here
needs the temporal structure PPO requires, so plain minibatch SGD over every
frame in the train split converges faster and more stably than anything
rollout-shaped would.

Usage:
    python -m src.train_supervised [args]
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data.datasets import MoViDataset
from src.models.discriminator import PoseSpace
from src.models.supervised import SupervisedPoseRegressor, gt_space_mse
from src.viz_pose import SupervisedPoseVizLogger


# ─── Data ───────────────────────────────────────────────────────────────────

def load_frames(h5_path: str, norm_stats_path: str, split: str,
                device: torch.device, verbose: bool = True):
    """
    Flatten every clip-camera in `split` into one big set of
    (lifted_pose, gt_pose, camera_idx) frames.

    Both pose tensors come out exactly as `processed_movi.h5` stores them —
    normalised, lifted with its per-camera stats and GT with `norm_stats_path`
    (see `src/data/datasets.py`'s module docstring). Neither is unnormalised
    here; `gt_space_mse` remaps the model's lifted-space output into
    GT-normalised space at loss time instead of doing it to the data up
    front, so the same tensors serve every minibatch without re-deriving
    anything per step.

    `keys=("poses",)` skips loading `trans`/`betas` per clip — this script
    never uses them, and MoViDataset otherwise reads both off disk on every
    `__getitem__`.
    """
    ds = MoViDataset(h5_path, norm_stats_path, split=split, keys=("poses",), verbose=verbose)
    cam_to_idx = {c: i for i, c in enumerate(PoseSpace.CAMERAS)}

    lifted_chunks, gt_chunks, cam_chunks = [], [], []
    for i in range(len(ds)):
        sample = ds[i]
        camera = sample["meta"]["camera"]
        lifted = sample["x"]["poses"].flatten(1)   # (T, 66)
        gt     = sample["y"]["poses"].flatten(1)   # (T, 66)
        lifted_chunks.append(lifted)
        gt_chunks.append(gt)
        cam_chunks.append(torch.full((lifted.shape[0],), cam_to_idx[camera], dtype=torch.long))
        if verbose and (i + 1) % 200 == 0:
            print(f"  [{split}] loaded {i + 1}/{len(ds)} clip-cameras")

    lifted_all = torch.cat(lifted_chunks, dim=0).to(device)
    gt_all     = torch.cat(gt_chunks, dim=0).to(device)
    cam_all    = torch.cat(cam_chunks, dim=0).to(device)
    print(f"{split}: {lifted_all.shape[0]:,} frames from {len(ds)} clip-cameras")
    return lifted_all, gt_all, cam_all


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    h5_path:         str = "data/processed_movi.h5"
    norm_stats_path: str = "data/normalization.json"

    hidden_dims:    tuple = (512, 256)   # matches PoseActor / (B) — same capacity
    batch_size:     int   = 4096
    learning_rate:  float = 1e-3
    weight_decay:   float = 0.0
    epochs:         int   = 30
    grad_norm_clip: float = 1.0

    out_dir:      str = "checkpoints/exp_supervised"
    log_interval: int = 100   # steps between printed running-loss updates
    # Epochs between periodic checkpoint saves (0 disables). Training is cheap
    # enough here (SGD over pre-loaded tensors, no env/rollout) that this is a
    # safety net rather than a real cost -- it exists so a run stopped early
    # (Ctrl+C, a machine that got too hot, a crash) doesn't lose everything;
    # see the try/finally around the epoch loop below for the Ctrl+C case
    # specifically.
    checkpoint_interval: int = 5

    # TensorBoard. A SummaryWriter is always created (cheap, and matches
    # src.train's PPO runs always writing one) -- these three only control the
    # periodic lifted/corrected/GT skeleton figure, which needs a rollout over
    # a few val clips and forward kinematics, so it is not free the way a
    # scalar write is. 0 disables the figure but keeps the scalar curves.
    viz_interval: int = 5    # epochs between figure logs
    viz_clips:    int = 3    # held-out val clips shown per figure
    viz_frames:   int = 4    # frames sampled per clip

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42


def _save(model: nn.Module, cfg: Config, tag: str = "") -> Path:
    """Write `regressor_final.pt` (or `regressor_<tag>.pt`), config alongside
    the weights. `config` goes in as a plain dict for the same reason
    `src/train.py`'s does: `python -m src.train_supervised` runs this module
    as `__main__`, so pickling the dataclass itself would record it as
    `__main__.Config` and any other reader would fail with
    `AttributeError: Can't get attribute 'Config' on <module '__main__'>`."""
    name = f"regressor_{tag}.pt" if tag else "regressor_final.pt"
    path = Path(cfg.out_dir) / name
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, str(path))
    return path


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    print("Loading train split...")
    X_train, Y_train, C_train = load_frames(cfg.h5_path, cfg.norm_stats_path, "train", device)
    # val is loaded for a diagnostic curve only — never used to pick a
    # checkpoint or a hyperparameter, matching this project's stated
    # convention (report.md: "val is not used as an evaluation split
    # anywhere"). Model selection, if any, belongs on `test` via
    # `src.evaluate`, same as (A)-(D).
    print("Loading val split (diagnostic only, not used for model selection)...")
    X_val, Y_val, C_val = load_frames(cfg.h5_path, cfg.norm_stats_path, "val", device)

    # All 22 joints, unlike the GAIL discriminator's default: excluding
    # global_orient is a plausibility-scoring choice specific to that
    # discriminator, not one this regression loss has a reason to inherit.
    space = PoseSpace(gt_stats_path=cfg.norm_stats_path, exclude_joints=(), device=str(device))

    model = SupervisedPoseRegressor(hidden_dims=cfg.hidden_dims).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    # Written straight into out_dir, same layout src.train's skrl runs use, so
    # `tensorboard --logdir checkpoints` finds every experiment (A)-(E) under
    # one tree.
    writer = SummaryWriter(log_dir=cfg.out_dir)

    viz_logger = None
    if cfg.viz_interval > 0:
        try:
            with open(cfg.norm_stats_path) as f:
                viz_stats = json.load(f)
            # Default keys (poses/trans/betas), unlike load_frames's
            # keys=("poses",) above -- the figure's forward-kinematics call
            # needs betas, and this dataset only ever serves `viz_clips` clips
            # on demand, not the whole split, so the extra columns cost nothing
            # in practice.
            viz_dataset = MoViDataset(cfg.h5_path, cfg.norm_stats_path, split="val", verbose=False)
            viz_logger = SupervisedPoseVizLogger(
                viz_dataset, model, viz_stats, device=str(device),
                n_clips=cfg.viz_clips, n_frames=cfg.viz_frames, seed=cfg.seed)
            print(f"Pose viz: orthographic world frame, every {cfg.viz_interval} epochs "
                  f"({cfg.viz_clips} clips x {cfg.viz_frames} frames)")
        except Exception as e:       # never let logging kill a run
            print("=" * 72)
            print(f"WARNING: POSE VIZ DISABLED — {type(e).__name__}: {e}")
            print("This run will log scalars but NO pose images.")
            print("=" * 72)

    n_train = X_train.shape[0]
    steps_per_epoch = max(1, n_train // cfg.batch_size)
    print(f"Training: {cfg.epochs} epochs x {steps_per_epoch} steps/epoch, batch={cfg.batch_size}")

    import time

    step = 0
    t_start = time.time()
    try:
        for epoch in range(cfg.epochs):
            t_epoch = time.time()
            perm = torch.randperm(n_train, device=device)
            model.train()
            running, running_n = 0.0, 0
            for b in range(steps_per_epoch):
                idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                x, y, cam = X_train[idx], Y_train[idx], C_train[idx]

                pred = model(x)
                loss = gt_space_mse(pred, y, cam, space)

                opt.zero_grad()
                loss.backward()
                if cfg.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
                opt.step()

                running += loss.item()
                running_n += 1
                step += 1
                if step % cfg.log_interval == 0:
                    avg = running / running_n
                    print(f"  epoch {epoch + 1}/{cfg.epochs} step {step}  "
                          f"loss={avg:.6f}")
                    writer.add_scalar("Loss/train (GT-space MSE)", avg, step)
                    running, running_n = 0.0, 0

            model.eval()
            with torch.no_grad():
                val_loss = gt_space_mse(model(X_val), Y_val, C_val, space).item()
            print(f"[epoch {epoch + 1}/{cfg.epochs}] val loss (GT-space MSE, diagnostic only) "
                  f"= {val_loss:.6f}  ({time.time() - t_epoch:.1f}s this epoch, "
                  f"{time.time() - t_start:.1f}s total)")
            writer.add_scalar("Loss/val (GT-space MSE, diagnostic only)", val_loss, step)

            if viz_logger is not None and (epoch + 1) % cfg.viz_interval == 0:
                for tag, value in viz_logger.correction_magnitude().items():
                    writer.add_scalar(tag, value, step)
                fig = viz_logger()
                if fig is not None:
                    writer.add_figure("pose/lifted_vs_corrected_vs_gt", fig, step)
                    import matplotlib.pyplot as plt
                    plt.close(fig)

            if cfg.checkpoint_interval and (epoch + 1) % cfg.checkpoint_interval == 0:
                p = _save(model, cfg)
                print(f"  checkpoint saved -> {p}")
    except KeyboardInterrupt:
        # The run was stopped on purpose (heat, time, "this looks bad already")
        # — save whatever the model currently is rather than losing the epochs
        # already spent. Re-raised so the process still exits non-zero and a
        # launcher script (e.g. a 3-seed loop) stops instead of silently moving
        # on to the next seed.
        p = _save(model, cfg)
        print(f"\nInterrupted after {step} steps — checkpoint saved -> {p}")
        writer.close()
        raise

    # ── Save ─────────────────────────────────────────────────────────────────
    writer.close()
    ckpt_path = _save(model, cfg)
    print(f"Model weights saved -> {ckpt_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5_path",         default="data/processed_movi.h5")
    parser.add_argument("--norm_stats_path", default="data/normalization.json")
    parser.add_argument("--batch_size",      type=int,   default=4096)
    parser.add_argument("--learning_rate",   type=float, default=1e-3)
    parser.add_argument("--weight_decay",    type=float, default=0.0)
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--grad_norm_clip",  type=float, default=1.0)
    parser.add_argument("--out_dir",         default="checkpoints/exp_supervised")
    parser.add_argument("--log_interval",    type=int,   default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="epochs between periodic checkpoint saves (0 disables); "
                             "also saved on Ctrl+C")
    parser.add_argument("--viz_interval",    type=int,   default=5,
                        help="log the lifted/corrected/GT skeleton figure every N "
                             "epochs to TensorBoard (0 disables the figure only; "
                             "scalar loss curves are always logged)")
    parser.add_argument("--viz_clips",       type=int,   default=3)
    parser.add_argument("--viz_frames",      type=int,   default=4)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    cfg = Config(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
