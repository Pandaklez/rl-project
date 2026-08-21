"""
GAIL-style motion discriminator for experiment (C).

Scores **single, pose-only** frames as GT-like ("real") or corrected-output-
like ("fake"), and turns that score into a per-step reward the policy is
trained to increase — i.e. a learned substitute for the frame-wise GT loss
that experiments (A)/(B) deliberately avoid. The discriminator is retrained
after every rollout on the batch skrl just collected, so unlike
`src/rewards.py`'s reprojection reward it is not a fixed function of the
data — it is a second model with its own optimizer, and the env only ever
holds a *reference* to it (`GAILRewardProvider`), never a copy. Updating its
weights in `src/gail_train.py` is what changes the reward the env hands out on
the next rollout; nothing has to be re-injected into the env for that to take
effect.

Design notes, since this deliberately does not reach for skrl's own GAIL
trainer (`skrl.utils.gail`, in newer skrl):

* skrl's GAIL is built around policies that produce whole (s, a) transitions
  matched against expert (s, a) pairs from a fixed demonstration buffer, and
  it owns the reward-shaping step itself. Here the "action" is a pose delta
  with no expert equivalent (MoVi has no recorded correction), the "expert"
  data is GT *motion*, not GT *actions*, and the reward needs to fold in
  cleanly next to the existing reprojection/smoothness terms rather than
  replace them. Bending skrl's GAIL utility to that shape would fight it at
  every step for no real benefit — a plain nn.Module trained with a plain
  Adam optimizer, driven from a custom `post_interaction` hook (see
  `src/gail_train.py::PPOWithGAIL`), is the more direct match. Nothing here is
  skrl-specific: mixing a vanilla torch net into an skrl training loop is
  completely ordinary — skrl's own actor/critic models *are* vanilla torch
  modules wrapped in a thin Mixin — so this costs nothing in compatibility.

* Single frames, **for now** — not the 2-frame consecutive windows an
  AMP-style discriminator (Peng et al. 2021) normally uses to also judge
  transition/velocity plausibility. Windows sound strictly better (they see
  motion, not just pose), but pairing `corrected_{t-1}` with `corrected_t`
  turned out to have a real bug: `MoviEnv.reset`/`NoTransMoviEnv.reset` seed
  `corrected_state = lifted_state.clone()` — a placeholder identity, not a
  policy output — for the frame before any action has been taken. A window
  built from consecutive memory rows can therefore pair a genuine correction
  with that raw-lifted placeholder at the first row of every episode, which
  is exactly the "rubbish" input the discriminator was never supposed to see.
  Single frames have no `t-1` to get wrong — the fix is not "mask the bad
  pairs" but "stop pairing." Revisiting a windowed/AMP-style discriminator
  later is still open (see `WINDOW`, kept below at 1 rather than deleted),
  but it would need the reset-boundary masking done properly first, likely
  in the recurrent-policy work already on the table rather than as a patch
  here. Note this does not remove *motion* realism from training — the
  existing acceleration-based smoothness reward (`src/rewards.py`) already
  covers that, on a genuinely GT-free basis; this discriminator now covers
  single-pose plausibility only, without overlapping it.

* Pose-only, deliberately. `trans` is SMPLer-X's non-metric virtual-camera
  depth (see `src/models/policy.py`'s `§2` notes), and GT `trans` lives in a
  different space entirely (metres, real camera). Mixing it in would hand the
  discriminator a trivially separable but meaningless feature — "is this
  GT-shaped trans or SMPLer-X-shaped trans" — which is exactly the shortcut a
  GAN discriminator finds first, and it would saturate the discriminator
  (and starve the policy's reward) long before it learned anything about
  motion plausibility.
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.datasets import gt_group

# Body joints only. The 30 finger joints are gone from the data
# (data/norm_upsample.py): MoVi has no finger mocap, so in GT they normalised to
# exactly +/-1 while SMPLer-X's genuine finger predictions never did, which let a
# discriminator separate real from fake at 100% by reading 90 dimensions that
# contain no motion at all.
N_JOINTS = 22
POSE_DIM = N_JOINTS * 3   # 66
WINDOW = 1                # frames per discriminator input; see module docstring


def _mlp(dims: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """Plain MLP, no dropout — see `src/models/policy.py::_mlp` for why dropout
    has no place in a network whose output feeds a policy's reward: it would
    make the discriminator's score for the *same* transition change from call
    to call, i.e. it would inject noise into the reward on top of whatever
    noise skrl's own train/eval mode switching would already introduce."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class MotionDiscriminator(nn.Module):
    """
    Binary classifier over `WINDOW`-frame pose-only windows (`WINDOW=1`: a
    single pose; see module docstring for why this isn't 2 consecutive frames
    right now).

    Input:  (..., WINDOW * POSE_DIM) — normalised pose(s), flattened and
            concatenated in time order. Build one with `make_window`.
    Output: (...,) raw logits. No sigmoid inside: the training loss applies
            `F.binary_cross_entropy_with_logits` and the reward applies
            `sigmoid` itself, and keeping the head linear is what lets the
            gradient penalty in `discriminator_step` act on an unsquashed
            score.
    """

    def __init__(self, window: int = WINDOW, hidden_dims: tuple[int, ...] = (256, 128),
                 pose_dim: int = POSE_DIM):
        super().__init__()
        self.window = window
        self.pose_dim = pose_dim
        self.net = _mlp([window * pose_dim, *hidden_dims, 1])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        # Small last-layer gain: an untrained discriminator should start near
        # logit 0 (score ~0.5 for everything), not confidently right or wrong
        # about real vs. fake before it has seen a single batch — a large
        # initial logit either saturates the BCE loss's gradient or hands the
        # policy a strong, meaningless reward signal from step one.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.net(windows).squeeze(-1)


def make_window(*poses: torch.Tensor) -> torch.Tensor:
    """
    Concatenate one or more `(..., 22, 3)` (or already-flat `(..., 66)`)
    pose tensors into one `(..., WINDOW*POSE_DIM)` discriminator input,
    oldest first. With `WINDOW=1` this is called with a single pose and is
    just a flatten; kept general so a wider window is a one-line call-site
    change later (see module docstring).
    """
    flat = [p.flatten(-2) if p.dim() >= 2 and p.shape[-1] == 3 else p for p in poses]
    return torch.cat(flat, dim=-1)


# ─── Real-motion data ───────────────────────────────────────────────────────

def load_gt_transitions(h5_path: str, split: str = "train",
                        device: str = "cpu") -> torch.Tensor:
    """
    Every individual GT pose in `split`, as one `(N, POSE_DIM)` tensor — the
    discriminator's "real" distribution (named `_transitions` for continuity
    with the windowed version this may become again; with `WINDOW=1` there is
    no transition, just a pose).

    Read directly from the HDF5 file's `gt` groups rather than through
    `MoViDataset`: GT does not depend on which camera lifted a clip, so
    iterating `dataset.samples` (one entry per (clip, camera)) would count
    every clip with two cameras twice and bias the real distribution toward
    it. Iterating the file's clip groups directly counts each clip once.
    """
    poses_per_clip = []
    with h5py.File(h5_path, "r") as f:
        for clip_name in f[split].keys():
            grp = gt_group(f[split][clip_name])
            poses_per_clip.append(grp["poses"][:].astype(np.float32).reshape(-1, POSE_DIM))
    if not poses_per_clip:
        raise ValueError(f"no GT poses found under {h5_path}:{split}")
    all_poses = np.concatenate(poses_per_clip, axis=0)
    return torch.from_numpy(all_poses).to(device)


# ─── Training step ──────────────────────────────────────────────────────────

def _r1_gradient_penalty(discriminator: MotionDiscriminator, real: torch.Tensor) -> torch.Tensor:
    """
    Zero-centred gradient penalty on real samples only (Mescheder et al. 2018;
    the same regulariser AMP, Peng et al. 2021, uses for exactly this kind of
    motion discriminator).

    Without it, a discriminator this small tends to separate real from fake
    almost perfectly within a handful of rollouts — GT motion and an
    early-training policy's output really are easy to tell apart — after
    which `sigmoid(D(fake))` saturates near 0 nearly everywhere and the GAIL
    reward stops carrying gradient just when the policy most needs it. The
    penalty discourages the discriminator from getting *too* confident by
    keeping its output smooth (small gradient norm) around real data.
    """
    real = real.detach().requires_grad_(True)
    logits = discriminator(real)
    grad, = torch.autograd.grad(outputs=logits.sum(), inputs=real,
                                create_graph=True, retain_graph=True)
    return grad.pow(2).sum(dim=-1).mean()


def discriminator_step(
    discriminator: MotionDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    grad_penalty_weight: float = 5.0,
) -> tuple[torch.Tensor, dict]:
    """
    One discriminator loss: binary cross-entropy (real=1, fake=0) plus an
    optional R1 gradient penalty. Does not step the optimizer — the caller
    (`src/gail_train.py::PPOWithGAIL._update_discriminator`) owns that, so it
    can also own gradient clipping / logging without this function reaching
    into skrl's Agent internals.

    Returns `(loss, logs)`; `logs` are plain floats, ready for
    `Agent.track_data`. `acc_real`/`acc_fake` are exactly the "scored on
    accuracy" signal — the fraction of each batch the discriminator currently
    classifies correctly — and are the number to watch for the failure mode
    the gradient penalty above guards against: both pinned at 1.0 for many
    rollouts in a row means the discriminator has stopped giving the policy
    anything to climb.
    """
    fake = fake.detach()
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)

    loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    loss = loss_real + loss_fake

    gp = torch.zeros((), device=real.device)
    if grad_penalty_weight > 0:
        gp = _r1_gradient_penalty(discriminator, real)
        loss = loss + grad_penalty_weight * gp

    logs = {
        "loss":       loss.item(),
        "loss_real":  loss_real.item(),
        "loss_fake":  loss_fake.item(),
        "grad_penalty": gp.item(),
        "acc_real":   (real_logits > 0).float().mean().item(),
        "acc_fake":   (fake_logits < 0).float().mean().item(),
    }
    return loss, logs


# ─── Reward, for the env ─────────────────────────────────────────────────────

class GAILRewardProvider:
    """
    The live handle `src/gail_env.py::GymMoviEnv` calls every step to score
    the current corrected pose against the discriminator currently being
    trained.

    This is deliberately *not* a snapshot: `discriminator`'s weights are
    updated in place by `PPOWithGAIL._update_discriminator` after each
    rollout, and because the env holds this same `GAILRewardProvider` instance
    (not a copy of its score at construction time), the very next step already
    sees the new weights. That is the whole answer to "the reward changes
    throughout training" — no channel back into the env is needed, only a
    reference held once at construction.

    One rollout's worth of steps is therefore always scored by the
    discriminator as of the *previous* rollout's update (the discriminator is
    trained on this rollout's data only after it has been collected). That
    lag is standard for this kind of concurrently-trained reward (it is how
    AMP and vanilla GAIL both operate) and is not a bug to fix.
    """

    def __init__(self, discriminator: MotionDiscriminator, device: str = "cpu"):
        self.discriminator = discriminator
        self.device = torch.device(device)

    @torch.no_grad()
    def score(self, pose: torch.Tensor) -> float:
        """Probability, under the current discriminator, that `pose` is a
        real GT pose. Bounded to (0, 1), so it stacks with the
        reprojection/smoothness terms (`src/rewards.py`) on the same rough
        scale without needing its own `reward_scale`."""
        was_training = self.discriminator.training
        self.discriminator.eval()
        window = make_window(pose).reshape(1, -1).to(self.device)
        reward = torch.sigmoid(self.discriminator(window)).item()
        self.discriminator.train(was_training)
        return reward
