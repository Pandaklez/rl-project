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

from src.data.datasets import gt_group

# Body joints only. The 30 finger joints are gone from the data
# (data/norm_upsample.py): MoVi has no finger mocap, so in GT they normalised to
# exactly +/-1 while SMPLer-X's genuine finger predictions never did, which let a
# discriminator separate real from fake at 100% by reading 90 dimensions that
# contain no motion at all.
N_JOINTS = 22
POSE_DIM = N_JOINTS * 3   # 66
WINDOW = 1                # frames per discriminator input; see module docstring


# Least-squares GAN targets, AMP eq. 4 (Peng et al. 2021). Real is +1 and fake
# is **-1**, not 0: the reward below is built around a score that is symmetric
# about zero, and 0/1 targets would shift its whole operating range.
REAL_TARGET = 1.0
FAKE_TARGET = -1.0


def amp_reward(logits: torch.Tensor) -> torch.Tensor:
    """
    AMP eq. 5: `r = max(0, 1 - 0.25 * (D - 1)^2)`.

    This is the half of the fix that matters. With a sigmoid reward, a
    confident discriminator hands back ~0 for every fake pose the policy could
    produce, so `r_gail` is constant and carries no gradient exactly when the
    policy needs it — and on this data the discriminator reaches 97-99%
    accuracy within ten updates no matter how large the R1 penalty is, because
    SMPLer-X regresses toward the conditional mean and its pose distribution is
    genuinely narrower than GT's (per-joint sd ratios 0.12 to 3.22).

    The least-squares score is not squashed, so it keeps ranking fakes long
    after it could classify them perfectly. `r` peaks at 1 for `D = 1`
    (indistinguishable from GT), falls to 0 at `D = -1` (confidently fake), and
    is clamped at 0 beyond — bounded to [0, 1] like the reprojection and
    smoothness terms it is added to.
    """
    return torch.clamp(1.0 - 0.25 * (logits - REAL_TARGET).pow(2), min=0.0)


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
    Output: (...,) raw scores, linear head, no squashing. The least-squares
            loss in `discriminator_step` regresses them onto +1 / -1 and
            `amp_reward` turns them into the reward, so the score must stay
            unbounded — that is also what lets the gradient penalty act on an
            unsquashed quantity.
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
        # score 0 (`amp_reward` = 0.75 for everything, a constant the reward
        # baseline removes), not confidently right or wrong
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


# ─── Pose space ─────────────────────────────────────────────────────────────

class PoseSpace:
    """
    Puts the discriminator's real and fake samples in **one** space, and picks
    which joints it sees.

    The bug this exists to fix. `data/processed_movi.h5` stores both poses
    already normalised, but with *different* statistics: GT with the GT stats
    (`data/normalization.json`), lifted with the per-camera stats
    (`data/normalization_lifted_{pg1,pg2}.json`). Each therefore has mean 0 and
    sd 1 in its own space, so the marginals match and every separability test
    comes back clean — while the same coordinate denotes a **different physical
    pose** on each side, because the two affine maps differ (per-joint sd ratios
    run 0.12 to 3.22). A discriminator trained on that comparison is
    well-behaved and rewards the wrong pose: a policy satisfying it perfectly
    would be driven to a pose wrong by 0.088 rad RMS per axis-angle component
    (5.1 deg, against a GT spread of 12.6) and by 31 deg on the root.

    The fix maps the **fake** side into the GT-normalised space the real bank
    already lives in:

        physical = mu_lifted + sd_lifted * z_lifted        (per camera)
        z_gt     = (physical - mu_gt) / sd_gt

    Only the fake side moves, so the bank needs no per-camera duplication. The
    camera matters and is not optional: pg1 and pg2 have different stats, so
    before this the fake batch was itself a mixture of two spaces.

    `exclude_joints` drops whole joints from the discriminator's input.
    Joint 0 is excluded by default: it is `global_orient`, the body's
    orientation in the world, which is a property of where the subject happened
    to face and not of whether the pose is a plausible human one. It is also the
    single largest contributor to the bias above (13.6 deg of the 5.1 deg RMS,
    against 4.7 deg for the 21 body joints together). AMP-style discriminators
    conventionally score local pose for the same reason.

        space = PoseSpace()                      # 21 joints, 63 dims
        real  = space.real(gt_bank)              # already GT-space: select only
        fake  = space.fake(corrected, cameras)   # remap, then select
    """

    CAMERAS = ("pg1", "pg2")

    def __init__(self, gt_stats_path: str = "data/normalization.json",
                 lifted_stats_fmt: str = "data/normalization_lifted_{cam}.json",
                 exclude_joints: tuple[int, ...] = (0,),
                 n_joints: int = N_JOINTS, device: str = "cpu"):
        import json

        dev = torch.device(device)
        g = json.load(open(gt_stats_path))["poses"]
        self.mu_gt = torch.tensor(g["mu"], dtype=torch.float32, device=dev).reshape(-1)
        self.sd_gt = torch.tensor(g["sigma"], dtype=torch.float32, device=dev).reshape(-1)
        if self.mu_gt.numel() != n_joints * 3:
            raise ValueError(
                f"{gt_stats_path} has {self.mu_gt.numel() // 3} joints, expected {n_joints}")

        mus, sds = [], []
        for cam in self.CAMERAS:
            l = json.load(open(lifted_stats_fmt.format(cam=cam)))["poses"]
            mus.append(torch.tensor(l["mu"], dtype=torch.float32, device=dev).reshape(-1))
            sds.append(torch.tensor(l["sigma"], dtype=torch.float32, device=dev).reshape(-1))
        self.mu_lift = torch.stack(mus)          # (2, 66)
        self.sd_lift = torch.stack(sds)

        # Guard the divisions once, here, rather than at every call site.
        self.sd_gt = torch.where(self.sd_gt.abs() < 1e-8,
                                 torch.ones_like(self.sd_gt), self.sd_gt)
        self.sd_lift = torch.where(self.sd_lift.abs() < 1e-8,
                                   torch.ones_like(self.sd_lift), self.sd_lift)

        self.exclude_joints = tuple(sorted(set(exclude_joints)))
        keep = [j for j in range(n_joints) if j not in self.exclude_joints]
        if not keep:
            raise ValueError("exclude_joints removes every joint")
        self.keep_joints = tuple(keep)
        self.keep_idx = torch.tensor([j * 3 + a for j in keep for a in range(3)],
                                     dtype=torch.long, device=dev)
        self.device = dev

    @property
    def dim(self) -> int:
        """Discriminator input width after joint selection."""
        return self.keep_idx.numel()

    def _cam_index(self, camera) -> torch.Tensor:
        """Accept a name, an index, or a batch of either."""
        if isinstance(camera, str):
            return torch.tensor(self.CAMERAS.index(camera.lower()),
                                dtype=torch.long, device=self.device)
        if isinstance(camera, torch.Tensor):
            return camera.long().to(self.device)
        return torch.tensor(int(camera), dtype=torch.long, device=self.device)

    def select(self, pose: torch.Tensor) -> torch.Tensor:
        """Keep only the joints the discriminator scores."""
        return pose.index_select(-1, self.keep_idx)

    def real(self, gt_norm: torch.Tensor) -> torch.Tensor:
        """GT poses are already in the target space; only the selection applies."""
        return self.select(gt_norm.to(self.device))

    def fake(self, lifted_norm: torch.Tensor, camera) -> torch.Tensor:
        """Lifted-per-camera-normalised poses -> GT-normalised, then selected.

        `camera` is a name, an index, or a per-row index tensor broadcastable
        against `lifted_norm`'s leading dimensions.
        """
        z = lifted_norm.to(self.device)
        idx = self._cam_index(camera)
        mu_l = self.mu_lift[idx]
        sd_l = self.sd_lift[idx]
        physical = mu_l + sd_l * z
        return self.select((physical - self.mu_gt) / self.sd_gt)


# ─── Real-motion data ───────────────────────────────────────────────────────

def split_demo_clips(h5_path: str, split: str = "train", demo_frac: float = 0.2,
                     seed: int = 0) -> tuple[list[str], list[str]]:
    """
    Partition `split`'s clips into `(policy_clips, demo_clips)` — the disjoint
    demonstration set experiment (C) trains against.

    Why disjoint. The discriminator's real bank and the policy's rollouts were
    drawn from the same clips, so nothing stopped the discriminator from
    memorising a clip's GT poses and then rewarding the policy for reproducing
    the ground truth of the very clip it was correcting. That is not the paired
    per-frame leak `GymMoviEnv` guards against — the sampler never lines a real
    sample up with the fake one — but it is a route from a clip's label to that
    same clip's reward, and it is the one AMP and vanilla GAIL both close by
    keeping the demonstration set separate from what the agent acts on.

    The partition is **clip-level, not frame-level**: neighbouring frames of one
    clip are nearly identical, so splitting frames would put a near-duplicate of
    every held-out pose back in the policy's own set and close nothing.

    `seed` is deliberately its own knob, defaulting to 0 rather than following
    the run seed. A partition that moved with the run seed would change the
    training set between seeds, so the seed sweep would measure partition
    variance and policy variance together and could not separate them.
    """
    if not 0.0 <= demo_frac < 1.0:
        raise ValueError(f"demo_frac must be in [0, 1), got {demo_frac}")
    with h5py.File(h5_path, "r") as f:
        clips = sorted(f[split].keys())      # sorted: HDF5 key order is not guaranteed
    if not clips:
        raise ValueError(f"no clips found under {h5_path}:{split}")
    if demo_frac == 0.0:
        return clips, clips               # overlapping, the pre-existing behaviour
    n_demo = max(1, int(round(demo_frac * len(clips))))
    if n_demo >= len(clips):
        raise ValueError(
            f"demo_frac={demo_frac} would leave the policy no clips "
            f"({n_demo} of {len(clips)})")
    order = np.random.default_rng(seed).permutation(len(clips))
    demo = {clips[i] for i in order[:n_demo]}
    return [c for c in clips if c not in demo], [c for c in clips if c in demo]


def load_gt_transitions(h5_path: str, split: str = "train",
                        device: str = "cpu",
                        clips: list[str] | None = None) -> torch.Tensor:
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

    `clips` restricts the bank to a subset of the split — pass
    `split_demo_clips(...)[1]` for a demonstration set disjoint from the clips
    the policy rolls out. `None` uses every clip in the split.
    """
    wanted = None if clips is None else set(clips)
    poses_per_clip = []
    with h5py.File(h5_path, "r") as f:
        if wanted is not None:
            missing = wanted - set(f[split].keys())
            if missing:
                raise ValueError(
                    f"{len(missing)} requested clip(s) are not in {split}, "
                    f"e.g. {sorted(missing)[:3]}")
        for clip_name in f[split].keys():
            if wanted is not None and clip_name not in wanted:
                continue
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
    which it can classify perfectly. With AMP's least-squares reward that is
    survivable — `amp_reward` keeps ranking fakes after they are separable —
    but a very sharp discriminator still makes the reward jumpy. The
    penalty discourages the discriminator from getting *too* confident by
    keeping its output smooth (small gradient norm) around real data.

    **It delays that failure; it does not prevent it.** Measured on this data
    over the real 160-step schedule, going from weight 5 to 50 moves final
    accuracy 99.7% -> 98.6%, and 250 buys nothing further (see
    `Config.disc_grad_penalty`). The classes are simply far apart once they are
    in a common space, so treat this as a knob that buys a little headroom, not
    as the answer to a saturated discriminator.
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
    grad_penalty_weight: float = 50.0,
) -> tuple[torch.Tensor, dict]:
    """
    One discriminator loss: the **least-squares** objective of AMP eq. 4
    (real -> +1, fake -> -1) plus an optional R1 gradient penalty. Does not step the optimizer — the caller
    (`src/gail_train.py::PPOWithGAIL._update_discriminator`) owns that, so it
    can also own gradient clipping / logging without this function reaching
    into skrl's Agent internals.

    Returns `(loss, logs)`; `logs` are plain floats, ready for
    `Agent.track_data`. Zero is still the decision boundary (the targets are
    symmetric about it), so `acc_real`/`acc_fake` mean what they always did:
    they are the "scored on accuracy" signal — the fraction of each batch the discriminator currently
    classifies correctly — and are the number to watch for the failure mode
    the gradient penalty above guards against: both pinned at 1.0 for many
    rollouts in a row means the discriminator has stopped giving the policy
    anything to climb.
    """
    fake = fake.detach()
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)

    # Least-squares objective, AMP eq. 4 (Peng et al. 2021): real -> +1,
    # fake -> -1, squared error rather than cross-entropy. See the module
    # docstring for why this replaced BCE.
    loss_real = (real_logits - REAL_TARGET).pow(2).mean()
    loss_fake = (fake_logits - FAKE_TARGET).pow(2).mean()
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

    def __init__(self, discriminator: MotionDiscriminator, device: str = "cpu",
                 space: "PoseSpace | None" = None):
        self.discriminator = discriminator
        self.device = torch.device(device)
        # Without a space the corrected pose would be scored in lifted
        # per-camera units by a discriminator trained in GT units — see
        # `PoseSpace` for what that costs. Required rather than optional.
        if space is None:
            raise ValueError(
                "GAILRewardProvider needs a PoseSpace: the corrected pose is "
                "normalised in lifted per-camera space and the discriminator's "
                "real samples are in GT space, so it has to be mapped before "
                "scoring.")
        self.space = space

    @torch.no_grad()
    def score(self, pose: torch.Tensor, camera) -> float:
        """AMP's style reward for `pose` under the current discriminator:
        `max(0, 1 - 0.25*(D - 1)^2)`, 1 when the pose is indistinguishable from
        GT motion and 0 once the discriminator is confident it is not. Bounded
        to [0, 1], so it stacks with the reprojection/smoothness terms
        (`src/rewards.py`) on the same rough scale without needing its own
        `reward_scale` — and unlike a sigmoid it keeps ranking poses after the
        discriminator can already classify them perfectly, which on this data
        happens within ten updates (see `amp_reward`).

        `pose` is the env's corrected pose, i.e. **lifted per-camera
        normalised**; `camera` says which camera's statistics it carries so
        `PoseSpace.fake` can map it into the space the discriminator was
        trained in. The camera is not optional — pg1 and pg2 normalise
        differently.
        """
        was_training = self.discriminator.training
        self.discriminator.eval()
        window = self.space.fake(make_window(pose).reshape(1, -1), camera)
        reward = amp_reward(self.discriminator(window)).item()
        self.discriminator.train(was_training)
        return reward
