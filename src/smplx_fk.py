"""
smplx_fk.py
───────────
Turn SMPL-X axis-angle pose parameters into 3D joint positions.

PA-MPJPE is a *position* error in metres, so it has to be measured on joints
produced by the body model — not on the axis-angle vectors themselves, which
do not live in a Euclidean space where distances or Procrustes alignment mean
anything.

Pose layout matches the HDF5 files written by scripts/movi_smplx_processing.py:

    poses[:,  0    ] -> global_orient   (3,)
    poses[:,  1:22 ] -> body_pose       (21, 3)
    poses[:, 22:37 ] -> left_hand_pose  (15, 3)
    poses[:, 37:52 ] -> right_hand_pose (15, 3)

Translation is deliberately zeroed: PA-MPJPE Procrustes-aligns before measuring,
so global position cancels anyway, and the lifted `trans` is virtual-camera depth
rather than metres (see scripts/fit_camera_offset.py) — feeding it in would only
inject noise.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent.parent / "common/utils/human_model_files/smplx"

# SMPL-X emits 127 joints; 0 is the pelvis and 1-21 are the body. The hand and
# face joints that follow are not part of a standard body PA-MPJPE.
N_BODY_JOINTS = 22
NUM_BETAS = 10


@lru_cache(maxsize=4)
def _body_model(gender: str, device: str):
    """Cached SMPL-X model. Loading the .npz costs ~100 MB, so build it once."""
    import smplx

    model = smplx.create(
        str(MODEL_DIR / f"SMPLX_{gender.upper()}.npz"),
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=NUM_BETAS,
        batch_size=1,
    )
    return model.to(device).eval()


@torch.no_grad()
def joints_from_poses(
    poses:      torch.Tensor | np.ndarray,
    betas:      torch.Tensor | np.ndarray | None = None,
    n_joints:   int = N_BODY_JOINTS,
    gender:     str = "neutral",
    device:     str = "cpu",
    chunk:      int = 512,
) -> torch.Tensor:
    """
    Run SMPL-X forward and return joint positions in metres.

    poses : (T, 52, 3) axis-angle
    betas : (16,) or (10,) shape coefficients, or None for the mean shape
    returns (T, n_joints, 3)

    Runs in chunks so a long clip does not allocate one huge batch. Every
    optional parameter is passed explicitly — this smplx build otherwise falls
    back to its stored batch_size=1 defaults and fails to broadcast.
    """
    poses = torch.as_tensor(np.asarray(poses), dtype=torch.float32)
    total = poses.shape[0]

    if betas is None:
        beta_vec = torch.zeros(NUM_BETAS, dtype=torch.float32)
    else:
        beta_vec = torch.as_tensor(np.asarray(betas), dtype=torch.float32)[:NUM_BETAS]

    model = _body_model(gender, device)
    out_chunks = []

    for start in range(0, total, chunk):
        p = poses[start : start + chunk].to(device)
        n = p.shape[0]
        zeros = lambda d: torch.zeros(n, d, device=device)  # noqa: E731

        out = model(
            global_orient   = p[:, 0],
            body_pose       = p[:, 1:22].reshape(n, -1),
            left_hand_pose  = p[:, 22:37].reshape(n, -1),
            right_hand_pose = p[:, 37:52].reshape(n, -1),
            betas           = beta_vec.to(device).expand(n, -1),
            transl          = zeros(3),
            jaw_pose        = zeros(3),
            leye_pose       = zeros(3),
            reye_pose       = zeros(3),
            expression      = zeros(model.num_expression_coeffs),
        )
        out_chunks.append(out.joints[:, :n_joints].cpu())

    return torch.cat(out_chunks)
