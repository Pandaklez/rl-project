"""
render_smplx.py
───────────────
Render one clip from lifted_movi_smplx.h5 as a video using the
pyrender-based approach from Meta's Embody3D (facebookresearch/embody-3d).

Usage:
    python render_smplx.py \
        --h5        lifted_movi_smplx.h5 \
        --clip      train/Subject_66__crawling \
        --norm_json normalization.json \
        --out       renders/Subject_66__crawling.mp4
"""
from __future__ import annotations

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"   # headless EGL — must be set before any GL import

import argparse
import json
from pathlib import Path

import glob
import re

import h5py
import numpy as np
import pyrender
import scipy.io as sio
import torch
import trimesh
from pyrender.constants import RenderFlags
import cv2
import smplx

SMPLX_MODEL_PATH = Path(__file__).parent / \
    "smpler-x-main/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
FPS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Renderer  (adapted from Meta Embody3D src/visualize.py)
# ─────────────────────────────────────────────────────────────────────────────

class SMPLXRenderer:
    def __init__(self, faces: np.ndarray, height: int = 512, width: int = 512):
        self.faces = faces[:, [0, 2, 1]]   # flip winding order (Embody3D convention)
        self.H = height
        self.W = width
        self._setup_scene()

    # --- helpers from Embody3D ------------------------------------------

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / torch.linalg.norm(x)

    @staticmethod
    def _viewmatrix(center, up, pos) -> torch.Tensor:
        lookat = center - pos
        vec2 = SMPLXRenderer._normalize(lookat)
        vec1_avg = SMPLXRenderer._normalize(up)
        vec0 = SMPLXRenderer._normalize(torch.cross(vec1_avg, vec2))
        vec1 = SMPLXRenderer._normalize(torch.cross(vec2, vec0))
        return torch.stack([vec0, -vec1, -vec2, pos], dim=1)

    def _setup_scene(self):
        self.scene = pyrender.Scene(
            ambient_light=np.array([0.4, 0.4, 0.4]),
            bg_color=(0.1, 0.1, 0.1),
        )

        # Camera — body is centred at [0, ~0.9, 0]; camera sits 4 m in front
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), aspectRatio=self.W / self.H)
        at  = torch.tensor([0.0, 0.9, 0.0])   # mid-torso height
        pos = torch.tensor([0.0, 0.9, 4.0])   # same height, 4 m away
        up  = torch.tensor([0.0, 1.0, 0.0])
        RT  = self._viewmatrix(at, up, pos)
        RT  = torch.cat([RT, torch.tensor([[0, 0, 0, 1]])], dim=0).numpy()
        self.scene.add(camera, pose=RT)

        # Lights
        for light_pos, intensity in [
            ([0, 3, 4],  4.0),
            ([3, 1, 1],  2.0),
            ([-3, 1, 1], 1.5),
        ]:
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
            lp = torch.tensor(light_pos, dtype=torch.float)
            LRT = self._viewmatrix(at, up, lp)
            LRT = torch.cat([LRT, torch.tensor([[0, 0, 0, 1]])], dim=0).numpy()
            self.scene.add(light, pose=LRT)

        self.renderer = pyrender.OffscreenRenderer(self.W, self.H)
        self.flags = RenderFlags.SHADOWS_DIRECTIONAL

    def render_frame(self, verts: np.ndarray) -> np.ndarray:
        """Render one frame.  verts: (V, 3) in metres.  Returns (H, W, 3) uint8."""
        v = verts.copy()

        vertex_colors = np.tile([0.4, 0.7, 1.0, 1.0], (len(v), 1))
        mesh_tri = trimesh.Trimesh(vertices=v, faces=self.faces,
                                   vertex_colors=vertex_colors, process=False)
        trimesh.repair.fix_normals(mesh_tri)
        mesh_node = self.scene.add(pyrender.Mesh.from_trimesh(mesh_tri, smooth=True))

        rgb, _ = self.renderer.render(self.scene, flags=self.flags)
        self.scene.remove_node(mesh_node)
        return rgb  # (H, W, 3) uint8

    def render_clip(self, verts_seq: np.ndarray) -> np.ndarray:
        """Render a full clip.  verts_seq: (T, V, 3).  Returns (T, H, W, 3) uint8."""
        frames = [self.render_frame(verts_seq[t]) for t in range(len(verts_seq))]
        return np.stack(frames, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x * np.where(sigma < 1e-8, 1.0, sigma) + mu).astype(np.float32)


def load_clip(h5_path: str, clip_path: str, norm_json: str | None):
    """Load and denormalize one clip from the H5.  clip_path e.g. 'train/Subject_66__crawling'."""
    with h5py.File(h5_path, "r") as f:
        poses = f[clip_path]["poses"][:]   # (T, 52, 3)
        trans = f[clip_path]["trans"][:]   # (T, 3)
        betas = f[clip_path]["betas"][:]   # (16,)
        attrs = dict(f[clip_path].attrs)

    if norm_json:
        with open(norm_json) as fj:
            nj = json.load(fj)
        mu_p  = np.array(nj["poses"]["mu"],   dtype=np.float32)   # (52, 3)
        sig_p = np.array(nj["poses"]["sigma"], dtype=np.float32)
        mu_t  = np.array(nj["trans"]["mu"],   dtype=np.float32)
        sig_t = np.array(nj["trans"]["sigma"], dtype=np.float32)
        mu_b  = np.array(nj["betas"]["mu"],   dtype=np.float32)
        sig_b = np.array(nj["betas"]["sigma"], dtype=np.float32)
        poses = denormalize(poses, mu_p, sig_p)
        trans = denormalize(trans, mu_t, sig_t)
        betas = denormalize(betas, mu_b, sig_b)

    return poses, trans, betas, attrs


def build_smplx_vertices(poses: np.ndarray, trans: np.ndarray,
                         betas: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run SMPL-X forward pass for all T frames.  Returns (T, V, 3)."""
    T = poses.shape[0]
    model = smplx.create(
        str(SMPLX_MODEL_PATH),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=16,
        batch_size=T,
    ).to(device)

    def t(x): return torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(
            global_orient     = t(poses[:, 0:1, :].reshape(T, 3)),
            body_pose         = t(poses[:, 1:22, :].reshape(T, 63)),
            left_hand_pose    = t(poses[:, 22:37, :].reshape(T, 45)),
            right_hand_pose   = t(poses[:, 37:52, :].reshape(T, 45)),
            betas             = t(np.tile(betas, (T, 1))),
            transl            = t(trans),
        )
    return out.vertices.cpu().numpy()   # (T, 10475, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Original-video frame loader
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.flat[0]
    return x


def get_action_interval(v3d_dir: str, subject_id: int, action: str):
    """Return (start, end) 0-based frame indices from the v3d mat for a given action."""
    path = Path(v3d_dir) / f"F_v3d_Subject_{subject_id}.mat"
    mat  = sio.loadmat(str(path), struct_as_record=False, squeeze_me=False)
    top  = next(k for k in mat if not k.startswith("__"))
    subj = _unwrap(mat[top])
    move = _unwrap(subj.move)
    names = [str(_unwrap(a[0])).strip("['\\n ]") for a in move.motions_list]
    inds  = [[int(t[0]), int(t[1])] for t in move.flags30]
    action_norm = action.replace(" ", "_").replace("/", "_")
    for name, (s, e) in zip(names, inds):
        if name.replace(" ", "_").replace("/", "_") == action_norm:
            return s, e
    raise ValueError(f"Action '{action}' not found for subject {subject_id}")


def load_original_frames(img_dir: str, start: int, end: int, height: int) -> np.ndarray:
    """
    Load frames [start, end) (0-based) from an extracted-frames directory.
    jpgs are named %06d.jpg (1-based), so frame index N → file N+1.
    Returns (T, height, W, 3) uint8 resized to target height.
    """
    frames = []
    for fi in range(start, end):
        path = Path(img_dir) / f"{fi + 1:06d}.jpg"
        if not path.exists():
            # fill with black if frame is missing
            if frames:
                frames.append(np.zeros_like(frames[-1]))
            continue
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        new_w = int(w * height / h)
        img = cv2.resize(img, (new_w, height))
        frames.append(img)
    return np.stack(frames, axis=0)   # (T, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",        required=True,  help="path to lifted_movi_smplx.h5")
    parser.add_argument("--clip",      required=True,  help="clip path inside H5, e.g. train/Subject_66__crawling")
    parser.add_argument("--norm_json", default=None,   help="normalization.json (required if H5 is z-scored)")
    parser.add_argument("--out",       required=True,  help="output .mp4 path")
    parser.add_argument("--height",    type=int, default=512)
    parser.add_argument("--width",     type=int, default=512)
    parser.add_argument("--device",    default="cpu")
    # side-by-side comparison
    parser.add_argument("--img_dir",   default=None,
                        help="extracted frames dir for original video (enables side-by-side)")
    parser.add_argument("--v3d_dir",   default=None,
                        help="dir with F_v3d_Subject_X.mat files (needed to find action interval)")
    args = parser.parse_args()

    print(f"Loading clip: {args.clip}")
    poses, trans, betas, attrs = load_clip(args.h5, args.clip, args.norm_json)
    T = poses.shape[0]
    action    = attrs.get("action", "")
    subject   = attrs.get("subject", "")
    print(f"  T={T}  action={action}  subject={subject}")

    print("Running SMPL-X forward pass...")
    verts = build_smplx_vertices(poses, trans, betas, device=args.device)
    print(f"  verts: {verts.shape}")

    # Pin root to [0, mean_y, 0] every frame: strips noisy x/z camera-space drift
    # while keeping the body at a consistent floor height.
    mean_y = float(verts[:, 0, 1].mean())
    root_anchor = np.array([0.0, mean_y, 0.0], dtype=np.float32)
    verts = verts - verts[:, 0:1, :] + root_anchor[None, None, :]

    print("Rendering SMPL-X frames...")
    smplx_faces = smplx.create(
        str(SMPLX_MODEL_PATH), model_type="smplx", gender="neutral",
        use_pca=False, num_betas=16, batch_size=1).faces
    renderer = SMPLXRenderer(faces=smplx_faces, height=args.height, width=args.width)
    render_frames = renderer.render_clip(verts)   # (T, H, W, 3) uint8

    # ── side-by-side ────────────────────────────────────────────────────────
    if args.img_dir and args.v3d_dir:
        subj_id = int(re.search(r"(\d+)", subject).group(1))
        start, end = get_action_interval(args.v3d_dir, subj_id, action)
        print(f"Loading original frames [{start}, {end}) from {args.img_dir}")
        orig_frames = load_original_frames(args.img_dir, start, end, args.height)
        n = min(len(orig_frames), len(render_frames))
        combined = np.concatenate([orig_frames[:n], render_frames[:n]], axis=2)  # (T, H, W*2, 3)
        out_w = combined.shape[2]
    else:
        combined = render_frames
        out_w    = args.width

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (out_w, args.height))
    for frame in combined:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
