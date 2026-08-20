"""
render_gt_amass.py
──────────────────
Render ground-truth clips from Gmovi.h5 using the AMASS/SMPLH body model.
Follows the official AMASS tutorial style (human_body_prior BodyModel).

Model files expected at:
  common/utils/human_model_files/smpl/smplh/{female,male,neutral}/model.npz

Usage:
    # list all clips
    python render_gt_amass.py --list

    # render one clip
    python render_gt_amass.py --clip test/Subject_66__crawling --out renders/gt_crawling.mp4

    # pick gender and device
    python render_gt_amass.py --clip test/Subject_66__crawling \
        --gender female --device cuda --out renders/gt_crawling.mp4
"""
from __future__ import annotations

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"   # headless EGL – must be set before any GL import

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
import pyrender
from pyrender.constants import RenderFlags
import cv2

# ── monkey-patch: human_body_prior calls lbs(…, dtype=…) but newer smplx
#    dropped that kwarg.  Remove it before forwarding. ─────────────────────
import human_body_prior.body_model.body_model as _bm_mod
from smplx.lbs import lbs as _smplx_lbs

def _lbs_compat(**kwargs):
    kwargs.pop("dtype", None)
    return _smplx_lbs(**kwargs)

_bm_mod.lbs = _lbs_compat
# ─────────────────────────────────────────────────────────────────────────────

from human_body_prior.body_model.body_model import BodyModel

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
H5_PATH     = ROOT / "Gmovi.h5"
SMPLH_ROOT  = ROOT / "common/utils/human_model_files/smpl/smplh"
RENDERS_DIR = ROOT / "renders"

FPS        = 30
SUBSAMPLE  = 4        # Gmovi is 120 fps → keep every 4th frame → 30 fps
NUM_BETAS  = 16


# ── body model ────────────────────────────────────────────────────────────────

def load_body_model(gender: str = "female", batch_size: int = 1,
                    device: str = "cpu") -> BodyModel:
    bm_path = str(SMPLH_ROOT / gender / "model.npz")
    bm = BodyModel(bm_path, model_type="smplh", num_betas=NUM_BETAS,
                   batch_size=batch_size).to(device)
    return bm


def smplh_forward(poses: np.ndarray, trans: np.ndarray,
                  betas: np.ndarray, gender: str = "female",
                  device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run SMPLH forward pass.

    poses : (T, 52, 3)  axis-angle per joint
    trans : (T, 3)
    betas : (16,)
    Returns: verts (T, 6890, 3), faces (13776, 3)
    """
    T = poses.shape[0]
    bm = load_body_model(gender=gender, batch_size=T, device=device)

    def t(x): return torch.tensor(x, dtype=torch.float32, device=device)

    # Gmovi pose layout: [root(1), body(21), lhand(15), rhand(15)] = 52 joints
    root_orient = t(poses[:, 0, :])                         # (T, 3)
    pose_body   = t(poses[:, 1:22, :].reshape(T, 63))       # (T, 63)
    pose_lhand  = t(poses[:, 22:37, :].reshape(T, 45))      # (T, 45)
    pose_rhand  = t(poses[:, 37:52, :].reshape(T, 45))      # (T, 45)
    pose_hand   = torch.cat([pose_lhand, pose_rhand], dim=1) # (T, 90)
    betas_t     = t(np.tile(betas, (T, 1)))                  # (T, 16)
    trans_t     = t(trans)                                   # (T, 3)

    with torch.no_grad():
        body = bm(
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas_t,
            trans=trans_t,
        )

    verts = body.v.cpu().numpy()           # (T, 6890, 3)
    faces = bm.f.cpu().numpy().astype(np.int32)  # (13776, 3)
    return verts, faces


# ── data loading ──────────────────────────────────────────────────────────────

def list_clips() -> list[str]:
    clips = []
    with h5py.File(H5_PATH, "r") as f:
        def _collect(name, obj):
            if isinstance(obj, h5py.Group) and "poses" in obj:
                clips.append(name)
        f.visititems(_collect)
    return sorted(clips)


def load_clip(clip_path: str, subsample: int = SUBSAMPLE
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(H5_PATH, "r") as f:
        grp   = f[clip_path]
        poses = grp["poses"][::subsample].astype(np.float32)  # (T, 52, 3)
        trans = grp["trans"][::subsample].astype(np.float32)  # (T, 3)
        betas = grp["betas"][:].astype(np.float32)            # (16,)
    return poses, trans, betas


# ── renderer (pyrender) ───────────────────────────────────────────────────────

class SMPLHRenderer:
    def __init__(self, faces: np.ndarray, height: int = 512, width: int = 512):
        self.faces = faces
        self.H, self.W = height, width
        self._build_scene()

    @staticmethod
    def _norm(v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.norm(v)

    @staticmethod
    def _viewmatrix(center, up, pos) -> np.ndarray:
        lookat = center - pos
        z = SMPLHRenderer._norm(lookat)
        x = SMPLHRenderer._norm(torch.cross(SMPLHRenderer._norm(up), z))
        y = SMPLHRenderer._norm(torch.cross(z, x))
        RT = torch.stack([x, -y, -z, pos], dim=1)
        RT = torch.cat([RT, torch.tensor([[0., 0., 0., 1.]])], dim=0)
        return RT.numpy()

    def _build_scene(self):
        self.scene = pyrender.Scene(
            ambient_light=np.array([0.3, 0.3, 0.3]),
            bg_color=(0.15, 0.15, 0.15),
        )

        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(55),
                                            aspectRatio=self.W / self.H)
        at  = torch.tensor([0., 0.9, 0.])
        pos = torch.tensor([0., 0.9, 4.0])
        up  = torch.tensor([0., 1., 0.])
        self.scene.add(camera, pose=self._viewmatrix(at, up, pos))

        for lp, intens in [([0, 3, 4], 5.0), ([3, 1, 1], 2.5), ([-3, 1, 1], 2.0)]:
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=intens)
            lpos  = torch.tensor(lp, dtype=torch.float32)
            self.scene.add(light, pose=self._viewmatrix(at, up, lpos))

        self.renderer = pyrender.OffscreenRenderer(self.W, self.H)
        self.flags = RenderFlags.SHADOWS_DIRECTIONAL

    def render_frame(self, verts: np.ndarray,
                     color: tuple = (0.2, 0.75, 0.35)) -> np.ndarray:
        """Render a single frame. verts: (V, 3). Returns (H, W, 3) uint8."""
        r, g, b = color
        vcol  = np.tile([r, g, b, 1.0], (len(verts), 1))
        mesh  = trimesh.Trimesh(vertices=verts, faces=self.faces,
                                vertex_colors=vcol, process=False)
        trimesh.repair.fix_normals(mesh)
        node  = self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        rgb, _ = self.renderer.render(self.scene, flags=self.flags)
        self.scene.remove_node(node)
        return rgb

    def render_clip(self, verts_seq: np.ndarray,
                    color: tuple = (0.2, 0.75, 0.35)) -> np.ndarray:
        """Render all frames. verts_seq: (T, V, 3). Returns (T, H, W, 3)."""
        return np.stack([self.render_frame(verts_seq[i], color)
                         for i in range(len(verts_seq))], axis=0)


# ── helpers ───────────────────────────────────────────────────────────────────

def pin_root(verts: np.ndarray) -> np.ndarray:
    """Translate so the root vertex sits at the time-averaged y, x=z=0."""
    mean_y = verts[:, 0, 1].mean()
    anchor = np.array([0., mean_y, 0.], dtype=np.float32)
    return verts - verts[:, 0:1, :] + anchor[None, None, :]


def write_video(frames: np.ndarray, out_path: str, fps: int = FPS):
    T, H, W = frames.shape[:3]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render Gmovi.h5 GT clips using AMASS/SMPLH body model")
    parser.add_argument("--list",      action="store_true",
                        help="list all available clips and exit")
    parser.add_argument("--clip",      default="test/Subject_66__crawling",
                        help="clip path inside Gmovi.h5  (default: test/Subject_66__crawling)")
    parser.add_argument("--gender",    default="female", choices=["female", "male", "neutral"],
                        help="SMPLH gender model to use  (default: female)")
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE,
                        help=f"take every Nth frame, {SUBSAMPLE} → 120 fps to 30 fps  (default: {SUBSAMPLE})")
    parser.add_argument("--device",    default="cpu",
                        help="torch device  (default: cpu)")
    parser.add_argument("--height",    type=int, default=512)
    parser.add_argument("--width",     type=int, default=512)
    parser.add_argument("--out",       default=None,
                        help="output .mp4 path  (auto-named if omitted)")
    args = parser.parse_args()

    if args.list:
        clips = list_clips()
        print(f"Found {len(clips)} clips in {H5_PATH}:")
        for c in clips:
            print(" ", c)
        return

    # ── load data ────────────────────────────────────────────────────────────
    print(f"Loading clip: {args.clip}  (subsample={args.subsample})")
    poses, trans, betas = load_clip(args.clip, subsample=args.subsample)
    T = poses.shape[0]
    print(f"  T={T}  betas={betas[:4].round(3)}…")

    # ── body model forward ───────────────────────────────────────────────────
    print(f"Running SMPLH forward pass ({args.gender}, {args.device})…")
    verts, faces = smplh_forward(poses, trans, betas,
                                  gender=args.gender, device=args.device)

    # centre the animation so it stays in frame regardless of capture position
    verts = pin_root(verts)

    # ── render ───────────────────────────────────────────────────────────────
    print(f"Rendering {T} frames at {args.height}×{args.width}…")
    renderer = SMPLHRenderer(faces=faces, height=args.height, width=args.width)
    frames = renderer.render_clip(verts, color=(0.2, 0.75, 0.35))  # green for GT

    # ── write video ──────────────────────────────────────────────────────────
    if args.out is None:
        clip_name = args.clip.replace("/", "__")
        args.out = str(RENDERS_DIR / f"gt_{clip_name}.mp4")

    write_video(frames, args.out, fps=FPS)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
