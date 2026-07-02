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

import re

import h5py
import numpy as np
import pyrender
import scipy.io as sio
from scipy.spatial.transform import Rotation as SciRot
import torch
import trimesh
from pyrender.constants import RenderFlags
import cv2
import smplx

SMPLX_MODEL_PATH = Path(__file__).parent / \
    "smpler-x-main/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
SMPLH_MODEL_PATH = Path(__file__).parent / \
    "common/utils/human_model_files/smpl/smplh/neutral/model.npz"
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

    def render_frame(self, verts: np.ndarray,
                     color: tuple[float, float, float] = (0.4, 0.7, 1.0)) -> np.ndarray:
        """Render one frame.  verts: (V, 3) in metres.  Returns (H, W, 3) uint8."""
        v = verts.copy()
        r, g, b = color
        vertex_colors = np.tile([r, g, b, 1.0], (len(v), 1))
        mesh_tri = trimesh.Trimesh(vertices=v, faces=self.faces,
                                   vertex_colors=vertex_colors, process=False)
        trimesh.repair.fix_normals(mesh_tri)
        mesh_node = self.scene.add(pyrender.Mesh.from_trimesh(mesh_tri, smooth=True))

        rgb, _ = self.renderer.render(self.scene, flags=self.flags)
        self.scene.remove_node(mesh_node)
        return rgb  # (H, W, 3) uint8

    def render_clip(self, verts_seq: np.ndarray,
                    color: tuple[float, float, float] = (0.4, 0.7, 1.0)) -> np.ndarray:
        """Render a full clip.  verts_seq: (T, V, 3).  Returns (T, H, W, 3) uint8."""
        frames = [self.render_frame(verts_seq[t], color) for t in range(len(verts_seq))]
        return np.stack(frames, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x * np.where(sigma < 1e-8, 1.0, sigma) + mu).astype(np.float32)


def load_clip(h5_path: str, clip_path: str, norm_json: str | None,
              pg: str | None = None, subsample: int = 1):
    """Load and denormalize one clip from the H5.  clip_path e.g. 'train/Subject_66__crawling'.
    If pg is given (e.g. 'PG1', 'PG2'), reads from that subgroup.
    subsample: take every Nth frame (e.g. 4 to go from 120fps → 30fps)."""
    with h5py.File(h5_path, "r") as f:
        grp   = f[clip_path][pg] if pg else f[clip_path]
        poses = grp["poses"][::subsample]   # (T, 52, 3)
        trans = grp["trans"][::subsample]   # (T, 3)
        betas = grp["betas"][:]             # (16,)
        # prefer subgroup attrs, fall back to clip-level
        attrs = dict(grp.attrs) if grp.attrs else dict(f[clip_path].attrs)

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


def align_gt_to_lifted(gt_poses: np.ndarray, gt_trans: np.ndarray,
                        lifted_poses: np.ndarray, lifted_trans: np.ndarray):
    """
    Align GT global_orient to lifted by computing R_align from the FIRST FRAME
    only (R_lifted_0 * R_gt_0^-1) and applying it as a fixed rotation to all
    frames.  Body pose (joints 1‥51) is unchanged.
    Returns (aligned_poses, aligned_trans).
    """
    R_gt_0     = SciRot.from_rotvec(gt_poses[0, 0, :])
    R_lifted_0 = SciRot.from_rotvec(lifted_poses[0, 0, :])
    R_align    = R_lifted_0 * R_gt_0.inv()

    new_global = (R_align * SciRot.from_rotvec(gt_poses[:, 0, :])).as_rotvec()
    new_poses  = gt_poses.copy()
    new_poses[:, 0, :] = new_global.astype(np.float32)

    gt_trans_rot = R_align.apply(gt_trans).astype(np.float32)
    offset       = lifted_trans[0] - gt_trans_rot[0]   # match first frame translation
    new_trans    = (gt_trans_rot + offset).astype(np.float32)

    return new_poses, new_trans


def _pin_root(verts: np.ndarray) -> np.ndarray:
    """Translate all frames so the root vertex sits at [0, mean_y, 0]."""
    mean_y = float(verts[:, 0, 1].mean())
    anchor = np.array([0.0, mean_y, 0.0], dtype=np.float32)
    return verts - verts[:, 0:1, :] + anchor[None, None, :]


def _load_smplh_struct():
    """
    Load model.npz and patch in zeroed hand-PCA fields (hands_componentsl/r,
    hands_meanl/r). This SMPL-H export was saved without them; smplx.SMPLH's
    __init__ reads those fields unconditionally even when use_pca=False, so
    stub them out (harmless since use_pca=False + flat_hand_mean=True mean
    they're never actually applied to the pose).
    """
    from smplx.utils import Struct
    raw = dict(np.load(str(SMPLH_MODEL_PATH), allow_pickle=True))
    raw.setdefault("hands_meanl", np.zeros(45, dtype=np.float32))
    raw.setdefault("hands_meanr", np.zeros(45, dtype=np.float32))
    raw.setdefault("hands_componentsl", np.zeros((45, 45), dtype=np.float32))
    raw.setdefault("hands_componentsr", np.zeros((45, 45), dtype=np.float32))
    return Struct(**raw)


def _create_body_model(model_type: str, num_betas: int, batch_size: int):
    """
    Instantiate the SMPL-X or SMPL-H body model.

    smplx.create() infers model_type from the model filename (expects
    SMPLX_*/SMPLH_* naming); our SMPL-H export is just "model.npz", so we
    construct SMPLH directly instead of going through create().
    """
    if model_type == "smplh":
        return smplx.SMPLH(
            model_path=str(SMPLH_MODEL_PATH),
            data_struct=_load_smplh_struct(),
            ext="npz",
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
            num_betas=num_betas,
            batch_size=batch_size,
        )
    return smplx.create(
        str(SMPLX_MODEL_PATH),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=num_betas,
        batch_size=batch_size,
    )


def build_smplx_vertices(poses: np.ndarray, trans: np.ndarray,
                         betas: np.ndarray, device: str = "cpu",
                         model_type: str = "smplx") -> np.ndarray:
    """Run SMPL-X/SMPL-H forward pass for all T frames.  Returns (T, V, 3)."""
    T = poses.shape[0]
    num_betas  = min(betas.shape[0], 10)  # smplx lib hard-caps old-format (<300-dim) models to 10 betas
    model = _create_body_model(model_type, num_betas, T).to(device)

    def t(x): return torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(
            global_orient     = t(poses[:, 0:1, :].reshape(T, 3)),
            body_pose         = t(poses[:, 1:22, :].reshape(T, 63)),
            left_hand_pose    = t(poses[:, 22:37, :].reshape(T, 45)),
            right_hand_pose   = t(poses[:, 37:52, :].reshape(T, 45)),
            betas             = t(np.tile(betas[:num_betas], (T, 1))),
            transl            = t(trans),
        )
    return out.vertices.cpu().numpy()   # (T, V, 3)


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
    parser.add_argument("--h5",           required=True,  help="lifted H5 (upd2, etc.)")
    parser.add_argument("--clip",         required=True,  help="clip path, e.g. train/Subject_66__crawling")
    parser.add_argument("--norm_json",    default=None,   help="normalization JSON for lifted H5 if z-scored")
    parser.add_argument("--pg",           default=None,   help="PG subgroup in lifted H5, e.g. PG2")
    parser.add_argument("--subsample",    type=int, default=1, help="subsample lifted frames (rarely needed)")
    parser.add_argument("--gt_h5",        default=None,   help="GT H5 (Gmovi.h5); enables 3-panel video|gt|lifted")
    parser.add_argument("--gt_norm_json", default=None,   help="normalization JSON for GT H5 if z-scored")
    parser.add_argument("--gt_subsample", type=int, default=4, help="subsample GT frames (4 for 120fps→30fps)")
    parser.add_argument("--out",          required=True,  help="output .mp4 path")
    parser.add_argument("--height",       type=int, default=512)
    parser.add_argument("--width",        type=int, default=512)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--model_type",   default="smplx", choices=["smplx", "smplh"],
                        help="Body model to render with. MoVi betas/poses were "
                             "fit for SMPL-H, so use smplh to check against the "
                             "model the data actually matches.")
    parser.add_argument("--zero_betas",   action="store_true",
                        help="Ignore betas from the data and use the model's "
                             "default neutral shape instead (diagnostic).")
    parser.add_argument("--img_dir",      default=None,   help="extracted video frames dir (enables side-by-side)")
    parser.add_argument("--v3d_dir",      default=None,   help="dir with F_v3d_Subject_X.mat files")
    args = parser.parse_args()

    print(f"Loading lifted clip: {args.clip}")
    lifted_poses, lifted_trans, lifted_betas, attrs = load_clip(
        args.h5, args.clip, args.norm_json, args.pg, args.subsample)
    T = lifted_poses.shape[0]
    action  = attrs.get("action", "")
    subject = attrs.get("subject", "")
    print(f"  lifted T={T}  action={action}  subject={subject}")

    if args.zero_betas:
        print("  --zero_betas set: using neutral shape instead of data betas")
        lifted_betas = np.zeros_like(lifted_betas)

    num_betas_faces = min(lifted_betas.shape[0], 10)
    body_faces = _create_body_model(args.model_type, num_betas_faces, 1).faces
    renderer = SMPLXRenderer(faces=body_faces, height=args.height, width=args.width)

    # ── lifted mesh ─────────────────────────────────────────────────────────
    print(f"Running {args.model_type.upper()} forward pass (lifted)...")
    lifted_verts = build_smplx_vertices(lifted_poses, lifted_trans, lifted_betas,
                                        device=args.device, model_type=args.model_type)
    lifted_verts = _pin_root(lifted_verts)

    print("Rendering lifted frames...")
    lifted_frames = renderer.render_clip(lifted_verts, color=(0.4, 0.7, 1.0))  # blue

    # ── GT mesh (optional) ──────────────────────────────────────────────────
    if args.gt_h5:
        print(f"Loading GT clip from {args.gt_h5} (subsample={args.gt_subsample})...")
        gt_poses, gt_trans, gt_betas, _ = load_clip(
            args.gt_h5, args.clip, args.gt_norm_json, None, args.gt_subsample)
        n = min(T, gt_poses.shape[0])
        print(f"  GT T={gt_poses.shape[0]} → trimmed to {n} to match lifted")
        gt_poses, gt_trans = gt_poses[:n], gt_trans[:n]
        lp, lt = lifted_poses[:n], lifted_trans[:n]

        print("Aligning GT orientation to lifted...")
        gt_poses_aligned, gt_trans_aligned = align_gt_to_lifted(
            gt_poses, gt_trans, lp, lt)

        print(f"Running {args.model_type.upper()} forward pass (GT)...")
        # GT betas are SMPL-H betas; only valid directly when rendering with SMPL-H.
        gt_betas_in = gt_betas if args.model_type == "smplh" else np.zeros(16, dtype=np.float32)
        gt_verts = build_smplx_vertices(gt_poses_aligned, gt_trans_aligned,
                                        gt_betas_in, device=args.device,
                                        model_type=args.model_type)
        gt_verts = _pin_root(gt_verts)

        print("Rendering GT frames...")
        gt_frames = renderer.render_clip(gt_verts, color=(0.2, 0.8, 0.3))  # green

        lifted_frames = lifted_frames[:n]
    else:
        gt_frames = None
        n = T

    # ── original video ──────────────────────────────────────────────────────
    orig_frames = None
    if args.img_dir and args.v3d_dir:
        subj_id = int(re.search(r"(\d+)", str(subject)).group(1))
        start, end = get_action_interval(args.v3d_dir, subj_id, action)
        print(f"Loading original frames [{start}, {end}) from {args.img_dir}")
        orig_frames = load_original_frames(args.img_dir, start, end, args.height)
        orig_frames = orig_frames[:n]

    # ── assemble panels ─────────────────────────────────────────────────────
    panels = []
    if orig_frames is not None:
        panels.append(orig_frames)
    if gt_frames is not None:
        panels.append(gt_frames)
    panels.append(lifted_frames)

    combined = np.concatenate(panels, axis=2)   # (T, H, sum_W, 3)
    out_w = combined.shape[2]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (out_w, args.height))
    for frame in combined:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
