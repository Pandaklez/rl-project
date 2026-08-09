"""
extract_2d.py
─────────────
Re-derive the two things the reprojection reward needs and that were lost when
the SMPLer-X `output/*/result/` directories were cleared:

  1. the per-frame person **bbox** (after `process_bbox`), which is what converts
     SMPLer-X's virtual-camera depth into metres, and
  2. per-frame **2D keypoints** (ViTPose-B, COCO-17) to reproject against.

Neither uses ground truth, so nothing here leaks GT into the reward.

Detection replicates the original pipeline exactly — same mmdet Faster R-CNN
checkpoint, same largest-box selection, same `process_bbox` (aspect 384/512,
ratio 1.25) — so the recovered bbox is the one that produced the lifted poses.
Videos are natively 30 fps, matching the `ffmpeg -vf fps=30/1` the original run
used, so frame indices line up 1:1.

Output HDF5:
    keypoints2d.h5
    └── PG1/Subject_12/
        ├── bbox    (N, 4) float32  xywh after process_bbox
        ├── score   (N,)   float32  detector confidence, 0 where nothing found
        └── kp2d    (N, 17, 3) float32  x, y (original image px), confidence

Usage:
    python scripts/extract_2d.py --out data/keypoints2d.h5
    python scripts/extract_2d.py --out /tmp/bench.h5 --max_videos 1 --max_frames 300
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
SMPLERX = REPO / "smpler-x-main"

# SMPLer-X crop geometry (config_smpler_x_b32.py:86-87)
INPUT_IMG_SHAPE = (512, 384)      # H, W
BBOX_RATIO = 1.25
# ViTPose-B input
VIT_H, VIT_W = 256, 192
N_KP = 17


# ─────────────────────────────────────────────────────────────────────────────
# ViTPose-B  (implemented directly; the installed mmpose is SMPLer-X's fork and
# does not import — transformer_utils/mmpose/models/detectors/poseur.py:13)
# ─────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, dim=768, heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTPose(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, n_kp=N_KP):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 16, 16)
        self.gh, self.gw = VIT_H // 16, VIT_W // 16          # 16 x 12
        self.pos_embed = nn.Parameter(torch.zeros(1, self.gh * self.gw + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.last_norm = nn.LayerNorm(dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.final = nn.Conv2d(256, n_kp, 1)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, C)
        x = x + self.pos_embed[:, 1:]                        # slot 0 is an unused cls token
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.gh, self.gw)
        return self.final(self.deconv(x))                    # (B, K, 64, 48)


def load_vitpose(ckpt: Path, device: str) -> ViTPose:
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    model = ViTPose()
    new = {}
    for k, v in sd.items():
        k = k.replace("backbone.", "")
        if k.startswith("patch_embed.proj."):
            new[k.replace("patch_embed.proj.", "patch_embed.")] = v
        elif k.startswith("blocks.") and ".attn.qkv." in k:
            new[k.replace(".attn.qkv.", ".attn.in_proj_")] = v
        elif k.startswith("blocks.") and ".attn.proj." in k:
            new[k.replace(".attn.proj.", ".attn.out_proj.")] = v
        elif k.startswith("blocks.") and ".mlp.fc1." in k:
            new[k.replace(".mlp.fc1.", ".mlp.0.")] = v
        elif k.startswith("blocks.") and ".mlp.fc2." in k:
            new[k.replace(".mlp.fc2.", ".mlp.2.")] = v
        elif k.startswith("keypoint_head.deconv_layers."):
            new[k.replace("keypoint_head.deconv_layers.", "deconv.")] = v
        elif k.startswith("keypoint_head.final_layer."):
            new[k.replace("keypoint_head.final_layer.", "final.")] = v
        else:
            new[k] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    if missing or unexpected:
        print(f"  vitpose load: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(f"    missing e.g. {missing[:4]}")
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# bbox handling — mirrors smpler-x-main/common/utils/preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_bbox(bbox, w_img, h_img):
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2 = min(w_img - 1, x1 + max(0, w - 1))
    y2 = min(h_img - 1, y1 + max(0, h - 1))
    if w * h > 0 and x2 > x1 and y2 > y1:
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    return None


def process_bbox(bbox, w_img, h_img, ratio=BBOX_RATIO):
    bbox = sanitize_bbox(bbox, w_img, h_img)
    if bbox is None:
        return None
    w, h = bbox[2], bbox[3]
    cx, cy = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    aspect = INPUT_IMG_SHAPE[1] / INPUT_IMG_SHAPE[0]
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    out = np.empty(4, dtype=np.float32)
    out[2], out[3] = w * ratio, h * ratio
    out[0], out[1] = cx - out[2] / 2.0, cy - out[3] / 2.0
    return out


def crop_for_vit(img, bbox):
    """Affine crop of `bbox` (xywh) to VIT_W x VIT_H, plus the inverse transform."""
    cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
    aspect = VIT_W / VIT_H
    w, h = bbox[2], bbox[3]
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    src = np.array([[cx, cy], [cx + w / 2, cy], [cx, cy + h / 2]], dtype=np.float32)
    dst = np.array([[VIT_W / 2, VIT_H / 2], [VIT_W, VIT_H / 2], [VIT_W / 2, VIT_H]], dtype=np.float32)
    fwd = cv2.getAffineTransform(src, dst)
    inv = cv2.getAffineTransform(dst, src)
    patch = cv2.warpAffine(img, fwd, (VIT_W, VIT_H), flags=cv2.INTER_LINEAR)
    return patch, inv


def decode_heatmaps(hm, invs):
    """argmax + quarter-offset refinement, mapped back to image coordinates."""
    b, k, hh, hw = hm.shape
    flat = hm.reshape(b, k, -1)
    idx = flat.argmax(axis=2)
    conf = flat.max(axis=2)
    ys, xs = np.divmod(idx, hw)
    coords = np.stack([xs, ys], axis=-1).astype(np.float32)
    for i in range(b):
        for j in range(k):
            x, y = int(xs[i, j]), int(ys[i, j])
            if 0 < x < hw - 1:
                coords[i, j, 0] += 0.25 * np.sign(hm[i, j, y, x + 1] - hm[i, j, y, x - 1])
            if 0 < y < hh - 1:
                coords[i, j, 1] += 0.25 * np.sign(hm[i, j, y + 1, x] - hm[i, j, y - 1, x])
    # heatmap (48x64) -> patch (192x256) -> image
    coords[..., 0] *= VIT_W / hw
    coords[..., 1] *= VIT_H / hh
    out = np.empty_like(coords)
    for i in range(b):
        m = invs[i]
        out[i, :, 0] = m[0, 0] * coords[i, :, 0] + m[0, 1] * coords[i, :, 1] + m[0, 2]
        out[i, :, 1] = m[1, 0] * coords[i, :, 0] + m[1, 1] * coords[i, :, 1] + m[1, 2]
    return np.concatenate([out, conf[..., None]], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def frame_mask(subject: str, v3d_dir: Path, n_frames: int):
    """
    Boolean mask of the frames any clip actually uses, from the v3d flags30
    ranges (the same metadata scripts/movi_smplx_processing.py reads). Frames
    between takes are never referenced by the lifted data, so detecting on them
    is wasted work. Returns None if the metadata is unavailable.
    """
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(v3d_dir / f"F_v3d_{subject}.mat"),
                          struct_as_record=False, squeeze_me=False)
        top = next(k for k in mat if not k.startswith("__"))
        s = mat[top]
        while isinstance(s, np.ndarray) and s.size == 1:
            s = s.flat[0]
        move = s.move
        while isinstance(move, np.ndarray) and move.size == 1:
            move = move.flat[0]
        mask = np.zeros(n_frames, dtype=bool)
        for a, b in move.flags30:
            mask[max(0, int(a) - 1): min(n_frames, int(b))] = True
        return mask
    except Exception as e:
        print(f"    (no flags30 for {subject}: {type(e).__name__}; processing all frames)")
        return None


def video_list(video_root: Path, cameras):
    out = []
    for cam in cameras:
        for p in sorted((video_root / f"{cam}_avi").glob(f"F_{cam}_Subject_*_L.avi")):
            subject = "Subject_" + p.stem.split("Subject_")[1].split("_")[0]
            out.append((cam, subject, p))
    return out


def process_video(path, detector, vit, device, batch, max_frames, inference_detector,
                  wanted=None):
    """
    `wanted` is an optional boolean mask over frame indices. Frames outside it are
    still decoded (cheap, ~1 ms) but skip detection, which is ~93% of the cost.
    """
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n = min(n, max_frames)
    bboxes = np.zeros((n, 4), np.float32)
    scores = np.zeros((n,), np.float32)
    kps = np.zeros((n, N_KP, 3), np.float32)

    buf_patch, buf_inv, buf_idx = [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def flush():
        if not buf_patch:
            return
        # Upload uint8 and do BGR->RGB + normalise on the GPU: ~10x cheaper than
        # building a float32 RGB copy on the host (0.35 -> 0.03 ms/frame).
        x = torch.from_numpy(np.stack(buf_patch)).to(device, non_blocking=True)
        x = x.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean) / std
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            hm = vit(x.contiguous()).float().cpu().numpy()
        dec = decode_heatmaps(hm, buf_inv)
        for slot, d in zip(buf_idx, dec):
            kps[slot] = d
        buf_patch.clear(); buf_inv.clear(); buf_idx.clear()

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if wanted is not None and i < len(wanted) and not wanted[i]:
            continue
        h_img, w_img = frame.shape[:2]
        det = inference_detector(detector, frame)
        person = det[0] if isinstance(det, (list, tuple)) else det
        if len(person) == 0:
            continue
        # same selection as inference.py: single largest-confidence person
        best = person[person[:, 4].argmax()]
        x1, y1, x2, y2, sc = best[:5]
        if (x2 - x1) < 50 or (y2 - y1) < 150:      # inference.py bbox_thr / bbox_thr*3
            continue
        bb = process_bbox(np.array([x1, y1, x2 - x1, y2 - y1]), w_img, h_img)
        if bb is None:
            continue
        bboxes[i], scores[i] = bb, sc
        patch, inv = crop_for_vit(frame, bb)
        buf_patch.append(patch); buf_inv.append(inv); buf_idx.append(i)
        if len(buf_patch) >= batch:
            flush()
    flush()
    cap.release()
    return bboxes, scores, kps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", default="demo/videos")
    ap.add_argument("--out", default="data/keypoints2d.h5")
    ap.add_argument("--vitpose", default="vitpose_base.pth")
    ap.add_argument("--cameras", nargs="+", default=["PG1", "PG2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--v3d_dir", default="F_Subjects_meta",
                    help="v3d .mat files, used to skip frames no clip references")
    ap.add_argument("--all_frames", action="store_true",
                    help="process every frame instead of only clip ranges")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    sys.path.insert(0, str(SMPLERX))
    from mmdet.apis import init_detector, inference_detector

    detector = init_detector(
        str(REPO / "pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py"),
        str(REPO / "pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"),
        device=args.device,
    )
    vit = load_vitpose(REPO / args.vitpose, args.device)

    vids = video_list(REPO / args.video_root, args.cameras)
    if args.max_videos:
        vids = vids[: args.max_videos]
    print(f"{len(vids)} videos -> {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done_frames, t0 = 0, time.time()
    with h5py.File(args.out, "a") as f:
        f.attrs["source"] = "mmdet faster_rcnn_r50_fpn + ViTPose-B (COCO-17)"
        f.attrs["note"] = ("bbox is post-process_bbox xywh, matching SMPLer-X inference; "
                           "kp2d is x,y in original image pixels plus confidence")
        for n_done, (cam, subject, path) in enumerate(vids, 1):
            key = f"{cam}/{subject}"
            if key in f:
                print(f"  [{n_done}/{len(vids)}] {key} already present, skipping")
                continue
            n_frames = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
            wanted = None if args.all_frames else frame_mask(
                subject, REPO / args.v3d_dir, n_frames)
            bb, sc, kp = process_video(path, detector, vit, args.device,
                                       args.batch, args.max_frames, inference_detector,
                                       wanted=wanted)
            g = f.create_group(key)
            for name, arr in (("bbox", bb), ("score", sc), ("kp2d", kp)):
                g.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
            g.attrs["video"] = path.name
            f.flush()
            done_frames += len(bb)
            el = time.time() - t0
            print(f"  [{n_done}/{len(vids)}] {key}: {len(bb)} frames, "
                  f"{(sc > 0).mean():.1%} detected | {done_frames/el:.1f} fps | "
                  f"eta {(len(vids)-n_done)*el/n_done/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
