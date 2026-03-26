"""Write depth2l/*.png under each <root>/<scenes_subdir>/<Scene>/ for UDPNet. Run from repo: cd code/Depth-Anything-V2."""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="parent of scenes_subdir")
    parser.add_argument("--scenes_subdir", type=str, default="test")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    test_root = os.path.join(args.root, args.scenes_subdir)
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Not found: {test_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt_path = os.path.join(_ROOT, "checkpoints", f"depth_anything_v2_{args.encoder}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model = model.to(device).eval()

    scene_dirs = [d for d in sorted(os.listdir(test_root)) if os.path.isdir(os.path.join(test_root, d))]
    total = 0
    for scene in scene_dirs:
        train_dir = os.path.join(test_root, scene, "train")
        depth_dir = os.path.join(test_root, scene, "depth2l")
        if not os.path.isdir(train_dir):
            continue
        os.makedirs(depth_dir, exist_ok=True)
        images = sorted([f for f in os.listdir(train_dir) if f.upper().endswith(".JPG")])
        for i, name in enumerate(images):
            stem = os.path.splitext(name)[0]
            depth_path = os.path.join(depth_dir, stem + ".png")
            if os.path.isfile(depth_path) and not args.overwrite:
                continue
            img_path = os.path.join(train_dir, name)
            raw = cv2.imread(img_path)
            if raw is None:
                print(f"Skip (read failed): {img_path}")
                continue
            depth = model.infer_image(raw, args.input_size)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            if depth.max() > depth.min():
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            else:
                depth = np.zeros_like(depth)
            depth = depth.astype(np.uint8)
            cv2.imwrite(depth_path, depth)
            total += 1
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{scene}] {i+1}/{len(images)} -> {depth_dir}")
    print(f"Done. Wrote {total} depth maps to depth2l/.")


if __name__ == "__main__":
    main()
