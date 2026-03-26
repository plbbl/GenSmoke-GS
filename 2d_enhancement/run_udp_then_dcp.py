#!/usr/bin/env python3
"""UDPNet (RGB + depth2l) then DCP; writes <out>/<Scene>/train/*.png and optional transforms JSON."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

_BUNDLE_ROOT = os.path.dirname(os.path.abspath(__file__))
_UDPNET_ROOT = os.path.join(_BUNDLE_ROOT, "code", "UDPNet")
_DCP_ROOT = os.path.join(_BUNDLE_ROOT, "dcp_dehaze")

sys.path.insert(0, _UDPNET_ROOT)
sys.path.insert(0, _DCP_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import batch_dehaze as _batch_dcp  # noqa: E402
import cv2  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

from Dehazing.OTS.models.ConvIR_UDPNet import build_net as build_convir_udp  # noqa: E402
from Dehazing.OTS.models.FSNet_UDPNet import build_net as build_fsnet_udp  # noqa: E402


def _collect_scenes(roots: List[str]) -> Dict[str, str]:
    scene_sources: Dict[str, str] = {}
    for root in roots:
        root = os.path.abspath(root.rstrip("/\\"))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"scene root not found: {root}")
        for scene in sorted(os.listdir(root)):
            scene_path = os.path.join(root, scene)
            if not os.path.isdir(scene_path):
                continue
            train_dir = os.path.join(scene_path, "train")
            depth_dir = os.path.join(scene_path, "depth2l")
            if not os.path.isdir(train_dir) or not os.path.isdir(depth_dir):
                continue
            scene_sources[scene] = scene_path
    if not scene_sources:
        raise RuntimeError("no scenes with train/ and depth2l/")
    return scene_sources


def _rgb_paths_in_train(train_dir: str) -> List[str]:
    paths: List[str] = []
    for pat in ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        paths.extend(glob.glob(os.path.join(train_dir, pat)))
    return sorted(set(paths))


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, int]:
    ckpt_name = os.path.basename(ckpt_path).lower()
    if "fsnet" in ckpt_name:
        model = build_fsnet_udp()
        pad_factor = 8
    elif "convir" in ckpt_name:
        model = build_convir_udp()
        pad_factor = 32
    else:
        raise ValueError(f"checkpoint name must contain convir or fsnet: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, pad_factor


def restore_one(
    model: torch.nn.Module,
    pad_factor: int,
    rgb_path: str,
    depth_path: str,
    device: torch.device,
) -> Image.Image:
    rgb = TF.to_tensor(Image.open(rgb_path).convert("RGB"))
    dep = TF.to_tensor(Image.open(depth_path).convert("L"))
    x = torch.cat([rgb, dep], dim=0).unsqueeze(0).to(device)
    h, w = x.shape[-2:]
    H = ((h + pad_factor) // pad_factor) * pad_factor
    W = ((w + pad_factor) // pad_factor) * pad_factor
    padh = H - h if h % pad_factor != 0 else 0
    padw = W - w if w % pad_factor != 0 else 0
    if padh or padw:
        x = F.pad(x, (0, padw, 0, padh), mode="reflect")
    with torch.no_grad():
        out = model(x)[2][:, :, :h, :w]
    out = torch.clamp(out, 0.0, 1.0).squeeze(0).cpu()
    return TF.to_pil_image(out, mode="RGB")


def _load_dcp_params(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    required = (
        "tmin",
        "omega",
        "gain",
        "gamma",
        "amax",
        "w",
        "p",
        "r",
        "eps",
        "guided",
    )
    for k in required:
        if k not in d:
            raise KeyError(f"DCP params missing key: {k}")
    return d


def run_udp_stage(
    scene_sources: Dict[str, str],
    udp_out_root: str,
    ckpt_path: str,
    device: torch.device,
) -> int:
    model, pad_factor = load_model(ckpt_path, device)
    count = 0
    for scene, source_path in sorted(scene_sources.items()):
        train_dir = os.path.join(source_path, "train")
        depth_dir = os.path.join(source_path, "depth2l")
        out_dir = os.path.join(udp_out_root, scene)
        os.makedirs(out_dir, exist_ok=True)
        for rgb_path in _rgb_paths_in_train(train_dir):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(depth_dir, stem + ".png")
            if not os.path.isfile(depth_path):
                raise FileNotFoundError(f"missing depth: {depth_path}")
            out_img = restore_one(model, pad_factor, rgb_path, depth_path, device)
            out_img.save(os.path.join(out_dir, stem + ".png"))
            count += 1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return count


def run_dcp_stage(
    udp_dir_per_scene: str,
    final_parent: str,
    scene_names: List[str],
    dcp_base: dict,
    guided: bool,
    cli_overrides: dict,
) -> int:
    tmin = float(cli_overrides.get("tmin", dcp_base["tmin"]))
    omega = float(cli_overrides.get("omega", dcp_base["omega"]))
    gain = float(cli_overrides.get("gain", dcp_base["gain"]))
    gamma = float(cli_overrides.get("gamma", dcp_base["gamma"]))
    amax = int(cli_overrides.get("amax", dcp_base["amax"]))
    w = int(cli_overrides.get("w", dcp_base["w"]))
    p = float(cli_overrides.get("p", dcp_base["p"]))
    r = int(cli_overrides.get("r", dcp_base["r"]))
    eps = float(cli_overrides.get("eps", dcp_base["eps"]))

    n_ok = 0
    for scene in scene_names:
        in_dir = os.path.join(udp_dir_per_scene, scene)
        train_out = os.path.join(final_parent, scene, "train")
        os.makedirs(train_out, exist_ok=True)
        for name in sorted(os.listdir(in_dir)):
            _, ext = os.path.splitext(name)
            if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                continue
            src_path = os.path.join(in_dir, name)
            dst_path = os.path.join(train_out, name)
            bgr = cv2.imread(src_path)
            if bgr is None:
                print(f"[dcp skip] unreadable: {src_path}", file=sys.stderr)
                continue
            out = _batch_dcp.dehaze_bgr_uint8(
                bgr,
                tmin=tmin,
                Amax=amax,
                w=w,
                p=p,
                omega=omega,
                guided=guided,
                r=r,
                eps=eps,
                gain=gain,
                gamma=gamma,
            )
            if not cv2.imwrite(dst_path, out):
                print(f"[dcp fail] write: {dst_path}", file=sys.stderr)
                continue
            n_ok += 1
    return n_ok


def copy_transforms(scene_sources: Dict[str, str], final_parent: str) -> None:
    for scene, source_path in sorted(scene_sources.items()):
        dest_scene = os.path.join(final_parent, scene)
        os.makedirs(dest_scene, exist_ok=True)
        for fname in ("transforms_train.json", "transforms_test.json"):
            src = os.path.join(source_path, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_scene, fname))


def copy_udp_only_to_train(udp_dir_per_scene: str, final_parent: str, scene_names: List[str]) -> int:
    n = 0
    for scene in scene_names:
        in_dir = os.path.join(udp_dir_per_scene, scene)
        train_out = os.path.join(final_parent, scene, "train")
        os.makedirs(train_out, exist_ok=True)
        for name in sorted(os.listdir(in_dir)):
            src = os.path.join(in_dir, name)
            if not os.path.isfile(src):
                continue
            shutil.copy2(src, os.path.join(train_out, name))
            n += 1
    return n


def main() -> None:
    default_ckpt = os.path.join(_UDPNET_ROOT, "UDPNet_checkpoints", "ConvIR_UDPNet_ITS.ckpt")
    default_dcp_params = os.path.join(_DCP_ROOT, "params.json")

    ap = argparse.ArgumentParser(description="UDPNet then DCP batch pipeline.")
    ap.add_argument(
        "--scene_roots",
        nargs="+",
        required=True,
        help="parent dirs; each contains <Scene>/{train,depth2l}",
    )
    ap.add_argument(
        "--out_root",
        required=True,
        help="output dataset root: <out_root>/<Scene>/train/*.png",
    )
    ap.add_argument("--ckpt", default=default_ckpt, help="UDPNet .ckpt (convir or fsnet in filename)")
    ap.add_argument("--work_dir", default="", help="UDP cache root; default <out_root>/.udp_stage")
    ap.add_argument("--keep-work", action="store_true", help="keep UDP cache after success")
    ap.add_argument("--skip-dcp", action="store_true", help="UDP only -> train/")
    ap.add_argument("--dcp-params", default=default_dcp_params, help="DCP params.json")
    ap.add_argument("--no-guided", action="store_true", help="disable DCP guided filter")
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--amax", type=int, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    args = ap.parse_args()

    scene_sources = _collect_scenes(args.scene_roots)
    scene_names = sorted(scene_sources.keys())
    out_root = os.path.abspath(args.out_root)
    work_dir = args.work_dir.strip() or os.path.join(out_root, ".udp_stage")
    work_dir = os.path.abspath(work_dir)
    udp_scene_root = os.path.join(work_dir, "udp")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcp_base = _load_dcp_params(args.dcp_params)
    guided = bool(dcp_base["guided"]) and not args.no_guided
    cli_o = {k: v for k, v in {
        "tmin": args.tmin,
        "omega": args.omega,
        "amax": args.amax,
        "gain": args.gain,
        "gamma": args.gamma,
    }.items() if v is not None}

    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    n_udp = run_udp_stage(scene_sources, udp_scene_root, args.ckpt, device)
    print(f"[udp] {n_udp} images, {time.time() - t0:.2f}s, cache={udp_scene_root}")

    if args.skip_dcp:
        n_out = copy_udp_only_to_train(udp_scene_root, out_root, scene_names)
        print(f"[udp-only] copied {n_out} files -> train/")
    else:
        t1 = time.time()
        n_dcp = run_dcp_stage(udp_scene_root, out_root, scene_names, dcp_base, guided, cli_o)
        print(f"[dcp] {n_dcp} images, {time.time() - t1:.2f}s -> {out_root}")

    copy_transforms(scene_sources, out_root)

    if not args.keep_work and os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[clean] removed {work_dir}")
    elif args.keep_work:
        print(f"[keep] {work_dir}")

    print(f"done. next: GPT-Image-1.5 per image on {out_root}, then ../3d_reconstruction/train.sh <dataset_parent> with post-MLLM train/.")


if __name__ == "__main__":
    main()
