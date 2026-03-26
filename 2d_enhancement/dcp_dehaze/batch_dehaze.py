#!/usr/bin/env python3
"""Batch dark-channel-prior dehazing (CVPR 2009) with guided filter refinement."""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "dark-channel-prior-dehazing", "src")
sys.path.insert(0, _SRC)
from dehaze import dehaze_raw, get_radiance  # noqa: E402

_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}
_PARAMS_PATH = os.path.join(_ROOT, "params.json")


def _load_runtime_params() -> dict:
    with open(_PARAMS_PATH, encoding="utf-8") as f:
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
            raise KeyError(f"params.json missing key: {k}")
    return d


def dehaze_bgr_uint8(
    bgr: np.ndarray,
    *,
    tmin: float,
    Amax: int,
    w: int,
    p: float,
    omega: float,
    guided: bool,
    r: int,
    eps: float,
    gain: float,
    gamma: float,
) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    _, A, _, refined_t = dehaze_raw(
        rgb,
        tmin=tmin,
        Amax=Amax,
        w=w,
        p=p,
        omega=omega,
        guided=guided,
        r=r,
        eps=eps,
    )
    J = get_radiance(rgb, A, refined_t)
    if gain != 1.0:
        J = J * float(gain)
    if gamma > 0.0 and gamma != 1.0:
        J = np.clip(J / 255.0, 0.0, 1.0) ** float(gamma) * 255.0
    J = np.clip(J, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(J, cv2.COLOR_RGB2BGR)


def main() -> None:
    base = _load_runtime_params()

    ap = argparse.ArgumentParser(description="Batch DCP dehazing.")
    ap.add_argument("--input", required=True, help="Root directory of input images")
    ap.add_argument("--output", required=True, help="Root directory for outputs")
    ap.add_argument("--tmin", type=float, default=base["tmin"])
    ap.add_argument("--omega", type=float, default=base["omega"])
    ap.add_argument("--amax", type=int, default=base["amax"])
    ap.add_argument("--w", type=int, default=base["w"])
    ap.add_argument("--p", type=float, default=base["p"])
    ap.add_argument("--r", type=int, default=base["r"])
    ap.add_argument("--eps", type=float, default=base["eps"])
    ap.add_argument("--gain", type=float, default=base["gain"])
    ap.add_argument("--gamma", type=float, default=base["gamma"])
    ap.add_argument(
        "--no-guided",
        action="store_true",
        help="Disable guided filter (otherwise use params.json guided)",
    )
    args = ap.parse_args()
    guided = bool(base["guided"]) and not args.no_guided

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)

    if not os.path.isdir(input_root):
        sys.exit(f"Input directory not found: {input_root}")
    if not os.path.isdir(_SRC):
        sys.exit(f"Missing DCP sources: {_SRC}")

    n_ok = 0
    for dirpath, _, filenames in os.walk(input_root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            if ext not in _EXT:
                continue
            src_path = os.path.join(dirpath, name)
            rel = os.path.relpath(src_path, input_root)
            dst_path = os.path.join(output_root, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            bgr = cv2.imread(src_path)
            if bgr is None:
                print(f"[skip] cannot read: {src_path}", file=sys.stderr)
                continue

            out = dehaze_bgr_uint8(
                bgr,
                tmin=args.tmin,
                Amax=args.amax,
                w=args.w,
                p=args.p,
                omega=args.omega,
                guided=guided,
                r=args.r,
                eps=args.eps,
                gain=args.gain,
                gamma=args.gamma,
            )
            if not cv2.imwrite(dst_path, out):
                print(f"[fail] write failed: {dst_path}", file=sys.stderr)
                continue
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"processed {n_ok} images …")

    print(f"done: {n_ok} images -> {output_root}")


if __name__ == "__main__":
    main()
