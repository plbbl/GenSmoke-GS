#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders_dir", type=str, required=True, help="e.g. .../test/ours_30000/renders")
    ap.add_argument("--scene_lower", type=str, required=True, help="scene prefix, e.g. hinoki")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory for flat JPGs")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    r = Path(args.renders_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not r.is_dir():
        raise FileNotFoundError(r)

    for png in sorted(r.rglob("*.png")):
        stem = png.stem
        dst = out / f"{args.scene_lower}_{stem}.JPG"
        Image.open(png).convert("RGB").save(dst, quality=args.quality, format="JPEG")
        print(dst)


if __name__ == "__main__":
    main()
