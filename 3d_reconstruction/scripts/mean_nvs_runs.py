#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _root() -> Path:
    e = os.environ.get("MCMCFASTER_ROOT", "").strip()
    if e:
        return Path(e).resolve()
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None, help="repo root (default: MCMCFASTER_ROOT or parent of scripts/)")
    ap.add_argument("--subdir-prefix", type=str, default="nvs_cap100000_g_resized_run", help="under output/")
    ap.add_argument("--run-first", type=int, required=True)
    ap.add_argument("--run-last", type=int, required=True)
    ap.add_argument("--out-name", type=str, required=True, help="subdir name under output/ for mean images")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    root = args.root.resolve() if args.root else _root()
    out_root = root / "output"
    rf, rl = args.run_first, args.run_last
    if rl < rf:
        print("ERROR: run-last < run-first", file=sys.stderr)
        sys.exit(1)

    def run_dir(i: int) -> Path:
        return out_root / f"{args.subdir_prefix}{i:02d}"

    ref = run_dir(rf)
    if not ref.is_dir():
        try:
            rshow = ref.relative_to(root)
        except ValueError:
            rshow = ref
        print(f"ERROR: missing {rshow}", file=sys.stderr)
        sys.exit(1)

    names = sorted(p.name for p in ref.iterdir() if p.suffix.upper() == ".JPG")
    if not names:
        print("ERROR: no JPG in reference run", file=sys.stderr)
        sys.exit(1)

    out_dir = out_root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    skip = 0
    for name in names:
        paths = [run_dir(r) / name for r in range(rf, rl + 1)]
        if not all(p.is_file() for p in paths):
            skip += 1
            continue
        arrs = []
        shape = None
        mismatch = False
        for p in paths:
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            if shape is None:
                shape = a.shape
            elif a.shape != shape:
                mismatch = True
                break
            arrs.append(a)
        if mismatch or len(arrs) != len(paths):
            skip += 1
            continue
        m = np.clip(np.mean(np.stack(arrs, axis=0), axis=0) + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(m).save(out_dir / name, quality=args.quality, format="JPEG")
        ok += 1
    try:
        oshow = out_dir.relative_to(root)
    except ValueError:
        oshow = out_dir
    print(f"written {ok} JPG -> {oshow} (skipped {skip})", flush=True)


if __name__ == "__main__":
    main()
