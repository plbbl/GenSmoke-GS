#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_colmap_loader():
    path = SCRIPT_DIR / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("colmap_loader_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cl = _load_colmap_loader()
read_extrinsics_binary = _cl.read_extrinsics_binary
read_points3D_binary = _cl.read_points3D_binary
qvec2rotmat = _cl.qvec2rotmat


def store_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def umeyama_similarity(src_pts: np.ndarray, dst_pts: np.ndarray):
    assert src_pts.shape == dst_pts.shape and src_pts.shape[1] == 3
    n = src_pts.shape[0]
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)
    src_c = src_pts - mu_src
    dst_c = dst_pts - mu_dst
    var_src = (src_c**2).sum() / n
    scale = np.sqrt((dst_c**2).sum() / n / (var_src + 1e-12))
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def colmap_camera_center(qvec, tvec):
    R_w2c = qvec2rotmat(qvec)
    return -R_w2c.T @ np.asarray(tvec, dtype=np.float64)


def fix_transforms_json(path: Path, train_dir: Path) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for frame in data.get("frames", []):
        fp = frame.get("file_path", "")
        rel = Path(fp)
        cand = train_dir / rel.name
        if not cand.exists():
            stem = rel.stem
            for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
                if (train_dir / f"{stem}{ext}").exists():
                    frame["file_path"] = f"{rel.parent.as_posix()}/{stem}{ext}"
                    break
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_colmap_v37(images_dir: Path, work_dir: Path, use_gpu: bool) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    sparse = work_dir / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    gpu_primary = "1" if use_gpu else "0"

    def run(cmd):
        subprocess.run(cmd, check=True, cwd=str(work_dir), env=env)

    if db.exists():
        db.unlink()

    def feat_match_mapper(gpu: str):
        run(
            [
                "colmap",
                "feature_extractor",
                "--database_path",
                str(db),
                "--image_path",
                str(images_dir),
                "--ImageReader.single_camera",
                "1",
                "--SiftExtraction.use_gpu",
                gpu,
            ]
        )
        run(
            [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                str(db),
                "--SiftMatching.use_gpu",
                gpu,
            ]
        )
        run(
            [
                "colmap",
                "mapper",
                "--database_path",
                str(db),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse),
            ]
        )

    try:
        feat_match_mapper(gpu_primary)
    except subprocess.CalledProcessError:
        if gpu_primary == "1":
            print("[COLMAP] GPU Sift failed (headless/OpenGL); retrying with CPU Sift/Match.")
            if db.exists():
                db.unlink()
            if sparse.exists():
                shutil.rmtree(sparse)
            sparse.mkdir(parents=True)
            feat_match_mapper("0")
        else:
            raise
    subs = sorted(d for d in sparse.iterdir() if d.is_dir() and (d / "images.bin").exists())
    if not subs:
        raise FileNotFoundError(f"No COLMAP model under {sparse}")
    return subs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_scene", type=str, required=True, help="Source dir with train/ and transforms_*.json")
    ap.add_argument("--out_scene", type=str, required=True, help="Output scene root directory")
    ap.add_argument("--no_gpu", action="store_true", help="COLMAP feature extraction / matching on CPU")
    ap.add_argument("--max_points", type=int, default=100_000, help="Max points in aligned PLY after subsampling")
    args = ap.parse_args()

    src = Path(args.src_scene).resolve()
    out = Path(args.out_scene).resolve()
    src_train = src / "train"
    if not src_train.is_dir():
        raise FileNotFoundError(src_train)

    out.mkdir(parents=True, exist_ok=True)
    out_train = out / "train"
    if out_train.exists():
        shutil.rmtree(out_train)
    shutil.copytree(src_train, out_train)

    for name in ("transforms_train.json", "transforms_test.json"):
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, out / name)
    for name in ("transforms_train.json", "transforms_test.json"):
        p = out / name
        if p.exists():
            fix_transforms_json(p, out_train)

    test_dir = out / "test"
    tt = out / "transforms_test.json"
    if tt.exists() and (not test_dir.is_dir() or not any(test_dir.iterdir())):
        print(
            "[INFO] No test/ images on disk; keeping transforms_test.json poses for NVS "
            "(eval may use placeholder GT)."
        )

    colmap_work = out / "_colmap_work"
    if colmap_work.exists():
        shutil.rmtree(colmap_work)
    print("[COLMAP] feature + match + mapper ...")
    sparse0 = run_colmap_v37(out_train, colmap_work, use_gpu=not args.no_gpu)

    out_sparse = out / "sparse" / "0"
    out_sparse.parent.mkdir(parents=True, exist_ok=True)
    if out_sparse.exists():
        shutil.rmtree(out_sparse)
    shutil.copytree(sparse0, out_sparse)

    images = read_extrinsics_binary(out_sparse / "images.bin")
    name_to_img = {Path(img.name).name: img for img in images.values()}

    with open(out / "transforms_train.json", encoding="utf-8") as f:
        train_json = json.load(f)

    src_centers, dst_centers = [], []
    for frame in train_json["frames"]:
        fp = frame["file_path"]
        stem = Path(fp).stem
        colmap_name = None
        for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
            cand = f"{stem}{ext}"
            if cand in name_to_img:
                colmap_name = cand
                break
        if colmap_name is None:
            continue
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        C_off = c2w[:3, 3]
        img = name_to_img[colmap_name]
        C_col = colmap_camera_center(img.qvec, img.tvec)
        src_centers.append(C_off)
        dst_centers.append(C_col)

    src_centers = np.array(src_centers)
    dst_centers = np.array(dst_centers)
    if len(src_centers) < 3:
        raise RuntimeError(
            f"Need >=3 train views registered in COLMAP and matched to JSON; got {len(src_centers)}."
        )

    scale, R_sim, t_sim = umeyama_similarity(src_centers, dst_centers)
    print(f"[ALIGN] Umeyama scale={scale:.6f}, pairs={len(src_centers)}")

    xyz, rgb, _ = read_points3D_binary(out_sparse / "points3D.bin")
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz_aligned = (R_sim.T @ (xyz - t_sim).T).T / scale

    n = xyz_aligned.shape[0]
    if n > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=args.max_points, replace=False)
        xyz_aligned = xyz_aligned[idx]
        rgb = rgb[idx]
        print(f"[PLY] Subsampled {n} -> {args.max_points} points")
    else:
        print(f"[PLY] Using all {n} points")

    ply_path = out / "points3d_sfm.ply"
    store_ply(str(ply_path), xyz_aligned, rgb)
    print(f"Wrote aligned point cloud: {ply_path}")
    print(f"Scene ready: {out}")


if __name__ == "__main__":
    main()
