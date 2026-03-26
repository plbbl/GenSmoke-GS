#!/usr/bin/env python3
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from g_resized_pool_config import (
    NUM_GPUS,
    ROOT,
    SCENES,
    expected_nvs_jpg_count,
    nvs_jpg_excluding_skipped,
)

SCRIPT = os.path.join(ROOT, "scripts", "run_convIR_g_resized_cap100k_repeat.sh")
RUN_FIRST = 1
RUN_LAST = 100


def run_first_from_env() -> int:
    raw = os.environ.get("G_RESIZED_RUN_FIRST")
    if not raw or not raw.strip():
        return RUN_FIRST
    try:
        return int(raw.strip())
    except ValueError:
        return RUN_FIRST


def run_last_from_env() -> int:
    raw = os.environ.get("G_RESIZED_RUN_LAST")
    if not raw or not raw.strip():
        return RUN_LAST
    try:
        return int(raw.strip())
    except ValueError:
        return RUN_LAST


def skip_for_launch() -> set[str]:
    raw = os.environ.get("G_RESIZED_SKIP_SCENES")
    if raw is None:
        return set()
    if not raw.strip():
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def build_tasks(skip: set[str], run_first: int, run_last: int) -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    for run in range(run_first, run_last + 1):
        for scene in SCENES:
            if scene in skip:
                continue
            tasks.append((run, scene))
    return tasks


def run_one(
    run_num: int,
    scene: str,
    gpu_pool: queue.Queue,
    log_dir: str,
    print_lock: threading.Lock,
) -> tuple[int, str, int, int]:
    gpu_id = gpu_pool.get()
    tag = f"gpu{gpu_id}_run{run_num:03d}_{scene}"
    log_path = os.path.join(log_dir, f"{tag}.txt")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = ["bash", SCRIPT, str(gpu_id), scene, str(run_num)]
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            ret = subprocess.run(
                cmd,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
        with print_lock:
            print(f"OK {tag} exit={ret.returncode}", flush=True)
        return (run_num, scene, gpu_id, ret.returncode)
    finally:
        gpu_pool.put(gpu_id)


def strict_run_batch_enabled() -> bool:
    return os.environ.get("G_RESIZED_STRICT_RUN_BATCH", "").lower() in (
        "1",
        "true",
        "yes",
    )


def main() -> None:
    dry = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    skip = skip_for_launch()
    run_first = run_first_from_env()
    run_last = run_last_from_env()
    if run_last < run_first:
        print(
            f"ERROR: G_RESIZED_RUN_LAST ({run_last}) < G_RESIZED_RUN_FIRST ({run_first})",
            file=sys.stderr,
        )
        sys.exit(1)
    tasks = build_tasks(skip, run_first, run_last)
    n_rep = run_last - run_first + 1
    strict_batch = strict_run_batch_enabled()
    print(
        f"[queue] runs {run_first}..{run_last} ({n_rep} repeats), "
        f"{len(SCENES) - len(skip)} scenes, skip={sorted(skip) or 'none'} -> {len(tasks)} tasks, "
        f"scheduling={'strict_run_batch' if strict_batch else 'fifo_global'}",
        flush=True,
    )
    if dry:
        for t in tasks[:30]:
            print(f"  {t}", flush=True)
        if len(tasks) > 30:
            print(f"  ... +{len(tasks) - 30} more", flush=True)
        return

    log_dir = os.path.join(
        ROOT,
        "output",
        f"g_resized_runs{run_first:02d}to{run_last:02d}_8gpu_dynamic_logs",
    )
    os.makedirs(log_dir, exist_ok=True)

    gpu_pool: queue.Queue = queue.Queue()
    for g in range(NUM_GPUS):
        gpu_pool.put(g)

    print_lock = threading.Lock()
    bad: list[tuple[int, str, int]] = []

    def drain_futures(futures: list) -> None:
        for fut in as_completed(futures):
            run_num, scene, _gpu_id, code = fut.result()
            if code != 0:
                bad.append((run_num, scene, code))

    if not strict_batch:
        with ThreadPoolExecutor(max_workers=NUM_GPUS) as ex:
            futures = [
                ex.submit(run_one, run, scene, gpu_pool, log_dir, print_lock)
                for run, scene in tasks
            ]
            drain_futures(futures)
    else:
        for run in range(run_first, run_last + 1):
            batch = [(run, s) for s in SCENES if s not in skip]
            if not batch:
                continue
            w = min(NUM_GPUS, len(batch))
            with print_lock:
                print(f"[run {run}] start {len(batch)} scenes, workers={w}", flush=True)
            with ThreadPoolExecutor(max_workers=w) as ex:
                futures = [
                    ex.submit(run_one, r, scene, gpu_pool, log_dir, print_lock)
                    for r, scene in batch
                ]
                drain_futures(futures)

    if bad:
        print(f"FAILED (first 20): {bad[:20]}", flush=True)
        sys.exit(1)

    exp = expected_nvs_jpg_count(skip)
    err = []
    for r in range(run_first, run_last + 1):
        rp = (
            subprocess.check_output(["bash", "-c", f'printf "%02d" {r}'])
            .decode()
            .strip()
        )
        nvs = os.path.join(ROOT, "output", f"nvs_cap100000_g_resized_run{rp}")
        if not os.path.isdir(nvs):
            err.append((nvs, "no_dir"))
            continue
        n = nvs_jpg_excluding_skipped(nvs, skip)
        if n != exp:
            err.append((nvs, n))

    if err:
        print(f"NVS check errors (expected {exp} JPG/run): {err[:15]}", flush=True)
        sys.exit(1)

    print(
        f"ALL_OK {len(tasks)} tasks, runs {run_first}-{run_last}, "
        f"NVS x {exp} JPG per run (skip={sorted(skip) or 'none'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
