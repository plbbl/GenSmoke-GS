#!/usr/bin/env python3
"""Run in the same env as training (e.g. conda run -n mcmc-nvs-open python scripts/check_faster_env.py)."""
from __future__ import annotations

import sys


def main() -> None:
    print("[check] FasterGSCudaBackend (raster + FusedAdam)")
    try:
        from FasterGSCudaBackend.torch_bindings import (  # noqa: F401
            FusedAdam,
            RasterizerSettings,
            diff_rasterize,
        )
    except Exception as e:
        print("  FAIL:", e)
        sys.exit(1)
    print("  OK")

    print("[check] diff_gaussian_rasterization (MCMC submodule, compute_relocation) + simple_knn")
    try:
        from diff_gaussian_rasterization import compute_relocation  # noqa: F401
        import simple_knn  # noqa: F401
    except Exception as e:
        print("  FAIL:", e)
        sys.exit(1)
    print("  OK")

    print("[check] defaults: USE_FASTERGS_RASTERIZER / USE_FASTERGS_ADAM = True in mcmc_faster")


if __name__ == "__main__":
    main()
