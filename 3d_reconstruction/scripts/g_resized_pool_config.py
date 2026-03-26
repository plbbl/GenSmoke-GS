from __future__ import annotations

import os

if os.environ.get("MCMCFASTER_ROOT", "").strip():
    ROOT = os.path.abspath(os.environ["MCMCFASTER_ROOT"].strip())
else:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCENES = [
    "Hinoki",
    "Futaba",
    "Koharu",
    "Midori",
    "Natsume",
    "Shirohana",
    "Tsubaki",
]
NUM_GPUS = 8
_JPG_PER_SCENE = 4


def expected_nvs_jpg_count(skip: set[str]) -> int:
    active = [s for s in SCENES if s not in skip]
    return _JPG_PER_SCENE * len(active)


def nvs_jpg_excluding_skipped(nvs_dir: str, skip: set[str]) -> int:
    if not skip:
        return len([x for x in os.listdir(nvs_dir) if x.upper().endswith(".JPG")])
    prefs = tuple(f"{s.lower()}_" for s in skip)

    def keep(name: str) -> bool:
        if not name.upper().endswith(".JPG"):
            return False
        low = name.lower()
        return not any(low.startswith(p) for p in prefs)

    return len([x for x in os.listdir(nvs_dir) if keep(x)])
