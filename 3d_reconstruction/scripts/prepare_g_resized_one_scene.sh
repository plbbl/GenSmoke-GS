#!/usr/bin/env bash
set -euo pipefail
SCENE="${1:?scene}"

ROOT="${MCMCFASTER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MC="${ROOT}/mcmc_faster"
DATA_ROOT="${MCMCFASTER_DATA_ROOT:-}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "ERROR: set MCMCFASTER_DATA_ROOT to the parent directory of scene folders" >&2
  exit 1
fi
SRC="${DATA_ROOT}/${SCENE}"
PREP="${ROOT}/prepared_scenes/ConvIR_g_resized_${SCENE}_prep"
PY="${MCMCFASTER_PYTHON:-python3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "[PREP] ${SCENE} src=${SRC} -> ${PREP}"
if [[ ! -d "${SRC}/train" ]]; then
  echo "ERROR: ${SRC}/train missing"
  exit 1
fi

COLMAP_EXTRA=()
if [[ "${PREPARE_COLMAP_NO_GPU:-}" == "1" ]]; then
  COLMAP_EXTRA+=(--no_gpu)
fi
"${PY}" "${MC}/prepare_developgpt_colmap.py" \
  --src_scene "${SRC}" \
  --out_scene "${PREP}" \
  --max_points "${MAX_POINTS:-150000}" \
  "${COLMAP_EXTRA[@]}"

echo "[PREP] done ${SCENE}"
