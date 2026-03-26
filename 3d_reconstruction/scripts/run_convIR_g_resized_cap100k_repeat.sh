#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?gpu}"
SCENE="${2:?scene}"
RUN_NUM="${3:?run}"

ROOT="${MCMCFASTER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0+PTX}"

PY="${MCMCFASTER_PYTHON:-python3}"
MC="${ROOT}/mcmc_faster"
PREP="${ROOT}/prepared_scenes/ConvIR_g_resized_${SCENE}_prep"
RUNPAD=$(printf "%02d" "${RUN_NUM}")
OUT="${ROOT}/output/g_resized_repeat_runs/run${RUNPAD}/ConvIR_${SCENE}_cap100000_30k"
NVS="${ROOT}/output/nvs_cap100000_g_resized_run${RUNPAD}"
DONE="${ROOT}/output/g_resized_repeat_done/run${RUNPAD}"
LOGDIR="${ROOT}/output/g_resized_repeat_logs"
mkdir -p "${OUT}" "${NVS}" "${DONE}" "${LOGDIR}"

SCENE_LOWER="$(echo "${SCENE}" | tr '[:upper:]' '[:lower:]')"
LOG="${LOGDIR}/run${RUNPAD}_${SCENE}.log"
exec > >(tee -a "${LOG}") 2>&1

echo "========== $(date -Is) g_resized RUN=${RUNPAD} SCENE=${SCENE} GPU=${GPU_ID} =========="

if [[ ! -d "${PREP}/train" ]]; then
  echo "ERROR: missing prepared scene (expected prepared_scenes/ConvIR_g_resized_<Scene>_prep/train). Run prepare_g_resized_one_scene.sh first."
  exit 1
fi

"${PY}" "${MC}/train.py" \
  --source_path "${PREP}" \
  --init_type sfm \
  --resolution 1 \
  --cap_max 100000 \
  -m "${OUT}" \
  --config "${MC}/configs/counter_100w.json" \
  --eval \
  --iterations 30000 \
  --save_iterations 30000 \
  --test_iterations 7000 15000 30000

"${PY}" "${MC}/render.py" -m "${OUT}" --iteration 30000 --skip_train

"${PY}" "${MC}/export_nvs_flat_jpg.py" \
  --renders_dir "${OUT}/test/ours_30000/renders" \
  --scene_lower "${SCENE_LOWER}" \
  --out_dir "${NVS}"

touch "${DONE}/${SCENE}.ok"
echo "========== $(date -Is) DONE RUN=${RUNPAD} ${SCENE} =========="
