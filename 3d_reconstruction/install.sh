#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
ENV_NAME="${CONDA_ENV_NAME:-mcmc-nvs-open}"
ENV_YAML="${ENV_YAML:-environment.yml}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_PIN="${TORCH_PIN:-1}"

for c in conda gcc g++; do command -v "$c" >/dev/null || { echo "ERROR: need $c"; exit 1; }; done

YAML="${ROOT}/${ENV_YAML}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  conda run -n "${ENV_NAME}" true || { echo "ERROR: no conda env ${ENV_NAME}"; exit 1; }
elif conda run -n "${ENV_NAME}" true 2>/dev/null; then
  conda env update -f "${YAML}" --prune
else
  conda env create -f "${YAML}"
fi

RUN=(conda run --no-capture-output -n "${ENV_NAME}")
if [[ "${TORCH_PIN}" == "1" ]]; then
  "${RUN[@]}" pip install torch==2.10.0 torchvision==0.25.0 --index-url "${TORCH_INDEX}"
else
  "${RUN[@]}" pip install torch torchvision --index-url "${TORCH_INDEX}"
fi
"${RUN[@]}" pip install -r "${ROOT}/requirements-pip.txt"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0+PTX}"
( cd "${ROOT}/mcmc_faster" && "${RUN[@]}" pip install submodules/diff-gaussian-rasterization submodules/simple-knn )
( cd "${ROOT}/faster-gaussian-splatting/FasterGSCudaBackend" && "${RUN[@]}" pip install . --no-build-isolation )
"${RUN[@]}" python "${ROOT}/scripts/check_faster_env.py"
