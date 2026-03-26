#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export MCMCFASTER_ROOT="${MCMCFASTER_ROOT:-$ROOT}"

if [[ -z "${MCMCFASTER_PYTHON:-}" ]] && command -v conda >/dev/null 2>&1; then
  _ce="${MCMCFASTER_CONDA_ENV:-mcmc-nvs-open}"
  if conda run -n "${_ce}" true 2>/dev/null; then
    export MCMCFASTER_PYTHON="$(conda run -n "${_ce}" python -c "import sys; print(sys.executable)")"
  fi
fi
PY="${MCMCFASTER_PYTHON:-python3}"
export MCMCFASTER_PYTHON="${PY}"

[[ -n "${1:-}" ]] && export MCMCFASTER_DATA_ROOT="$1"
[[ -n "${MCMCFASTER_DATA_ROOT:-}" ]] || { echo "Usage: $0 /path/to/dataset_parent"; exit 1; }

for s in Futaba Hinoki Koharu Midori Natsume Shirohana Tsubaki; do
  [[ -d "${MCMCFASTER_DATA_ROOT}/${s}/train" ]] || { echo "ERROR: missing ${MCMCFASTER_DATA_ROOT}/${s}/train"; exit 1; }
done

[[ "${SKIP_PREPARE:-}" == "1" ]] || bash "${ROOT}/scripts/prepare_g_resized_all_parallel.sh"
"${PY}" "${ROOT}/scripts/launch_g_resized_cap100k_run01to100_8gpu_dynamic.py"
if [[ "${SKIP_MEAN:-}" != "1" ]]; then
  "${PY}" "${ROOT}/scripts/mean_nvs_runs.py" --run-first 1 --run-last 100 --out-name nvs_mean_run01_to_run100
fi
