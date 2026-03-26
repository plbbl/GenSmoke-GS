#!/usr/bin/env bash
set -euo pipefail
ROOT="${MCMCFASTER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SCENES=(Futaba Hinoki Koharu Midori Natsume Shirohana Tsubaki)
LOGDIR="${ROOT}/output/g_resized_prep_logs"
mkdir -p "${LOGDIR}"
for i in "${!SCENES[@]}"; do
  s="${SCENES[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="${i}"
    bash "${ROOT}/scripts/prepare_g_resized_one_scene.sh" "${s}"
  ) > "${LOGDIR}/prep_${s}.log" 2>&1 &
  echo "[PREP] started ${s} on GPU ${i} -> ${LOGDIR}/prep_${s}.log"
done
wait
echo "[PREP] all 7 scenes done $(date -Is)"
