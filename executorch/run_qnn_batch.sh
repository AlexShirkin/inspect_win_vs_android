#!/usr/bin/env bash
# Batch PTE inference via qnn_executor_runner (run inside ExecuTorch Docker).
#
# Usage (from repo root pte_qualcomm, paths inside container):
#   cd /workspace/inspect_win_vs_android
#   ./executorch/run_qnn_batch.sh mocopi_body \
#     /workspace/workspace/executorch/deeplab_v3/mmd_nca_net_qualcomm_12_mocopi_body_front_proj_70999.pte
#
# Env (same as workspace/install/build_mmd_nca_net.sh):
#   WORKSPACE=/workspace/workspace
#   EXECUTORCH_ROOT=$WORKSPACE/executorch
#   QNN_SDK_ROOT, BUILD_X64, LD_LIBRARY_PATH

set -euo pipefail

EXPERIMENT="${1:?experiment name: mocopi_body | mocopi_legs}"
PTE_PATH="${2:?path to .pte inside container}"
MAX_SAMPLES="${3:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${REPO_ROOT}/results/executorch/${EXPERIMENT}/pte_run"

: "${WORKSPACE:=/workspace/workspace}"
: "${EXECUTORCH_ROOT:=${WORKSPACE}/executorch}"
: "${QNN_SDK_ROOT:=${WORKSPACE}/qualcommAiengineSDK/qairt/2.26.0.240828}"
: "${BUILD_X64:=${EXECUTORCH_ROOT}/build-x86}"
: "${LD_LIBRARY_PATH:=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${BUILD_X64}/lib:${LD_LIBRARY_PATH:-}}"

RUNNER="${BUILD_X64}/examples/qualcomm/executor_runner/qnn_executor_runner"
export LD_LIBRARY_PATH

mkdir -p "${WORK_DIR}/outputs"
rm -f "${WORK_DIR}"/outputs/*.raw

PREPARE=(python executorch/scripts/prepare_runner_inputs.py
  --experiment "${EXPERIMENT}"
  --out-dir "${WORK_DIR}")
if [[ -n "${MAX_SAMPLES}" ]]; then
  PREPARE+=(--max-samples "${MAX_SAMPLES}")
fi
(cd "${REPO_ROOT}" && "${PREPARE[@]}")

echo "Running ${RUNNER}"
(cd "${WORK_DIR}" && "${RUNNER}" \
  --model_path "${PTE_PATH}" \
  --input_list_path input_list.txt \
  --output_folder_path outputs)

python "${REPO_ROOT}/executorch/scripts/collect_pte_outputs.py" \
  --outputs-dir "${WORK_DIR}/outputs" \
  --out-npz "${WORK_DIR}/embeddings_pte.npz"

python "${REPO_ROOT}/executorch/scripts/compare_pte_with_windows.py" \
  --experiment "${EXPERIMENT}" \
  --pte-embeddings "${WORK_DIR}/embeddings_pte.npz" \
  --label "PTE QNN x86"

echo "Done. Artifacts: ${WORK_DIR}"
