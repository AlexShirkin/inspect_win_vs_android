#!/usr/bin/env bash
# ETDump on legs worst samples (inside Docker).
#
# Prerequisite: rebuild .pte with dump_intermediate_outputs=True in mmd_nca_net.py
#
# Usage:
#   ./executorch/run_etdump_worst.sh /path/to/legs.pte

set -euo pipefail

PTE_PATH="${1:?path to legs .pte}"
EXPERIMENT="${2:-mocopi_legs}"
WORST="${3:-279,831,441,559,53}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${REPO_ROOT}/results/executorch/${EXPERIMENT}/etdump_run"

: "${WORKSPACE:=/workspace/workspace}"
: "${EXECUTORCH_ROOT:=${WORKSPACE}/executorch}"
: "${QNN_SDK_ROOT:=${WORKSPACE}/qualcommAiengineSDK/qairt/2.26.0.240828}"
: "${BUILD_X64:=${EXECUTORCH_ROOT}/build-x86}"
: "${LD_LIBRARY_PATH:=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${BUILD_X64}/lib:${LD_LIBRARY_PATH:-}}"

RUNNER="${BUILD_X64}/examples/qualcomm/executor_runner/qnn_executor_runner"
export LD_LIBRARY_PATH

mkdir -p "${WORK_DIR}/outputs"

python "${REPO_ROOT}/executorch/scripts/prepare_runner_inputs.py" \
  --experiment "${EXPERIMENT}" \
  --out-dir "${WORK_DIR}" \
  --indices "${WORST}"

cd "${WORK_DIR}"
"${RUNNER}" \
  --model_path "${PTE_PATH}" \
  --input_list_path input_list.txt \
  --output_folder_path outputs \
  --dump_intermediate_outputs \
  --etdump_path etdump.etdp \
  --debug_output_path debug_output.bin

conda run -n executorch python "${REPO_ROOT}/executorch/scripts/analyze_etdump.py" \
  --etdump "${WORK_DIR}/etdump.etdp" \
  --out-dir "${REPO_ROOT}/results/executorch/${EXPERIMENT}"

echo "ETDump: ${WORK_DIR}/etdump.etdp"
echo "Report: ${REPO_ROOT}/results/executorch/${EXPERIMENT}/layer_gaps_report.txt"
