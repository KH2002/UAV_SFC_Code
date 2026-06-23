#!/usr/bin/env bash
# 在同一 config（同一评估集）上分别测试 MPopLoc 与 MADRL2Stages，并对比结果。
#
# 用法:
#   bash Test/run_tests.sh <CHECKPOINT> [CONFIG] [NUM_EPISODES]
#
# 示例:
#   bash Test/run_tests.sh \
#     MADRL2Stages/output/<run>/checkpoint/policy_final.pt \
#     MADRL2Stages/config/config_large.yaml \
#     20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${1:-}"
CONFIG="${2:-${REPO_ROOT}/MADRL2Stages/config/config_large.yaml}"
NUM_EPISODES="${3:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: 需要提供 MADRL2Stages checkpoint 路径。" >&2
  echo "用法: bash Test/run_tests.sh <CHECKPOINT> [CONFIG] [NUM_EPISODES]" >&2
  exit 1
fi

PY="${PYTHON:-python}"
EP_ARG=()
if [[ -n "${NUM_EPISODES}" ]]; then
  EP_ARG=(--num-episodes "${NUM_EPISODES}")
fi

cd "${REPO_ROOT}"

echo "======================================================"
echo "Config     : ${CONFIG}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Episodes   : ${NUM_EPISODES:-<config default>}"
echo "======================================================"

echo ""
echo ">>> [1/2] MPopLoc"
"${PY}" "${SCRIPT_DIR}/test_mpoploc.py" --config "${CONFIG}" "${EP_ARG[@]}"

echo ""
echo ">>> [2/2] MADRL2Stages"
"${PY}" "${SCRIPT_DIR}/test_madrl2stages.py" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" "${EP_ARG[@]}"

echo ""
echo "全部测试完成，结果见 ${SCRIPT_DIR}/output/<scene_tag>/{MPopLoc,MADRL2Stages}/"
