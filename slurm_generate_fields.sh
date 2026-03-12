#!/bin/bash
#SBATCH --job-name=kf_field_snapshots
#SBATCH --output=logs/kf_field_snapshots_%j.out
#SBATCH --error=logs/kf_field_snapshots_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G

set -euo pipefail

# Kolmogorov 場快照對比生成腳本
# 預設比較 Stage A Adam vs Stage A SOAP。維護原則：只在本地 repo 修改，遠端同步後使用。

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/examples/kolmogorov_flow/comparison}"
PIRATE_CONFIG_ALIAS="${PIRATE_CONFIG_ALIAS:-stage1}"
SOAP_CONFIG_ALIAS="${SOAP_CONFIG_ALIAS:-stage1_soap}"
PIRATE_CHECKPOINT_PATH="${PIRATE_CHECKPOINT_PATH:-${PROJECT_DIR}/runs/kf_stage1/ckpt}"
SOAP_CHECKPOINT_PATH="${SOAP_CHECKPOINT_PATH:-${PROJECT_DIR}/runs/kf_stage1_soap/ckpt}"
SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-0.5 1.0 1.5 1.8}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found" >&2
  exit 1
fi

if [ ! -d "${PIRATE_CHECKPOINT_PATH}" ]; then
  echo "PIRATE checkpoint path not found: ${PIRATE_CHECKPOINT_PATH}" >&2
  exit 1
fi

if [ ! -d "${SOAP_CHECKPOINT_PATH}" ]; then
  echo "SOAP checkpoint path not found: ${SOAP_CHECKPOINT_PATH}" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export JAX_PLATFORMS="cpu"
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

cat <<EOF
===========================
Kolmogorov 場快照對比生成
===========================
Job ID: ${SLURM_JOB_ID}
Project: ${PROJECT_DIR}
Output: ${OUTPUT_DIR}
Pirate config: ${PIRATE_CONFIG_ALIAS}
Soap config: ${SOAP_CONFIG_ALIAS}
Pirate ckpt: ${PIRATE_CHECKPOINT_PATH}
Soap ckpt: ${SOAP_CHECKPOINT_PATH}
Snapshot times: ${SNAPSHOT_TIMES}
===========================
EOF

# shellcheck disable=SC2086
uv run python examples/kolmogorov_flow/generate_field_snapshots.py \
  --config_pirate "${PIRATE_CONFIG_ALIAS}" \
  --config_soap "${SOAP_CONFIG_ALIAS}" \
  --checkpoint_path_pirate "${PIRATE_CHECKPOINT_PATH}" \
  --checkpoint_path_soap "${SOAP_CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --times ${SNAPSHOT_TIMES}

echo "Generated outputs under: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}" | sed -n '1,120p'
