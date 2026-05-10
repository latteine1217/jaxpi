#!/bin/bash
#SBATCH --job-name=post_kf_ckpt_field
#SBATCH --output=logs/post_kf_ckpt_field_%j.out
#SBATCH --error=logs/post_kf_ckpt_field_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm/postprocess"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
CONFIG_PATH="${CONFIG_PATH:-examples/kolmogorov_flow/configs/paper_repro_soap.py}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/re1e6_n512_ds4_soap/ckpt}"
WINDOW_IDX="${WINDOW_IDX:-1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
TIME_INDEX="${TIME_INDEX:--1}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
PLOT_STRIDE="${PLOT_STRIDE:-4}"
APPROX_VORTICITY="${APPROX_VORTICITY:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_DIR}/examples/kolmogorov_flow/comparison/window${WINDOW_IDX}_field_comparison.png}"

slurm_prepare_project "${PROJECT_DIR}" 0
slurm_require_path "${PROJECT_DIR}/${CONFIG_PATH}" "Config path not found"
slurm_require_path "${CHECKPOINT_ROOT}" "Checkpoint root not found"
slurm_setup_cuda_env

mkdir -p "$(dirname "${OUTPUT_PATH}")"

slurm_print_header \
  "Kolmogorov Checkpoint Field Comparison" \
  "Project" "${PROJECT_DIR}" \
  "Config" "${CONFIG_PATH}" \
  "CkptRoot" "${CHECKPOINT_ROOT}" \
  "Window" "${WINDOW_IDX}" \
  "Step" "${CHECKPOINT_STEP:-latest}" \
  "Output" "${OUTPUT_PATH}"
slurm_print_python_info
slurm_print_gpu_info

CMD=(
  "${UV_BIN}" run python scripts/analysis/render_checkpoint_field_comparison.py
  --config "${CONFIG_PATH}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --window "${WINDOW_IDX}"
  --time-index "${TIME_INDEX}"
  --chunk-size "${CHUNK_SIZE}"
  --plot-stride "${PLOT_STRIDE}"
  --output "${OUTPUT_PATH}"
)

if [ -n "${CHECKPOINT_STEP}" ]; then
  CMD+=(--step "${CHECKPOINT_STEP}")
fi

if [ "${APPROX_VORTICITY}" = "1" ]; then
  CMD+=(--approx-vorticity)
fi

"${CMD[@]}"
