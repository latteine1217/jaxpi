#!/bin/bash
#SBATCH --job-name=sweep_kf_w1_weights
#SBATCH --output=logs/sweep_kf_w1_weights_%j.out
#SBATCH --error=logs/sweep_kf_w1_weights_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm/sweep"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
N_TRIALS="${N_TRIALS:-40}"
MAX_STEPS="${MAX_STEPS:-50000}"
THRESHOLD="${THRESHOLD:-5e-5}"
STUDY_NAME="${STUDY_NAME:-kf_w1_weight_sweep}"
STORAGE="${STORAGE:-sqlite:///sweep_w1.db}"
CONFIG_PATH="${CONFIG_PATH:-examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py}"

slurm_prepare_project "${PROJECT_DIR}" 0
# sweep 為單 GPU 順序執行，降低 XLA 預分配比例，避免 CUDA kernel 編譯時 OOM
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
slurm_setup_cuda_env
export PYTHONUNBUFFERED=1

slurm_print_header \
  "Kolmogorov Window-1 Weight Sweep" \
  "Project"    "${PROJECT_DIR}" \
  "Study"      "${STUDY_NAME}" \
  "N_trials"   "${N_TRIALS}" \
  "Max_steps"  "${MAX_STEPS}" \
  "Threshold"  "${THRESHOLD}" \
  "Storage"    "${STORAGE}" \
  "Config"     "${CONFIG_PATH}"

slurm_print_python_info
slurm_print_gpu_info

START_TIME="$(date +%s)"

"${UV_BIN}" run python scripts/sweep/sweep_weights_window1.py \
  --n-trials    "${N_TRIALS}" \
  --max-steps   "${MAX_STEPS}" \
  --threshold   "${THRESHOLD}" \
  --study-name  "${STUDY_NAME}" \
  --storage     "${STORAGE}" \
  --config      "${CONFIG_PATH}"

EXIT_CODE=$?
slurm_print_footer "${EXIT_CODE}" "${START_TIME}"
