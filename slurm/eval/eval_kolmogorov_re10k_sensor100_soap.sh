#!/bin/bash
#SBATCH --job-name=eval_kf_re10k_sensor100_soap
#SBATCH --output=logs/eval_kf_re10k_sensor100_soap_%j.out
#SBATCH --error=logs/eval_kf_re10k_sensor100_soap_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm/eval"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
EVAL_DRIVER="${EVAL_DRIVER:-eval_re10k_sensor100.py}"
EVAL_CKPT_ROOT="${EVAL_CKPT_ROOT:-${PROJECT_DIR}/2026-03-23_train_re10k_n256_soap/ckpt}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_DIR}/2026-03-28_eval_re10k_sensor100_soap}"

slurm_prepare_project "${PROJECT_DIR}" 0
slurm_require_path "${EVAL_DRIVER}" "Eval driver not found"
slurm_require_path "${EVAL_CKPT_ROOT}" "Checkpoint root not found"
mkdir -p "${EVAL_OUTPUT_DIR}"
slurm_setup_cuda_env

export EVAL_CKPT_ROOT
export EVAL_OUTPUT_DIR

slurm_print_header \
  "Kolmogorov Re=1e4 Sensor100 SOAP Eval" \
  "Project" "${PROJECT_DIR}" \
  "Driver" "${EVAL_DRIVER}" \
  "Ckpt" "${EVAL_CKPT_ROOT}" \
  "Output" "${EVAL_OUTPUT_DIR}"

slurm_print_python_info
slurm_print_gpu_info

srun "${UV_BIN}" run python "${EVAL_DRIVER}"
