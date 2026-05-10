#!/bin/bash
#SBATCH --job-name=eval_kf_re1e6_sensor100_w25_soap
#SBATCH --output=logs/eval_kf_re1e6_sensor100_w25_soap_%j.out
#SBATCH --error=logs/eval_kf_re1e6_sensor100_w25_soap_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm/eval"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
EVAL_DRIVER="${EVAL_DRIVER:-eval_sensor100_w25.py}"
EVAL_CKPT_ROOT="${EVAL_CKPT_ROOT:-${PROJECT_DIR}/2026-03-26_train_re1e6_n2048_ke024_soap_sensor100_w25/ckpt}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_DIR}/2026-03-28_eval_re1e6_n2048_ke024_soap_sensor100_w25}"

slurm_prepare_project "${PROJECT_DIR}" 0
slurm_require_path "${EVAL_DRIVER}" "Eval driver not found"
slurm_require_path "${EVAL_CKPT_ROOT}" "Checkpoint root not found"
mkdir -p "${EVAL_OUTPUT_DIR}"
slurm_setup_cuda_env

export EVAL_CKPT_ROOT
export EVAL_OUTPUT_DIR

slurm_print_header \
  "Kolmogorov Re=1e6 Sensor100 W25 SOAP Eval" \
  "Project" "${PROJECT_DIR}" \
  "Driver" "${EVAL_DRIVER}" \
  "Ckpt" "${EVAL_CKPT_ROOT}" \
  "Output" "${EVAL_OUTPUT_DIR}"

slurm_print_python_info
slurm_print_gpu_info

srun "${UV_BIN}" run python "${EVAL_DRIVER}"
