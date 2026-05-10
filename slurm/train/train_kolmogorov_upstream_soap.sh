#!/bin/bash
#SBATCH --job-name=train_kf_upstream_soap
#SBATCH --output=logs/train_kf_upstream_soap_%j.out
#SBATCH --error=logs/train_kf_upstream_soap_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm/train"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi_upstream_soap_run}"
CONFIG_PATH="${CONFIG_PATH:-examples/kolmogorov_flow/configs/upstream_soap.py}"
RUN_SLUG="${RUN_SLUG:-train_kf_upstream_soap}"
RUN_NAME="${RUN_NAME:-${RUN_SLUG}_$(slurm_job_id)}"
WORKDIR="${WORKDIR:-${PROJECT_DIR}/runs/${RUN_NAME}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_CONFIG_ALIAS="${EVAL_CONFIG_ALIAS:-${CONFIG_PATH}}"
EVAL_MODE="${EVAL_MODE:-final_step}"
EVAL_DEVICE="${EVAL_DEVICE:-gpu}"

slurm_prepare_project "${PROJECT_DIR}" 1
slurm_require_path "${CONFIG_PATH}" "Config not found"
slurm_setup_cuda_env

slurm_print_header \
  "Kolmogorov Upstream SOAP Train" \
  "Project" "${PROJECT_DIR}" \
  "Config" "${CONFIG_PATH}" \
  "Workdir" "${WORKDIR}" \
  "Wandb" "${WANDB_MODE}" \
  "Extra" "${EXTRA_ARGS:-<none>}"

slurm_print_python_info
slurm_print_gpu_info

START_TIME="$(date +%s)"
# shellcheck disable=SC2086
srun "${UV_BIN}" run python examples/kolmogorov_flow/main.py \
  --config="${CONFIG_PATH}" \
  --workdir="${WORKDIR}" \
  ${EXTRA_ARGS}
TRAIN_EXIT_CODE=$?

slurm_print_footer "${TRAIN_EXIT_CODE}" "${START_TIME}"

if [ "${RUN_EVAL}" = "1" ] && [ "${TRAIN_EXIT_CODE}" -eq 0 ]; then
  "${UV_BIN}" run python examples/kolmogorov_flow/evaluate_checkpoint.py \
    --config "${EVAL_CONFIG_ALIAS}" \
    --checkpoint_path "${WORKDIR}" \
    --mode "${EVAL_MODE}" \
    --device "${EVAL_DEVICE}"
fi
