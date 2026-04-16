#!/bin/bash
#SBATCH --job-name=post_kf_generate_fields
#SBATCH --output=logs/post_kf_generate_fields_%j.out
#SBATCH --error=logs/post_kf_generate_fields_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/examples/kolmogorov_flow/comparison}"
PIRATE_CONFIG_ALIAS="${PIRATE_CONFIG_ALIAS:-stage1}"
SOAP_CONFIG_ALIAS="${SOAP_CONFIG_ALIAS:-stage1_soap}"
PIRATE_CHECKPOINT_PATH="${PIRATE_CHECKPOINT_PATH:-${PROJECT_DIR}/runs/kf_stage1/ckpt}"
SOAP_CHECKPOINT_PATH="${SOAP_CHECKPOINT_PATH:-${PROJECT_DIR}/runs/kf_stage1_soap/ckpt}"
SNAPSHOT_TIMES="${SNAPSHOT_TIMES:-0.5 1.0 1.5 1.8}"

slurm_prepare_project "${PROJECT_DIR}" 0
mkdir -p "${OUTPUT_DIR}"
slurm_require_path "${PIRATE_CHECKPOINT_PATH}" "PIRATE checkpoint path not found"
slurm_require_path "${SOAP_CHECKPOINT_PATH}" "SOAP checkpoint path not found"
slurm_setup_cpu_env

slurm_print_header \
  "Kolmogorov Generate Fields" \
  "Project" "${PROJECT_DIR}" \
  "Output" "${OUTPUT_DIR}" \
  "Pirate" "${PIRATE_CHECKPOINT_PATH}" \
  "Soap" "${SOAP_CHECKPOINT_PATH}" \
  "Times" "${SNAPSHOT_TIMES}"

# shellcheck disable=SC2086
"${UV_BIN}" run python examples/kolmogorov_flow/generate_field_snapshots.py \
  --config_pirate "${PIRATE_CONFIG_ALIAS}" \
  --config_soap "${SOAP_CONFIG_ALIAS}" \
  --checkpoint_path_pirate "${PIRATE_CHECKPOINT_PATH}" \
  --checkpoint_path_soap "${SOAP_CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --times ${SNAPSHOT_TIMES}
