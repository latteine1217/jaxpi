#!/bin/bash
#SBATCH --job-name=post_kf_w1_sweep
#SBATCH --output=logs/post_kf_w1_sweep_%j.out
#SBATCH --error=logs/post_kf_w1_sweep_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm/postprocess"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
NO_DATA_CONFIG="${NO_DATA_CONFIG:-examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py}"
NO_DATA_CKPT_ROOT="${NO_DATA_CKPT_ROOT:-${PROJECT_DIR}/re1e6_n512_ds4_soap_w1_ablation/ckpt}"
SENSOR_CONFIG="${SENSOR_CONFIG:-examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_ablation.py}"
SENSOR_CKPT_ROOT="${SENSOR_CKPT_ROOT:-${PROJECT_DIR}/re1e6_n512_ds4_soap_sensor100_w50_w1_ablation/ckpt}"
WINDOW_IDX="${WINDOW_IDX:-1}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-10000,20000,30000,40000,50000,60000,70000,80000,90000,100000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/eval_runs/window1_checkpoint_sweep}"
TIME_INDEX="${TIME_INDEX:--1}"
TIME_CHUNK_SIZE="${TIME_CHUNK_SIZE:-}"
SPACE_CHUNK_SIZE="${SPACE_CHUNK_SIZE:-}"
FIELD_CHUNK_SIZE="${FIELD_CHUNK_SIZE:-4096}"
PLOT_STRIDE="${PLOT_STRIDE:-2}"
SKIP_FIELDS="${SKIP_FIELDS:-0}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

slurm_prepare_project "${PROJECT_DIR}" 0
slurm_require_path "${PROJECT_DIR}/${NO_DATA_CONFIG}" "No-data config not found"
slurm_require_path "${PROJECT_DIR}/${SENSOR_CONFIG}" "Sensor config not found"
slurm_require_path "${NO_DATA_CKPT_ROOT}" "No-data checkpoint root not found"
slurm_require_path "${SENSOR_CKPT_ROOT}" "Sensor checkpoint root not found"
slurm_setup_cuda_env

mkdir -p "${OUTPUT_DIR}"

slurm_print_header \
  "Kolmogorov Window-1 Checkpoint Sweep" \
  "Project" "${PROJECT_DIR}" \
  "NoDataConfig" "${NO_DATA_CONFIG}" \
  "NoDataCkpt" "${NO_DATA_CKPT_ROOT}" \
  "SensorConfig" "${SENSOR_CONFIG}" \
  "SensorCkpt" "${SENSOR_CKPT_ROOT}" \
  "Window" "${WINDOW_IDX}" \
  "Steps" "${CHECKPOINT_STEPS}" \
  "Output" "${OUTPUT_DIR}"
slurm_print_python_info
slurm_print_gpu_info

CMD=(
  "${UV_BIN}" run python scripts/analysis/evaluate_window1_checkpoint_sweep.py
  --no-data-config "${NO_DATA_CONFIG}"
  --no-data-checkpoint-root "${NO_DATA_CKPT_ROOT}"
  --sensor-config "${SENSOR_CONFIG}"
  --sensor-checkpoint-root "${SENSOR_CKPT_ROOT}"
  --window "${WINDOW_IDX}"
  --steps "${CHECKPOINT_STEPS}"
  --output-dir "${OUTPUT_DIR}"
  --time-index "${TIME_INDEX}"
  --field-chunk-size "${FIELD_CHUNK_SIZE}"
  --plot-stride "${PLOT_STRIDE}"
)

if [ -n "${TIME_CHUNK_SIZE}" ]; then
  CMD+=(--time-chunk-size "${TIME_CHUNK_SIZE}")
fi

if [ -n "${SPACE_CHUNK_SIZE}" ]; then
  CMD+=(--space-chunk-size "${SPACE_CHUNK_SIZE}")
fi

if [ "${SKIP_FIELDS}" = "1" ]; then
  CMD+=(--skip-fields)
fi

if [ "${ALLOW_MISSING}" = "1" ]; then
  CMD+=(--allow-missing)
fi

"${CMD[@]}"
