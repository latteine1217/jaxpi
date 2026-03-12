#!/bin/bash
#SBATCH --job-name=kf_stage1_adam_2gpu
#SBATCH --output=logs/kf_stage1_adam_2gpu_%j.out
#SBATCH --error=logs/kf_stage1_adam_2gpu_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

set -euo pipefail

# Kolmogorov Stage A (Adam) 雙 GPU 訓練腳本
# 維護原則：只在本地 repo 修改，遠端機器透過 git sync 後使用。

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
CONFIG_PATH="${CONFIG_PATH:-examples/kolmogorov_flow/stage_ab/pirate_les_stage1.py}"
RUN_NAME="${RUN_NAME:-kf_stage1_adam_2gpu_${SLURM_JOB_ID}}"
WORKDIR="${WORKDIR:-${PROJECT_DIR}/runs/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-offline}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"
cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found" >&2
  exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_MODE
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

PY_VER=$(uv run python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="${PROJECT_DIR}/.venv/lib/${PY_VER}/site-packages"
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cuda_cupti/lib:${SITE_PACKAGES}/nvidia/cufft/lib:${SITE_PACKAGES}/nvidia/cusolver/lib:${SITE_PACKAGES}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

EVAL_CONFIG_ALIAS="${EVAL_CONFIG_ALIAS:-stage1}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_MODE="${EVAL_MODE:-final_step}"
EVAL_DEVICE="${EVAL_DEVICE:-gpu}"

cat <<EOF
===========================
Kolmogorov Stage A Adam 雙 GPU 訓練
===========================
Job ID: ${SLURM_JOB_ID}
Node: ${SLURM_NODELIST}
Project: ${PROJECT_DIR}
Config: ${CONFIG_PATH}
Workdir: ${WORKDIR}
Wandb mode: ${WANDB_MODE}
Extra args: ${EXTRA_ARGS}
===========================
EOF

uv --version
uv run python -c "import jax; print('JAX', jax.__version__); print('devices:', jax.devices())"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

START_TIME=$(date +%s)

# shellcheck disable=SC2086
srun uv run python examples/kolmogorov_flow/main.py \
  --config="${CONFIG_PATH}" \
  --workdir="${WORKDIR}" \
  ${EXTRA_ARGS}

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "Training exit code: ${TRAIN_EXIT_CODE}"
echo "Elapsed: ${ELAPSED}s"

if [ "${RUN_EVAL}" = "1" ] && [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
  echo "Running evaluation..."
  uv run python examples/kolmogorov_flow/evaluate_checkpoint.py \
    --config "${EVAL_CONFIG_ALIAS}" \
    --checkpoint_path "${WORKDIR}" \
    --mode "${EVAL_MODE}" \
    --device "${EVAL_DEVICE}"
fi

echo "stdout: ${PROJECT_DIR}/logs/kf_stage1_adam_2gpu_${SLURM_JOB_ID}.out"
echo "stderr: ${PROJECT_DIR}/logs/kf_stage1_adam_2gpu_${SLURM_JOB_ID}.err"
echo "workdir: ${WORKDIR}"
