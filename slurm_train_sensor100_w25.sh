#!/bin/bash
#SBATCH --job-name=kf_sensor100_w25
#SBATCH --output=logs/kf_sensor100_w25_%j.out
#SBATCH --error=logs/kf_sensor100_w25_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/jaxpi}"
CONFIG_PATH="${CONFIG_PATH:-examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_w25.py}"
RUN_NAME="${RUN_NAME:-kf_sensor100_w25_${SLURM_JOB_ID}}"
WORKDIR="${WORKDIR:-${PROJECT_DIR}/runs/${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-offline}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python not found: ${PYTHON_BIN}" >&2; exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}" >&2; exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_MODE
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

PY_VER=$(${PYTHON_BIN} -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="${PROJECT_DIR}/.venv/lib/${PY_VER}/site-packages"
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cuda_cupti/lib:${SITE_PACKAGES}/nvidia/cufft/lib:${SITE_PACKAGES}/nvidia/cusolver/lib:${SITE_PACKAGES}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

cat <<INFO
===========================
Kolmogorov SOAP + 100 QR-pivot sensors, 25 windows
===========================
Job ID   : ${SLURM_JOB_ID}
Node     : ${SLURM_NODELIST}
Config   : ${CONFIG_PATH}
Workdir  : ${WORKDIR}
Wandb    : ${WANDB_MODE}
===========================
INFO

${PYTHON_BIN} -V
${PYTHON_BIN} -c "import jax; print('JAX', jax.__version__); print('devices:', jax.devices())"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

START_TIME=$(date +%s)

# shellcheck disable=SC2086
srun ${PYTHON_BIN} examples/kolmogorov_flow/main.py \
  --config="${CONFIG_PATH}" \
  --workdir="${WORKDIR}" \
  ${EXTRA_ARGS}

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
echo "Exit code: ${TRAIN_EXIT_CODE}  Elapsed: $((END_TIME - START_TIME))s"
echo "stdout: ${PROJECT_DIR}/logs/kf_sensor100_w25_${SLURM_JOB_ID}.out"
echo "workdir: ${WORKDIR}"
