#!/bin/bash
#SBATCH --job-name=kf_upstream_soap
#SBATCH --output=logs/kf_upstream_soap_%j.out
#SBATCH --error=logs/kf_upstream_soap_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G

set -euo pipefail
source ~/.zshrc 2>/dev/null || true

REPO_ROOT="${REPO_ROOT:-/home/junyi/jaxpi_upstream_soap_run}"
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=offline
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_GPU_ALLOCATOR=cuda_malloc_async

CONFIG="examples/kolmogorov_flow/configs/upstream_soap.py"
WORKDIR="runs/kf_upstream_soap_${SLURM_JOB_ID}"

echo "==========================="
echo "Kolmogorov Re=1e4 Upstream SOAP"
echo "==========================="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Repo     : ${REPO_ROOT}"
echo "Config   : ${CONFIG}"
echo "Workdir  : ${WORKDIR}"
echo "Wandb    : offline"
echo "==========================="

nvidia-smi --query-gpu=index,name,"memory.total","memory.used" --format=csv,noheader

srun uv run python examples/kolmogorov_flow/main.py \
    --config="${CONFIG}" \
    --workdir="${WORKDIR}"

echo "=== Finished : $(date) ==="
