#!/bin/bash
#SBATCH --job-name=eval_s100w25
#SBATCH --output=logs/eval_sensor100_w25_%j.out
#SBATCH --error=logs/eval_sensor100_w25_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail
cd /home/junyi/jaxpi
mkdir -p logs

export PYTHONPATH="/home/junyi/jaxpi:${PYTHONPATH:-}"
export WANDB_MODE=offline
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "=== Job info ==="
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : $(hostname)"
echo "  Started  : $(date)"
echo "  Ckpt dir : /home/junyi/jaxpi/re1e6_n2048_ke024_soap_sensor100_w25/ckpt"
echo "================"

./.venv/bin/python3 eval_sensor100_w25.py

echo "=== Finished : $(date) ==="
