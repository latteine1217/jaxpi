#!/bin/bash
#SBATCH --job-name=eval_prs
#SBATCH --output=logs/eval_paper_repro_soap_%j.out
#SBATCH --error=logs/eval_paper_repro_soap_%j.err
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
echo "  Ckpt dir : /home/junyi/jaxpi/paper_repro_soap/ckpt"
echo "================"

./.venv/bin/python3 eval_paper_repro_soap.py

echo "=== Finished : $(date) ==="
