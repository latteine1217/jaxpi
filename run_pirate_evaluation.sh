#!/bin/bash
# PIRATE Checkpoint 評估腳本（在 headnode 運行）

set -e

# 設置工作目錄
cd ~/jaxpi

echo "========================================="
echo "PIRATE Checkpoint 評估"
echo "========================================="
echo "開始時間: $(date)"
echo ""

# 載入 CUDA 模塊
echo "載入 CUDA 11.4..."
module load cuda/11.4

# 設置 LD_LIBRARY_PATH (使用 NVIDIA Python 套件中的 CUDA 庫)
export LD_LIBRARY_PATH="${HOME}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cublas/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cufft/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cusolver/lib:${HOME}/.local/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

# 設置 Python 路徑
export PYTHONPATH="${HOME}/jaxpi:${PYTHONPATH}"

# 檢查環境
echo "=== 環境檢查 ==="
echo "CUDA Version:"
nvcc --version | grep "release"
echo ""
echo "Python Version:"
python3 --version
echo ""
echo "JAX 設備:"
python3 -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())" 2>&1 | grep -v "^[EW]0000"
echo ""

# 運行評估
echo "========================================="
echo "開始評估 PIRATE checkpoint"
echo "========================================="
echo ""

python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
    --config pirate \
    --checkpoint_path ~/jaxpi/pirate/ckpt \
    --output ~/jaxpi/pirate_evaluation_$(date +%Y%m%d_%H%M%S).npz

echo ""
echo "========================================="
echo "評估完成！"
echo "結束時間: $(date)"
echo "========================================="
