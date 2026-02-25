#!/bin/bash
#SBATCH --job-name=kf_soap_2gpu
#SBATCH --output=logs/kf_soap_2gpu_%j.out
#SBATCH --error=logs/kf_soap_2gpu_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48G

# ===========================
# Kolmogorov Flow - SOAP 雙 GPU 訓練腳本
# ===========================

PROJECT_DIR="${HOME}/jaxpi"
EXAMPLE_DIR="${PROJECT_DIR}/examples/kolmogorov_flow"
WORKDIR="${EXAMPLE_DIR}/results_soap_2gpu_${SLURM_JOB_ID}"

mkdir -p ${PROJECT_DIR}/logs

echo "==========================="
echo "Kolmogorov Flow - SOAP 雙 GPU 訓練"
echo "==========================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "==========================="

# 環境變數設定
# 動態取得 Python 版本與 site-packages 路徑
PY_VER=$(uv run python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="${PROJECT_DIR}/.venv/lib/${PY_VER}/site-packages"

# 設定 CUDA 相關路徑 (指向 .venv 內部)
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cuda_cupti/lib:${SITE_PACKAGES}/nvidia/cufft/lib:${SITE_PACKAGES}/nvidia/cusolver/lib:${SITE_PACKAGES}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="daf43f72d9f4f636dc69479c446ace76a4a3eb92"
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

# uv 環境設定
# 確保在正確的 project 目錄下，uv 會自動讀取 pyproject.toml
export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venv"

cd ${PROJECT_DIR}

echo ""
echo "環境檢查:"
echo "uv 版本:"
uv --version
echo ""
echo "JAX 環境:"
uv run python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.device_count(), 'GPUs'); print('Device list:', jax.devices())"

echo ""
echo "GPU 狀態:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo ""
echo "========================================="
echo "訓練配置 (SOAP)"
echo "========================================="
echo "架構: PirateNet (3 layers, 256 hidden dim)"
echo "優化器: SOAP (Schedule-Free Adam)"
echo "訓練步數: 20,000 steps per window"
echo "時間窗口: 25 windows"
echo "總訓練步數: 500,000 iterations"
echo ""
echo "Batch Size:"
echo "  - Per device: 4096 (與 PIRATE 相同)"
echo "  - Total: 8,192 (雙 GPU)"
echo "  - 配置已修正為與 PIRATE 相同架構"
echo ""
echo "優化器: SOAP (lr=1e-3, schedule_free=True)"
echo "Wandb: ${WANDB_MODE} mode"
echo "========================================="
echo ""

START_TIME=$(date +%s)

# 開始訓練
# SOAP 配置已修正：hidden_dim=256, batch_size=4096 (與 PIRATE 相同)
# 使用 uv run python 取代 python3
srun uv run python examples/kolmogorov_flow/main.py \
    --config=examples/kolmogorov_flow/configs/soap.py \
    --workdir=${WORKDIR} \
    --config.training.max_steps=20000 \
    --config.training.batch_size_per_device=2048 \
    --config.training.num_time_windows=25

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "==========================="
echo "訓練完成"
echo "Exit Code: ${TRAIN_EXIT_CODE}"
echo "End Time: $(date)"
echo "總耗時: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "==========================="

# 如果訓練成功，執行評估
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "訓練成功！開始評估模型..."
    uv run python examples/kolmogorov_flow/main.py \
        --config=examples/kolmogorov_flow/configs/soap.py \
        --workdir=${WORKDIR} \
        --config.mode=eval
    
    EVAL_EXIT_CODE=$?
    echo "評估完成 (Exit Code: ${EVAL_EXIT_CODE})"
    
    # 如果使用 offline 模式，提示如何同步
    if [ "${WANDB_MODE}" = "offline" ]; then
        echo ""
        echo "========================================="
        echo "Wandb 同步指令"
        echo "========================================="
        echo "cd ~/jaxpi"
        echo "uv run python -m wandb sync wandb/offline-run-*"
        echo "========================================="
    fi
else
    echo ""
    echo "訓練失敗 Exit Code: ${TRAIN_EXIT_CODE}"
    echo "請檢查錯誤日誌: ~/jaxpi/logs/kf_soap_2gpu_${SLURM_JOB_ID}.err"
fi

echo ""
echo "==========================="
echo "結果檔案位置"
echo "==========================="
echo "工作目錄: ${WORKDIR}"
echo "Checkpoints: ~/jaxpi/soap_Re10000/ckpt/"
echo "日誌檔案:"
echo "  - stdout: ~/jaxpi/logs/kf_soap_2gpu_${SLURM_JOB_ID}.out"
echo "  - stderr: ~/jaxpi/logs/kf_soap_2gpu_${SLURM_JOB_ID}.err"
if [ "${WANDB_MODE}" = "online" ]; then
    echo "Wandb: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow"
else
    echo "Wandb: offline mode - manual sync required"
fi
echo "==========================="

echo ""
echo "--- 最終 GPU 狀態 ---"
nvidia-smi

echo ""
echo "--- 磁碟使用 ---"
du -sh ${WORKDIR} 2>/dev/null || echo "工作目錄不存在"
du -sh ~/jaxpi/soap_Re10000/ckpt/ 2>/dev/null || echo "Checkpoint 目錄不存在"
