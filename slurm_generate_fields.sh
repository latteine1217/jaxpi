#!/bin/bash
#SBATCH --job-name=field_viz
#SBATCH --output=/home/junyi/jaxpi/logs/field_viz_%j.out
#SBATCH --error=/home/junyi/jaxpi/logs/field_viz_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G

# ===========================
# 流場視覺化腳本（統一版本）
# ===========================
# 功能：生成 DNS vs PINN 流場對比圖
# 支援：PIRATE 和 SOAP 模型
# 配置：通過環境變數調整行為
# ===========================

PROJECT_DIR="${HOME}/jaxpi"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/jaxpi/field_comparison_plots}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-generate_field_comparison.py}"

# 配置要生成的窗口（可通過環境變數覆蓋）
SOAP_WINDOWS="${SOAP_WINDOWS:-1 10 17 25}"
PIRATE_WINDOWS="${PIRATE_WINDOWS:-1 4 7 10}"

# 模型配置
RUN_SOAP="${RUN_SOAP:-1}"
RUN_PIRATE="${RUN_PIRATE:-1}"

echo "=========================================="
echo "流場視覺化生成任務"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""
echo "配置："
echo "  - Python 腳本: ${PYTHON_SCRIPT}"
echo "  - 輸出目錄: ${OUTPUT_DIR}"
echo "  - SOAP 窗口: ${SOAP_WINDOWS}"
echo "  - PIRATE 窗口: ${PIRATE_WINDOWS}"
echo "=========================================="
echo ""

# 環境設置
cd ${PROJECT_DIR}

# JAX 強制使用 CPU（Python 腳本內已設定，這裡再次確保）
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# 創建輸出目錄
mkdir -p ${OUTPUT_DIR}

echo "環境檢查:"
uv --version || echo "警告: uv 未找到"
echo ""

# ===========================
# 生成 SOAP 流場對比圖
# ===========================
if [ "${RUN_SOAP}" = "1" ]; then
    echo "=========================================="
    echo "生成 SOAP 流場對比圖"
    echo "=========================================="
    echo ""

    for window in ${SOAP_WINDOWS}; do
        echo "--- SOAP Window ${window} ---"
        uv run python ${PYTHON_SCRIPT} \
            --config soap \
            --checkpoint_path ~/jaxpi/soap_Re10000/ckpt \
            --window ${window} \
            --time_step -1 \
            --output_dir ${OUTPUT_DIR}

        if [ $? -ne 0 ]; then
            echo "警告: SOAP Window ${window} 生成失敗"
        fi
        echo ""
    done
fi

# ===========================
# 生成 PIRATE 流場對比圖
# ===========================
if [ "${RUN_PIRATE}" = "1" ]; then
    echo "=========================================="
    echo "生成 PIRATE 流場對比圖"
    echo "=========================================="
    echo ""

    for window in ${PIRATE_WINDOWS}; do
        echo "--- PIRATE Window ${window} ---"
        uv run python ${PYTHON_SCRIPT} \
            --config pirate \
            --checkpoint_path ~/jaxpi/pirate/ckpt \
            --window ${window} \
            --time_step -1 \
            --output_dir ${OUTPUT_DIR}

        if [ $? -ne 0 ]; then
            echo "警告: PIRATE Window ${window} 生成失敗"
        fi
        echo ""
    done
fi

EXIT_CODE=$?

echo "=========================================="
echo "生成完成"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 流場對比圖生成成功"
    echo ""
    echo "結果位置: ${OUTPUT_DIR}/"
    echo ""
    echo "生成的圖片:"
    ls -lh ${OUTPUT_DIR}/*.png 2>/dev/null || echo "沒有找到 PNG 檔案"
else
    echo "✗ 生成失敗 (Exit Code: ${EXIT_CODE})"
    echo "請檢查錯誤日誌"
fi

echo ""
echo "磁碟使用:"
du -sh ${OUTPUT_DIR} 2>/dev/null || echo "輸出目錄不存在"
