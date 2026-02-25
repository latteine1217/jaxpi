#!/bin/bash

# Kolmogorov Flow 訓練啟動腳本
# 用法: ./start_training.sh [config_name] [work_dir]
# 範例: ./start_training.sh pirate results_pirate

set -e

CONFIG=${1:-pirate}
WORKDIR=${2:-results_$CONFIG}

echo "==========================="
echo "Kolmogorov Flow 訓練"
echo "==========================="
echo "配置: configs/${CONFIG}.py"
echo "工作目錄: ${WORKDIR}"
echo "==========================="

# 檢查是否在正確目錄
if [ ! -f "main.py" ]; then
    echo "錯誤: 請在 examples/kolmogorov_flow 目錄下執行此腳本"
    exit 1
fi

# 檢查配置檔案
if [ ! -f "configs/${CONFIG}.py" ]; then
    echo "錯誤: 配置檔案 configs/${CONFIG}.py 不存在"
    exit 1
fi

# 建立工作目錄
mkdir -p ${WORKDIR}

# 顯示 GPU 資訊
echo ""
echo "--- GPU 資訊 ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# 開始訓練
echo "開始訓練..."
python main.py \
    --config=configs/${CONFIG}.py \
    --workdir=${WORKDIR}
