#!/bin/bash
# 最小可行測試：驗證Transfer Optimizer State功能
# 執行環境：伺服器 (junyi@140.114.120.128)

set -e

echo "=========================================="
echo "  Transfer Optimizer State 最小可行測試"
echo "=========================================="
echo ""
echo "測試配置："
echo "  - 訓練步數：1000 steps/window"
echo "  - 窗口數：1或2"
echo "  - 優化器：SOAP"
echo "  - 預計時間：~5-10分鐘"
echo ""

# 切換到工作目錄
cd ~/jaxpi/examples/kolmogorov_flow

# 創建測試輸出目錄
TEST_DIR="minimal_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $TEST_DIR
cd $TEST_DIR

echo "測試輸出目錄：$(pwd)"
echo ""

# ========== 測試1：單窗口訓練（基準測試） ==========
echo "=========================================="
echo "測試1：單窗口訓練（無遷移學習）"
echo "=========================================="
echo "目的：驗證基本訓練功能正常"
echo ""

python ../../train.py \
    --config ../configs/soap.py \
    --config.training.num_time_windows=1 \
    --config.training.max_steps=1000 \
    --config.wandb.name="test1_single_window" \
    --config.wandb.tags="['minimal_test','single_window']" \
    2>&1 | tee test1_single_window.log

if [ $? -eq 0 ]; then
    echo "✓ 測試1通過"
else
    echo "✗ 測試1失敗"
    exit 1
fi
echo ""

# ========== 測試2：雙窗口訓練（params-only遷移） ==========
echo "=========================================="
echo "測試2：雙窗口訓練（params-only遷移）"
echo "=========================================="
echo "目的：驗證向後相容性"
echo ""

python ../../train.py \
    --config ../configs/soap.py \
    --config.training.num_time_windows=2 \
    --config.training.max_steps=1000 \
    --config.transfer_learning=True \
    --config.transfer_optimizer_state=False \
    --config.wandb.name="test2_params_only" \
    --config.wandb.tags="['minimal_test','params_only']" \
    2>&1 | tee test2_params_only.log

if [ $? -eq 0 ]; then
    echo "✓ 測試2通過"
    # 檢查log中是否有正確的訊息
    if grep -q "僅傳遞params" test2_params_only.log; then
        echo "✓ 確認使用params-only模式"
    else
        echo "⚠ 警告：未找到預期的log訊息"
    fi
else
    echo "✗ 測試2失敗"
    exit 1
fi
echo ""

# ========== 測試3：雙窗口訓練（full state遷移） ==========
echo "=========================================="
echo "測試3：雙窗口訓練（full state遷移）"
echo "=========================================="
echo "目的：驗證新功能正常運作"
echo ""

python ../../train.py \
    --config ../configs/soap.py \
    --config.training.num_time_windows=2 \
    --config.training.max_steps=1000 \
    --config.transfer_learning=True \
    --config.transfer_optimizer_state=True \
    --config.wandb.name="test3_full_state" \
    --config.wandb.tags="['minimal_test','full_state']" \
    2>&1 | tee test3_full_state.log

if [ $? -eq 0 ]; then
    echo "✓ 測試3通過"
    # 檢查log中是否有正確的訊息
    if grep -q "傳遞完整優化器狀態" test3_full_state.log; then
        echo "✓ 確認使用full state模式"
    else
        echo "⚠ 警告：未找到預期的log訊息"
    fi
else
    echo "✗ 測試3失敗"
    exit 1
fi
echo ""

# ========== 測試結果總結 ==========
echo "=========================================="
echo "  測試結果總結"
echo "=========================================="
echo ""

echo "測試通過情況："
echo "  ✓ 測試1：單窗口訓練"
echo "  ✓ 測試2：params-only遷移"
echo "  ✓ 測試3：full state遷移"
echo ""

echo "Log檔案位置："
echo "  - test1_single_window.log"
echo "  - test2_params_only.log"
echo "  - test3_full_state.log"
echo ""

echo "檢查要點："
echo "1. 三個測試都無錯誤終止"
echo "2. test2的log包含'僅傳遞params'"
echo "3. test3的log包含'傳遞完整優化器狀態'"
echo ""

# 簡單分析initial loss
echo "初始Loss對比分析："
echo "（Window 1的初始loss，應該：test3 < test2）"
echo ""

echo "Test2 (params-only):"
grep "Window 1.*step 0" test2_params_only.log | head -3 || echo "  未找到"

echo ""
echo "Test3 (full state):"
grep "Window 1.*step 0" test3_full_state.log | head -3 || echo "  未找到"

echo ""
echo "=========================================="
echo "  ✅ 最小可行測試完成"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 檢查上述log確認功能正確"
echo "  2. 若全部通過，可進行快速驗證實驗（3×3窗口×10K步）"
echo "  3. 參考文檔：docs/transfer_optimizer_state/03_QUICK_REFERENCE.md"
echo ""
