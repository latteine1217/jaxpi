#!/bin/bash
# Wandb 同步腳本 - 快速上傳訓練日誌到雲端

SERVER="junyi@140.114.120.128"

echo "=========================================="
echo "📊 Wandb Sync Tool"
echo "=========================================="
echo ""

# 選單
echo "選擇操作："
echo "  1) 同步最新的 run（SOAP 當前訓練）"
echo "  2) 同步所有 offline runs"
echo "  3) 查看 Wandb 專案 URL"
echo "  4) 查看同步狀態"
echo ""
read -p "請輸入選項 (1-4): " choice

case $choice in
  1)
    echo ""
    echo "🔄 正在同步最新的 run..."
    echo "----------------------------------------"
    ssh $SERVER "cd ~/jaxpi && python3 -m wandb sync wandb/latest-run"
    echo ""
    echo "✅ 同步完成！"
    echo "📊 查看結果: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow"
    ;;
  
  2)
    echo ""
    echo "🔄 正在同步所有 offline runs..."
    echo "這可能需要幾分鐘..."
    echo "----------------------------------------"
    ssh $SERVER "cd ~/jaxpi && python3 -m wandb sync wandb/offline-run-*"
    echo ""
    echo "✅ 同步完成！"
    echo "📊 查看結果: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow"
    ;;
  
  3)
    echo ""
    echo "📊 Wandb 專案資訊："
    echo "----------------------------------------"
    echo "專案 URL: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow"
    echo "Entity: felix-tc-tw-national-tsinghua-university"
    echo "Project: PINN-Kolmogorov_flow"
    echo ""
    echo "當前運行的實驗："
    ssh $SERVER "ls -lt ~/jaxpi/wandb/offline-run-* 2>/dev/null | head -3"
    ;;
  
  4)
    echo ""
    echo "📊 Wandb 狀態："
    echo "----------------------------------------"
    ssh $SERVER "wandb status"
    echo ""
    echo "最新 run 資訊："
    ssh $SERVER "ls -lh ~/jaxpi/wandb/latest-run/"
    ;;
  
  *)
    echo "❌ 無效選項"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "💡 提示："
echo "  - SOAP 訓練使用 offline 模式（需手動同步）"
echo "  - PIRATE 訓練已改為 online 模式（自動同步）"
echo "  - 建議每 6-12 小時同步一次 SOAP 查看進度"
echo "=========================================="
