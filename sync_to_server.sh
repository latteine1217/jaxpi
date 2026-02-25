#!/bin/bash
# 同步Transfer Optimizer State功能至伺服器

set -e  # Exit on error

echo "=== 同步Transfer Optimizer State至伺服器 ==="
echo ""

# 伺服器資訊
SERVER="junyi@140.114.120.128"
REMOTE_DIR="~/jaxpi"

echo "步驟1: 檢查SSH連線..."
if ! ssh -o ConnectTimeout=5 $SERVER "echo 'SSH連線成功'"; then
    echo "錯誤: 無法連線至伺服器 $SERVER"
    exit 1
fi
echo "✓ SSH連線正常"
echo ""

echo "步驟2: 同步核心程式碼..."
# 只同步必要的修改檔案
rsync -avz --progress \
    --include='jaxpi/models.py' \
    --include='examples/kolmogorov_flow/configs/soap.py' \
    --include='examples/kolmogorov_flow/configs/pirate.py' \
    --include='examples/kolmogorov_flow/train.py' \
    --include='docs/transfer_optimizer_state/***' \
    --exclude='*' \
    ./ $SERVER:$REMOTE_DIR/

echo "✓ 核心程式碼同步完成"
echo ""

echo "步驟3: 同步文檔..."
rsync -avz --progress docs/transfer_optimizer_state/ $SERVER:$REMOTE_DIR/docs/transfer_optimizer_state/

echo "✓ 文檔同步完成"
echo ""

echo "步驟4: 驗證伺服器端檔案..."
ssh $SERVER << 'ENDSSH'
cd ~/jaxpi
echo "檢查關鍵檔案..."
ls -lh jaxpi/models.py
ls -lh examples/kolmogorov_flow/configs/soap.py
ls -lh examples/kolmogorov_flow/train.py
echo ""
echo "檢查文檔目錄..."
ls -lh docs/transfer_optimizer_state/ 2>/dev/null || echo "文檔目錄尚未創建"
ENDSSH

echo "✓ 驗證完成"
echo ""
echo "=== 同步完成 ==="
echo "下一步: ssh $SERVER 並執行最小可行測試"
