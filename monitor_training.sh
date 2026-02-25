#!/bin/bash
# 訓練監控腳本 - 快速檢查 SOAP 和 PIRATE 訓練狀態

SERVER="junyi@140.114.120.128"

echo "=========================================="
echo "🚀 JAXpi Training Monitor"
echo "=========================================="
echo ""

# 1. 檢查 SLURM 佇列
echo "📊 SLURM Job Queue:"
ssh $SERVER "squeue -u junyi" || echo "❌ Failed to connect"
echo ""

# 2. SOAP 訓練狀態 (Job 2574)
echo "🧼 SOAP Training Status (Job 2574):"
echo "----------------------------------------"
SOAP_RUNNING=$(ssh $SERVER "squeue -u junyi -j 2574 2>/dev/null | grep -c 2574")
if [ "$SOAP_RUNNING" -gt 0 ]; then
    echo "✅ Status: RUNNING"
    
    # 最新進度
    echo ""
    echo "Latest Progress:"
    ssh $SERVER "tail -30 ~/jaxpi/logs/kf_soap_2gpu_2574.err | grep -E '(Iter:|u_error|v_error)' | tail -9"
    
    # 時間窗口
    echo ""
    echo "Time Windows:"
    ssh $SERVER "grep 'Training time window' ~/jaxpi/logs/kf_soap_2gpu_2574.err | tail -2"
    
    # 執行時間
    echo ""
    RUNTIME=$(ssh $SERVER "squeue -u junyi -j 2574 -o '%M' | tail -1")
    echo "Runtime: $RUNTIME"
else
    echo "❌ Status: NOT RUNNING"
    echo "Last 10 lines of log:"
    ssh $SERVER "tail -10 ~/jaxpi/logs/kf_soap_2gpu_2574.err"
fi
echo ""

# 3. PIRATE 訓練狀態 (Job 2575)
echo "🏴‍☠️ PIRATE Training Status (Job 2575):"
echo "----------------------------------------"
PIRATE_RUNNING=$(ssh $SERVER "squeue -u junyi -j 2575 2>/dev/null | grep -c 2575")
if [ "$PIRATE_RUNNING" -gt 0 ]; then
    PIRATE_STATUS=$(ssh $SERVER "squeue -u junyi -j 2575 -o '%T' | tail -1")
    echo "Status: $PIRATE_STATUS"
    
    if [ "$PIRATE_STATUS" == "RUNNING" ]; then
        # 最新進度
        echo ""
        echo "Latest Progress:"
        ssh $SERVER "tail -30 ~/jaxpi/logs/kf_pirate_2gpu_2575.err | grep -E '(Iter:|u_error|v_error)' | tail -9"
        
        # 時間窗口
        echo ""
        echo "Time Windows:"
        ssh $SERVER "grep 'Training time window' ~/jaxpi/logs/kf_pirate_2gpu_2575.err | tail -2"
        
        # 執行時間
        echo ""
        RUNTIME=$(ssh $SERVER "squeue -u junyi -j 2575 -o '%M' | tail -1")
        echo "Runtime: $RUNTIME"
    fi
else
    echo "❌ Status: NOT IN QUEUE"
fi
echo ""

# 4. GPU 使用狀況
echo "🎮 GPU Status on acmt20:"
echo "----------------------------------------"
ssh $SERVER "srun --nodelist=acmt20 --pty nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null" || echo "⚠️ Cannot query GPU (normal if not on node)"
echo ""

echo "=========================================="
echo "📝 Quick Commands:"
echo "  Cancel SOAP:   ssh $SERVER 'scancel 2574'"
echo "  Cancel PIRATE: ssh $SERVER 'scancel 2575'"
echo "  Full SOAP log: ssh $SERVER 'tail -100 ~/jaxpi/logs/kf_soap_2gpu_2574.err'"
echo "  Full PIRATE log: ssh $SERVER 'tail -100 ~/jaxpi/logs/kf_pirate_2gpu_2575.err'"
echo "=========================================="
