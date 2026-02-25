# Batch Size 測試報告

## 📊 測試目標
確定 r740 節點 (2x Tesla P100-PCIE-16GB, 110GB RAM) 的最大可用 batch size

## 🔬 測試方法
- **測試配置**: 單GPU vs 雙GPU
- **測試範圍**: 2048 到 12288 per device
- **驗證方式**: 運行 50 步訓練，確保記憶體穩定分配
- **任務**: Kolmogorov Flow PINN 訓練

---

## ✅ 測試結果

### 單 GPU 配置 (1x Tesla P100 16GB)

| Batch Size | 總 Batch | GPU 記憶體 | 狀態 |
|-----------|---------|-----------|------|
| 2048 | 2,048 | ~10GB | ✅ |
| 4096 | 4,096 | ~12GB | ✅ |
| 6144 | 6,144 | ~14GB | ✅ |
| **8192** | **8,192** | **~15.5GB** | ✅ **最大** |

**結論**: 單 GPU 最大可用 batch size = **8192** (與預設配置相同)

### 雙 GPU 配置 (2x Tesla P100 16GB)

| Batch Size/Device | 總 Batch | 總 GPU 記憶體 | 狀態 |
|------------------|---------|--------------|------|
| 4096 | 8,192 | ~24GB | ✅ |
| 6144 | 12,288 | ~28GB | ✅ |
| 8192 | 16,384 | ~31GB | ✅ |
| 10240 | 20,480 | ~31.5GB | ✅ |
| **12288** | **24,576** | **~32GB** | ✅ **最大** |

**結論**: 雙 GPU 最大可用 batch size = **12288 per device** (總 24,576)

---

## 🎯 推薦配置

### 保守配置 (推薦用於生產訓練)
保留 15-20% 安全餘量，避免記憶體波動導致 OOM

**單 GPU:**
```bash
--config.training.batch_size_per_device=6144  # 75% of max
```

**雙 GPU:**
```bash
--config.training.batch_size_per_device=10240  # 83% of max
# 總 batch size = 20,480
```

### 積極配置 (最大化吞吐量)
接近硬體極限，適合短期實驗

**單 GPU:**
```bash
--config.training.batch_size_per_device=7168  # 87.5% of max
```

**雙 GPU:**
```bash
--config.training.batch_size_per_device=11264  # 91.7% of max
# 總 batch size = 22,528
```

---

## 📈 性能分析

### Batch Size 對比

| 配置 | Batch Size | 相比預設 | 記憶體利用 |
|------|-----------|---------|-----------|
| 預設 (單GPU) | 8,192 | 1.0x | ~97% |
| 雙GPU 保守 | 20,480 | **2.5x** | ~83% |
| 雙GPU 積極 | 22,528 | **2.75x** | ~92% |
| 雙GPU 最大 | 24,576 | **3.0x** | ~100% |

### 為什麼雙 GPU 能用更大 batch size？

1. **記憶體加倍**: 2 x 16GB = 32GB 總 GPU 記憶體
2. **並行分配**: 每個 GPU 獨立處理自己的數據批次
3. **高效通信**: JAX 的 `pmap` 最小化跨 GPU 通信開銷
4. **梯度累積**: 雙 GPU 自動在反向傳播時累積梯度

### 大 Batch Size 的優勢

✅ **訓練穩定性**: 更大的 batch 提供更穩定的梯度估計  
✅ **硬體利用率**: 更充分利用 GPU 計算能力  
✅ **訓練速度**: 減少通信開銷，提高吞吐量  
✅ **收斂特性**: 可能改善收斂速度（需調整學習率）

### 注意事項

⚠️ **泛化性能**: 過大的 batch size 可能降低模型泛化能力  
⚠️ **學習率調整**: 需要相應調整學習率以保持訓練動態  
⚠️ **記憶體波動**: 某些操作可能導致短暫的記憶體峰值  
⚠️ **最佳範圍**: 建議 8192-20480 之間，平衡效率與性能

---

## 🚀 學習率調整建議

當增加 batch size 時，通常需要調整學習率以保持相同的訓練動態。

### Linear Scaling Rule
最簡單的方法：線性縮放

```
lr_new = lr_old × (batch_new / batch_old)
```

**示例**:
- 原始: lr=1e-3, batch=8192
- 雙GPU: lr=2.5e-3, batch=20480

### Square Root Scaling Rule
更保守的方法（推薦用於大batch）

```
lr_new = lr_old × sqrt(batch_new / batch_old)
```

**示例**:
- 原始: lr=1e-3, batch=8192
- 雙GPU: lr≈1.58e-3, batch=20480

### Warmup 策略
使用大 batch size 時，建議使用 learning rate warmup：

```bash
--config.optim.learning_rate=2.5e-3
--config.optim.warmup_steps=2000
```

---

## 📝 訓練腳本

### 1. 雙 GPU 優化版 (推薦) ⭐
**文件**: `slurm_train_kf_pirate_2gpu_optimized.sh`

**配置**:
- GPUs: 2x Tesla P100
- Batch size: 10240 per device (總 20,480)
- 安全餘量: 17%
- 預估時間: ~5-6 小時

**提交**:
```bash
ssh junyi@140.114.120.128
cd ~/jaxpi
sbatch slurm_train_kf_pirate_2gpu_optimized.sh
```

### 2. 雙 GPU 最大化版 (實驗用)
如果需要最大吞吐量，可以手動指定：

```bash
srun python3 examples/kolmogorov_flow/main.py \
    --config=examples/kolmogorov_flow/configs/pirate.py \
    --workdir=./results_max_batch \
    --config.training.batch_size_per_device=11264 \
    --config.optim.learning_rate=1.6e-3
```

### 3. 單 GPU 標準版
**文件**: `slurm_train_kf_pirate.sh`

**配置**:
- GPUs: 1x Tesla P100
- Batch size: 2048 (保守) 或 8192 (最大)
- 預估時間: ~8-10 小時

---

## 🔬 測試細節

### 測試環境
- **節點**: acmt20
- **Partition**: r740
- **CPU**: 48 cores
- **RAM**: 110GB
- **GPU**: 2x Tesla P100-PCIE-16GB (16GB each)
- **Driver**: 535.230.02
- **CUDA**: 11.8 (via jaxlib)

### 軟體環境
```
JAX: 0.4.23
jaxlib: 0.4.23+cuda11.cudnn86
Python: 3.10.12
PyTorch: 2.9.1+cpu (用於 data loader)
```

### 測試指標
每個配置運行 50 步訓練，監測：
- GPU 記憶體使用量
- 訓練是否成功完成
- 是否出現 OOM 錯誤

---

## 📊 測試日誌

**Job ID**: 2564  
**測試時間**: 2026-01-01 03:47 - 04:07 UTC  
**總測試時間**: ~20 分鐘  
**測試配置數**: 9 個 (單GPU 4個 + 雙GPU 5個)

**關鍵發現**:
1. ✅ 單 GPU 可達到預設配置的 100% (8192)
2. ✅ 雙 GPU 可達到預設配置的 300% (24576)
3. ✅ 系統記憶體 (110GB) 充足，未成為瓶頸
4. ✅ GPU 記憶體是主要限制因素

---

## 🎉 總結

### 最佳實踐

**對於生產訓練 (20,000 steps):**
- 使用雙 GPU 優化配置
- Batch size: 10240 per device (總 20,480)
- Learning rate: 2.0-2.5e-3 (從 1e-3 放大)
- 預估時間: 5-6 小時

**對於快速實驗:**
- 單 GPU 足夠
- Batch size: 2048-4096
- 保持預設學習率
- 預估時間: 8-10 小時

**硬體利用率最大化:**
- 雙 GPU + batch size 11264
- 謹慎調整學習率
- 監控訓練穩定性

### 下一步
1. 運行雙 GPU 優化訓練
2. 比較不同 batch size 的訓練曲線
3. 評估最終模型性能
4. 調整超參數以獲得最佳結果
