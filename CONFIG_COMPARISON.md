# PIRATE vs SOAP 配置詳細對比

**對比日期**: 2026-01-05  
**目的**: 找出除了 optimizer、time_windows 之外的所有配置差異

---

## 📋 完整配置對比表

| 配置項 | PIRATE | SOAP | 差異 | 影響 |
|--------|--------|------|------|------|
| **Optimizer** | Adam | Soap | ✅ 已知 | 優化器算法 |
| **schedule_free** | False | True | ✅ 已知 | SOAP 特性 |
| **num_time_windows** | 10 | 25 | ✅ 已知 | 時間離散化策略 |
| **save_every_steps** | 10000 | 5000 | ⚠️ **新發現** | 保存頻率 |
| **All other params** | 相同 | 相同 | ✅ | 控制變量 |

---

## ⚠️ 新發現的差異

### 1. **save_every_steps（Checkpoint 保存頻率）**

| 參數 | PIRATE | SOAP | 差異 |
|------|--------|------|------|
| `saving.save_every_steps` | 10000 | 5000 | SOAP 保存更頻繁 |

**影響分析**：
- ❌ **不影響訓練結果**：僅影響 checkpoint 保存頻率
- ✅ **不影響比較公平性**：不涉及模型訓練過程
- 📝 **原因推測**：SOAP 有 25 個窗口，可能為了更細粒度的中間結果保存

**結論**：這個差異**不影響**性能比較，僅是工程上的便利性考慮。

---

## ✅ 確認相同的關鍵配置

### 網路架構（完全相同）
```python
arch.arch_name = "PirateNet"
arch.num_layers = 3
arch.hidden_dim = 256                    # ✅ 相同
arch.out_dim = 3
arch.activation = "tanh"
arch.fourier_emb.embed_scale = 2.0
arch.fourier_emb.embed_dim = 256         # ✅ 相同
arch.reparam.type = "weight_fact"
arch.reparam.mean = 1.0
arch.reparam.stddev = 0.1
```

### 優化器基礎參數（完全相同）
```python
optim.beta1 = 0.9                        # ✅ 相同
optim.beta2 = 0.999                      # ✅ 相同
optim.eps = 1e-8                         # ✅ 相同
optim.learning_rate = 1e-3               # ✅ 相同
optim.decay_rate = 0.9                   # ✅ 相同
optim.decay_steps = 2000                 # ✅ 相同
optim.warmup_steps = 2000                # ✅ 相同
optim.grad_accum_steps = 0               # ✅ 相同
```

### 訓練配置（除窗口數外相同）
```python
training.max_steps = 20000               # ✅ 相同（每窗口）
training.batch_size_per_device = 4096    # ✅ 相同
```

**總迭代次數差異**：
- PIRATE: 10 窗口 × 20000 步/窗口 = 200k 總迭代
- SOAP: 25 窗口 × 20000 步/窗口 = 500k 總迭代

### 損失權重配置（完全相同）
```python
weighting.scheme = "grad_norm"           # ✅ 相同
weighting.init_weights = {
    "u_ic": 100.0,                       # ✅ 相同
    "v_ic": 100.0,                       # ✅ 相同
    "ru": 1.0,                           # ✅ 相同
    "rv": 1.0,                           # ✅ 相同
    "rc": 1.0                            # ✅ 相同
}
weighting.momentum = 0.9                 # ✅ 相同
weighting.update_every_steps = 1000      # ✅ 相同
weighting.use_causal = True              # ✅ 相同
weighting.causal_tol = 1.0               # ✅ 相同
weighting.num_chunks = 16                # ✅ 相同
```

### 其他配置（完全相同）
```python
config.mode = "train"                    # ✅ 相同
config.transfer_learning = False         # ✅ 相同
config.time_fraction = 1.0               # ✅ 相同
config.seed = 42                         # ✅ 相同
config.input_dim = 3                     # ✅ 相同
logging.log_every_steps = 100            # ✅ 相同
saving.num_keep_ckpts = 2                # ✅ 相同
```

---

## 📊 配置差異總結

### 已知的核心差異（3 項）
1. ✅ **optimizer**: Adam vs SOAP
2. ✅ **schedule_free**: False vs True
3. ✅ **num_time_windows**: 10 vs 25

### 新發現的次要差異（1 項）
4. ⚠️ **save_every_steps**: 10000 vs 5000（不影響性能）

### 相同的配置（>50 項）
- ✅ 網路架構（層數、維度、激活函數等）
- ✅ 學習率及調度參數
- ✅ 批量大小
- ✅ 每窗口訓練步數
- ✅ Adam 的 beta1/beta2/eps
- ✅ 損失權重配置
- ✅ 因果權重設置
- ✅ 隨機種子
- ✅ 所有其他超參數

---

## 🎯 對比較公平性的影響

### ✅ 不影響比較的差異
1. **save_every_steps** (10000 vs 5000)
   - 僅影響 checkpoint 保存頻率
   - 不改變訓練過程
   - 不影響最終結果

### ⚠️ 影響比較的差異（已知）
1. **optimizer** (Adam vs SOAP)
   - 這是實驗的核心變量
2. **num_time_windows** (10 vs 25)
   - 時間離散化策略的差異
   - 導致總訓練次數差異（200k vs 500k）
3. **schedule_free** (False vs True)
   - SOAP 的內建特性

---

## ✅ 結論

### 發現摘要
除了已知的 3 個核心差異（optimizer、schedule_free、num_time_windows）外，
**只有 1 個次要差異**：`save_every_steps` (10000 vs 5000)

### 影響評估
**save_every_steps 差異不影響性能比較**，因為：
- 不改變訓練算法
- 不影響模型更新
- 不影響損失計算
- 僅影響工程便利性（保存頻率）

### 對報告的影響
**無需更新報告**，因為：
- 這個差異不影響核心結論
- 不改變時間對齊比較的有效性
- 不影響"時間窗口策略是關鍵因素"的結論

### 實驗設計評估
✅ **配置控制良好**：
- 除了 3 個有意的變量外，其他配置完全相同
- 符合對比實驗的基本要求
- 唯一的次要差異（save_every_steps）不影響結果

---

## 📋 建議

### 對當前實驗
- ✅ 當前配置已足夠公平
- ✅ save_every_steps 差異可以忽略
- ✅ 報告無需因此更新

### 對未來實驗（控制變量）
如果要進行嚴格的控制變量實驗，建議：

**選項 A: 只改變優化器**
```python
# 兩個配置完全相同，除了：
PIRATE: optimizer="Adam", schedule_free=False
SOAP:   optimizer="Soap", schedule_free=True
# 其他包括 num_time_windows=10 都相同
```

**選項 B: 保持當前時間窗口策略**
```python
# 保持當前的窗口數，只改變優化器：
PIRATE_25: optimizer="Adam", num_time_windows=25
SOAP_25:   optimizer="Soap", num_time_windows=25
# 可以直接比較相同時間窗口下的優化器差異
```

---

**分析者**: AI Assistant  
**分析日期**: 2026-01-05  
**配置文件**:
- `~/jaxpi/examples/kolmogorov_flow/configs/pirate.py`
- `~/jaxpi/examples/kolmogorov_flow/configs/soap.py`

