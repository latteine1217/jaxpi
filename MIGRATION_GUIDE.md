# JAXpi 遷移指南：升級到最新依賴版本

## 背景

本專案已更新以支援最新版本的 JAX/Optax 生態系統，並移除了相容性有問題的可選依賴（SOAP, PSGD）。

## 主要變更

### 1. 移除的功能

#### SOAP Optimizer
- **原因**: `soap_jax` 與 optax 0.1.9 不相容，需要的 `tree_update_moment` 函數不存在
- **替代方案**: 使用標準 Adam 或 AdamW optimizer
- **影響的配置檔案**:
  - `examples/kolmogorov_flow/configs/soap.py` → 改用 Adam，hidden_dim 保持 384
  - 其他 `*_soap.py` 配置檔案 → 建議改用對應的 `pirate.py`

#### PSGD/Kron Optimizer
- **原因**: 依賴管理複雜，且不是主流方法
- **替代方案**: 使用 Adam, AdamW, Muon 等 optax 內建 optimizer

#### Schedule-Free Wrapper
- **原因**: `optax.contrib.schedule_free` 在舊版 optax 中不存在
- **替代方案**: 直接使用 optimizer，或手動實現 schedule-free 邏輯

### 2. 新增的 Optimizer

#### AdamW
```python
config.optim.optimizer = "AdamW"
config.optim.weight_decay = 0.01  # L2 regularization
```

#### 支援的 Optimizer 列表
- `Adam` - 標準 Adam optimizer（默認推薦）
- `AdamW` - Adam with decoupled weight decay
- `Muon` - 新型 optimizer，可能有更好的收斂性
- `Lamb` - Layer-wise Adaptive Moments optimizer
- `Adagrad` - Adaptive gradient algorithm
- `RMSProp` - Root Mean Square Propagation

### 3. 新增的配置選項

#### 梯度裁剪（Gradient Clipping）
```python
config.optim.grad_clip = 1.0  # Clip gradients by global norm
```

如果 `grad_clip > 0.0`，會自動套用 `optax.clip_by_global_norm`。

### 4. 依賴版本更新

#### 更新前（舊版）
```python
# setup.py
install_requires=[
    "jax",
    "jaxlib",
    "optax",
    "flax",
    # ... 沒有版本約束
]
```

#### 更新後（新版）
```python
# setup.py
install_requires=[
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "optax>=0.1.7",
    "flax>=0.7.0",
    "numpy>=1.22.0,<2.0.0",  # JAX 0.4.x 需要 numpy 1.x
    "scipy>=1.9.0,<2.0.0",
    # ... 明確的版本約束
]
```

## 遷移步驟

### 對於使用者

#### 1. 更新配置檔案

如果你的配置使用了 `optimizer = "Soap"`：

```python
# 舊配置
config.optim.optimizer = "Soap"
config.optim.schedule_free = True

# 新配置
config.optim.optimizer = "Adam"  # 或 "AdamW"
config.optim.grad_clip = 1.0  # 可選：添加梯度裁剪
```

#### 2. 移除 schedule_free 設定

```python
# 移除這一行（不再支援）
config.optim.schedule_free = True

# 如需梯度裁剪，改用
config.optim.grad_clip = 1.0
```

#### 3. 重新安裝依賴

```bash
# 方法 A: 從 setup.py 安裝
pip install -e .

# 方法 B: 手動安裝關鍵依賴（推薦用於 CUDA 環境）
pip install 'jax[cuda11_local]==0.4.23' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install 'optax>=0.1.7' 'flax>=0.7.0' 'numpy<2.0' 'scipy<2.0'
```

### 對於開發者

#### 1. Checkpoint 相容性

現有的 checkpoint 應該仍然相容，因為：
- 模型架構沒有改變
- TrainState 結構沒有改變
- 只是 optimizer 類型改變

但建議測試載入舊 checkpoint 並驗證是否正常工作。

#### 2. 訓練腳本更新

遠端伺服器的訓練腳本需要更新：

```bash
# 舊的 SOAP 訓練腳本
slurm_train_2gpu_soap.sh  # 需要更新配置

# 建議：重新命名並修改
slurm_train_2gpu_adam_large.sh  # 使用 Adam + larger hidden_dim
```

#### 3. 測試清單

- [ ] 訓練可以正常啟動
- [ ] 損失正常收斂
- [ ] 記憶體使用量沒有異常增加
- [ ] Wandb logging 正常工作
- [ ] Checkpoint 保存/載入正常

## 效能影響

### Adam vs SOAP

根據文獻，SOAP 的主要優勢是：
- Schedule-free（不需要學習率衰減）
- 更快的收斂速度（某些情況下）

使用 Adam 替代時的建議：
1. **保持學習率調度**：使用 warmup + exponential decay
2. **可能需要更多 training steps**：SOAP 聲稱可以減少 10-20% 的訓練時間
3. **考慮使用 AdamW**：如果模型傾向過擬合，weight decay 可能有幫助

### Muon Optimizer（實驗性）

如果想要嘗試更先進的 optimizer：
```python
config.optim.optimizer = "Muon"
config.optim.learning_rate = 1e-3  # 可能需要調整
```

Muon 是一個較新的 optimizer，在某些任務上表現優於 Adam。

## 故障排除

### 問題 1: `ImportError: cannot import name 'schedule_free'`

**原因**: 配置檔案中仍有 `schedule_free = True`

**解決方案**: 移除或設為 `False`，或改用 `grad_clip`

### 問題 2: `ValueError: Optimizer 'Soap' not supported`

**原因**: SOAP optimizer 已被移除

**解決方案**: 改用 `Adam` 或其他支援的 optimizer

### 問題 3: `AttributeError: 'Config' object has no attribute 'weight_decay'`

**原因**: 使用 AdamW 但沒有提供 `weight_decay` 參數

**解決方案**: 
```python
config.optim.optimizer = "AdamW"
config.optim.weight_decay = 0.01  # 添加這一行
```

## 回滾計畫

如果升級後遇到無法解決的問題，可以回滾到舊版本：

```bash
# 1. 恢復舊的 models.py
git checkout <old_commit> jaxpi/models.py

# 2. 恢復舊的 setup.py
git checkout <old_commit> setup.py

# 3. 重新安裝舊版依賴
pip install 'jax==0.4.23' 'optax==0.1.9' 'flax==0.7.5'
```

## 參考資料

- **JAX Release Notes**: https://github.com/google/jax/releases
- **Optax Documentation**: https://optax.readthedocs.io/
- **AdamW Paper**: https://arxiv.org/abs/1711.05101
- **Muon Optimizer**: https://arxiv.org/abs/2310.05374

## 更新日誌

- **2026-01-02**: 初始版本
  - 移除 SOAP 和 PSGD 支援
  - 添加 AdamW 和梯度裁剪支援
  - 更新依賴版本約束
