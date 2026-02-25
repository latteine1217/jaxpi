# JAXpi Causal Training 機制詳解

## 🎯 核心概念

**Causal Training** 是一種針對時間演化問題（IVP, Initial Value Problem）的特殊訓練策略，目的是讓神經網路**從前往後**學習時間演化，避免因果關係違反。

---

## 📚 Time Window 訓練流程

### 1. 時間分割（Time Windowing）

**代碼位置**: `train.py` 第 113-128 行

```python
# 將完整時間域分割成多個 windows
num_time_steps = len(t_star) // config.training.num_time_windows
t = t_star[:num_time_steps]

# 每個 window 的時間範圍
t0 = t[0]
t1 = t[-1] + 1.1 * dt
```

**PIRATE 配置**:
- 總時間步: 50
- Windows: 10
- 每個 window: 5 個時間步

**SOAP 配置**:
- 總時間步: 50  
- Windows: 25
- 每個 window: 2 個時間步

---

### 2. 逐窗口訓練（Sequential Window Training）

**代碼位置**: `train.py` 第 129-172 行

```python
for idx in range(config.training.num_time_windows):
    # Step 1: 取得當前 window 的參考解
    u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :]
    
    # Step 2: 初始化模型（使用上一個 window 的最終狀態作為 IC）
    model = models.NavierStokes(config, t, coords, u0, v0, w0, nu)
    
    # Step 3: 訓練當前 window
    model = train_one_window(config, workdir, model, samplers, ...)
    
    # Step 4: 預測下一個 window 的初始條件
    u0 = model.u_ic_pred_fn(params, t_star[num_time_steps], coords[:, 0], coords[:, 1])
```

**關鍵特性**:
- ✅ **每個 window 都是一個獨立的訓練任務**
- ✅ **前一個 window 的終點 = 下一個 window 的起點**
- ✅ **每個 window 訓練 20,000 步**
- ⚠️ **不是從頭訓練整個時間域，而是滾動式前進**

---

### 3. 模型整合方式

#### 方法 A: 獨立模型（當前實作）

**每個 window 訓練一個獨立的模型**:
```
Window 1: model_1(t ∈ [0, 0.1])     → save to ckpt/time_window_1/
Window 2: model_2(t ∈ [0.1, 0.2])   → save to ckpt/time_window_2/
...
Window 10: model_10(t ∈ [0.9, 1.0]) → save to ckpt/time_window_10/
```

**推理時**:
- t ∈ [0, 0.1] → 載入 model_1
- t ∈ [0.1, 0.2] → 載入 model_2
- ...依此類推

**優點**:
- ✅ 每個 window 專注於局部時間範圍，精度高
- ✅ 避免長時間累積誤差

**缺點**:
- ⚠️ 需要儲存 10 個模型
- ⚠️ Window 交界可能有不連續

---

#### 方法 B: Transfer Learning（可選）

**代碼位置**: `train.py` 第 139-147 行

```python
if config.transfer_learning:
    if idx > 0:
        # 從上一個 window 載入 checkpoint
        ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", 
                                  "time_window_{}".format(idx))
        state = restore_checkpoint(model.state, ckpt_path)
        # 用舊參數初始化新模型
        model.state = _create_train_state(config, tx=model.tx, params=state.params)
```

**流程**:
```
Window 1: 隨機初始化 → 訓練 → params_1
Window 2: 用 params_1 初始化 → 訓練 → params_2
Window 3: 用 params_2 初始化 → 訓練 → params_3
...
```

**當前配置**: `config.transfer_learning = False` (未啟用)

**如果啟用**:
- ✅ 模型可以累積之前學到的知識
- ✅ 可能加快後續 window 的收斂
- ⚠️ 但可能過度依賴前面的錯誤

---

## 🔬 Causal Weighting 機制

### 核心思想

**問題**: 在同一個 time window 內，神經網路可能同時學習 t=0.0 和 t=0.1 的狀態，導致**違反因果關係**（未來影響過去）。

**解決**: 使用 **causal weights** 讓網路**優先學習早期時間**，再逐步學習後期時間。

---

### 實作細節

**代碼位置**: `models.py` 第 91-115 行

```python
@partial(jit, static_argnums=(0,))
def res_and_w(self, params, batch):
    # Step 1: 將 batch 按時間排序
    t_sorted = batch[:, 0].sort()
    
    # Step 2: 計算 residual
    ru_pred, rv_pred, rc_pred = self.r_pred_fn(params, t_sorted, batch[:, 1], batch[:, 2])
    
    # Step 3: 分成 num_chunks 個時間塊
    ru_pred = ru_pred.reshape(self.num_chunks, -1)  # [16, batch_size/16]
    
    # Step 4: 計算每個時間塊的平均 loss
    ru_l = jnp.mean(ru_pred**2, axis=1)  # [16]
    
    # Step 5: 計算 causal weight
    ru_gamma = jnp.exp(-self.tol * (self.M @ ru_l))
    
    # Step 6: 取所有 PDE residual 的最小 weight
    gamma = jnp.vstack([ru_gamma, rv_gamma, rc_gamma])
    gamma = gamma.min(0)
    
    return ru_l, rv_l, rc_l, gamma
```

---

### Causal Matrix M

**代碼位置**: `models.py` 第 265 行

```python
self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T
```

**M 的結構** (num_chunks=4 為例):
```
M = [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0]]
```

**作用**:
```python
# 假設 4 個時間塊的 loss 為
loss = [l_0, l_1, l_2, l_3]

# M @ loss 的結果
M @ loss = [
    0,                    # 第 0 塊：不受未來影響
    l_0,                  # 第 1 塊：受第 0 塊影響
    l_0 + l_1,            # 第 2 塊：受第 0, 1 塊影響
    l_0 + l_1 + l_2       # 第 3 塊：受第 0, 1, 2 塊影響
]

# Weight 計算
gamma = exp(-tol * M @ loss)
```

**物理意義**:
- **早期時間塊 (t=0.0)**: weight = exp(0) = 1.0 (全力學習)
- **中期時間塊**: weight = exp(-tol × 前期 loss 累積)
- **後期時間塊**: 只有當前期 loss 很小時，weight 才會大

**效果**: **從前往後逐步學習，符合因果關係**

---

### 參數設定

**PIRATE & SOAP 配置**:
```python
config.weighting.use_causal = True
config.weighting.causal_tol = 1.0
config.weighting.num_chunks = 16
```

**num_chunks = 16** 的意義:
- 每個 window 內的 batch 被分成 16 個時間塊
- 更細緻的因果控制

**causal_tol = 1.0** 的意義:
- 容忍度參數，控制 weight 衰減的速度
- 太大 → 後期時間學習太慢
- 太小 → 因果約束太弱

---

## 📊 完整訓練流程（以 PIRATE 為例）

### 全域視角

```
總時間域: t ∈ [0, 1.0], 50 個時間步
分割為 10 個 windows，每個 5 步

Window 1: t ∈ [0.0, 0.1]
    ├─ IC: u0 = u_ref[0] (ground truth)
    ├─ 訓練 20,000 步
    │  └─ 每步內：16 個時間塊，causal weighting
    ├─ 儲存: ckpt/time_window_1/
    └─ 預測: u_final = model(t=0.1)

Window 2: t ∈ [0.1, 0.2]
    ├─ IC: u0 = u_final (來自 Window 1 的預測)
    ├─ 訓練 20,000 步
    ├─ 儲存: ckpt/time_window_2/
    └─ 預測: u_final = model(t=0.2)

...

Window 10: t ∈ [0.9, 1.0]
    ├─ IC: u0 = 來自 Window 9
    ├─ 訓練 20,000 步
    └─ 儲存: ckpt/time_window_10/

總訓練步數: 10 windows × 20,000 = 200,000 iterations
```

---

### 單個 Window 內部（Window 3 為例）

```
時間範圍: t ∈ [0.2, 0.3]
初始條件: u0 = model_2 預測的 t=0.2 狀態

每個訓練步:
  1. Sample IC batch (t=0.2 的空間點)
  2. Sample residual batch (t ∈ [0.2, 0.3] 的時空點)
  3. 計算 IC loss: ||u(t=0.2) - u0||²
  4. 計算 residual loss with causal weighting:
     - 將 batch 分成 16 個時間塊
     - 塊 0: t ∈ [0.20, 0.206]  → weight ≈ 1.0
     - 塊 1: t ∈ [0.206, 0.213] → weight = exp(-tol × loss_0)
     - 塊 2: t ∈ [0.213, 0.219] → weight = exp(-tol × (loss_0 + loss_1))
     - ...
     - 塊 15: t ∈ [0.294, 0.30] → weight = exp(-tol × Σ(前15塊的loss))
  5. 總 loss = w_ic × IC_loss + Σ(gamma_i × residual_loss_i)
  6. 更新參數

重複 20,000 步
```

---

## 🎯 Causal Training 的優勢

### 1. 符合物理因果性
- ✅ 過去決定未來，不是未來影響過去
- ✅ 符合 Navier-Stokes 的初值問題本質

### 2. 數值穩定性
- ✅ 避免長時間累積誤差
- ✅ 每個 window 都是短期預測，精度高

### 3. 訓練效率
- ✅ 每個 window 只需關注局部時間
- ✅ 不需要在整個時間域上同時滿足 PDE

---

## ⚠️ 潛在問題

### 1. 誤差累積
- 每個 window 的初始條件來自前一個的預測
- IC 誤差會累積到後面的 windows
- **Window 10 的精度 < Window 1**

### 2. Window 交界不連續
- 不同 model 在交界處可能不平滑
- 需要評估 u(t=0.1⁻) vs u(t=0.1⁺)

### 3. 儲存開銷
- 需要儲存 10 個獨立模型
- 每個模型 ~數十 MB

---

## 🔍 實驗設計對比

| 特性 | PIRATE | SOAP |
|------|--------|------|
| **Windows** | 10 | 25 |
| **每個 window 長度** | 長 (5 步) | 短 (2 步) |
| **優勢** | 更快完成 | 更少誤差累積 |
| **劣勢** | 累積誤差大 | 訓練時間長 |

**為什麼 SOAP 用 25 windows?**
- 更短的 window → 每次預測只需跨越更短時間
- 減少單次預測的誤差
- 但代價是需要更多次的 window 銜接

---

## 📚 相關論文

**Causal Training 的理論基礎**:
- Wang et al., "When and why PINNs fail to train: A neural tangent kernel perspective" (2020)
- Wang et al., "Respecting causality for training physics-informed neural networks" (2022)

**核心思想**: 
傳統 PINN 在訓練時會同時考慮所有時間點，導致梯度流動混亂。Causal training 強制網路按照時間順序學習，避免因果關係違反。

---

## 🎓 總結

### Time Window 策略
- **不整合成單一模型**
- **10 個獨立模型，各自負責不同時間段**
- **滾動式前進：前一個的終點 = 下一個的起點**

### Causal Weighting
- **Window 內部的因果約束**
- **優先學習早期時間，再逐步學習後期**
- **透過 16 個時間塊 + 權重矩陣 M 實現**

### 完整流程
```
全域: 10 windows (獨立訓練)
    └─ Window 1: 20k steps
        └─ 每步: 16 chunks (causal weighting)
    └─ Window 2: 20k steps (IC 來自 Window 1)
        └─ 每步: 16 chunks
    ...
```

**這是一個兩層的因果機制**：
1. **外層 (Time Window)**: 宏觀時間順序
2. **內層 (Causal Weight)**: 微觀時間順序

---

**創建時間**: 2026-01-01  
**作者**: JAXpi Context Analysis
