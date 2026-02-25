# 2507.08972v2（Simulating Three-dimensional Turbulence with PINNs）對照本專案報告

本報告根據 `2507.08972v2.pdf`（arXiv:2507.08972v2, 2025-10-11）整理該論文相對於本 repo（PirateNets/JAX-PI 分支）的**方法改進**與**超參數/設定差異**，並對照本 repo 目前在 `examples/kolmogorov_flow` 的可對應實作。

**最後更新：2026-01-09** - 加入架構深度的精確計數慣例與設計哲學分析

---

## 1) 論文做了哪些「關鍵改進」（相對於僅有 PirateNet 的基礎 PINN）

論文的核心主張不是提出單一新元件，而是把多個已知能改善 PINN 可訓練性的技巧「組裝成一個可在湍流上工作」的 training stack，並給出統一超參數（Table 2）以提升可重現性：

1. **大容量、可深訓練的網路骨幹：PirateNet + RFF + RWF**
   - 以 **PirateNet（Residual Adaptive / gating + 可學的 residual nonlinearity α）** 降低深網在 PINN 中的可訓練性問題與 spectral bias。
   - **Random Fourier Features (RFF)** 強化高頻/多尺度表徵能力。
   - **Random Weight Factorization (RWF)** 讓每個 neuron 具備類似「自適應 learning-rate」的效果，加速收斂與改善 ill-conditioning。

2. **時間因果性的訓練（Causal training）**
   - 用隨時間遞減/遞增的權重 `w(t)` 讓 PINN 先把早期時間學好，再逐步關注後期，避免「後面先學好、前面學不穩」的非物理訓練路徑。

3. **自適應 loss 權重（Self-adaptive weighting / Grad-norm 類方法）**
   - 以梯度範數平衡不同 loss term，降低 IC/BC/PDE residual 量級不平衡導致的訓練不穩。
   - 以 EMA 形式更新，控制震盪。

4. **準二階/預條件化的優化（SOAP optimizer）**
   - SOAP 在 layer-wise eigenspace 內做 Adam 更新，緩解「不同 loss term 梯度方向衝突」與 PINN 的 ill-conditioned landscape。

5. **Time-marching + Transfer learning（時間窗口 + 轉移學習）**
   - 把長時間區間拆成多個 window；每個 window 訓練一個 PINN。
   - 用上一個 window 的收斂參數初始化下一個 window，並用前一個 window 的末端輸出提供下一窗的初值，降低長時間混沌系統的優化難度。

6. （僅對特定 case）**Multi-stage 訓練以提升精度（Taylor–Green vortex）**
   - 在特定條件下（stage-0 PDE residual 已經極低 < 10⁻⁸）才有幫助；論文也明確指出對 Kolmogorov/channel flow 未必有效，因此不使用。

7. **評估指標從「點誤差」推進到「湍流統計」**
   - 除了相對誤差，強調能量頻譜、動能/渦量（enstrophy）演化、Reynolds stress、近壁統計（wall units）等，作為湍流是否「學對」的標準。

---

## 2) 論文的「統一超參數配置」（Table 2 + Appendix C 摘要）

以下為論文在三個 turbulence benchmark 共同採用的設定（原文 Table 2 + Appendix C）：

### 2.1 網路（PirateNet）
- Depth（residual blocks）：**2**
- Width（hidden size）：**768**
- Activation：**Swish**
- RFF：`B ~ N(0, 2)`（論文寫法；實作上需釐清是 **variance=2** 或 **std=2**）
- RWF：`µ=1.0, σ=0.1`

### 2.2 優化（Optimization）
- Optimizer：**SOAP**
- `β1=0.9, β2=0.999`
- Eigenspace update（precondition）frequency：**每 2 step**
- Base LR：`1e-3`
- Warmup：Table 2 標為 **2,000** steps（正文同時出現「前 22,000 step」的字樣；以 Table 2 為準較合理）
- Decay：每 **2,000** steps 乘上 `0.9`（論文文字描述為「factor of 0.9 every 2,000 steps」）
- 額外：**schedule-free 方法（β=0.9）+ global-norm gradient clipping=1**

### 2.3 訓練（Training）
- Iterations：**每個 time window 1e5**
- Batch size：**8,192**
- Time window size：**0.1**
- Loss weighting：Grad-norm 類方法，**每 1,000 iters 更新**
- Causal tolerance：`ε=1.0`

### 2.4 Benchmark-specific 補充（論文正文）
- **2D Kolmogorov flow**：`Re = 1e6`，`T = 5`，forcing `f = [0.1 sin(4πy), 0]`，DNS 參考為 pseudo-spectral `2048×2048`。
- **3D Taylor–Green vortex**：`Re = 1600`；time window：早期 `0.2`（t∈[0,8]），後期縮到 `0.05`；並在 t≤8 的 window 內使用 2-stage（`ε0=1, ε1=1e-3`）做 residual refinement。
- **3D turbulent channel flow**：`Re_τ ≈ 395`，域長 `Lx=2π, Lz=π, Ly=2`；x/z periodic、y 方向 no-slip；以 wall units 的 mean profile、Reynolds stress、vorticity RMS、energy spectra 驗證。

---

## 3) 對照本 repo：哪些已經有、哪些不一致、參數差在哪

本 repo 其實已經涵蓋論文 stack 的多數元件（至少在 `examples/kolmogorov_flow` 這條實驗線上）。下面用「是否具備」與「參數差異」做對照。

### 3.1 元件對應（Implementation mapping）

- PirateNet + α=0 初始化：`jaxpi/archs.py`
  - `PirateNet` + `PIModifiedBottleneck` 的 `alpha` 預設 `nonlinearity=0.0`，對應論文「α(l)=0 起始」的做法。
- RFF：`jaxpi/archs.py#FourierEmbs`
  - config 以 `fourier_emb.embed_scale` 控制 `Normal(std=embed_scale)` 初始化。
- RWF：`jaxpi/archs.py#Dense(reparam=weight_fact)`
  - 以 `mean/stddev` 生成 scale 參數後 `exp()`，對應論文的 `w = exp(s)·v`。
- Causal training（chunked）：`examples/kolmogorov_flow/models.py#res_and_w`
  - 以 chunked 方式估計 `gamma`，對應論文的 causality weighting 思路。
- Grad-norm loss weighting：`jaxpi/models.py#compute_weights`
  - repo 是「對每個 loss component」做 grad-norm 平衡（例如 `u_ic,v_ic,ru,rv,rc`），而論文表述偏向 ic/bc/pde 三項全域權重。
- SOAP optimizer：`jaxpi/models.py#_create_optimizer`
  - `soap(..., precondition_frequency=2)` 對應論文的 eigenspace 更新頻率 f=2。
- Time-marching：`examples/kolmogorov_flow/train.py`
  - 已是逐 window 訓練；並會把前一窗末端的 PINN 輸出當下一窗 IC。
- Transfer learning（參數初始化）：`examples/kolmogorov_flow/train.py`
  - 具備但預設關閉：`config.transfer_learning = False`（`examples/kolmogorov_flow/configs/*.py`）。

### 3.2 架構深度的精確計數慣例（關鍵釐清）

**PirateNet 的層數計算需要明確口徑，避免混淆：**

```python
# PirateNet 架構組成 (jaxpi/archs.py)
├─ U gate: Dense(hidden_dim)           # 1 個 Dense
├─ V gate: Dense(hidden_dim)           # 1 個 Dense
├─ L 個 PIModifiedBottleneck (for-loop，L = config.arch.num_layers)
│   └─ 每個 block 固定 3 個 Dense
└─ Output: Dense(out_dim) 或 jnp.dot   # 1 個線性投影

Dense 總數公式 = 2 (gates) + 3L (blocks) + 1 (output) = 3L + 3
```

**三種計數口徑對比：**

| 配置 | **L** (residual blocks) | **論文口徑** (blocks 內) | **完整口徑** (所有 Dense) | hidden_dim |
|------|------------------------|----------------------|------------------------|-----------|
| **當前專案實際** | `L=3` | 3×3 = **9 layers** | 2+9+1 = **12 layers** | 256 |
| **PirateNet 預設** | `L=2` | 3×2 = **6 layers** | 2+6+1 = **9 layers** | 256 |
| **論文配置** | `L=2` | 3×2 = **6 layers** | 2+6+1 = **9 layers** | 768 |

**重要說明：**
- 論文 Table 2 的 `Depth (residual blocks) = 2` 指的是 **L=2**（for-loop 次數）
- 論文正文 "two residual blocks (six layers total)" 的「six layers」只計算 blocks 內的 Dense（3×2=6），不含 U/V gates 與 output
- 當前專案 configs 多數設定 `arch.num_layers=3`（如 `configs/soap.py`、`configs/pirate.py`），因此 **L=3**
- **關鍵發現：當前專案比論文「多 1 個 residual block」（L=3 vs L=2）**

### 3.3 參數差異總表（Paper vs 本 repo Kolmogorov 設定）

以本 repo 的 `examples/kolmogorov_flow/configs/soap.py`（以及同目錄的 `pirate.py`）作為直接可比的 baseline：

| 類別 | 論文（2507.08972v2） | 本 repo（configs/*.py） | 差異解讀 |
|---|---:|---:|---|
| 任務 Re | Kolmogorov：`1e6` | `1e4` | 論文挑戰更硬（尺度分離更極端）。 |
| T / 時間跨度 | Kolmogorov：`T=5` | 資料：`t_end≈1.96` | 時間長度不同，影響統計收斂與混沌累積。 |
| forcing（Kolmogorov） | `[0.1 sin(4πy), 0]` | `2 sin(4πy)` | 力項量級差異 20 倍。 |
| **PirateNet depth (L)** | **2 blocks** | **3 blocks** | **當前專案更深 1 個 block（多 3 個 Dense）** |
| **PirateNet layers (完整)** | **9 layers** | **12 layers** | **當前：2+9+1，論文：2+6+1** |
| width | 768 | 256 | 論文寬度 3 倍。 |
| **參數量估計** | **~4.7M** | **~0.7M** | **論文約 6.6 倍參數量** |
| activation | Swish | tanh | 表徵/梯度性質不同。 |
| RWF (µ,σ) | (1.0, 0.1) | (1.0, 0.1) | ✓ 一致。 |
| RFF scale | `N(0,2)` | `embed_scale=2.0`（std=2） | 需確認論文 N(0,2) 的「2」是 variance 還是 std。 |
| optimizer | SOAP | SOAP | ✓ 一致。 |
| SOAP precond freq | 2 | 2 | ✓ 一致。 |
| LR schedule | 1e-3 + warmup + decay | 同左 | ✓ 一致（decay 預設連續非 staircase）。 |
| **gradient clipping** | **1.0 (global norm)** | **無（SOAP 跳過）** | ⚠️ **實作矛盾點 1** |
| schedule-free | 有（β=0.9） | `True` 但不套用到 SOAP | ⚠️ **實作矛盾點 2** |
| batch size | 8192 | 4096/device × 2 GPU = 8192 | ✓ 總量一致。 |
| iters / window | 1e5 | 2e4 | 本 repo 每窗訓練量少 5×。 |
| window size | 0.1 | ~0.08（取決於 num_windows） | 接近但不完全相同。 |
| **transfer learning** | **開啟** | **關閉** | ⚠️ **論文視為必要元件** |
| causal tol | 1.0 | 1.0 | ✓ 一致。 |
| causal chunks | 未明確 | 16 | 需實驗驗證。 |
| loss weight update | 每 1000 iters | 1000 | ✓ 一致。 |

---

## 4) 設計哲學的根本差異：深窄 vs 淺寬

### 4.1 架構設計策略對比

| 維度 | 當前專案 | 論文配置 | 物理意義 |
|------|---------|---------|---------|
| **策略** | **深而窄** (Deep & Narrow) | **淺而寬** (Shallow & Wide) | - |
| L (blocks) | 3 | 2 | 更多非線性變換 vs 更短梯度路徑 |
| hidden_dim | 256 | 768 | 有限表徵空間 vs 豐富並行容量 |
| 總 layers | 12 | 9 | 更深抽象 vs 更扁平多尺度 |
| 參數量 | 0.7M | 4.7M | 節省資源 vs 充分表徵 |

### 4.2 湍流物理的啟發

**為什麼論文選擇「淺而寬」？**

```
湍流的物理特性：
- 能量級聯：E(k) ~ k^(-5/3) 涵蓋多個數量級的波數
- 不同尺度的渦旋「同時」交互，而非「層次化」
- 需要並行表徵所有活躍尺度，而非逐層抽象

深度網絡的限制（針對湍流）：
→ 每層學習的是「抽象特徵」的層次結構（適合 CV/NLP）
→ 但湍流不需要「layer 1: 大渦 → layer 2: 中渦 → layer 3: 小渦」
→ 而需要「單層同時捕捉所有尺度的互動」

寬度網絡的優勢：
→ 768 dim 可以同時分配給：
   - 100 neurons 捕捉大尺度結構
   - 300 neurons 捕捉慣性範圍 (k^-5/3)
   - 200 neurons 捕捉小尺度耗散
   - 168 neurons 捕捉時間演化
```

**實驗證據：**
- 論文 Figure 2(e) 的 Kolmogorov flow 能量譜完美復現 k⁻⁵/³ 和 k⁻³ 兩段
- 這暗示網絡確實學到了「跨尺度」的並行表徵
- 寬網絡的並行容量可能是關鍵因素

---

## 5) 關鍵實作細節的深入分析

### 5.1 SOAP + Gradient Clipping 的矛盾

**論文要求：**
> "schedule-free method with momentum β = 0.9, together with **gradient clipping (global norm of 1)**"

**當前實作（`jaxpi/models.py:141-146`）：**
```python
# SOAP 本身就是 schedule-free optimizer，不需要額外 wrapper
if config.schedule_free and config.optimizer != "Soap":
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # ← 只對非 SOAP 套用
        optax.contrib.schedule_free(tx, lr, b1=config.beta1)
    )
```

**矛盾點：**
- SOAP 被視為「天生 schedule-free」，因此跳過 Optax 的 schedule-free wrapper
- 但也因此**跳過了 gradient clipping**
- 論文明確要求 SOAP + clipping，但當前實作未執行

**建議修正：**
```python
elif config.optimizer == "Soap":
    tx = soap(
        learning_rate=lr,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=0.0,
        precondition_frequency=2
    )
    # 論文明確要求 clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        tx
    )
```

### 5.2 Transfer Learning 的狀態轉移策略

**論文描述：**
> "trainable parameters of models for later time windows are initialized using the converged model parameters for earlier time windows"

**當前實作（`train.py:176-183`）：**
```python
if config.transfer_learning:
    state = restore_checkpoint(model.state, ckpt_path)
    model.state = _create_train_state(
        config, tx=model.tx, params=state.params  # ← 只傳遞參數
    )
```

**實作方案：**
- **方案 A（當前）：** 僅轉移網絡權重，優化器狀態重置
- **方案 B（可能更優）：** 同時轉移 SOAP 的協方差矩陣

**論文未說明採用哪種方案，但考慮到：**
- SOAP 的協方差矩陣（eigenspace）計算成本極高
- 轉移優化器狀態可以繼承前一窗口的「二階信息」
- 可能加速後續窗口的收斂

**建議實驗驗證兩種方案的效果差異。**

### 5.3 Multi-stage Training 的適用條件

**論文的微妙說法（Appendix C）：**
> "effective only when the first-stage network already achieves sufficiently low PDE residuals (e.g., < 10⁻⁸)"

**批判性思考：**
```
這是一個「循環論證」：
- 需要 Stage-0 residual < 10⁻⁸ 才能用 Stage-1
- 但如果 Stage-0 已經達到 10⁻⁸，Stage-1 的增益在哪？

可能的真實原因：
→ Stage-0 能達到 10⁻⁸ 的「點誤差」（L2 error）
→ 但在「統計量」（能譜、渦量峰值）上仍有系統偏差
→ Stage-1 的線性化修正可以消除這些高階誤差

論文未說明的關鍵：
1. Stage-1 的線性化是相對哪個「參考解」？
2. ε₁ = 0.001 的選擇依據是什麼？
3. 為何對 Kolmogorov/Channel 無效？
```

---

## 6) 對齊論文的改進建議（修正版）

### 6.1 立即執行（成本最低，驗證核心假說）

**優先級 🔴 P0（本週內）：**

```python
# configs/paper_aligned_v1.py

# 1. 對齊深度（減少 1 個 block）
arch.num_layers = 2           # L: 3→2（對齊論文）
arch.hidden_dim = 256         # 先維持寬度
arch.activation = "swish"     # 對齊論文

# 2. 修復 SOAP 實作矛盾
# 在 jaxpi/models.py 加入 gradient clipping

# 3. 打開 transfer learning
config.transfer_learning = True

# 4. 延長訓練（折衷方案）
training.max_steps = 50000    # 20K → 50K

預期成本：約 40 GPU hours (2× RTX 3090)
預期效益：
  - 驗證「淺寬優於深窄」的設計哲學
  - 確認 L=2 是否足夠（若成功，說明 L=3 冗餘）
  - 檢驗 gradient clipping 的穩定性提升
```

### 6.2 次階段執行（如果 P0 驗證成功）

**優先級 🟡 P1（1-2 週）：**

```python
# configs/paper_aligned_v2.py

arch.num_layers = 2           # 維持淺層
arch.hidden_dim = 384         # 漸進增加寬度（+50%）
arch.activation = "swish"

training.max_steps = 50000

預期成本：約 60 GPU hours
參數量：~1.3M（增加 1.8 倍）
預期效益：接近論文性能的 50-60%
```

**優先級 🟡 P1+：**

```python
# configs/paper_aligned_v3.py

arch.num_layers = 2
arch.hidden_dim = 512         # 中等寬度（+100%）

預期成本：約 80 GPU hours
參數量：~2.4M（增加 3.3 倍）
預期效益：接近論文性能的 70-80%
```

### 6.3 完整對齊（最終目標）

**優先級 🟢 P2（1 個月）：**

```python
# configs/paper_full.py

# 1. 完整網絡容量
arch.num_layers = 2
arch.hidden_dim = 768         # 完全對齊論文
arch.activation = "swish"

# 2. 論文級別訓練預算
training.max_steps = 100000

# 3. 物理參數對齊（需重新生成 DNS 數據）
Re = 1e6
T = 5.0
forcing = [0.1 * jnp.sin(4 * jnp.pi * y), 0]

預期成本：約 150 GPU hours (2× RTX 3090)
參數量：~4.7M（完全對齊）
預期效益：複現論文結果
```

---

## 7) 實驗驗證計劃

### 7.1 深度消融研究（Depth Ablation）

**固定 hidden_dim=256，變化 L：**

| Config | L | 總 layers | 參數量 | 預期發現 |
|--------|---|----------|--------|---------|
| ultra_shallow | 1 | 6 | ~0.4M | 容量是否不足？ |
| paper_depth | 2 | 9 | ~0.52M | 論文選擇的理由 |
| current | 3 | 12 | ~0.72M | 是否冗餘？ |
| over_deep | 4 | 15 | ~0.92M | 是否損害性能？ |

**評估指標：**
- 能量譜 E(k) 的 k⁻⁵/³ 擬合優度
- Enstrophy 峰值位置與幅度
- PDE residual 收斂曲線
- 訓練時間（wall-clock）

### 7.2 寬度消融研究（Width Ablation）

**固定 L=2（對齊論文），變化 hidden_dim：**

| Config | hidden_dim | 參數量 | 計算成本 | 預期發現 |
|--------|-----------|--------|---------|---------|
| baseline | 256 | 0.52M | 1.0× | 基準 |
| medium | 384 | 1.3M | 2.3× | 性能/成本權衡點？ |
| large | 512 | 2.4M | 4.0× | 收益是否遞減？ |
| paper_full | 768 | 4.7M | 8.5× | 論文級別性能 |

**假說驗證：**
- H1: 寬度對湍流統計（能譜、渦量）的影響 > 深度
- H2: 存在「性價比最優點」（可能在 384-512 dim）
- H3: 768 dim 對 Re=10⁴ 可能過剩（但對 Re=10⁶ 必要）

---

## 8) 物理問題的本質差異

### 8.1 Reynolds Number 的影響

| 參數 | 當前專案 | 論文配置 | 物理意義 |
|------|---------|---------|---------|
| Re | 10⁴ | 10⁶ | 尺度分離程度 |
| DNS 網格 | 數百點 | 2048² | 解析度需求 |
| 慣性範圍 | 有限 | 極寬 | k⁻⁵/³ 範圍 |
| 湍流強度 | 中等 | 極強 | 混沌程度 |

**關鍵洞察：**
- Re=10⁴ 是「中等湍流」的驗證場景
- Re=10⁶ 是「完全發展湍流」的挑戰場景
- 兩者不是簡單的「參數縮放」，而是不同的物理 regime

**建議：**
- 當前配置適合「方法驗證」與「架構搜索」
- 若目標是「與 DNS 競爭」，必須同時對齊 Re 與網絡容量

### 8.2 Forcing 振幅的影響

```python
論文：f = [0.1 sin(4πy), 0]  # 溫和驅動
當前：f = [2.0 sin(4πy), 0]  # 強驅動（20 倍）

影響：
- 溫和驅動：慢速發展湍流，統計更穩定
- 強驅動：快速達到飽和態，可能掩蓋一些細節

建議：
→ 對齊 forcing 振幅，以排除這個變數的干擾
```

---

## 9) 參考對照的 repo 檔案位置

- 論文文本抽取（方便 grep）：`context/2507.08972v2_extracted_clean.txt`
- Kolmogorov 設定：
  - `examples/kolmogorov_flow/configs/soap.py`
  - `examples/kolmogorov_flow/configs/pirate.py`
  - `examples/kolmogorov_flow/train.py`
  - `examples/kolmogorov_flow/models.py`
- 架構與 optimizer：
  - `jaxpi/archs.py` (PirateNet 定義，注意 `num_layers` = L = residual blocks 數量)
  - `jaxpi/models.py` (_create_optimizer 的 SOAP + clipping 邏輯)

---

## 10) 總結：核心發現與建議

### 10.1 架構設計的核心差異

```
當前專案：深而窄（L=3, dim=256, 12 layers, 0.7M params）
論文配置：淺而寬（L=2, dim=768, 9 layers, 4.7M params）

→ 這不是簡單的「參數量差異」
→ 而是兩種根本不同的建模哲學
→ 論文的「淺寬」設計可能更適合湍流的並行多尺度特性
```

### 10.2 三個最關鍵的修正方向

1. **減少深度至 L=2**（成本最低，驗證設計哲學）
2. **漸進增加寬度**（256→384→512→768）
3. **修復 SOAP + clipping 實作矛盾**

### 10.3 實驗優先級

```
P0（立即）：num_layers=2 + swish + transfer_learning=True + SOAP clipping
P1（2週）：hidden_dim=384/512 的漸進測試
P2（1月）：完整對齊 (768 dim + Re=10⁶)
```

### 10.4 計數慣例的重要提醒

**在討論架構時，務必明確使用哪種口徑：**
- **L (blocks)**: `config.arch.num_layers`（配置文件使用）
- **論文口徑**: blocks 內 Dense 數（3L）
- **完整口徑**: 所有 Dense（2+3L+1）

**避免歧義的建議：**
```python
# 在配置文件中加入註解
arch.num_layers = 2  # residual blocks (論文 Table 2)
# 對應完整架構：2 gates + 2×3 blocks + 1 output = 9 Dense layers
# 論文口徑："six layers" (只算 blocks 內的 6 個 Dense)
```

---

**最後更新：2026-01-09**
**分析者：Claude (Sonnet 4.5) + 人工校正**
