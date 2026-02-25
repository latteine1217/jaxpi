# 效能優化路線圖

**目標**: 將訓練速度提升 1.5x - 3x，記憶體使用降低 30-50%

---

## 優化階段概覽

```
Phase 1: Quick Wins (1-2天)        → 20-30% 加速
    ↓
Phase 2: 核心優化 (3-5天)          → 累計 50-80% 加速
    ↓
Phase 3: 記憶體優化 (1-2天)        → 記憶體 -30-50%
    ↓
Phase 4: 進階優化 (ongoing)        → 累計 1.5x-3x 加速
```

---

## Phase 1: Quick Wins（立即實施）

**時間**: 1-2 天
**預期加速**: 20-30%
**難度**: ⭐⭐☆☆☆

### 1.1 減少 Host-Device 傳輸 ⚡
**預期**: 5-10% 加速
**時間**: 30 分鐘
**難度**: ⭐⭐☆☆☆

**修改文件**: `examples/kolmogorov_flow/train.py`

```diff
  if step % config.logging.log_every_steps == 0:
-     state = _to_host_state(model.state)
-     batch_host = _to_host_batch(batch)
-     log_dict = evaluator(state, batch_host, ...)
+     # 在 GPU 上計算所有 metrics
+     log_dict_device = compute_metrics_on_device(model, model.state, batch, ...)
+     # 僅傳輸標量結果
+     log_dict = jax.device_get(log_dict_device)
      wandb.log(log_dict, step)
```

**檢查點**:
- [ ] 實施新的 `compute_metrics_on_device` 函數
- [ ] 測試 logging 輸出一致
- [ ] 測量加速效果

---

### 1.2 使用 JAX 原生採樣器 ⚡⚡
**預期**: 10-15% 加速
**時間**: 1-2 小時
**難度**: ⭐⭐⭐☆☆

**新增文件**: `examples/kolmogorov_flow/jax_samplers.py`

**修改文件**: `examples/kolmogorov_flow/train.py`

```diff
- ics_sampler = ICSampler(...)
- res_sampler = iter(LocalUniformSampler(...))
+ from jax_samplers import JaxUniformSampler, JaxICSampler
+ ics_sampler = JaxICSampler(...)
+ res_sampler = JaxUniformSampler(...)

  for step in range(max_steps):
-     batch = {key: next(sampler) for key, sampler in samplers.items()}
-     batch = _split_batch(batch)
-     batch = _device_put(batch)
+     rng_key, *subkeys = random.split(rng_key, 3)
+     batch = {
+         "ics": ics_sampler.sample(subkeys[0]),
+         "res": res_sampler.sample(subkeys[1]),
+     }
```

**檢查點**:
- [ ] 實施 JAX 採樣器類別
- [ ] 修改訓練循環
- [ ] 驗證採樣分佈一致
- [ ] 測量加速效果

---

### 1.3 減少 Logging 頻率 ⚡
**預期**: 5-8% 加速
**時間**: 5 分鐘
**難度**: ⭐☆☆☆☆

**修改文件**: `examples/kolmogorov_flow/configs/pirate.py`, `soap.py`

```diff
- config.logging.log_every_steps = 10
+ config.logging.log_every_steps = 100

- config.logging.log_errors = True
- config.logging.log_grads = True
- config.logging.log_ntk = True
+ config.logging.log_errors = False  # 僅最後評估
+ config.logging.log_grads = False   # 訓練時不需要
+ config.logging.log_ntk = False     # 非常昂貴

- config.weighting.update_every_steps = 10
+ config.weighting.update_every_steps = 100
```

**檢查點**:
- [ ] 更新所有配置文件
- [ ] 確認仍可監控訓練進度
- [ ] 測量加速效果

---

### Phase 1 驗收標準
- ✅ 訓練速度提升 **20-30%**
- ✅ Loss 曲線與 baseline 一致（誤差 < 1%）
- ✅ 無額外記憶體開銷
- ✅ 所有測試通過

---

## Phase 2: 核心優化（重點突破）

**時間**: 3-5 天
**預期加速**: 累計 50-80%（含 Phase 1）
**難度**: ⭐⭐⭐⭐☆

### 2.1 優化 PDE Residual 計算 ⚡⚡⚡
**預期**: 30-50% 加速
**時間**: 1-2 天
**難度**: ⭐⭐⭐⭐☆

**問題**: 同一點重複前向傳播 4 次

**修改文件**: `examples/kolmogorov_flow/models.py`

**策略 A**: 使用 `jax.jvp` + `jax.vjp` 手動實現
```python
def r_net_optimized(self, params, t, x, y):
    # 一次前向傳播 + 高效導數計算
    def forward(args):
        t, x, y = args
        u, v, p = self.neural_net(params, t, x, y)
        return jnp.stack([u, v, p])

    primals = (t, x, y)

    # 一階導數（使用 forward-mode）
    tangents = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
    values, derivs = jax.vmap(
        lambda tang: jax.jvp(forward, (primals,), (tang,))
    )(jnp.array(tangents))

    # 二階導數（使用 jvp of vjp）
    # ... 實施細節 ...

    return ru, rv, rc
```

**策略 B**: 使用 `jax.checkpoint` 減少記憶體
```python
@partial(jax.checkpoint, prevent_cse=False)
def neural_net_cached(self, params, t, x, y):
    return self.neural_net(params, t, x, y)
```

**檢查點**:
- [ ] 實施新的 residual 計算
- [ ] 單元測試：與原版結果一致（誤差 < 1e-6）
- [ ] 整合到訓練流程
- [ ] 測量加速和記憶體使用

---

### 2.2 使用 `lax.scan` 重寫評估 ⚡⚡
**預期**: 2-5x 評估加速
**時間**: 1 天
**難度**: ⭐⭐⭐☆☆

**問題**: Python while loop 無法 JIT 編譯

**修改文件**: `examples/kolmogorov_flow/models.py`

```python
@partial(jit, static_argnums=(0,))
def compute_l2_error_time_chunked(self, params, t, coords, u_ref, v_ref, w_ref):
    def eval_chunk(carry, chunk_data):
        t_chunk, u_chunk, v_chunk, w_chunk = chunk_data

        # 計算預測
        u_pred = self.u_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])
        w_pred = self.w_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])

        # 累積誤差
        errors = jnp.array([
            jnp.sum((u_pred - u_chunk) ** 2),
            jnp.sum((v_pred - v_chunk) ** 2),
            jnp.sum((w_pred - w_chunk) ** 2),
            jnp.sum(u_chunk ** 2),
            jnp.sum(v_chunk ** 2),
            jnp.sum(w_chunk ** 2),
        ])

        return carry + errors, None

    # 預先分塊（在 CPU 上）
    chunks = self._create_time_chunks(t, u_ref, v_ref, w_ref)

    # scan 迭代（在 GPU 上，完全 JIT）
    totals, _ = jax.lax.scan(eval_chunk, jnp.zeros(6), chunks)

    # 計算相對誤差
    u_error = jnp.sqrt(totals[0]) / jnp.sqrt(totals[3])
    v_error = jnp.sqrt(totals[1]) / jnp.sqrt(totals[4])
    w_error = jnp.sqrt(totals[2]) / jnp.sqrt(totals[5])

    return u_error, v_error, w_error
```

**檢查點**:
- [ ] 實施分塊邏輯
- [ ] 使用 `lax.scan` 替換 while loop
- [ ] 驗證結果一致
- [ ] 測量評估加速

---

### 2.3 合併權重更新到訓練步驟 ⚡
**預期**: 5-8% 加速
**時間**: 半天
**難度**: ⭐⭐⭐☆☆

**修改文件**: `examples/kolmogorov_flow/train.py`

```python
def _jit_step_with_conditional_weights(state, batch, should_update_weights):
    # 計算梯度並更新參數
    grads = jax.grad(model.loss)(state.params, state.weights, batch)
    new_state = state.apply_gradients(grads=grads)

    # 條件性更新權重（使用 lax.cond 避免重複計算）
    def update_weights(s):
        weights = model.compute_weights(s.params, batch)
        return s.apply_weights(weights=weights)

    new_state = jax.lax.cond(
        should_update_weights,
        update_weights,
        lambda s: s,
        new_state
    )

    return new_state

# 編譯
jit_step = jit(_jit_step_with_conditional_weights, ...)

# 使用
for step in range(max_steps):
    should_update = (step % config.weighting.update_every_steps == 0)
    model.state = jit_step(model.state, batch, should_update)
```

**檢查點**:
- [ ] 實施條件性權重更新
- [ ] 驗證權重更新頻率正確
- [ ] 測量加速效果

---

### Phase 2 驗收標準
- ✅ 訓練速度提升 **50-80%**（累計）
- ✅ 評估速度提升 **2-5x**
- ✅ Loss 曲線與 baseline 一致
- ✅ 單元測試覆蓋核心優化

---

## Phase 3: 記憶體優化（擴大規模）

**時間**: 1-2 天
**預期效果**: 記憶體使用 -30-50%，支援更大 batch size
**難度**: ⭐⭐⭐☆☆

### 3.1 vmap 分塊處理 💾
**預期**: 記憶體 -30%
**時間**: 半天
**難度**: ⭐⭐☆☆☆

**修改文件**: `examples/kolmogorov_flow/models.py`

```python
def u_pred_fn_chunked(self, params, t, x, y, chunk_size=4096):
    """分塊計算，避免記憶體爆炸"""
    n_space = x.shape[0]
    n_time = t.shape[0]

    # 空間維度分塊
    results = []
    for i in range(0, n_space, chunk_size):
        x_chunk = x[i:i+chunk_size]
        y_chunk = y[i:i+chunk_size]

        # 僅對當前 chunk vmap
        u_chunk = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))(
            params, t, x_chunk, y_chunk
        )
        results.append(u_chunk)

    return jnp.concatenate(results, axis=1)
```

**檢查點**:
- [ ] 實施分塊 vmap
- [ ] 調整最佳 chunk_size
- [ ] 測量記憶體使用
- [ ] 驗證結果一致

---

### 3.2 混合精度訓練 💾⚡
**預期**: 記憶體 -50%, 速度 +20%
**時間**: 1 天
**難度**: ⭐⭐⭐⭐☆

**修改文件**: `jaxpi/models.py`

```python
from jax.experimental import mixed_precision

# 創建混合精度策略
policy = mixed_precision.create_policy(
    compute_dtype=jnp.float16,  # 計算用 FP16
    param_dtype=jnp.float32,    # 參數保持 FP32
    output_dtype=jnp.float32,   # 輸出轉回 FP32
)

# 應用到模型
arch = mixed_precision.apply_policy(arch, policy)
```

**檢查點**:
- [ ] 實施混合精度
- [ ] 調整 loss scaling（避免下溢）
- [ ] 驗證數值穩定性
- [ ] 測量記憶體和速度

---

### 3.3 梯度累積 💾
**預期**: 允許有效 batch size × 2-4
**時間**: 半天
**難度**: ⭐⭐⭐☆☆

**修改文件**: `examples/kolmogorov_flow/train.py`

```python
def _jit_step_with_gradient_accumulation(state, batches):
    """累積多個小 batch 的梯度"""
    def accumulate_grads(carry, batch):
        grads = jax.grad(model.loss)(state.params, state.weights, batch)
        return tree_map(jnp.add, carry, grads), None

    # 累積
    total_grads, _ = jax.lax.scan(
        accumulate_grads,
        tree_map(jnp.zeros_like, state.params),
        batches
    )

    # 平均並應用
    avg_grads = tree_map(lambda g: g / len(batches), total_grads)
    return state.apply_gradients(grads=avg_grads)

# 配置
config.training.batch_size_per_device = 1280  # 減半
config.training.gradient_accumulation_steps = 2  # 有效 = 2560
```

**檢查點**:
- [ ] 實施梯度累積
- [ ] 測試等效性（與大 batch 比較）
- [ ] 測量記憶體節省

---

### Phase 3 驗收標準
- ✅ 記憶體使用降低 **30-50%**
- ✅ 支援 batch size 增加 **50-100%**
- ✅ 訓練收斂性不變
- ✅ 可在 2 GPU 上訓練原本 OOM 的配置

---

## Phase 4: 進階優化（持續改進）

**時間**: Ongoing
**預期效果**: 細節優化，累計達到 1.5x-3x 加速
**難度**: ⭐⭐⭐⭐⭐

### 4.1 自定義 CUDA Kernels
- 為 PDE residual 寫專用 kernel
- 融合多個操作（kernel fusion）

### 4.2 分散式訓練優化
- 改進多 GPU 通信
- Pipeline parallelism
- ZeRO optimizer

### 4.3 編譯器優化
- XLA 優化標誌調整
- 自定義 HLO passes

---

## 測試與驗證策略

### 正確性測試
```python
# tests/test_optimization.py
def test_optimized_vs_baseline():
    """確保優化後結果一致"""
    # 固定隨機種子
    rng = random.PRNGKey(42)

    # 訓練 baseline
    loss_baseline = train_baseline(rng, steps=100)

    # 訓練優化版
    loss_optimized = train_optimized(rng, steps=100)

    # 驗證 loss 曲線一致（容許小誤差）
    assert jnp.allclose(loss_baseline, loss_optimized, rtol=1e-3)
```

### 效能測試
```python
# benchmark/bench_training.py
import time

def benchmark_step(jit_step, state, batch, warmup=10, repeat=100):
    # Warmup（JIT 編譯）
    for _ in range(warmup):
        state = jit_step(state, batch)

    # 測量
    start = time.time()
    for _ in range(repeat):
        state = jit_step(state, batch)
    jax.block_until_ready(state)  # 等待 GPU 完成
    elapsed = time.time() - start

    return elapsed / repeat * 1000  # ms per step

# 比較
baseline_time = benchmark_step(baseline_jit_step, ...)
optimized_time = benchmark_step(optimized_jit_step, ...)
speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x")
```

### 記憶體測試
```python
def measure_memory_usage(fn, *args):
    """測量函數的記憶體使用"""
    # 清空快取
    for device in jax.devices():
        device.clear_cache()

    # 執行函數
    result = fn(*args)
    jax.block_until_ready(result)

    # 讀取記憶體統計
    stats = jax.devices()[0].memory_stats()
    return stats['peak_bytes_in_use'] / 1e9  # GB
```

---

## 里程碑與時間表

```
Week 1: Phase 1 (Quick Wins)
├─ Day 1-2: 實施 1.1, 1.2
├─ Day 3: 實施 1.3, 驗證
└─ Day 4-5: 測試與文檔

Week 2-3: Phase 2 (核心優化)
├─ Day 6-8: 實施 2.1 (PDE residual)
├─ Day 9-11: 實施 2.2 (lax.scan)
├─ Day 12-13: 實施 2.3 (合併權重更新)
└─ Day 14-15: 整合測試

Week 3-4: Phase 3 (記憶體優化)
├─ Day 16-17: 實施 3.1, 3.2
├─ Day 18-19: 實施 3.3
└─ Day 20-21: 驗證與調優

Week 4+: Phase 4 (進階優化)
└─ 持續改進
```

---

## 回退計畫

如果某個優化導致問題：

1. **保留 baseline 代碼**
   ```python
   # 使用配置開關
   if config.use_optimized_residual:
       residual = optimized_r_net(...)
   else:
       residual = original_r_net(...)
   ```

2. **版本控制**
   ```bash
   git checkout -b optimization-phase1
   # 實施優化
   git commit -m "Phase 1: Quick wins"

   # 如果有問題
   git checkout main
   ```

3. **A/B 測試**
   - 同時運行兩個版本
   - 比較 loss 曲線和最終準確度

---

## 成功指標

### Phase 1 完成
- ✅ 訓練吞吐量: baseline × 1.2 - 1.3
- ✅ 記憶體使用: 無增加
- ✅ 準確度: 與 baseline 一致（誤差 < 1%）

### Phase 2 完成
- ✅ 訓練吞吐量: baseline × 1.5 - 1.8
- ✅ 評估速度: baseline × 2 - 5
- ✅ 準確度: 與 baseline 一致

### Phase 3 完成
- ✅ 記憶體使用: baseline × 0.5 - 0.7
- ✅ 最大 batch size: baseline × 1.5 - 2
- ✅ 吞吐量: 維持或提升

### 最終目標
- 🎯 **訓練速度**: **1.5x - 3x**
- 🎯 **記憶體效率**: **-30% - -50%**
- 🎯 **準確度**: **保持不變**
- 🎯 **可訓練規模**: **增加 50-100%**

---

## 總結

這個路線圖提供了系統性的效能優化方案：
- **漸進式**: 從簡單到複雜
- **可測量**: 每階段有明確指標
- **低風險**: 保留 baseline，可回退
- **高回報**: 預期 1.5x-3x 加速

**建議**: 先完成 Phase 1-2，再根據實際需求決定是否進行 Phase 3-4。
