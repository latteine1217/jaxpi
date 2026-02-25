# 效能優化快速修正指南

**目標**: 以最小改動獲得最大加速
**預期**: 20-30% 訓練加速，立即可實施

---

## 🚀 Quick Win #1: 減少 Host-Device 傳輸

### 問題
每次 logging 都將整個 state 傳回 CPU（數十 MB）

### 修正
**文件**: `examples/kolmogorov_flow/train.py`

```python
# ❌ 原始代碼（Line 283-296）
if step % config.logging.log_every_steps == 0:
    state = _to_host_state(model.state)  # 昂貴！
    batch_host = _to_host_batch(batch)
    log_dict = evaluator(state, batch_host, ...)

# ✅ 優化後
@partial(jit, static_argnums=(0,))
def compute_metrics_on_device(model, state, batch, t, coords, u_ref, v_ref, w_ref):
    """在 GPU 上計算所有 metrics，僅返回標量"""
    # 計算 losses
    losses = model.losses(state.params, batch)

    # 計算 errors（如果需要）
    if config.logging.log_errors:
        u_error, v_error, w_error = model.compute_l2_error(
            state.params, t, coords, u_ref, v_ref, w_ref
        )
    else:
        u_error = v_error = w_error = 0.0

    # 返回標量字典（僅幾個浮點數）
    return {
        "u_ic_loss": losses["u_ic"],
        "v_ic_loss": losses["v_ic"],
        "ru_loss": losses["ru"],
        "rv_loss": losses["rv"],
        "rc_loss": losses["rc"],
        "u_error": u_error,
        "v_error": v_error,
        "w_error": w_error,
    }

# 在訓練循環中
if step % config.logging.log_every_steps == 0:
    # 在 GPU 上計算 metrics
    log_dict_device = compute_metrics_on_device(
        model, model.state, batch, t_eval, coords_eval, u_eval, v_eval, w_eval
    )
    # 僅傳輸標量結果（<1 KB）
    log_dict = jax.device_get(log_dict_device)
    wandb.log(log_dict, step + step_offset)
```

**預期加速**: 5-10%

---

## 🚀 Quick Win #2: 使用 GPU 採樣器

### 問題
每步在 CPU 生成樣本，然後傳輸到 GPU

### 修正
**新增文件**: `examples/kolmogorov_flow/jax_samplers.py`

```python
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

class JaxUniformSampler:
    """完全在 GPU 上運行的採樣器"""

    def __init__(self, dom, batch_size, rng_key):
        self.dom = jnp.array(dom)
        self.batch_size = batch_size
        self.rng_key = rng_key

    def sample(self, key):
        """生成一個 batch（在 GPU 上）"""
        return random.uniform(
            key,
            shape=(self.batch_size, self.dom.shape[0]),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

class JaxICSampler:
    """初始條件採樣器"""

    def __init__(self, u0, v0, w0, coords, batch_size, rng_key):
        self.u0 = jnp.array(u0)
        self.v0 = jnp.array(v0)
        self.w0 = jnp.array(w0)
        self.coords = jnp.array(coords)
        self.batch_size = batch_size
        self.rng_key = rng_key

    def sample(self, key):
        idx = random.choice(key, self.coords.shape[0], shape=(self.batch_size,))
        coords_batch = self.coords[idx, :]
        u_batch = self.u0[idx]
        v_batch = self.v0[idx]
        w_batch = self.w0[idx]
        return coords_batch, u_batch, v_batch, w_batch
```

**修改訓練循環**: `examples/kolmogorov_flow/train.py`

```python
# ❌ 原始代碼（Line 519-525）
ics_sampler = ICSampler(u0, v0, w0, coords, global_batch_size * 2)
res_sampler = iter(LocalUniformSampler(dom, global_batch_size))

samplers = {
    "ics": iter(ics_sampler),
    "res": iter(res_sampler),
}

# ✅ 優化後
from jax_samplers import JaxUniformSampler, JaxICSampler

ics_sampler = JaxICSampler(u0, v0, w0, coords, global_batch_size * 2, random.PRNGKey(config.seed))
res_sampler = JaxUniformSampler(dom, global_batch_size, random.PRNGKey(config.seed + 1))

# 訓練循環修改（Line 256-267）
rng_key = random.PRNGKey(config.seed)
for step in range(config.training.max_steps):
    start_time = time.time()

    # 在 GPU 上生成樣本
    rng_key, ics_key, res_key = random.split(rng_key, 3)

    ics_batch = ics_sampler.sample(ics_key)
    res_batch = res_sampler.sample(res_key)

    batch = {"ics": ics_batch, "res": res_batch}

    # 不需要 _split_batch 和 _device_put（已經在 GPU 上）
    with mesh_context:
        model.state = jit_step(model.state, batch)
```

**預期加速**: 10-15%

---

## 🚀 Quick Win #3: 減少 Logging 頻率

### 問題
過於頻繁的 logging 拖慢訓練

### 修正
**文件**: `examples/kolmogorov_flow/configs/pirate.py` 和 `soap.py`

```python
# ❌ 原始
config.logging = ml_collections.ConfigDict()
config.logging.log_every_steps = 10  # 太頻繁

# ✅ 優化
config.logging.log_every_steps = 100  # 減少 10 倍

# 同時調整其他頻率
config.weighting.update_every_steps = 100  # 從 10 增加
config.saving.save_every_steps = 1000  # 從 100 增加
```

**預期加速**: 5-8%

---

## 🚀 Quick Win #4: 停用不必要的 Logging

### 問題
計算大量不必要的 metrics

### 修正
**文件**: `examples/kolmogorov_flow/configs/pirate.py`

```python
# ❌ 原始
config.logging.log_errors = True  # 每次都計算誤差（昂貴）
config.logging.log_grads = True   # 重複計算梯度
config.logging.log_ntk = True     # 非常昂貴

# ✅ 優化（訓練時）
config.logging.log_errors = False  # 僅最後評估時計算
config.logging.log_grads = False   # 不需要時關閉
config.logging.log_ntk = False     # 僅調試時開啟

# 僅保留必要的
config.logging.log_losses = True   # 監控訓練進度
config.logging.log_weights = True  # 監控 weighting scheme
```

**預期加速**: 3-5%

---

## 🚀 Quick Win #5: 使用更大的 Batch Size

### 問題
小 batch size 無法充分利用 GPU 並行能力

### 修正
**文件**: `examples/kolmogorov_flow/configs/pirate.py`

```python
# ❌ 原始（雙 GPU）
config.training.batch_size_per_device = 2851  # OOM

# ✅ 策略 1: 使用能整除的較小值
config.training.batch_size_per_device = 2560  # 16 * 160（能被 num_chunks=16 整除）

# ✅ 策略 2: 梯度累積（模擬大 batch）
config.training.batch_size_per_device = 1280  # 實際使用
config.training.gradient_accumulation_steps = 2  # 累積 2 步 = 等效 2560
```

**實作梯度累積** (optional):
```python
# train.py 中修改訓練步驟
def _jit_step_with_accumulation(state, batches):
    """累積多個 batch 的梯度"""
    def compute_grads(carry, batch):
        grads = jax.grad(model.loss)(state.params, state.weights, batch)
        return tree_map(jnp.add, carry, grads), None

    # 累積梯度
    total_grads, _ = jax.lax.scan(
        compute_grads,
        tree_map(jnp.zeros_like, state.params),
        batches
    )

    # 平均並應用
    avg_grads = tree_map(lambda g: g / len(batches), total_grads)
    return state.apply_gradients(grads=avg_grads)
```

---

## 實施檢查清單

### 立即可做（10 分鐘）
- [ ] 修改 `config.logging.log_every_steps = 100`
- [ ] 停用 `config.logging.log_errors = False`（訓練時）
- [ ] 停用 `config.logging.log_grads = False`
- [ ] 停用 `config.logging.log_ntk = False`

### 30 分鐘內完成
- [ ] 實施 Quick Win #1（減少 device_get）
- [ ] 調整 batch size（找到不 OOM 的最大值）

### 1-2 小時完成
- [ ] 實施 Quick Win #2（JAX 採樣器）
- [ ] 測試並驗證加速效果

---

## 測試加速效果

### 基準測試腳本
```bash
# 1. 記錄原始速度
python examples/kolmogorov_flow/train.py \
  --config=examples/kolmogorov_flow/configs/pirate.py \
  --config.training.max_steps=100 \
  2>&1 | tee baseline.log

# 提取每步時間
grep "Step time" baseline.log | awk '{sum+=$3; n++} END {print "Average:", sum/n, "ms"}'

# 2. 應用優化後測試
python examples/kolmogorov_flow/train.py \
  --config=examples/kolmogorov_flow/configs/pirate_optimized.py \
  --config.training.max_steps=100 \
  2>&1 | tee optimized.log

grep "Step time" optimized.log | awk '{sum+=$3; n++} END {print "Average:", sum/n, "ms"}'
```

### 記憶體測試
```python
# 在訓練循環中添加
import jax

if step == 0 or step % 100 == 0:
    for device in jax.devices():
        stats = device.memory_stats()
        print(f"[Step {step}] GPU {device.id} memory: {stats['bytes_in_use']/1e9:.2f} GB")
```

---

## 預期結果摘要

| 優化項目 | 預期加速 | 難度 | 時間 |
|---------|---------|------|------|
| 減少 logging 頻率 | 5-8% | 簡單 | 5 min |
| 停用不必要 logging | 3-5% | 簡單 | 5 min |
| 減少 device_get | 5-10% | 中等 | 30 min |
| JAX 採樣器 | 10-15% | 中等 | 1-2 hr |
| **總計** | **23-38%** | - | **2-3 hr** |

---

## 注意事項

1. **正確性優先**: 每次修改後務必檢查訓練 loss 曲線
2. **逐步驗證**: 一次修改一個，確認效果
3. **配置管理**: 保留原始配置文件作為 baseline
4. **監控記憶體**: 確保優化不會導致 OOM

---

## 下一步

完成 Quick Fixes 後，參考 `PERFORMANCE_OPTIMIZATION_REPORT.md` 的 Phase 2：
- 優化 PDE residual 計算（30-50% 加速）
- 使用 lax.scan 重寫評估（2-5x 評估加速）
- 記憶體優化（支援更大 batch size）
