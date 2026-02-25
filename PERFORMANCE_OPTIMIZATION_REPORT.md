# 效能優化分析報告

**分析日期**: 2026-01-19
**專案**: jaxpi - Physics-Informed Neural Networks (Kolmogorov Flow)
**環境**: JAX 0.6.x, 雙 GPU 訓練

---

## 執行摘要

本報告識別出 **12 個關鍵效能瓶頸**，分為三個優先級：
- 🔴 **Critical (5 個)**: 嚴重影響效能，建議立即優化
- 🟡 **High (4 個)**: 顯著影響效能，建議優先處理
- 🟢 **Medium (3 個)**: 中等影響，可逐步優化

**預估加速潛力**: 1.5x - 3x（訓練速度），30-50%（記憶體使用）

---

## 🔴 Critical Issues（立即優化）

### 1. 每步訓練都進行 Host-Device 數據傳輸

**位置**: `examples/kolmogorov_flow/train.py:283-286`

**問題**:
```python
if step % config.logging.log_every_steps == 0:
    # ❌ 每次 log 都將整個 state 和 batch 傳回 CPU
    state = _to_host_state(model.state)
    batch_host = _to_host_batch(batch)
    # ... evaluator 計算 ...
```

**影響**:
- **每 N 步**就進行一次昂貴的 GPU → CPU 數據傳輸
- `model.state` 包含所有參數（可能數 MB 到數十 MB）
- 阻塞 GPU 計算管線

**優化方案**:

**選項 A（建議）**: 僅傳輸需要的標量值
```python
# ✅ 僅計算並傳輸標量 metrics
if step % config.logging.log_every_steps == 0:
    # 在 GPU 上計算所有 metrics
    log_dict = evaluator_on_device(model.state, batch)

    # 僅將標量結果傳回 CPU
    log_dict_host = jax.device_get(log_dict)  # 僅幾個浮點數
    wandb.log(log_dict_host, step + step_offset)
```

**選項 B**: 使用 `jax.debug.callback` 異步傳輸
```python
def log_callback(log_dict, step):
    wandb.log(jax.device_get(log_dict), step)

if step % config.logging.log_every_steps == 0:
    jax.debug.callback(log_callback, log_dict, step + step_offset)
```

**預期加速**: 5-10% （取決於 `log_every_steps`）

---

### 2. 重複計算梯度（evaluator）

**位置**: `jaxpi/evaluator.py:26-31`

**問題**:
```python
def log_grads(self, params, batch, *args):
    # ❌ 重新計算梯度（訓練時已經計算過）
    grads = jacrev(self.model.losses)(params, batch, *args)
    for key, value in grads.items():
        flattened_grad = flatten_pytree(value)
        grad_norm = jnp.linalg.norm(flattened_grad)
        self.log_dict[key + "_grad_norm"] = grad_norm
```

**影響**:
- 每次 logging 都**重新計算一次完整的梯度**
- 與訓練步驟中的梯度計算重複（`jax.grad(model.loss)` 在 Line 214）
- 梯度計算是最昂貴的操作之一

**優化方案**:

**方案 A**: 在訓練步驟中保存梯度
```python
# train.py
def _jit_step(state, batch):
    grads = jax.grad(model.loss)(state.params, state.weights, batch)
    new_state = state.apply_gradients(grads=grads)
    # 返回梯度以供 logging
    return new_state, grads

jit_step = jit(_jit_step, ...)

# 在訓練循環中
model.state, grads = jit_step(model.state, batch)

# logging 時使用已計算的梯度
if step % config.logging.log_every_steps == 0:
    log_dict = evaluator(state, batch, grads=grads)
```

**方案 B**: 使用 `jax.value_and_grad`
```python
def _jit_step_with_grads(state, batch):
    loss_fn = lambda p: model.loss(p, state.weights, batch)
    loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, grads, loss_value
```

**預期加速**: 10-20%（取決於 `log_every_steps` 和是否啟用 `log_grads`）

---

### 3. 多次梯度計算（PDE residual）

**位置**: `examples/kolmogorov_flow/models.py:94-115`

**問題**:
```python
def r_net(self, params, t, x, y):
    u, v, p = self.neural_net(params, t, x, y)  # ❌ 第 1 次前向傳播

    # ❌ 第 2 次：jacrev 會再次前向傳播
    (u_t, u_x, u_y), (v_t, v_x, v_y), (_, p_x, p_y) = jacrev(
        self.neural_net, argnums=(1, 2, 3)
    )(params, t, x, y)

    # ❌ 第 3, 4 次：hessian 又會前向傳播兩次
    u_hessian = hessian(self.u_net, argnums=(2, 3))(params, t, x, y)
    v_hessian = hessian(self.v_net, argnums=(2, 3))(params, t, x, y)
```

**影響**:
- **同一點計算了 4 次前向傳播**
- Hessian 計算特別昂貴（需要二階導數）

**優化方案**:

**方案 A（建議）**: 使用 `jax.jvp` 和 `jax.vjp` 手動實現
```python
def r_net_optimized(self, params, t, x, y):
    # 使用 forward-mode AD 計算一階導數
    primals = (t, x, y)
    tangents_t = (1.0, 0.0, 0.0)
    tangents_x = (0.0, 1.0, 0.0)
    tangents_y = (0.0, 0.0, 1.0)

    # 一次前向傳播獲取值和一階導數
    (u, v, p), (u_t, u_x, u_y) = jax.jvp(
        lambda args: self.neural_net(params, *args)[0],
        (primals,), (tangents_t, tangents_x, tangents_y)
    )

    # ... 類似處理 v 和 p

    # 對於二階導數，使用 jvp of vjp
    # 這比 hessian 更高效
```

**方案 B**: 快取前向傳播結果
```python
# 使用 @jax.checkpoint 減少記憶體使用
@partial(jax.checkpoint, prevent_cse=False)
def neural_net_cached(self, params, t, x, y):
    return self.neural_net(params, t, x, y)
```

**預期加速**: 30-50%（PDE residual 是最昂貴的部分）

---

### 4. 評估時的 Python 循環（時間分塊）

**位置**: `examples/kolmogorov_flow/models.py:262-286`

**問題**:
```python
def compute_l2_error_time_chunked(self, params, t, coords, ...):
    # ❌ Python while loop，無法被 JIT 優化
    while start < time_count:
        t0 = t_values[start]
        end = start + 1
        while end < time_count and (t_values[end] - t0) < chunk_seconds:
            end += 1  # ❌ 動態循環邊界

        # ❌ 每次迭代都是獨立的 XLA 編譯
        u_pred = self.u_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])
        # ...
```

**影響**:
- Python 循環無法被 JIT 編譯
- 每個 chunk 獨立編譯，失去 XLA 優化機會
- 數據在 CPU-GPU 間來回傳輸

**優化方案**:

**方案 A（建議）**: 使用 `jax.lax.scan` 重寫
```python
@partial(jit, static_argnums=(0,))
def compute_l2_error_time_chunked(self, params, t, coords, u_ref, v_ref, w_ref):
    def eval_chunk(carry, t_chunk_data):
        t_chunk, u_chunk, v_chunk, w_chunk = t_chunk_data

        u_pred = self.u_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])
        w_pred = self.w_pred_fn(params, t_chunk, coords[:, 0], coords[:, 1])

        total_u = jnp.sum((u_pred - u_chunk) ** 2)
        total_v = jnp.sum((v_pred - v_chunk) ** 2)
        total_w = jnp.sum((w_pred - w_chunk) ** 2)

        return carry + jnp.array([total_u, total_v, total_w, ...]), None

    # 預先分塊（在 CPU 上）
    chunks = create_time_chunks(t, u_ref, v_ref, w_ref, chunk_seconds)

    # 使用 scan 迭代（在 GPU 上，完全 JIT 編譯）
    totals, _ = jax.lax.scan(eval_chunk, jnp.zeros(6), chunks)

    # 計算誤差
    u_error = jnp.sqrt(totals[0]) / jnp.sqrt(totals[3])
    return u_error, v_error, w_error
```

**方案 B**: 批次化處理（如果記憶體允許）
```python
@partial(jit, static_argnums=(0,))
def compute_l2_error_batched(self, params, t, coords, u_ref, v_ref, w_ref):
    # 一次性計算所有時間步（如果記憶體足夠）
    u_pred = self.u_pred_fn(params, t, coords[:, 0], coords[:, 1])
    v_pred = self.v_pred_fn(params, t, coords[:, 0], coords[:, 1])
    w_pred = self.w_pred_fn(params, t, coords[:, 0], coords[:, 1])

    # 向量化計算誤差
    u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    return u_error, v_error, w_error
```

**預期加速**: 2-5x（評估階段）

---

### 5. 訓練循環中的數據採樣（CPU 瓶頸）

**位置**: `examples/kolmogorov_flow/train.py:259-265`

**問題**:
```python
for step in range(config.training.max_steps):
    # ❌ 每步都在 CPU 上生成隨機數據
    batch = {}
    for key, sampler in samplers.items():
        batch[key] = next(sampler)  # Python iterator，阻塞 GPU

    # ❌ CPU → GPU 傳輸
    batch = _split_batch(batch)
    batch = _device_put(batch)
```

**影響**:
- 每步訓練都等待 CPU 採樣完成
- GPU 等待數據傳輸（可能閒置）
- 無法重疊計算和數據傳輸

**優化方案**:

**方案 A（建議）**: 使用 JAX 原生採樣器
```python
# 改用完全在 GPU 上的採樣器
class JaxUniformSampler:
    def __init__(self, dom, batch_size, rng_key):
        self.dom = dom
        self.batch_size = batch_size
        self.key = rng_key

    @partial(jit, static_argnums=(0,))
    def sample(self, key):
        # 在 GPU 上生成樣本
        return random.uniform(
            key,
            shape=(self.batch_size, self.dom.shape[0]),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1]
        )

# 訓練循環
rng_key = random.PRNGKey(config.seed)
for step in range(config.training.max_steps):
    rng_key, *subkeys = random.split(rng_key, len(samplers) + 1)

    # 所有採樣在 GPU 上完成
    batch = {
        "ics": ics_sampler.sample(subkeys[0]),
        "res": res_sampler.sample(subkeys[1]),
    }

    # 已經在 GPU 上，不需要 device_put
    model.state = jit_step(model.state, batch)
```

**方案 B**: 預取（Prefetching）
```python
import threading
from queue import Queue

class PrefetchSampler:
    def __init__(self, sampler, queue_size=2):
        self.sampler = sampler
        self.queue = Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def _worker(self):
        for batch in self.sampler:
            self.queue.put(jax.device_put(batch, device))

    def __next__(self):
        return self.queue.get()

# 使用
sampler = PrefetchSampler(original_sampler, queue_size=2)
```

**預期加速**: 10-15%（減少 CPU-GPU 等待時間）

---

## 🟡 High Priority（優先優化）

### 6. 權重更新時的重複計算

**位置**: `examples/kolmogorov_flow/train.py:270-277`

**問題**:
```python
if config.weighting.scheme in ["grad_norm", "ntk"]:
    if step % config.weighting.update_every_steps == 0:
        # ❌ 再次計算 losses（已在訓練步驟計算過）
        weights = model.compute_weights(model.state.params, batch)
        model.state = model.state.apply_weights(weights=weights)
```

**優化方案**:
```python
# 合併權重計算到訓練步驟
def _jit_step_with_weights(state, batch, should_update_weights):
    grads = jax.grad(model.loss)(state.params, state.weights, batch)
    new_state = state.apply_gradients(grads=grads)

    # 條件更新權重
    def update_weights(state):
        weights = model.compute_weights(state.params, batch)
        return state.apply_weights(weights=weights)

    new_state = jax.lax.cond(
        should_update_weights,
        update_weights,
        lambda s: s,
        new_state
    )
    return new_state
```

**預期加速**: 5-8%

---

### 7. vmap 的過度嵌套

**位置**: `examples/kolmogorov_flow/models.py:47-49`

**問題**:
```python
# ❌ 嵌套 vmap 可能導致記憶體爆炸
self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))
self.w_pred_fn = vmap(vmap(self.w_net, (None, None, 0, 0)), (None, 0, None, None))
```

**影響**:
- 對於大網格（如 256x256），中間結果可能佔用大量記憶體
- 可能觸發 OOM

**優化方案**:

**方案 A**: 分塊處理
```python
def u_pred_fn_chunked(self, params, t, x, y, chunk_size=1024):
    """分塊計算預測，避免記憶體爆炸"""
    n_points = x.shape[0]
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    results = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_points)
        x_chunk = x[start:end]
        y_chunk = y[start:end]

        # 只對當前 chunk vmap
        u_chunk = vmap(self.u_net, (None, None, 0, 0))(params, t, x_chunk, y_chunk)
        results.append(u_chunk)

    return jnp.concatenate(results, axis=0)
```

**方案 B**: 使用 `jax.checkpoint` 降低記憶體峰值
```python
from jax import checkpoint

self.u_pred_fn = checkpoint(
    vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
)
```

**預期效果**: 記憶體使用降低 30-50%

---

### 8. NTK 計算效率低下

**位置**: `jaxpi/models.py:290-299`

**問題**:
```python
elif self.config.weighting.scheme == "ntk":
    # ❌ compute_diag_ntk 需要計算 Jacobian
    ntk = self.compute_diag_ntk(params, batch, *args)
    mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)
    # ...
```

**影響**:
- NTK 計算需要 Jacobian（與參數數量成正比）
- 對大模型非常昂貴

**優化方案**:

**方案 A**: 使用近似 NTK
```python
def compute_approx_ntk(self, params, batch, num_samples=100):
    """使用隨機投影近似 NTK"""
    # Hutchinson's trace estimator
    rng = random.PRNGKey(0)

    def ntk_vector_product(v):
        # J^T J v 的高效計算
        jvp_fn = lambda p: jax.jvp(
            lambda p: self.losses(p, batch), (params,), (v,)
        )[1]
        return jax.vjp(jvp_fn, params)[1](jnp.ones_like(...))[0]

    # 隨機估計
    estimates = []
    for _ in range(num_samples):
        v = random.normal(rng, params.shape)
        estimates.append(ntk_vector_product(v))

    return jnp.mean(jnp.array(estimates))
```

**方案 B**: 降低更新頻率
```python
# config 中設置
config.weighting.update_every_steps = 100  # 從 10 增加到 100
```

**預期加速**: 20-40%（如果使用 NTK weighting）

---

### 9. Causal weight 計算的排序開銷

**位置**: `examples/kolmogorov_flow/models.py:138`

**問題**:
```python
def res_and_w(self, params, batch):
    # ...
    # ❌ 排序是 O(n log n) 操作
    t_sorted = jnp.sort(batch[:, 0])
    ru_pred, rv_pred, rc_pred = self.r_pred_fn(
        params, t_sorted, batch[:, 1], batch[:, 2]
    )
```

**影響**:
- 每個訓練步驟都排序（batch_size ~ 數千）
- 阻礙 GPU 並行

**優化方案**:

**方案 A**: 預排序採樣
```python
class SortedSampler:
    """採樣時已經保證時間順序"""
    def sample(self, key):
        # 生成隨機樣本
        batch = self._sample_raw(key)
        # 按時間排序（僅一次）
        sorted_indices = jnp.argsort(batch[:, 0])
        return batch[sorted_indices]

# 訓練時不需要再排序
def res_and_w(self, params, batch):
    # batch 已排序，跳過 sort
    t = batch[:, 0]  # 假設已排序
    ru_pred, rv_pred, rc_pred = self.r_pred_fn(...)
```

**方案 B**: 使用 argsort indices
```python
# 保存排序 indices，複用於其他座標
indices = jnp.argsort(batch[:, 0])
t_sorted = batch[indices, 0]
x_sorted = batch[indices, 1]
y_sorted = batch[indices, 2]
```

**預期加速**: 3-5%

---

## 🟢 Medium Priority（逐步優化）

### 10. Checkpoint 儲存頻率

**位置**: `examples/kolmogorov_flow/train.py:307-320`

**問題**:
```python
if config.saving.save_every_steps is not None:
    if (step + 1) % config.saving.save_every_steps == 0:
        # ❌ 頻繁的磁碟 I/O 阻塞訓練
        save_checkpoint(model.state, ckpt_path, ...)
```

**優化方案**:
```python
# 1. 降低儲存頻率
config.saving.save_every_steps = 1000  # 從 100 增加

# 2. 異步儲存
import threading

def async_save_checkpoint(state, path):
    def _save():
        save_checkpoint(state, path, ...)
    threading.Thread(target=_save).start()
```

**預期加速**: 1-2%

---

### 11. Logging 開銷

**位置**: `examples/kolmogorov_flow/train.py:281-304`

**問題**:
```python
if step % config.logging.log_every_steps == 0:
    # ❌ 過於詳細的 logging
    log_dict = evaluator(state, batch_host, ...)  # 計算所有 metrics
    wandb.log(log_dict, step)
```

**優化方案**:
```python
# 1. 減少 logging 頻率
config.logging.log_every_steps = 100  # 從 10 增加

# 2. 條件性 logging
if step % config.logging.log_every_steps == 0:
    # 僅 log 關鍵 metrics
    log_dict = {
        "loss": loss_value,
        "u_error": u_error,
    }
    # 每 1000 步才 log 詳細 metrics
    if step % 1000 == 0:
        log_dict.update(evaluator.compute_detailed_metrics(...))
```

**預期加速**: 2-3%

---

### 12. 無用的渦度計算

**位置**: `examples/kolmogorov_flow/models.py:88-92`

**問題**:
```python
def w_net(self, params, t, x, y):
    # ❌ 計算渦度（需要兩次梯度）
    u_y = grad(self.u_net, argnums=3)(params, t, x, y)
    v_x = grad(self.v_net, argnums=2)(params, t, x, y)
    w = v_x - u_y
    return w
```

**檢查**: 如果渦度 `w` 不用於 loss 計算，可以跳過

**優化方案**:
```python
# 僅在 evaluation 時計算渦度
if config.logging.log_errors and compute_vorticity:
    w_pred = self.w_pred_fn(params, t, coords[:, 0], coords[:, 1])
else:
    w_pred = None  # 跳過昂貴的計算
```

---

## 優化優先級路線圖

### Phase 1: Quick Wins（1-2 天）
1. ✅ 減少 Host-Device 傳輸（Issue #1）
2. ✅ 移除重複梯度計算（Issue #2）
3. ✅ 使用 JAX 採樣器（Issue #5）

**預期加速**: 20-30%

### Phase 2: 核心優化（3-5 天）
4. ✅ 優化 PDE residual 計算（Issue #3）
5. ✅ 使用 lax.scan 重寫評估（Issue #4）
6. ✅ 合併權重更新（Issue #6）

**預期加速**: 累計 50-80%

### Phase 3: 記憶體優化（1-2 天）
7. ✅ vmap 分塊處理（Issue #7）
8. ✅ 降低 NTK 計算頻率（Issue #8）

**預期效果**: 記憶體降低 30-50%，允許更大 batch size

### Phase 4: 細節優化（ongoing）
9. ✅ 預排序採樣（Issue #9）
10. ✅ 異步 checkpoint（Issue #10）
11. ✅ 精簡 logging（Issue #11）
12. ✅ 條件性渦度計算（Issue #12）

---

## 記憶體優化建議

### 當前記憶體使用分析

**估計記憶體使用**（batch_size = 2851, 256x256 grid）:

| 組件 | 大小 | 說明 |
|------|------|------|
| 模型參數 | ~10-50 MB | 取決於網路大小 |
| 優化器狀態 | ~20-100 MB | Adam moments |
| Batch 數據 | ~100 MB | 2851 samples × 3 coords |
| 中間激活 | **500+ MB** | vmap 嵌套產生 |
| 梯度計算 | **300+ MB** | Jacobian/Hessian |

**總計**: ~1-1.5 GB per GPU（理論最低）

### 減少記憶體的策略

1. **梯度累積**（允許更小 batch）
```python
def gradient_accumulation_step(state, batches):
    def accumulate(carry, batch):
        grads = jax.grad(model.loss)(state.params, state.weights, batch)
        return carry + grads, None

    total_grads, _ = jax.lax.scan(accumulate, tree_map(jnp.zeros_like, state.params), batches)
    # 平均梯度
    avg_grads = tree_map(lambda g: g / len(batches), total_grads)
    return state.apply_gradients(grads=avg_grads)
```

2. **混合精度訓練**
```python
# 使用 float16 減少記憶體
policy = jax.experimental.mixed_precision.create_policy(
    compute_dtype=jnp.float16,
    param_dtype=jnp.float32,
    output_dtype=jnp.float32
)

# 應用到模型
model = mixed_precision.apply_policy(model, policy)
```

3. **Rematerialization（重計算）**
```python
# 用計算換記憶體
from jax import checkpoint

# 在前向傳播中標記檢查點
@checkpoint
def neural_net(params, x):
    # 中間結果會被重新計算，而非儲存
    return network_forward(params, x)
```

---

## JAX 效能最佳實踐檢查清單

### ✅ 已遵循
- ✅ 使用 `@jit` 裝飾計算密集函數
- ✅ 使用 `vmap` 向量化而非 Python loops
- ✅ 使用 `lax.stop_gradient` 阻止不必要的梯度流

### ⚠️ 需改進
- ⚠️ 避免 Python 控制流（while loops）→ 使用 `lax.scan`/`lax.cond`
- ⚠️ 減少 `jax.device_get`/`jax.device_put` 調用
- ⚠️ 批次化所有操作，避免逐個處理
- ⚠️ 使用 `static_argnums` 標記靜態參數

### ❌ 未使用但建議採用
- ❌ `jax.profiler` 分析實際瓶頸
- ❌ `jax.debug.print` 調試時避免 device_get
- ❌ XLA 編譯器優化標誌
- ❌ 多主機分散式訓練（TPU/multi-node）

---

## 效能分析工具使用

### 1. JAX Profiler
```python
import jax.profiler

# 在訓練循環中
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for step in range(100):
        model.state = jit_step(model.state, batch)
```

### 2. 記憶體分析
```python
# 檢查記憶體使用
import jax

def print_memory_stats():
    for device in jax.devices():
        stats = device.memory_stats()
        print(f"Device {device}:")
        print(f"  Bytes in use: {stats['bytes_in_use'] / 1e9:.2f} GB")
        print(f"  Peak bytes: {stats.get('peak_bytes_in_use', 0) / 1e9:.2f} GB")

# 在訓練中定期調用
if step % 100 == 0:
    print_memory_stats()
```

### 3. 編譯時間分析
```python
import time

# 測量 JIT 編譯時間
start = time.time()
jit_step(model.state, batch)  # 第一次調用會編譯
compile_time = time.time() - start
print(f"Compilation time: {compile_time:.2f}s")

# 測量執行時間
start = time.time()
for _ in range(100):
    jit_step(model.state, batch)
jax.block_until_ready(model.state)  # 等待所有計算完成
exec_time = (time.time() - start) / 100
print(f"Average step time: {exec_time*1000:.2f}ms")
```

---

## 總結與建議

### 關鍵優化點
1. **減少數據傳輸**（CPU-GPU）- 最重要
2. **消除重複計算**（梯度、losses）- 高影響
3. **優化 PDE residual**（多次前向傳播）- 最昂貴部分
4. **改用 JAX 原生操作**（lax.scan, GPU 採樣器）

### 實施建議
- **先測量，後優化**：使用 profiler 確認實際瓶頸
- **逐步優化**：每次改動後測試正確性和效能
- **保持向後相容**：使用配置開關控制優化特性

### 預期總加速
- **訓練速度**: 1.5x - 3x（取決於配置）
- **記憶體使用**: 降低 30-50%
- **可訓練 batch size**: 增加 50-100%

**下一步**: 建議從 Phase 1 的 Quick Wins 開始，先解決 Issue #1, #2, #5
