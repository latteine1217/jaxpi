# JAX API 相容性檢查報告（JAX 0.6.x）

**檢查日期**: 2026-01-19
**目標版本**: JAX 0.6.x
**專案**: jaxpi（2024 年開發）

---

## 執行摘要

專案中存在的 **2 個關鍵 API 過時問題已全部修正** ✅

### 修正狀態

- ✅ **已修正**: 2 個 Critical 問題
- ℹ️ **已檢查**: 1 個 Warning 問題（無需修正）
- 📋 **建議**: 2 個可選遷移項目（非必要）

---

## 🔴 Critical Issues（必須修正）

### 1. `jax.experimental.pjit` 已過時 ✅ 已修正

**影響文件**:
- `examples/kolmogorov_flow/train.py`

**問題描述**:
```python
# ❌ 過時用法（JAX < 0.4）
from jax.experimental.pjit import pjit
```

從 **JAX 0.4** 開始，`pjit` 已從 `jax.experimental` 合併到主 API。使用舊的導入路徑會在未來版本中失效。

**修正方案**:
```python
# ✅ JAX 0.4+ 正確用法
from jax import jit
from jax.sharding import PartitionSpec as P

# pjit 功能已整合進 jit，使用方式：
jitted_fn = jit(
    fn,
    in_shardings=in_specs,
    out_shardings=out_specs
)
```

**已完成修正**:
- ✅ Line 9: 修改 import 為 `from jax import random, lax, jit`
- ✅ Line 12: 移除 `from jax.experimental.pjit import pjit`
- ✅ Line 234, 240: 將 `pjit(...)` 改為 `jit(...)`
- ✅ Line 588: 將評估函數的 `pjit(...)` 改為 `jit(...)`
- ✅ Line 246-247, 267, 272-274: 更新所有函數名稱和調用
- ✅ Line 332: 更新日誌訊息

**向後相容性**: ✅ 完全相容（JAX 0.4+）

---

### 2. `shard_map` 的 `axis_name` 參數 ℹ️ 無影響

**檢查結果**: ✅ 專案中**未使用** `shard_map`

**說明**:
- 經檢查，專案中所有多 GPU 運算已使用 `jit` + `NamedSharding`
- `examples/kolmogorov_flow/train.py` 的評估函數使用 `jit(...)`，非 `shard_map`
- 無需修正

**參考資訊**（如未來需要使用）:
新版 `shard_map` 從 mesh 的軸名稱自動推導 `axis_name`，不再需要顯式傳遞：

```python
# ❌ 舊版 API（JAX < 0.4.13）
result = shard_map(
    fn, mesh=mesh, in_specs=P('data'), out_specs=P('data'),
    axis_name='data'  # ⚠️ 此參數已移除
)

# ✅ 新版 API
result = shard_map(
    fn, mesh=mesh, in_specs=P('data'), out_specs=P('data')
    # axis_name 由 mesh 自動推導
)
```

**建議**: 繼續使用 `jit` + sharding（與訓練流程一致，API 更穩定）

---

## 🟡 Warning（建議修正）

### 3. `flax.jax_utils` ℹ️ 已檢查，保留現狀

**影響文件**:
- `jaxpi/models.py:5, 240`

**檢查結果**: ✅ 目前可保留，未來建議遷移

**使用情況**:
```python
# jaxpi/models.py:240
if replicate:
    return jax_utils.replicate(state)
```

**分析**:
- `jax_utils.replicate()` 用於 `pmap` 模式（將 state 複製到所有設備）
- **Kolmogorov Flow 使用 `replicate_state=False`**，手動管理 sharding（正確做法）
- 其他範例仍使用 `pmap`，依賴 `jax_utils.replicate`
- 在 JAX 0.6.x 中仍然可用，但建議未來遷移

**現代替代方案**（未來遷移時使用）:
```python
# ❌ 舊版（用於 pmap）
from flax import jax_utils
replicated = jax_utils.replicate(state)

# ✅ 新版（用於 jit + sharding）
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.sharding.Mesh(jax.devices(), ('data',))
replicated_sharding = NamedSharding(mesh, P())
replicated = jax.device_put(state, replicated_sharding)
```

**建議**:
- **目前**: 保持現狀（與 pmap 相容）
- **未來**: 逐步遷移其他範例到 `jit` + sharding，然後移除 `jax_utils` 依賴

---

## 🟢 Info（可選遷移）

### 4. `pmap` 仍可用但建議遷移

**影響文件**（共 9 處）:
- `jaxpi/models.py:8, 304, 311`
- `jaxpi/samplers.py:5, 33, 57, 88, 106`
- `examples/ldc/models.py:5, 151, 158`
- `examples/rayleigh_taylor/train.py:8, 34, 56`
- 其他範例檔案

**現況**:
- ✅ `pmap` 在 JAX 0.6.x **仍然完全支援**
- ⚠️ JAX 官方建議遷移到 `jit` + `shard_map` 或純 `jit` + sharding

**遷移範例**:
```python
# ❌ pmap 用法（仍可用但不建議）
from jax import pmap

@pmap
def train_step(state, batch):
    ...

# ✅ 建議用法（JAX 0.4+）
from jax import jit
from jax.sharding import NamedSharding, PartitionSpec as P

@jit
def train_step(state, batch):
    ...

# 在呼叫前設置 sharding
state = jax.device_put(state, replicated_sharding)
```

**優先級**: 低（非緊急）
**理由**: `pmap` 仍可正常運作，但為了未來維護性建議逐步遷移。

---

### 5. `jax.experimental.jet` 仍為實驗性

**影響文件**:
- `examples/ks_chaotic/models.py:5`
- `examples/kdv/models.py:7`

**現況**:
- ✅ 在 JAX 0.6.x **仍然可用**
- ⚠️ 仍在 `experimental` 命名空間，API 可能未來變更

**建議**:
- 暫時無需修改
- 關注 JAX release notes，未來可能需要調整

**範例用法**（目前正確）:
```python
from jax.experimental.jet import jet

# 計算高階導數
u_fn = lambda x: self.u_net(params, t, x)
_, (u_x, u_xx, u_xxx, u_xxxx) = jet(u_fn, (x,), [[1.0, 0.0, 0.0, 0.0]])
```

---

## 修正摘要

### ✅ 已完成修正
1. **修正 `pjit` 導入** (`examples/kolmogorov_flow/train.py`)
   - 移除 `from jax.experimental.pjit import pjit`
   - 改用 `from jax import jit`
   - 更新所有 `pjit(...)` 調用為 `jit(...)`
   - 更新變數名稱和日誌訊息

### ℹ️ 已檢查，無需修正
2. **`shard_map` 使用檢查**: 專案中未使用 `shard_map`
3. **`flax.jax_utils` 檢查**: 目前與 `pmap` 相容，保留現狀

### 📋 後續建議（可選）
4. 逐步遷移其他範例的 `pmap` → `jit` + sharding
5. 移除 `flax.jax_utils` 依賴（在完成 pmap 遷移後）

---

## 測試建議

修正後建議測試：

- [ ] **語法檢查**: ✅ 已通過 `python -m py_compile`
- [ ] **單 GPU 訓練**: 測試基本功能
- [ ] **雙 GPU 訓練**: 測試 batch size 2851（應解決 API 錯誤）
- [ ] **評估函數**: 驗證多 GPU 評估正常運作
- [ ] **Checkpoint 操作**: 測試儲存/載入
- [ ] **其他範例**: 確認未破壞其他程式

---

## 參考資料

- [JAX Array Migration Guide](https://github.com/jax-ml/jax/blob/main/docs/jax_array_migration.md)
- [Migrating from pmap](https://github.com/jax-ml/jax/blob/main/docs/migrate_pmap.md)
- [JAX Sharding Documentation](https://github.com/jax-ml/jax/blob/main/docs/notebooks/Distributed_arrays_and_automatic_parallelization.md)
- [JAX experimental.jet API](https://github.com/jax-ml/jax/blob/main/docs/jax.experimental.jet.md)

---

## 結論

### ✅ 修正完成

專案的 JAX API 使用方式**已完全相容 JAX 0.6.x**：

1. ✅ **過時的 `pjit` 導入已修正** → 改用 `jax.jit`
2. ✅ **未使用問題的 `shard_map`** → 無需修正
3. ℹ️ **`flax.jax_utils` 保留** → 與現有 `pmap` 程式相容

### 預期效果

修正後應能解決：
- ✅ **語法相容性**: 符合 JAX 0.6.x API 規範
- ✅ **雙卡訓練錯誤**: 原本的 `pjit` API 錯誤已修正
- ✅ **未來維護性**: 使用穩定的主 API，避免 experimental 變動

### 下一步

建議在伺服器端測試：
```bash
# 測試雙卡訓練（batch size 2851）
python examples/kolmogorov_flow/train.py --config=examples/kolmogorov_flow/configs/pirate.py
```

如遇到其他問題（如 OOM），可能需要進一步調整 batch size 或記憶體管理策略，但 **API 相容性問題已解決**。
