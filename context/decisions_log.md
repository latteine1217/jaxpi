# 決策日誌 (Decisions Log)

記錄所有重要的技術決策、實作選擇、以及其依據。

---

## 2026-01-01: 專案初始化

### 決策 #001: 選擇 JAX 0.4.23 + CUDA 11
**原因**:
- 伺服器的 CUDA driver 535.230.02 支援 CUDA 11.x
- JAX 0.4.23 是穩定版本，與 scipy 1.11.4 相容
- 避免使用最新版 scipy (1.15.3) 造成的 API 不相容

**影響**:
- ✅ GPU 成功被識別
- ✅ cuDNN 8.6.0 與 jaxlib 完美配合
- ⚠️ 需要手動設定 LD_LIBRARY_PATH

**參考**: Job 2561 成功訓練

---

### 決策 #002: PyTorch CPU 版本
**原因**:
- PyTorch GPU 版需要 cuDNN 9，與 JAX 的 cuDNN 8.6 衝突
- JAXpi 僅使用 PyTorch 的 DataLoader，不需要 GPU 加速
- 避免 CUDA 庫版本衝突

**影響**:
- ✅ 成功避免 cuDNN 版本衝突
- ✅ 不影響訓練性能（PyTorch 僅用於資料載入）

**參考**: TRAINING_RESULTS_SUMMARY.md 第 146-148 行

---

### 決策 #003: 單 GPU 使用 batch_size=2048
**原因**:
- 預設 batch_size=8192 在 16GB GPU 上會 OOM
- 2048 是安全值，記憶體使用量 ~10GB (62.5%)
- 優先確保訓練成功完成

**影響**:
- ✅ Job 2561 成功完成測試訓練
- ✅ 證實物理模型與損失函數正確
- ⚠️ 未充分利用 GPU 記憶體

**後續優化**: 決策 #004

**參考**: Job 2561, BATCH_SIZE_TEST_REPORT.md

---

### 決策 #004: 推薦雙 GPU batch_size=10240 per device
**原因**:
- Batch size 測試 (Job 2564) 顯示最大可達 12288
- 保留 17% 安全餘量避免記憶體波動造成 OOM
- 相比單 GPU (8192) 提升 2.5x 吞吐量

**影響**:
- ✅ 理論上大幅提升訓練速度
- ❌ Job 2565 實際執行失敗 (JIT 編譯階段)

**狀態**: 待 Debug

**參考**: BATCH_SIZE_TEST_REPORT.md, Job 2565

---

---

### 決策 #005: 雙 GPU OOM 根因與解決方案

**問題**:
- Job 2565 使用 batch_size=10240 per device 時 OOM
- 錯誤: 嘗試分配 22.4GB，但 P100 只有 16GB

**根本原因**:
1. **梯度計算記憶體需求是 per-device 的**
2. `update_weights` 中的 `vmap(transpose(jvp(...)))` 計算 Jacobian 時：
   - 需要 15+ 個 150MB 中間緩衝區
   - XLA 總共需要分配 22.4GB 臨時記憶體
   - 這是**每個 GPU 獨立**的需求，pmap 無法分散
3. batch_size=10240 時，Shape: `f32[5,10240,3,256]` 太大

**解決方案**:
- 降低 batch_size 到 **4096 per device**
- 這讓單 GPU 的記憶體需求降到 ~12GB（安全範圍）
- 總 batch size = 8,192（與原始單 GPU 預設值相同）

**影響**:
- ✅ Job 2566 (batch_size=4096) 成功運行
- ✅ 訓練穩定，損失正常收斂
- ⚠️ 未能達到 2.5x batch size 提升（原期望 20,480）
- ✅ 相比單 GPU (batch_size=2048)，仍有 4x 提升

**狀態**: 已解決並驗證

**參考**: Job 2565 (失敗), Job 2566 (成功)

---

### 決策 #006: 最終訓練配置 - 雙 GPU

**選擇**: 雙 GPU 訓練，batch_size=4096 per device (總 8,192)

**原因**:
1. **穩定性**: Job 2566 證實此配置可穩定運行
2. **性能**: 相比單 GPU (batch_size=2048) 有 4x batch size 提升
3. **時間**: 預估 7.8 小時完成 20,000 步訓練（可接受）
4. **收斂**: 訓練質量良好（u_error ↓99.5%）

**相比其他方案**:
- vs 單 GPU batch_size=2048: ✅ 4x batch, ~相同時間
- vs 單 GPU batch_size=8192: ⚠️ batch 相同，但需測試穩定性
- vs 雙 GPU batch_size=10240: ❌ OOM，不可行

**未來優化空間**:
- 可測試 batch_size=5120 或 6144 per device
- 可能介於 4096 (成功) 和 10240 (失敗) 之間

**狀態**: 已選定，運行中 (Job 2566)

**參考**: Job 2566

---

### 決策 #007: Wandb 離線模式改為線上模式

**問題**:
- 原始腳本設定 `WANDB_MODE=offline`
- 訓練 logs 只儲存在本地，不會自動上傳到 wandb 雲端
- 需要手動同步才能在 wandb dashboard 查看

**原因分析**:
1. **離線模式的好處**：避免網路問題導致訓練中斷
2. **離線模式的缺點**：無法即時監控訓練進度

**解決方案**:
1. **手動同步現有 runs**：
   ```bash
   cd ~/jaxpi
   python3 -m wandb sync wandb/offline-run-*
   ```
   - 已同步 15 個 offline runs
   
2. **創建 online 模式腳本**：
   - 新腳本：`slurm_train_kf_pirate_2gpu_online.sh`
   - 設定：`WANDB_MODE=online`
   - 優點：可即時在 wandb dashboard 監控

**建議**:
- **長時間訓練（>1小時）**：使用 offline 模式，訓練完成後手動同步
- **短時間實驗/debug**：使用 online 模式，方便即時監控

**狀態**: 已解決

**參考**: 
- Job 2566 wandb: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow/runs/u1o0ib02
- Project dashboard: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow

---

### 決策 #008: **反轉** - 修補 SOAP 相容性而非移除

**背景問題**:
- SOAP optimizer 與 optax 0.1.9 不相容（缺少 `tree_update_moment` 函數）
- 嘗試升級 optax 導致 JAX 連帶升級到 0.6.2（不支援 CUDA 11）
- 初始方案是移除 SOAP，但**使用者明確要求保留 SOAP 進行比較**

**決策**: **保持 JAX 0.4.23 + optax 0.1.9，修補缺失的 SOAP 依賴函數**

1. **創建 optax 補丁模組** (`jaxpi/optax_soap_patch.py`):
   - 實作 `tree_update_moment()` - 使用 JAX tree_map 更新動量
   - 實作 `tree_update_moment_per_elem_norm()` - Per-element 正規化版本
   - 在 import 時自動 monkey-patch `optax.tree_utils`
   
2. **修復 soap_jax 套件的 JAX API 相容性**:
   - 問題：`jnp.argsort(x, descending=True)` 在 JAX 0.4.23 不存在
   - 解法：直接修改套件原始碼改為 `jnp.argsort(-x)`
   - 位置：`~/.local/lib/python3.10/site-packages/soap_jax/soap.py`

3. **修復模組名稱衝突**:
   - 重新命名 `jaxpi/logging.py` → `jaxpi/wandb_logging.py`
   - 避免與 Python 標準庫 `logging` 衝突

**原因**:
- ✅ **使用者需求優先**：明確要求比較 PIRATE vs SOAP
- ✅ 保持穩定的 CUDA 11 + JAX 0.4.23 環境
- ✅ 避免升級連鎖反應（numpy 2.x 不相容、CUDA 12 需求等）
- ✅ 補丁實作簡單且不侵入式（僅 20 行程式碼）

**影響**:
- ✅ Job 2574 (SOAP) 成功運行並正常收斂
- ✅ PIRATE 配置保持不變（Job 2575 待執行）
- ⚠️ 需維護補丁模組，未來升級時可能需要調整
- ⚠️ 直接修改 site-packages（非 best practice，但有效）

**實作完成**:
1. [x] 創建 `jaxpi/optax_soap_patch.py`
2. [x] 修改 `jaxpi/models.py` 引入補丁
3. [x] 修正 soap_jax argsort 問題
4. [x] 重新命名 logging 模組
5. [x] 更新所有 import 路徑
6. [x] 測試 SOAP 訓練（Job 2574 成功）
7. [x] 創建環境重建腳本 `rebuild_env.sh`

**狀態**: ✅ 已完成並驗證

**參考**: 
- Job 2574 (SOAP 訓練中，正常收斂)
- Job 2575 (PIRATE 等待中)
- 檔案：`jaxpi/optax_soap_patch.py`, `rebuild_env.sh`

---

---

### 決策 #009: SOAP vs PIRATE 訓練配置對比

**目標**: 公平比較兩個 optimizer 的收斂速度與最終精度

**SOAP 配置** (Job 2574):
- Optimizer: Soap (schedule-free optimizer)
- Hidden dim: 384 (較大網路)
- Time windows: 25
- Iterations per window: 2000
- Batch size: 3072 per device (較小，因網路較大)
- 預估完成時間: ~50-60 hours

**PIRATE 配置** (Job 2575):
- Optimizer: Adam
- Hidden dim: 256 (標準網路)
- Time windows: 10
- Iterations per window: 2000
- Batch size: 4096 per device (較大)
- 預估完成時間: ~12-15 hours

**設計原理**:
1. **網路容量差異**: SOAP 使用較大網路 (384 vs 256) 來測試其對高容量模型的效果
2. **時間窗口數**: SOAP 使用更多窗口 (25 vs 10) 以測試長期訓練穩定性
3. **Batch size**: 因 SOAP 網路較大，降低 batch size 避免 OOM
4. **公平性考量**: 總迭代次數相同 (SOAP: 25×2000=50k, PIRATE: 10×2000=20k) - **待調整**

**當前狀態** (2026-01-02 17:32 JST):
- SOAP (Job 2574): ✅ Window 1/25 完成
  - Iter 2000/2000 完成
  - u_error: 2.984 → 0.430 (↓85.6%)
  - v_error: 2.604 → 0.483 (↓81.5%)
  - 訓練時間: ~11 分鐘/window
  - 收斂穩定，無 NaN 或發散

- PIRATE (Job 2575): ⏳ 等待 GPU 資源
  - 狀態: PENDING (Resources)
  - 預計在 SOAP 完成或取消後開始

**下一步**:
- [ ] 監控 SOAP 是否成功進入 Window 2
- [ ] 等待 PIRATE 開始訓練
- [ ] 訓練完成後比較：
  - 收斂速度（每個 window 的 error 下降率）
  - 最終精度（最後 window 的 error）
  - 訓練穩定性（是否有震盪或發散）
  - 總訓練時間

**狀態**: 進行中

**參考**: Job 2574 (SOAP), Job 2575 (PIRATE)

---

### 決策 #010: Wandb 混合模式策略

**背景**:
- SOAP (Job 2574) 已使用 offline 模式開始訓練
- PIRATE (Job 2575) 尚未開始，可調整配置
- 需要監控兩個長時間訓練的進度

**決策**: **採用混合模式** - SOAP offline + PIRATE online

1. **SOAP 保持 offline 模式**:
   - 已經開始訓練，無法中途切換
   - 每 6-12 小時手動同步一次查看進度
   - 指令: `ssh ... "cd ~/jaxpi && python3 -m wandb sync wandb/latest-run"`

2. **PIRATE 改為 online 模式**:
   - 修改 `slurm_train_2gpu_final.sh`: `WANDB_MODE=online`
   - Job 2575 開始時自動即時上傳
   - 方便在 wandb dashboard 即時對比兩個實驗

3. **工具開發**:
   - 創建 `sync_wandb.sh` 互動式同步腳本
   - 創建 `WANDB_SYNC_GUIDE.md` 完整操作指南

**原因**:
- ✅ SOAP 無法中途改模式（已開始 13+ 小時）
- ✅ PIRATE 尚未開始，可預先配置
- ✅ 混合模式兼顧穩定性（offline）與即時性（online）
- ✅ 避免網路問題影響長時間 SOAP 訓練

**實作完成**:
- [x] 同步 SOAP 當前進度到 wandb（已完成，可查看）
- [x] 修改 PIRATE 腳本為 online 模式
- [x] 創建 `sync_wandb.sh` 工具
- [x] 撰寫 `WANDB_SYNC_GUIDE.md` 文檔

**影響**:
- ✅ 可立即在 wandb 查看 SOAP 前期結果
- ✅ PIRATE 開始後自動即時更新（無需手動同步）
- ✅ 兩個實驗可在 dashboard 並排比較
- ⚠️ 需定期手動同步 SOAP（建議每天 2 次）

**狀態**: ✅ 已實施

**參考**: 
- Wandb project: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow/runs/pqf1pwby
- `WANDB_SYNC_GUIDE.md`
- `sync_wandb.sh`

---

### 決策 #011: Wandb Group/Tags/Sweep 增強功能

**背景**:
- 現有訓練只使用基本的 wandb.init(project, name)
- 無法有效組織和對比多個實驗（SOAP vs PIRATE）
- 缺乏超參數調優自動化工具

**決策**: **全面升級 Wandb 整合**，支持 Group/Tags/Sweep

#### 1. 配置文件增強
在所有配置文件（pirate.py, soap.py）中添加：
```python
wandb.group = "optimizer_comparison"  # 實驗分組
wandb.tags = ["adam", "2gpu", "Re10000"]  # 標籤列表
wandb.notes = "Experiment description"  # 備註說明
wandb.sweep_id = None  # Sweep ID (自動化調優)
```

#### 2. 訓練程式更新 (`train.py`)
- 重構 `train_and_evaluate()` 的 `wandb.init()` 邏輯
- 支持條件式載入 group/tags/notes
- 添加 Sweep 模式檢測與配置更新
- 實作 `update_config_from_sweep()` 函數處理超參數注入

#### 3. Sweep 配置創建
創建 `configs/sweep_config.yaml`：
- 使用貝葉斯優化搜索最佳超參數
- 優化目標：最小化 u_error
- 參數範圍：learning_rate, hidden_dim, num_layers, batch_size等
- Early stopping: Hyperband 提前終止表現差的 run

#### 4. 文檔創建
- `WANDB_ADVANCED_GUIDE.md`: 完整使用指南（30 頁）
- 包含：實際案例、故障排除、進階技巧

**原因**:
- ✅ **實驗組織**: Group 讓 SOAP/PIRATE 可在同一視圖對比
- ✅ **靈活過濾**: Tags 支持多維度篩選（optimizer, hardware, Reynolds number）
- ✅ **可追溯性**: Notes 記錄實驗目的和特殊配置
- ✅ **自動化優化**: Sweep 減少手動調參工作量，智能搜索最佳配置
- ✅ **可維護性**: 配置與程式碼分離，易於複現和分享

**實作完成**:
- [x] 更新 `configs/pirate.py` 添加 group/tags/notes
- [x] 更新 `configs/soap.py` 添加 group/tags/notes
- [x] 重構 `train.py` 的 wandb 初始化邏輯
- [x] 實作 `update_config_from_sweep()` 函數
- [x] 創建 `configs/sweep_config.yaml`
- [x] 撰寫 `WANDB_ADVANCED_GUIDE.md` 文檔
- [x] 部署所有更新到遠端伺服器
- [x] 驗證配置文件可正常載入

**影響**:
- ✅ SOAP 和 PIRATE 現在有 `group="optimizer_comparison"`，可並排對比
- ✅ Tags 幫助快速識別實驗類型（optimizer, hardware setup）
- ✅ Sweep 功能可用於未來超參數優化（現在訓練完成後）
- ⚠️ Sweep 需要 online 模式（與當前 SOAP offline 模式不衝突）
- ✅ 所有改動向後相容，不影響正在運行的 Job 2574, 2575

**使用範例**:
```python
# 配置文件
wandb.group = "optimizer_comparison"
wandb.tags = ["soap", "schedule_free"]

# 訓練時自動應用
python3 main.py --config=configs/soap.py

# 或命令列覆蓋
python3 main.py --config=configs/pirate.py \
  --config.wandb.group="ablation_study" \
  --config.wandb.tags='["test"]'

# Sweep 使用
wandb sweep configs/sweep_config.yaml  # 創建 sweep
wandb agent <entity>/<project>/<sweep-id>  # 啟動 agent
```

**下一步建議**:
1. ✅ 當前 SOAP/PIRATE 訓練完成後，在 Wandb 查看 group 對比效果
2. ⏳ 根據初步結果，決定是否需要使用 Sweep 進行超參數調優
3. ⏳ 如需 Sweep，在新的 GPU 時段啟動（避免干擾當前訓練）

**狀態**: ✅ 已完成並部署

**參考**: 
- `WANDB_ADVANCED_GUIDE.md`
- `configs/sweep_config.yaml`
- `train.py` (lines 101-233)

---

## 待補充決策

以下決策待訓練完成後補充：

- [ ] **決策 #010**: SOAP vs PIRATE 最終結果比較
- [ ] **決策 #011**: 是否需要進一步優化 batch size 或網路架構

---

## 決策模板

```markdown
### 決策 #XXX: [簡短標題]
**原因**:
- [為什麼做這個決定]
- [考慮了哪些替代方案]

**影響**:
- ✅ [正面影響]
- ⚠️ [中性/警示]
- ❌ [負面影響或失敗]

**狀態**: [進行中 / 已驗證 / 待改進]

**參考**: [相關檔案、Job ID、測試結果]
```
