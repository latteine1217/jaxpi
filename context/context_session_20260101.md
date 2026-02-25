# JAXpi Kolmogorov Flow 訓練專案 - 上下文記錄

**建立時間**: 2026-01-01  
**專案目標**: 在遠端 SLURM 集群上訓練 Kolmogorov Flow 的 Physics-Informed Neural Network

---

## 🎯 專案狀態

**當前階段**: ✅ SOAP vs PIRATE 並行訓練進行中
**伺服器**: junyi@140.114.120.128  
**專案路徑**: ~/jaxpi/  
**更新時間**: 2026-01-02 17:35 JST

---

## ✅ 已完成里程碑

### 1. 環境配置 (完成)
- ✅ JAX GPU 版本安裝 (0.4.23 + CUDA 11)
- ✅ 依賴衝突解決 (scipy, PyTorch, cuDNN)
- ✅ LD_LIBRARY_PATH 配置
- ✅ 雙 GPU 硬體偵測確認

### 2. 單 GPU 測試訓練 (完成)
- ✅ Job 2561: 成功完成 10,000 步訓練
- ✅ 訓練時間: 24 分鐘
- ✅ 損失收斂: u_error ↓50%, v_error ↓64%
- ✅ 配置: batch_size=2048, 1 GPU

### 3. Batch Size 優化測試 (完成)
- ✅ Job 2564: 完整測試單/雙 GPU 的記憶體極限
- ✅ 單 GPU 最大: 8192
- ✅ 雙 GPU 最大: 12288 per device (總 24,576)
- ✅ 雙 GPU 推薦: 10240 per device (總 20,480, 83% 容量)

### 4. SOAP Optimizer 相容性修復 (完成) ⭐
- ✅ 問題: `soap_jax` 與 optax 0.1.9 不相容
- ✅ 解法: 創建 `jaxpi/optax_soap_patch.py` 補丁模組
- ✅ 修復: `soap_jax` 的 JAX 0.4.23 API 相容性（argsort）
- ✅ 解決: 模組名稱衝突（logging.py → wandb_logging.py）
- ✅ 工具: `rebuild_env.sh` 環境重建腳本
- ✅ 驗證: Job 2574 (SOAP) 成功運行並收斂

### 5. 雙 GPU 訓練配置確定 (完成)
- ✅ PIRATE: batch_size=4096/device, 10 windows, Adam
- ✅ SOAP: batch_size=3072/device, 25 windows, SOAP optimizer
- ✅ 腳本: `slurm_train_2gpu_final.sh` (PIRATE), `slurm_train_2gpu_soap.sh` (SOAP)
- ✅ 環境: JAX 0.4.23 + CUDA 11 + optax 0.1.9（穩定配置）

---

## ✅ 問題已解決：雙 GPU 訓練配置

### Job 2565: 雙 GPU OOM 根因分析

**失敗配置**:
- Batch size: 10240 per device (總 20,480)
- 錯誤: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 24033839880 bytes (22.4 GB)`

**根本原因**:
1. **單 GPU 記憶體需求 22.4GB**，超過 P100 的 16GB
2. 問題出在 `update_weights` 中的梯度計算（`vmap(transpose(jvp(...)))`）
3. XLA 需要為 Jacobian 計算分配 ~22.4GB 的臨時緩衝區
4. **pmap 並未將這部分記憶體分散到多 GPU**（是 per-device 的需求）

**記憶體分析**:
- 15 個 150MB buffers (f32[153600,256]) = 2.25GB
- 加上梯度、中間變數、XLA overhead = **22.4GB per GPU**
- 這遠超 P100 的 16GB 限制

### Job 2566: 成功的雙 GPU 訓練 ✅

**工作配置**:
- GPUs: 2x Tesla P100-PCIE-16GB
- Batch size: **4096 per device** (總 8,192)
- Steps: 20,000 × 10 windows
- Status: **運行中** (已完成 Window 1-2，Window 3 進行中)

**性能指標**:
- JIT 編譯時間: ~62 秒（第一次）
- 每次迭代: ~0.6 秒（穩定）
- 每個 window: ~47 分鐘
- 預估總時間: **~7.8 小時**

**訓練質量**:
- u_error: 2.829 → 0.0138 (↓99.5%)
- v_error: 2.757 → 0.0246 (↓99.1%)
- 所有損失穩定收斂

**Job ID**: 2566  
**開始時間**: 2026-01-01 04:19:14 UTC  
**預計完成**: 2026-01-01 ~12:00 UTC  
**Wandb Run**: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow/runs/u1o0ib02

---

## 📋 當前任務與下一步

### 進行中 🔥

#### Job 2574 (SOAP) - ✅ 運行中
- **配置**: 
  - Optimizer: SOAP (schedule-free)
  - Batch size: 3072 per device (總 6,144)
  - Network: hidden_dim=384 (較大)
  - Time windows: 25
  - Iterations: 2000 per window
- **進度**:
  - Window 1/25 完成 ✅
  - 已完成 Iter 2000/2000
  - 當前: Window 1 → Window 2 轉換中
  - 已運行: 13:35 (11 分鐘/window)
- **性能**:
  - u_error: 2.984 → 0.430 (↓85.6%)
  - v_error: 2.604 → 0.483 (↓81.5%)
  - 訓練速度: ~0.58s/step
  - 收斂穩定，無 NaN 或發散
- **預估完成時間**: ~50-60 小時 (25 windows)
- **日誌**: `~/jaxpi/logs/kf_soap_2gpu_2574.{out,err}`
- **腳本**: `~/jaxpi/slurm_train_2gpu_soap.sh`

#### Job 2575 (PIRATE) - ⏳ 排隊中
- **配置**:
  - Optimizer: Adam
  - Batch size: 4096 per device (總 8,192)
  - Network: hidden_dim=256 (標準)
  - Time windows: 10
  - Iterations: 2000 per window
- **狀態**: PENDING (Resources)
  - 等待 GPU 資源（SOAP 占用中）
  - 將在 SOAP 完成或取消後自動開始
- **預估完成時間**: ~12-15 小時 (10 windows)
- **日誌**: `~/jaxpi/logs/kf_pirate_2gpu_2575.{out,err}`
- **腳本**: `~/jaxpi/slurm_train_2gpu_final.sh`

### 待完成 📝

1. **監控 SOAP 訓練** (持續)
   - 檢查是否成功進入 Window 2
   - 監控各 window 的收斂速度
   - 確認 25 個 windows 都能穩定完成
   - 使用工具: `./monitor_training.sh` ⭐

2. **等待 PIRATE 開始** (取決於資源)
   - 選項 A: 等待 SOAP 完成（2-3 天）
   - 選項 B: 取消 SOAP，優先跑 PIRATE（12-15 小時）
   - 建議: 讓 SOAP 先跑完幾個 windows，評估效果後決定

3. **對比分析** (兩個 job 都完成後)
   - 收斂速度比較（每 window 的 error 下降率）
   - 最終精度比較（最後 window 的 u_error, v_error）
   - 訓練穩定性（是否有震盪、發散、NaN）
   - Optimizer 效果（SOAP vs Adam）
   - 網路容量影響（384 vs 256 hidden_dim）

4. **結果報告** (分析完成後)
   - 生成訓練曲線圖（error vs iterations）
   - 撰寫實驗報告（SOAP_vs_PIRATE_RESULTS.md）
   - 更新 README.md
   - 決定最佳配置建議

### 可選操作 🎮

- **取消 SOAP**: `ssh junyi@140.114.120.128 'scancel 2574'`
- **取消 PIRATE**: `ssh junyi@140.114.120.128 'scancel 2575'`
- **監控訓練**: `./monitor_training.sh`
- **查看 SOAP 日誌**: `ssh junyi@140.114.120.128 'tail -100 ~/jaxpi/logs/kf_soap_2gpu_2574.err'`
- **同步 wandb**: `ssh junyi@140.114.120.128 "cd ~/jaxpi && python3 -m wandb sync wandb/offline-run-*"`

---

## 📂 重要檔案路徑

### 本地 (開發機)
- 專案根目錄: `/Users/latteine/Documents/coding/jaxpi/`
- Context: `context/context_session_20260101.md`
- 決策日誌: `context/decisions_log.md` (待建立)
- 任務目錄: `tasks/` (待建立)

### 遠端伺服器
- 專案根目錄: `~/jaxpi/`
- **主要訓練腳本**: 
  - `~/jaxpi/slurm_train_2gpu_final.sh` (PIRATE) ⭐
  - `~/jaxpi/slurm_train_2gpu_soap.sh` (SOAP) ⭐
- **監控工具**: 
  - 本地監控腳本: `./monitor_training.sh` ⭐ (快速檢查訓練狀態)
- **SOAP 補丁模組**:
  - `~/jaxpi/jaxpi/optax_soap_patch.py` - optax 相容性補丁
  - `~/jaxpi/rebuild_env.sh` - 環境重建腳本
- 腳本歸檔: `~/jaxpi/scripts_archive/` (舊版本)
- 文檔:
  - `~/jaxpi/TRAINING_GUIDE.md` - 訓練指南（待更新）
  - `~/jaxpi/PIRATE_VS_SOAP.md` - 實驗對比說明（待更新）
  - `~/jaxpi/context/decisions_log.md` - 技術決策日誌 ⭐
- 日誌: `~/jaxpi/logs/`
  - `kf_soap_2gpu_2574.{out,err}` - SOAP 訓練日誌
  - `kf_pirate_2gpu_2575.{out,err}` - PIRATE 訓練日誌
- 結果: `~/jaxpi/examples/kolmogorov_flow/results_*/`
- Checkpoints: `~/jaxpi/soap/ckpt/`, `~/jaxpi/pirate/ckpt/`

### 配置檔案
- PIRATE 配置: `examples/kolmogorov_flow/configs/pirate.py`
- SOAP 配置: `examples/kolmogorov_flow/configs/soap.py` ⭐
- 訓練邏輯: `examples/kolmogorov_flow/train.py`
- 模型定義: `examples/kolmogorov_flow/models.py`
- 核心架構: `jaxpi/models.py` (包含 SOAP 補丁引入), `jaxpi/archs.py`
- Wandb 日誌: `jaxpi/wandb_logging.py` (已重命名避免衝突)

---

## 🔧 技術規格

### 硬體
- 節點: acmt20 (r740 partition)
- GPU: 2x Tesla P100-PCIE-16GB (16GB each)
- CPU: 48 cores
- RAM: 110GB
- CUDA Driver: 535.230.02

### 軟體環境
```bash
JAX: 0.4.23
jaxlib: 0.4.23+cuda11.cudnn86
Python: 3.10.12
PyTorch: 2.9.1+cpu
scipy: 1.11.4
numpy: 1.26.4  # 重要：1.26.x，不能用 2.x
flax: 0.7.5
optax: 0.1.9
soap_jax: latest (from GitHub)
cuDNN: 8.6.0.163
```

**重要依賴約束**:
- NumPy 必須 < 2.0（JAX 0.4.23 不支援 2.x）
- optax 鎖定 0.1.9（SOAP 補丁基於此版本）
- JAX 鎖定 0.4.23（CUDA 11 相容性）

### 關鍵環境變數
```bash
export LD_LIBRARY_PATH="${HOME}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:..."
```

---

## 📊 性能基線

### 單 GPU (已驗證)
- Batch size: 2048
- 時間/window: ~2.3 分鐘
- 總訓練時間 (10,000 步): 24 分鐘
- 推算 20,000 步: ~48 分鐘

### 雙 GPU (目標)
- Batch size: 10240 per device (20,480 總)
- 預期加速: 1.5-2.0x
- 目標訓練時間: 30-40 分鐘 (20,000 步)

---

## 🎯 成功驗收標準

1. ✅ 訓練完成 20,000 步，10 個時間窗口
2. ✅ Exit Code = 0
3. ✅ 所有損失收斂 (u_error, v_error, w_error 均下降)
4. ✅ Checkpoints 成功儲存
5. ✅ 評估指標達到或超越單 GPU baseline

---

## 📝 備註

- Wandb API Key: `daf43f72d9f4f636dc69479c446ace76a4a3eb92` (已配置)
- 測試訓練已證實物理模型正確、損失函數收斂
- 主要挑戰是工程問題（雙 GPU 配置），非物理或數學問題
