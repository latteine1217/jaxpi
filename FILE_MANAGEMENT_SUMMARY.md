# 文件管理總結

實驗報告和結果已完成整理，文件分布如下：

## 📍 本地文件（主要工作目錄）

**位置**: `/Users/latteine/Documents/coding/jaxpi/`

### 📊 報告文件
- ⭐ **EXECUTIVE_SUMMARY.md** (6.3 KB) - 執行摘要，快速了解結果
- 📊 **EXPERIMENT_REPORT.md** (14 KB) - 完整技術報告
- 📋 **INDEX.md** (7.2 KB) - 文件總覽和導航
- 🎨 **VISUALIZATION_GUIDE.md** (6.6 KB) - 視覺化指南
- 📏 **EVALUATION_GUIDE.md** (8.5 KB) - 評估指南

### 📈 視覺化圖表
- **training_comparison.png** (580 KB) - 訓練對比圖
- **error_growth_linear.png** (323 KB) - 誤差增長圖
- **field_comparison_pirate_demo.png** (935 KB) - PIRATE 場對比示例
- **field_comparison_soap_demo.png** (912 KB) - SOAP 場對比示例

### 🛠️ 分析腳本
- **parse_pirate_errors.py** - 從日誌提取誤差
- **compare_results.py** - 生成對比報告
- **plot_training_comparison.py** - 生成訓練對比圖
- **generate_field_comparison.py** - 生成流場對比圖
- **slurm_generate_field_comparison.sh** - SLURM 批量生成腳本

### 📚 其他文檔
- **README.md** - 專案概述
- **AGENTS.md** - Agent 開發指引
- **WANDB_SYNC_GUIDE.md** - Wandb 同步指南
- 等等...

---

## 🖥️ 伺服器文件（精簡版）

**位置**: `junyi@140.114.120.128:~/jaxpi/`

### ✅ 保留的文件

#### 必要腳本（16 個文件）
- `slurm_train_2gpu_final.sh` - PIRATE 訓練腳本
- `slurm_train_2gpu_soap.sh` - SOAP 訓練腳本
- `parse_pirate_errors.py` - 誤差提取
- `compare_results.py` - 對比報告生成
- `plot_training_comparison.py` - 圖表生成
- `generate_field_comparison.py` - 場對比圖生成
- `slurm_generate_field_comparison.sh` - 批量生成
- 其他實用腳本...

#### 重要數據
- `pirate/ckpt/` (168 MB) - PIRATE checkpoints (10 windows)
- `soap_Re10000/ckpt/` (1.1 GB) - SOAP checkpoints (17 windows)
- `logs/` - 訓練日誌
- `examples/kolmogorov_flow/data/` - DNS 參考數據

#### 文檔
- `README.md` - 基本說明
- `SERVER_README.md` - 伺服器腳本使用指南

### 🗂️ 歸檔的文件

**位置**: `~/jaxpi/archive_reports/` (2.8 MB)

包含之前生成的報告和圖表：
- 所有 `.md` 報告文件
- 所有 `.txt` 數據摘要
- 所有 `.png` 視覺化圖表

**注意**: 這些文件已備份到本地，可以安全刪除以節省空間：
```bash
ssh junyi@140.114.120.128 'rm -rf ~/jaxpi/archive_reports/'
```

---

## 📊 核心實驗結果

### SOAP vs PIRATE 性能對比

| 指標 | PIRATE (平均) | SOAP (平均) | 改善 |
|------|---------------|-------------|------|
| u_error | 10.24% | 0.99% | **90.3% ⬆️** |
| v_error | 11.66% | 1.70% | **85.4% ⬆️** |
| w_error | 33.30% | 6.90% | **79.3% ⬆️** |

### 訓練狀態

| 模型 | 狀態 | 進度 | 最終誤差 (Window 10) |
|------|------|------|---------------------|
| PIRATE | ✅ 完成 | 10/10 | u=35.15%, v=33.49%, w=93.55% |
| SOAP | 🟡 進行中 | 17/25 (68%) | u=1.74%, v=2.63%, w=12.89% |

### 結論

🎯 **SOAP 顯著優於 PIRATE**，強烈推薦用於高雷諾數湍流 PINN 問題！

---

## 🚀 快速使用指南

### 查看本地報告
```bash
cd /Users/latteine/Documents/coding/jaxpi

# 閱讀執行摘要
cat EXECUTIVE_SUMMARY.md

# 查看圖表
open training_comparison.png
open error_growth_linear.png
```

### 從伺服器下載最新數據
```bash
# 下載訓練日誌
scp junyi@140.114.120.128:~/jaxpi/logs/kf_soap_2gpu_2580.err ./

# 提取最新誤差
python3 parse_pirate_errors.py kf_soap_2gpu_2580.err
```

### 在伺服器上生成場對比圖
```bash
ssh junyi@140.114.120.128
cd ~/jaxpi
sbatch slurm_generate_field_comparison.sh
```

---

## 📁 文件大小總結

### 本地（完整版）
- 報告文件: ~50 KB
- 視覺化圖表: ~2.7 MB
- 腳本: ~50 KB
- **總計**: ~2.8 MB

### 伺服器（精簡版）
- 必要腳本: ~50 KB
- Checkpoints: ~1.3 GB (PIRATE 168 MB + SOAP 1.1 GB)
- 日誌文件: ~50 MB
- 歸檔報告: ~2.8 MB (可刪除)
- **總計**: ~1.35 GB (不含歸檔)

---

## ✅ 清理檢查清單

- [x] 本地保留所有報告和圖表
- [x] 伺服器移除重複的報告和圖表到歸檔目錄
- [x] 伺服器保留必要的腳本和 checkpoints
- [x] 創建伺服器使用說明 (SERVER_README.md)
- [x] 創建本地文件管理總結 (本文件)

---

## 🔮 後續工作

1. **等待 SOAP 完成**: Window 18-25 (預計今晚)
2. **生成真實流場對比圖**: 使用完成的 checkpoints
3. **更新報告**: 包含完整 25 窗口的結果
4. **清理伺服器歸檔**: 確認本地備份後刪除 `archive_reports/`

---

**整理完成時間**: 2026-01-05 10:30 CST  
**版本**: 1.0  
**狀態**: ✅ 文件已整理完成
