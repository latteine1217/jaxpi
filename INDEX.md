# JAXpi Kolmogorov Flow 實驗 - 文件總覽

本目錄包含 SOAP vs PIRATE 優化器比較實驗的完整文檔和結果。

---

## 📋 快速導航

### 🎯 核心報告（從這裡開始）

1. **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** ⭐ **推薦首先閱讀**
   - 時間對齊比較分析
   - 配置差異詳解
   - 關鍵發現和數據表格
   - 實驗設計限制說明
   - 5-8 分鐘閱讀

2. **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)** 📊 **完整技術報告**
   - 詳細的時間窗口映射分析
   - 時間對齊的結果比較
   - 配置差異的系統性分析
   - 建議的控制實驗設計
   - 15-20 分鐘閱讀

### 📊 數據和圖表

#### 視覺化圖表
- **[training_comparison.png](training_comparison.png)** (580 KB)
  - 4 子圖綜合對比
  - PIRATE vs SOAP 誤差曲線
  - 改善百分比柱狀圖

- **[error_growth_linear.png](error_growth_linear.png)** (323 KB)
  - 線性尺度誤差增長分析
  - 清楚展示 PIRATE 指數增長 vs SOAP 線性增長

- **[field_comparison_pirate_demo.png](field_comparison_pirate_demo.png)** (935 KB)
  - PIRATE 流場對比示例（DNS vs Prediction vs Error）

- **[field_comparison_soap_demo.png](field_comparison_soap_demo.png)** (912 KB)
  - SOAP 流場對比示例

#### 數據文件
- **[pirate_errors_summary.txt](pirate_errors_summary.txt)**
  - PIRATE 10 個窗口的完整誤差數據

- **[soap_errors_summary.txt](soap_errors_summary.txt)**
  - SOAP 17 個窗口（當前）的完整誤差數據

- **[comparison_report.txt](comparison_report.txt)**
  - 詳細的文字對比報告

### 🛠️ 使用指南

3. **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)**
   - 所有視覺化腳本的使用說明
   - 如何生成訓練對比圖
   - 如何生成流場對比圖
   - 故障排除

4. **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)**
   - 如何評估 checkpoint
   - 計算 L2 相對誤差
   - 批量評估流程

5. **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**
   - 訓練腳本使用說明
   - SLURM 配置
   - 常見問題

### 📚 其他文檔

6. **[CONFIG_COMPARISON.md](CONFIG_COMPARISON.md)**
   - 詳細的配置參數對比
   - 50+ 參數的逐項比較
   - 識別核心差異與次要差異

7. **[TIME_WINDOW_CORRECTION.md](TIME_WINDOW_CORRECTION.md)**
   - 時間窗口對齊問題說明
   - 為何需要時間對齊比較
   - 對結論的影響分析

8. **[REPORT_UPDATE_SUMMARY.md](REPORT_UPDATE_SUMMARY.md)**
   - 報告更新歷史
   - 從錯誤比較到正確分析的過程

9. **[README.md](README.md)**
   - 專案概述
   - 安裝和設置

10. **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)**
    - 從本地到 SLURM 的遷移指南

11. **[soap_failure_report.md](soap_failure_report.md)**
    - SOAP 第一次失敗（OOM）的分析報告

---

## 🎯 核心發現速覽

### 時間對齊比較（關鍵發現）

**相同物理時間點的性能比較**:

| 物理時間 | PIRATE | SOAP | 差異 |
|----------|--------|------|------|
| **t≈0.76** | u=1.65%, v=2.85%, w=12.27% | u=1.74%, v=2.63%, w=12.89% | <8% |
| **t≈1.32** | u=8.21%, v=14.04%, w=36.56% | u=7.96%, v=14.43%, w=37.75% | <5% |

**結論**: 在相同時間點，優化器選擇對準確度影響很小（<8%）

---

### 平均誤差比較

| 指標 | PIRATE (10窗口) | SOAP (前17窗口) | 差異 |
|------|-----------------|----------------|------|
| u_error (平均) | 10.24% | 2.97% | 71% ⬇️ |
| v_error (平均) | 11.66% | 4.22% | 64% ⬇️ |
| w_error (平均) | 33.30% | 11.80% | 65% ⬇️ |

**注意**: SOAP 配置使用 2.5× 更多時間窗口（25 vs 10），無法分離優化器效果與時間窗口策略效果

---

### 配置差異

**核心差異**（僅 3 項）:
1. 優化器: Adam vs SOAP
2. schedule_free: False vs True  
3. 時間窗口數: 10 vs 25

**其餘 50+ 參數完全相同**

---

### 訓練效率

| 指標 | PIRATE | SOAP | 比率 |
|------|--------|------|------|
| 訓練時間 | 8.2h | ~28h | 3.4× |
| 時間窗口數 | 10 | 25 | 2.5× |
| 總迭代數 | 200k | 500k | 2.5× |
| 每窗口時間 | 49 min | 70 min | 1.43× |

**關鍵洞察**: SOAP 配置的較低平均誤差主要來自更細的時間離散化策略，而非優化器本身

---

## 🔄 實驗狀態

### PIRATE
- ✅ **完成**: 10/10 窗口
- 📅 完成時間: 2026-01-03 17:13 CST
- ⏱️ 總訓練時間: 8h 12m
- 📊 最終誤差: u=35.15%, v=33.49%, w=93.55%

### SOAP
- 🟡 **進行中**: 17/25 窗口完成（68%）
- 📅 開始時間: 2026-01-04 06:04 CST
- ⏱️ 當前運行時間: ~20h
- 📊 當前誤差: u=7.96%, v=14.43%, w=37.75% (Window 17)
- ⏰ 預計完成: 2026-01-05 晚間

---

## 🛠️ 可用工具

### Python 腳本

1. **parse_pirate_errors.py**
   - 從訓練日誌提取誤差數據
   - 用法: `python3 parse_pirate_errors.py <log_file>`

2. **compare_results.py**
   - 生成對比報告
   - 用法: `python3 compare_results.py > report.txt`

3. **plot_training_comparison.py**
   - 生成訓練對比圖
   - 用法: `python3 plot_training_comparison.py`
   - 輸出: training_comparison.png, error_growth_linear.png

4. **generate_field_comparison.py**
   - 從 checkpoint 生成流場對比圖
   - 需要 GPU，使用 SLURM 提交
   - 用法: 見 VISUALIZATION_GUIDE.md

### Shell 腳本

1. **slurm_generate_field_comparison.sh**
   - 批量生成流場對比圖
   - 用法: `sbatch slurm_generate_field_comparison.sh`

2. **slurm_train_2gpu_final.sh**
   - PIRATE 訓練腳本（已完成）

3. **slurm_train_2gpu_soap.sh**
   - SOAP 訓練腳本（進行中）

---

## 📂 目錄結構

```
jaxpi/
├── 📊 報告和文檔
│   ├── EXECUTIVE_SUMMARY.md          ⭐ 執行摘要
│   ├── EXPERIMENT_REPORT.md          📊 完整報告
│   ├── VISUALIZATION_GUIDE.md        🎨 視覺化指南
│   ├── EVALUATION_GUIDE.md           📏 評估指南
│   ├── TRAINING_GUIDE.md             🚀 訓練指南
│   └── INDEX.md                      📋 本文件
│
├── 📈 圖表
│   ├── training_comparison.png       (580 KB)
│   ├── error_growth_linear.png       (323 KB)
│   ├── field_comparison_pirate_demo.png  (935 KB)
│   └── field_comparison_soap_demo.png    (912 KB)
│
├── 📊 數據
│   ├── pirate_errors_summary.txt
│   ├── soap_errors_summary.txt
│   └── comparison_report.txt
│
├── 🛠️ 腳本
│   ├── parse_pirate_errors.py
│   ├── compare_results.py
│   ├── plot_training_comparison.py
│   ├── generate_field_comparison.py
│   └── slurm_generate_field_comparison.sh
│
├── 🎯 訓練配置和日誌
│   ├── slurm_train_2gpu_final.sh
│   ├── slurm_train_2gpu_soap.sh
│   └── logs/
│       ├── kf_pirate_2gpu_2578.{out,err}
│       └── kf_soap_2gpu_2580.{out,err}
│
└── 💾 Checkpoints
    ├── pirate/ckpt/time_window_{1-10}/
    └── soap_Re10000/ckpt/time_window_{1-17}/
```

---

## 🚀 快速開始

### 查看結果

```bash
# 1. 閱讀執行摘要（推薦）
cat EXECUTIVE_SUMMARY.md

# 2. 查看視覺化圖表
display training_comparison.png
display error_growth_linear.png

# 3. 查看詳細數據
cat pirate_errors_summary.txt
cat soap_errors_summary.txt
```

### 生成新圖表

```bash
# 1. 提取最新誤差數據
python3 parse_pirate_errors.py logs/kf_soap_2gpu_2580.err > soap_errors_latest.txt

# 2. 生成對比報告
python3 compare_results.py > comparison_latest.txt

# 3. 生成訓練對比圖
python3 plot_training_comparison.py

# 4. 生成流場對比圖（需要 GPU）
sbatch slurm_generate_field_comparison.sh
```

---

## 📞 問題反饋

如有問題或需要更多資訊，請參考：
- 視覺化問題: 見 [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)
- 評估問題: 見 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- 訓練問題: 見 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

## 📝 更新日誌

- **2026-01-05 (v2.0)**: 
  - 完成時間對齊分析，修正所有比較基準
  - 重寫 EXPERIMENT_REPORT.md 和 EXECUTIVE_SUMMARY.md
  - 更新所有圖表為物理時間軸
  - 創建 CONFIG_COMPARISON.md 和 TIME_WINDOW_CORRECTION.md
  - 識別實驗設計限制，建議控制實驗
- **2026-01-05 (v1.0)**: 創建完整實驗報告和視覺化（初版）
- **2026-01-04**: SOAP 訓練開始（第二次嘗試，hidden_dim=256）
- **2026-01-03**: PIRATE 訓練完成
- **2026-01-03**: SOAP 第一次嘗試失敗（OOM，hidden_dim=384）

---

**最後更新**: 2026-01-05 11:00 CST  
**版本**: 2.0 (Time-Aligned Analysis)  
**狀態**: PIRATE 完成 ✅ | SOAP 進行中 🟡 (68%)
