# 視覺化腳本使用指南

本目錄包含用於視覺化 PIRATE 和 SOAP 訓練結果的腳本和圖表。

## 📊 已生成的圖表

### 1. 訓練對比圖表
- **training_comparison.png** (580 KB)
  - 4 個子圖的綜合對比
  - u/v/w 錯誤的對數尺度對比
  - SOAP 相對 PIRATE 的改善百分比

- **error_growth_linear.png** (323 KB)
  - 3 個子圖：u/v/w 錯誤的線性尺度增長趨勢
  - 清楚展示誤差隨時間窗口的傳播

### 2. 場對比示例圖（模擬數據）
- **field_comparison_pirate_demo.png** (935 KB)
- **field_comparison_soap_demo.png** (912 KB)
  - 3×3 網格布局：DNS / Prediction / Error
  - 展示 u velocity, v velocity, vorticity
  - 使用模擬數據演示圖表格式

## 🛠️ 腳本說明

### 數據提取腳本

#### 1. `parse_pirate_errors.py` - 錯誤提取腳本
從訓練日誌中提取每個時間窗口的最終錯誤。

**使用方法**：
```bash
python3 parse_pirate_errors.py <log_file>

# 範例
python3 parse_pirate_errors.py ~/jaxpi/logs/kf_pirate_2gpu_2578.err
python3 parse_pirate_errors.py ~/jaxpi/logs/kf_soap_2gpu_2580.err
```

**輸出**：
- 每個窗口的 u_error, v_error, w_error
- 統計摘要（平均、最小、最大）

#### 2. `compare_results.py` - 對比報告生成
生成詳細的 PIRATE vs SOAP 對比報告。

**使用方法**：
```bash
python3 compare_results.py > comparison_report.txt
```

**輸出**：
- 訓練配置對比
- 逐窗口錯誤對比
- 改善百分比統計
- 關鍵發現總結

### 視覺化腳本

#### 3. `plot_training_comparison.py` - 訓練對比圖表
生成訓練錯誤的對比圖表。

**使用方法**：
```bash
python3 plot_training_comparison.py
```

**輸出**：
- `training_comparison.png` - 4 子圖綜合對比
- `error_growth_linear.png` - 3 子圖線性尺度增長

**依賴**：matplotlib, numpy

#### 4. `generate_field_comparison.py` - 場對比圖生成（真實數據）
從 checkpoint 載入模型並生成 DNS vs Prediction vs Error 的場對比圖。

**使用方法**：
```bash
python3 generate_field_comparison.py \
    --config {soap|pirate} \
    --checkpoint_path <checkpoint_dir> \
    --window <time_window> \
    --time_step <step_index> \
    --output_dir <output_dir>
```

**參數說明**：
- `--config`: 模型配置（soap 或 pirate）
- `--checkpoint_path`: checkpoint 根目錄
- `--window`: 時間窗口編號（1-10 for PIRATE, 1-25 for SOAP）
- `--time_step`: 窗口內時間步索引（-1 表示最後一步，默認）
- `--output_dir`: 輸出目錄（默認當前目錄）

**範例**：
```bash
# PIRATE Window 10 最終時間步
python3 generate_field_comparison.py \
    --config pirate \
    --checkpoint_path ~/jaxpi/pirate/ckpt \
    --window 10 \
    --time_step -1 \
    --output_dir ~/jaxpi/field_comparison_plots

# SOAP Window 17 最終時間步
python3 generate_field_comparison.py \
    --config soap \
    --checkpoint_path ~/jaxpi/soap_Re10000/ckpt \
    --window 17 \
    --time_step -1 \
    --output_dir ~/jaxpi/field_comparison_plots
```

**依賴**：JAX, Flax, matplotlib, numpy（需要 GPU 環境）

**重要**：此腳本需要在計算節點上運行（需要 GPU）！

### SLURM 腳本

#### 5. `slurm_generate_field_comparison.sh` - 批量生成場對比圖
使用 SLURM 在計算節點上批量生成多個時間窗口的場對比圖。

**使用方法**：
```bash
sbatch slurm_generate_field_comparison.sh
```

**生成的圖表**：
- PIRATE: Window 5, Window 10
- SOAP: Window 5, Window 10, Window 17（如果可用）

**輸出位置**：`~/jaxpi/field_comparison_plots/`

**配置**：
- 單個 GPU
- 30 分鐘時間限制
- 32GB 內存

## 📈 主要結果

### SOAP vs PIRATE 性能對比（前 10 個窗口）

| 指標 | PIRATE（平均）| SOAP（平均）| 改善 |
|------|--------------|-------------|------|
| u_error | 10.24% | 0.99% | **90.3%** ⬆️ |
| v_error | 11.66% | 1.70% | **85.4%** ⬆️ |
| w_error | 33.30% | 6.90% | **79.3%** ⬆️ |

### 最終窗口性能（Window 10）

| 指標 | PIRATE | SOAP | 改善 |
|------|--------|------|------|
| u_error | 35.15% | 1.74% | **95.0%** ⬆️ |
| v_error | 33.49% | 2.63% | **92.1%** ⬆️ |
| w_error | 93.55% | 12.89% | **86.2%** ⬆️ |

### 關鍵發現

1. **準確度提升**：SOAP 在所有指標上都顯著優於 PIRATE，平均改善 80-90%
2. **穩定性**：SOAP 的誤差增長更緩慢，特別是在後期窗口
3. **訓練時間**：SOAP 每窗口慢 43%（70min vs 49min），但精度提升值得
4. **建議**：對於高雷諾數湍流問題，**優先使用 SOAP 優化器**

## 🔄 工作流程

### 完整的視覺化流程

1. **等待訓練完成**
   ```bash
   squeue -u junyi  # 檢查 SOAP 訓練狀態
   ```

2. **提取錯誤數據**
   ```bash
   python3 parse_pirate_errors.py ~/jaxpi/logs/kf_pirate_2gpu_2578.err > pirate_errors.txt
   python3 parse_pirate_errors.py ~/jaxpi/logs/kf_soap_2gpu_2580.err > soap_errors.txt
   ```

3. **生成對比報告**
   ```bash
   python3 compare_results.py > comparison_report.txt
   ```

4. **生成訓練對比圖表**（可在 headnode 運行）
   ```bash
   python3 plot_training_comparison.py
   ```

5. **生成場對比圖**（需要提交到計算節點）
   ```bash
   sbatch slurm_generate_field_comparison.sh
   ```

6. **下載結果到本地**
   ```bash
   scp junyi@140.114.120.128:~/jaxpi/*.png ./
   scp junyi@140.114.120.128:~/jaxpi/field_comparison_plots/*.png ./
   scp junyi@140.114.120.128:~/jaxpi/*_report.txt ./
   ```

## 📝 注意事項

1. **GPU 需求**：`generate_field_comparison.py` 必須在有 GPU 的計算節點上運行
2. **記憶體**：場對比圖生成需要約 10-20GB 記憶體（載入 checkpoint + DNS 數據）
3. **時間**：每個場對比圖生成約需 2-5 分鐘
4. **Checkpoint 位置**：
   - PIRATE: `~/jaxpi/pirate/ckpt/time_window_{1-10}/`
   - SOAP: `~/jaxpi/soap_Re10000/ckpt/time_window_{1-25}/`

## 🐛 故障排除

### 問題：在 headnode 運行場對比腳本失敗
**原因**：headnode 的 GPU 環境不完整（cuSOLVER 版本不匹配）
**解決**：使用 SLURM 提交到計算節點：`sbatch slurm_generate_field_comparison.sh`

### 問題：OOM (Out of Memory)
**原因**：256×256 網格 + checkpoint 佔用太多記憶體
**解決**：在 SLURM 腳本中增加 `--mem` 參數（當前 32GB）

### 問題：找不到 checkpoint
**原因**：路徑錯誤或窗口尚未完成
**解決**：檢查 `ls ~/jaxpi/<model>/ckpt/` 確認可用的窗口

## 📚 相關文件

- `EVALUATION_GUIDE.md` - Checkpoint 評估指南
- `README.md` - 專案主文檔
- `pirate_errors_summary.txt` - PIRATE 錯誤摘要
- `soap_errors_summary.txt` - SOAP 錯誤摘要
- `comparison_report.txt` - 詳細對比報告

---

最後更新：2026-01-05
