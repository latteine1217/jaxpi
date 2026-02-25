# 執行最小可行測試指南

## 快速開始

### 方式1: 使用SLURM提交作業（推薦）

```bash
# SSH至伺服器
ssh junyi@140.114.120.128

# 切換目錄
cd ~/jaxpi/examples/kolmogorov_flow

# 提交測試作業
sbatch slurm_minimal_test.sh

# 查看作業狀態
squeue -u junyi

# 查看即時輸出（作業開始後）
tail -f minimal_test_<JOB_ID>.log
```

**預計時間**: 約15-30分鐘

### 方式2: 直接執行（需要互動式GPU）

```bash
# SSH至伺服器
ssh junyi@140.114.120.128

# 請求互動式GPU節點
srun --partition=r740 --gres=gpu:1 --mem=50G --time=01:00:00 --pty bash

# 載入環境
source ~/.bashrc
conda activate jaxpi

# 切換目錄並執行
cd ~/jaxpi/examples/kolmogorov_flow
bash minimal_validation_test.sh
```

---

## 測試內容

測試腳本會執行3個測試：

1. **測試1**: 單窗口訓練（1000 steps）
   - 驗證基本訓練功能

2. **測試2**: 雙窗口訓練 - params-only模式
   - 驗證向後相容性
   - 應看到log: "僅傳遞params"

3. **測試3**: 雙窗口訓練 - full state模式  
   - 驗證新功能
   - 應看到log: "傳遞完整優化器狀態"

---

## 預期結果

### 成功標準

✅ 所有3個測試無錯誤完成  
✅ 測試2的log包含 "僅傳遞params"  
✅ 測試3的log包含 "傳遞完整優化器狀態"  
✅ 測試3的window 1初始loss < 測試2的window 1初始loss

### 如果測試失敗

1. 查看錯誤log:
   ```bash
   cat minimal_test_<JOB_ID>.err
   cat minimal_test_*/test*.log
   ```

2. 檢查常見問題:
   - 記憶體不足 → 降低batch_size
   - GPU不可用 → 檢查SLURM配置
   - 模組import錯誤 → 檢查conda環境

3. 參考文檔:
   ```bash
   cat ~/jaxpi/docs/transfer_optimizer_state/03_QUICK_REFERENCE.md
   ```

---

## 測試後檢查

### 查看結果摘要

```bash
cd ~/jaxpi/examples/kolmogorov_flow
ls -lh minimal_test_*/

# 查看測試log
cat minimal_test_*/test1_single_window.log
cat minimal_test_*/test2_params_only.log  
cat minimal_test_*/test3_full_state.log
```

### 對比初始loss（關鍵指標）

```bash
# 提取Window 1的初始loss
echo "=== Test2 (params-only) Window 1 初始loss ==="
grep -A 5 "Window 1" minimal_test_*/test2_params_only.log | head -20

echo ""
echo "=== Test3 (full state) Window 1 初始loss ==="
grep -A 5 "Window 1" minimal_test_*/test3_full_state.log | head -20
```

**預期**: Test3的初始loss應該接近Test2的最終loss（因為保留了優化器狀態）

---

## 下一步

### 若測試全部通過 ✅

1. 進行**快速驗證實驗**（3組 × 3窗口 × 10K步）
   - 參考: `docs/transfer_optimizer_state/02_EXPERIMENT_PROTOCOL.md`
   - 預計時間: ~1天GPU時間

2. 或直接用於生產訓練:
   ```bash
   # 啟用full state transfer
   python train.py --config configs/soap.py \
       --config.transfer_optimizer_state=True
   ```

### 若測試失敗 ❌

1. 檢查錯誤原因
2. 回報issue並附上log
3. 參考技術規格: `docs/transfer_optimizer_state/01_TECHNICAL_SPEC.md`

---

## 參考文檔

- 完整文檔索引: `~/jaxpi/docs/transfer_optimizer_state/`
- 實現摘要: `IMPLEMENTATION_SUMMARY.md`
- 快速參考: `03_QUICK_REFERENCE.md`
- 技術規格: `01_TECHNICAL_SPEC.md`

---

**更新日期**: 2026-01-09  
**版本**: v1.0
