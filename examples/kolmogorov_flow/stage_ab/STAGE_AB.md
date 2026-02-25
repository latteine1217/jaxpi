本文件說明 Kolmogorov Flow 的 Stage A/B 設計與用法。

=== 目的 ===
Stage A：用 LES 全場 data constraint 導向穩定梯度與平滑 loss landscape。  
Stage B：保留 PDE，改用 DNS sensor 作 data constraint 做偏差校正。

=== 檔案位置 ===
Configs:
- `examples/kolmogorov_flow/stage_ab/pirate_les_stage1.py`
- `examples/kolmogorov_flow/stage_ab/pirate_les_stage1_windowed.py`
- `examples/kolmogorov_flow/stage_ab/pirate_les_stage2.py`

工具腳本:
- `examples/kolmogorov_flow/split_kolmogorov_les_windows.py`

=== Stage A（Dense LES） ===
1. 完整資料版本  
   - Config: `examples/kolmogorov_flow/stage_ab/pirate_les_stage1.py`
   - 特性: Dense LES data constraint + causal training

2. Windowed 版本（低記憶體）  
   - Config: `examples/kolmogorov_flow/stage_ab/pirate_les_stage1_windowed.py`
   - 特性: 只載入單一 time window 的 LES，降低 GPU 記憶體

Windowed 資料產生:
```bash
python3 examples/kolmogorov_flow/split_kolmogorov_les_windows.py \
  --source examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re100000.npy \
  --output-dir examples/kolmogorov_flow/data/kolmogorov_les/re100000_windows \
  --num-windows 10
```

Stage A 執行:
```bash
python3 examples/kolmogorov_flow/train.py \
  --config=examples/kolmogorov_flow/stage_ab/pirate_les_stage1.py
```

Windowed Stage A:
```bash
python3 examples/kolmogorov_flow/train.py \
  --config=examples/kolmogorov_flow/stage_ab/pirate_les_stage1_windowed.py
```

=== Stage B（DNS Sensor + PDE） ===
- Config: `examples/kolmogorov_flow/stage_ab/pirate_les_stage2.py`
- 特性: DNS sensor data constraint，關閉 causal

Stage B 執行:
```bash
python3 examples/kolmogorov_flow/train.py \
  --config=examples/kolmogorov_flow/stage_ab/pirate_les_stage2.py
```

=== 注意事項 ===
1. Windowed 模式僅載入單一 time window，訓練期記憶體較低。  
2. Full-series error logging 在 windowed 模式下會跳過，避免再次載入全量資料。  
3. Stage A/B 都使用 data loss，Stage A 來源是 LES，Stage B 來源是 DNS sensor。  
