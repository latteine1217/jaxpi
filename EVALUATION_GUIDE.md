# 評估指南（精簡版）

## 指標

使用相對 L2 誤差：`||pred-ref||_2 / ||ref||_2`。

## 常用指令

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config pirate \
  --checkpoint_path ./runs/kf_pirate/ckpt
```

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config soap \
  --checkpoint_path ./runs/kf_soap/ckpt \
  --window 10
```

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config soap \
  --checkpoint_path ./runs/kf_soap/ckpt \
  --mode final_step
```

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config stage1 \
  --checkpoint_path ./pirate_les_stage1/ckpt \
  --mode final_step \
  --device gpu
```

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config stage1 \
  --checkpoint_path ./pirate_les_stage1/ckpt \
  --mode window \
  --device gpu
```

## 驗證重點

- 同一物理時間點再做模型比較。
- 同時檢查 `u/v/w` 三個誤差，不只看單一指標。
- 記錄 checkpoint step、窗口範圍、資料來源。
- `--device` 僅支援 `auto` 或 `gpu`。
- `--config` 也支援 `stage_ab` alias：
  `stage1`、`stage1_windowed`、`stage1_soap`、`stage1_windowed_soap`、`stage2`、`stage2_soap`

## 訓練期誤差記錄（`log_errors=True`）

- full-series 誤差採固定 chunk 計算：
- `logging.eval_time_chunk_size`：每個時間塊的步數（優先）
- `logging.eval_time_chunk_seconds`：若未設定 `eval_time_chunk_size`，用秒數換算步數
- `logging.eval_space_chunk_size`：每個空間塊的點數
