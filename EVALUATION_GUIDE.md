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

## 驗證重點

- 同一物理時間點再做模型比較。
- 同時檢查 `u/v/w` 三個誤差，不只看單一指標。
- 記錄 checkpoint step、窗口範圍、資料來源。
