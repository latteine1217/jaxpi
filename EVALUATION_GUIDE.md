# 評估指南（精簡版）

## 指標

使用相對 L2 誤差：`||pred-ref||_2 / ||ref||_2`。

## 不可違反的對齊規則

- `time window` checkpoint 一律用 **window-local time** 評估；不可把 DNS absolute time 直接餵進模型。
- spectrum / Fourier 圖一律用 **資料實際 domain length** 建波數軸；目前 `re10k` 與 `re1e6 N512 ds4` DNS 檔都是 `L = 1.0`。
- 若 `len(t_star)` 不能被 `num_time_windows` 整除，必須由 config 顯式宣告 `config.eval.expected_time_remainder`；沒有宣告時，評估腳本應直接失敗，不可靜默截尾。
- window-1 checkpoint A/B 若要作為正式結論，優先使用 direct `apply_fn` 路徑，不要用舊 wrapper-based evaluator。

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
- 記錄 `trailing time steps` 是否為 `0`；若非 `0`，必須同時記錄 config 中的 `expected_time_remainder`。
- 記錄 spectrum 使用的 `Lx / Ly`，避免把 unit-domain 與 `2π`-domain 混在一起。
- `--device` 僅支援 `auto` 或 `gpu`。
- `--config` 也支援 `stage_ab` alias：
  `stage1`、`stage1_windowed`、`stage1_soap`、`stage1_windowed_soap`、`stage2`、`stage2_soap`

## 建議入口

- 主入口： [examples/kolmogorov_flow/evaluate_checkpoint.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/evaluate_checkpoint.py)
- full-window / figure driver：
  - [eval_paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py)
  - [eval_sensor100_w25.py](/Users/latteine/Documents/coding/jaxpi/eval_sensor100_w25.py)
  - [eval_re10k_n256_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_re10k_n256_soap.py)
  - [eval_re10k_sensor100.py](/Users/latteine/Documents/coding/jaxpi/eval_re10k_sensor100.py)
- 共用 helper： [examples/kolmogorov_flow/eval_common.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/eval_common.py)
- 已停用舊入口： [examples/kolmogorov_flow/eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/eval.py)

## 訓練期誤差記錄（`log_errors=True`）

- full-series 誤差採固定 chunk 計算：
- `logging.eval_time_chunk_size`：每個時間塊的步數（優先）
- `logging.eval_time_chunk_seconds`：若未設定 `eval_time_chunk_size`，用秒數換算步數
- `logging.eval_space_chunk_size`：每個空間塊的點數
