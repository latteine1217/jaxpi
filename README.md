# JAXpi Kolmogorov Flow Branch

本分支聚焦於 Kolmogorov Flow 的 PINN 實驗，目標是用可重現流程比較 PIRATE 與 SOAP 設定。

## 快速開始

```bash
uv venv
uv pip install -e .
```

```bash
uv run python examples/kolmogorov_flow/main.py \
  --config=examples/kolmogorov_flow/configs/pirate.py \
  --workdir=./runs/kf_pirate
```

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config pirate \
  --checkpoint_path ./runs/kf_pirate/ckpt
```

```bash
# 只評估每窗口最後一步
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config soap \
  --checkpoint_path ./runs/kf_soap/ckpt \
  --mode final_step
```

## 專案重點

- `examples/kolmogorov_flow/`：訓練、評估、資料腳本
- `jaxpi/`：模型與基礎工具
- `slurm_*.sh`：伺服器提交腳本

## 文檔入口

- `INDEX.md`
- `EXECUTIVE_SUMMARY.md`
- `EVALUATION_GUIDE.md`
- `server_setup_guide.md`

## 近期更新

- `examples/kolmogorov_flow/evaluate_checkpoint.py` 已整合模式切換，使用 `--mode`、`--device` 控制（`auto/gpu`）。
- 訓練結束的 full-series 誤差評估改為固定 chunk 的 JAX 掃描路徑（降低 host-device 來回）。
- 一次性分析腳本集中到 `scripts/analysis/`。

## 分析腳本（scripts/analysis）

```bash
# 時間對齊比較（需提供兩組 checkpoint 根目錄）
python3 scripts/analysis/time_aligned_comparison.py \
  --pirate-checkpoint-base ./runs/pirate/ckpt \
  --soap-checkpoint-base ./runs/soap/ckpt \
  --output ./runs/time_aligned_comparison.npz
```

```bash
# 檢查 DNS 資料並輸出 vorticity GIF
python3 scripts/analysis/check_dns_data.py \
  --data-path examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy
```

## 備註

- 大型 DNS/LES/checkpoint 資料不應直接納入 Git。
- 以 `README.md` + `INDEX.md` 為主要維護入口。
