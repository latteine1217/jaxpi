# JAXpi Kolmogorov Flow Branch

本分支聚焦於 Kolmogorov Flow 的 PINN 實驗，目標是用可重現流程比較 PIRATE 與 SOAP 設定，並探索稀疏感測資料對跨窗口場重建的效益。

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
# 原作者 upstream SOAP baseline（本地相容版）
uv run python examples/kolmogorov_flow/main.py \
  --config=examples/kolmogorov_flow/configs/upstream_soap.py \
  --workdir=./runs/kf_upstream_soap
```

```bash
# Re=1e6 SOAP 論文復現（paper_repro_soap.py）
uv run python examples/kolmogorov_flow/main.py \
  --config=examples/kolmogorov_flow/configs/paper_repro_soap.py \
  --workdir=./runs/re1e6_n512_soap
```

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config soap \
  --checkpoint_path ./runs/re1e6_n512_soap/ckpt \
  --mode final_step
```

## 目錄結構

```
examples/kolmogorov_flow/
  configs/             訓練設定（paper_repro / sensor / ablation）
  data/
    kolmogorov_dns/    DNS 全場資料（gitignore，需本地存放）
    kolmogorov_sensors/ QR-pivot 感測點位置與 DNS 對應值
    paper_dns_ref/     論文對比用參考資料
  stage_ab/            兩階段 LES → sensor config
  models.py            NavierStokes PINN 模型（含 bug fix 紀錄）
  train.py / evaluate_checkpoint.py

jaxpi/                 基礎模型與工具
slurm/
  train/               訓練提交腳本
  eval/                checkpoint 評估腳本
  postprocess/         後處理與圖像生成腳本
  lib/common.sh        共用 Slurm 模板函式
scripts/analysis/      一次性分析腳本
eval_runs/             已完成評估的結果（關鍵結果納入 Git）
```

## 文檔入口

- `INDEX.md` — 實驗分類索引
- `EXECUTIVE_SUMMARY.md` — 研究摘要
- `EXPERIMENT_RECORD.md` — 實驗外部狀態帳本（主要追蹤來源）
- `EVALUATION_GUIDE.md` — 評估流程說明
- `server_setup_guide.md` — 伺服器環境設定

## 主要 Config 對照

| Config | Re | 感測器 | 說明 |
| :--- | :--- | :--- | :--- |
| `upstream_soap.py` | 1e6 | 無 | 與 upstream pirate branch 對齊的 baseline |
| `paper_repro_soap.py` | 1e6 | 無 | 論文復現主設定（N=512，50 windows） |
| `paper_repro_soap_sensor100_n512_w50.py` | 1e6 | QR-K100 | sensor 約束版，50 windows |
| `paper_repro_soap_window1_ablation.py` | 1e6 | 無 | window-1 only，收斂速度對照 |
| `paper_repro_soap_sensor100_n512_w50_window1_ablation.py` | 1e6 | QR-K100 | window-1 only，sensor A/B 對照 |
| `re10k_soap.py` | 1e4 | 無 | Re=10000 baseline |
| `re10k_soap_sensor100.py` | 1e4 | QR-K100 | Re=10000 + sensor |

## Slurm 腳本

訓練腳本（`slurm/train/`）：
- `train_kolmogorov_paper_repro_soap.sh`
- `train_kolmogorov_re1e6_sensor100_w25_soap.sh`
- `train_kolmogorov_re10k_soap.sh`
- `train_kolmogorov_re10k_sensor100_soap.sh`
- `train_kolmogorov_upstream_soap.sh`

評估腳本（`slurm/eval/`）：
- `eval_kolmogorov_paper_repro_soap.sh`
- `eval_kolmogorov_re1e6_sensor100_w25_soap.sh`
- `eval_kolmogorov_re10k_n256_soap.sh`

後處理腳本（`slurm/postprocess/`）：
- `postprocess_kolmogorov_window1_checkpoint_sweep.sh` — window-1 checkpoint sweep

## 分析腳本（scripts/analysis/）

```bash
# Window-1 checkpoint sweep（no-data vs sensor，direct apply_fn 路徑）
uv run python scripts/analysis/evaluate_window1_checkpoint_sweep.py \
  --window 1 \
  --output-dir eval_runs/my_sweep
```

```bash
# Vorticity field 快照比較
uv run python scripts/analysis/render_vorticity_snapshot_grid.py
```

## 近期更新

- **`models.py` bug fix**：移除 `neural_net()` 中 `z[None, :] + outputs[0]` 的 scalar-wrapper 路徑。
  舊版在 double vmap（time × space）下讓 `apply_fn` 接收 `(T, 1, 3)` 而非 `(T, N, 3)`，
  導致 batch-size-dependent 錯誤評估結果（3265 audit 確認）。修復後 smoke test PASS。
- **window-1 checkpoint sweep**（job 3273）：no-data vs sensor 兩組在 window-1 收斂底線均約 `1e-3`、差距 < 6%，
  說明 sensor 約束對 window-1 本身擬合無顯著效益，效益需從跨窗誤差累積觀察。
  結果存於 `eval_runs/window1_checkpoint_sweep_direct_apply_20260416/`。
- `evaluate_window1_checkpoint_sweep.py` 使用 direct `apply_fn` 路徑（batch-invariant），繞開舊 wrapper bug。
- 新增 QR-pivot 感測點資料：Re=1e3/1e4/1e6，K=100/200，存於 `data/kolmogorov_sensors/`。
- `slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh` 新增後處理腳本。

## 備註

- 大型 DNS/LES/checkpoint 資料不應直接納入 Git（已加入 `.gitignore`）。
- 評估結果原則上不上傳，除非是關鍵 A/B 對照的 scalar 摘要（`.csv` / `.txt`）或代表性圖像。
- `EXPERIMENT_RECORD.md` 是唯一追蹤實驗狀態的外部帳本。
