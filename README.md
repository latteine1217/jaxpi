# JAXpi · Kolmogorov Flow Turbulence Reconstruction

![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![JAX](https://img.shields.io/badge/framework-JAX%20%2F%20Flax-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

以 Physics-Informed Neural Network（PINN）重建二維 Kolmogorov Flow 湍流場的研究分支。
核心問題：在**稀疏感測資料（K ≤ 100，QR-pivot 取點）**約束下，能否用 PINN 跨時間窗口重建全場，
並量化 sensor 約束相對純 PDE 訓練的真實效益。本分支提供可重現的 PIRATE / SOAP 配置對照流程，
涵蓋 Re=10⁴ 與 Re=10⁶ 兩個算例。

本專案 fork 自 [PredictiveIntelligenceLab/JAX-PI](https://github.com/PredictiveIntelligenceLab/JAX-PI)，
在其 PINN 框架上擴充稀疏感測重建、時間窗口策略、SOAP 優化器基線與 LES 資料生成工具鏈。

---

## 研究背景與目標

| 項目 | 內容 |
| :--- | :--- |
| 問題域 | 2D Kolmogorov Flow，`Re = 1e4 / 1e6` |
| 方法 | PINN + PirateNet，SOAP / AdamW → L-BFGS 優化 |
| 約束 | 稀疏感測（QR-pivot，`K ≤ 100`）+ PDE residual |
| 參考 | DNS 全場 vs 稀疏感測重建 |
| 策略 | Time-windowing（逐窗口權重轉移）|

研究目標是分離並量化三個因子對重建精度的影響：**優化器**（PIRATE vs SOAP）、
**時間窗口數**、以及**稀疏感測約束**。

---

## 主要發現

以下結論均有對應 checkpoint 評估證據（詳見 [`EXPERIMENT_RECORD.md`](EXPERIMENT_RECORD.md)），
包含具研究價值的 null result：

- **PIRATE vs SOAP**：在相同物理時間點比較時兩者精度差異通常不大；SOAP 平均誤差較低，
  但總訓練成本更高。時間窗口策略對結果的影響不小於優化器本身。
- **稀疏感測在 window-1 無可量測加速（null result，job `3491`）**：
  以 sweep rank-1 權重（`dw=38.06`）訓練 100k 步的 `sensor + PDE`，
  在所有比較步數上都落在 no-data 純 PDE 基線的 ±2% 內。
  稀疏 QR-pivot K=100 sensor 對 window-1 的 corrected field error **沒有可量測益處**。
- **residual ranking 與 field quality 解耦**：sweep 中以 PDE residual 收斂速度選出的 rank-1 權重，
  在實際場誤差上並未領先 —— 這是「不可僅憑單一 metric 下結論」原則的最強案例。
- **效益需從跨窗誤差累積觀察**：單一 window 的成功不代表整體重建成功；
  sensor 約束的潛在價值在於抑制 `window 2+` 的誤差累積，而非改善 window-1 本身。

> 研究結論以 `EXPERIMENT_RECORD.md` 為唯一外部狀態帳本；本 README 僅摘要已驗證結論，不取代帳本。

---

## 安裝

需求：Python `>=3.10, <3.14`、支援 CUDA 12 的 GPU（CPU 亦可執行但僅適合 smoke test）。

```bash
uv venv
uv pip install -e .
```

主要依賴：`jax[cuda12]`、`flax`、`optax`、`ml_collections`、
[`soap-jax`](https://github.com/haydn-jones/SOAP_JAX)、`optuna`、`wandb`。

---

## 快速開始

### 訓練

```bash
# PIRATE 配置
uv run python examples/kolmogorov_flow/main.py \
  --config=examples/kolmogorov_flow/configs/pirate.py \
  --workdir=./runs/kf_pirate

# upstream SOAP baseline（與原作者 pirate branch 對齊）
uv run python examples/kolmogorov_flow/main.py \
  --config=examples/kolmogorov_flow/configs/upstream_soap.py \
  --workdir=./runs/kf_upstream_soap

# Re=1e6 SOAP 論文復現（N=512，50 windows）
uv run python examples/kolmogorov_flow/main.py \
  --config=examples/kolmogorov_flow/configs/paper_repro_soap.py \
  --workdir=./runs/re1e6_n512_soap
```

### 評估

```bash
python3 examples/kolmogorov_flow/evaluate_checkpoint.py \
  --config soap \
  --checkpoint_path ./runs/re1e6_n512_soap/ckpt \
  --mode final_step
```

> 在伺服器（lab-server）上的訓練一律透過 Slurm 提交至計算節點，
> 不在 head node 直接執行；提交方式見 [Slurm 工作流](#slurm-工作流)。

---

## 專案結構

```
examples/kolmogorov_flow/
  configs/              訓練設定（PIRATE / SOAP / sensor / ablation / eval）
  data/
    kolmogorov_dns/     DNS 全場資料（gitignore，需本地存放）
    kolmogorov_les/     LES 場資料（gitignore）
    kolmogorov_sensors/ QR-pivot 感測點位置與 DNS 對應值（re1000 / re10000 / re100000 / re1000000）
    paper_dns_ref/      論文對比用參考資料
  stage_ab/             兩階段 LES → sensor 訓練腳本（stage1 / stage2，PIRATE / SOAP 變體）
  models.py             NavierStokes PINN 模型（含 batch-invariant bug fix 紀錄）
  train.py              訓練主迴圈（含 dense_les sampler、windowed transfer hooks）
  eval_common.py        評估共用模組（window-local time / domain-aware spectrum / 契約檢查）
  evaluate_checkpoint.py  對齊後的 checkpoint 評估 driver
  split_kolmogorov_les_windows.py  LES 資料切窗工具

jaxpi/                  基礎 PINN 模型、archs、優化器與工具
scripts/
  les/                  獨立 LES 資料生成與驗證（generate / validate）
  analysis/             一次性分析腳本（checkpoint sweep、vorticity 快照等）
  sweep/                Optuna 權重 sweep 工具
slurm/
  train/                訓練提交腳本
  eval/                 checkpoint 評估腳本
  postprocess/          後處理與圖像生成腳本
  sweep/                Optuna sweep 提交腳本
  lib/common.sh         共用 Slurm 模板函式
docs/superpowers/       子專案 spec 與實作 plan
eval_runs/              已完成評估結果（大型輸出 gitignore，僅關鍵 scalar 摘要納入 Git）
tests/                  pytest 測試（含 LES generator no-DNS 測試）
```

---

## 資料集

| 資料 | 用途 | 是否納入 Git |
| :--- | :--- | :--- |
| `kolmogorov_dns/` | DNS 全場（評估參考真值） | 否（大型，本地存放） |
| `kolmogorov_les/` | LES 場（two-stage Stage 1 來源） | 否（大型） |
| `kolmogorov_sensors/` | QR-pivot 感測點，Re=1e3/1e4/1e5/1e6，K=100/200 | 是 |
| `kolmogorov_flow_Re10000_256.npy` | Re=1e4 算例資料 | 是 |

大型 DNS/LES/checkpoint 一律不入 Git（見 [`.gitignore`](.gitignore)）。

---

## 設定檔（configs/）

設定以「物理算例 × 優化器 × 感測 × 時間窗口」組合命名，常用代表如下；
完整清單見 [`examples/kolmogorov_flow/configs/`](examples/kolmogorov_flow/configs/)。

| Config | Re | 感測器 | 說明 |
| :--- | :--- | :--- | :--- |
| `pirate.py` | — | 無 | PIRATE 基線 |
| `upstream_soap.py` | 1e6 | 無 | 與 upstream pirate branch 對齊的 SOAP baseline |
| `paper_repro_soap.py` | 1e6 | 無 | 論文復現主設定（N=512，50 windows） |
| `paper_repro_soap_sensor100_n512_w50.py` | 1e6 | QR-K100 | sensor 約束版，50 windows |
| `paper_repro_soap_window1_ablation.py` | 1e6 | 無 | window-1 only，收斂速度對照 |
| `paper_repro_soap_sensor100_n512_w50_window1_dw38_eval.py` | 1e6 | QR-K100 | window-1 fixed-weight（`dw=38.06`，50k）eval 配置 |
| `paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py` | 1e6 | QR-K100 | 上者延伸至 100k 步（job `3491`，null-result 驗證） |
| `re10k_soap.py` | 1e4 | 無 | Re=10000 baseline |
| `re10k_soap_sensor100.py` | 1e4 | QR-K100 | Re=10000 + sensor |

---

## LES 資料生成（Sub-project A · 已完成）

無 DNS 依賴的獨立 LES 生成器，避免任何 DNS 統計洩漏進訓練訊號（vorticity-streamfunction 形式，
hyperviscosity + Bardina mixed closure，線性摩擦由 turnover time 推估）。

```bash
# 生成 Re=1e6 LES（stand-alone 校準，無 DNS 參考）
uv run python scripts/les/generate_kolmogorov_les.py --no_dns \
  --manual_nu_h <hyperviscosity> --manual_r_fric <friction> \
  --output examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_Re1e6_N512_T5.npy

# 三項硬檢驗（no-decay / bounded / divergence），輸出 KE、enstrophy、spectrum、divergence 圖
uv run python scripts/les/validate_les.py \
  --input  examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_Re1e6_N512_T5.npy \
  --output-dir eval_runs/les_validation

# 切窗供 PINN 逐窗口讀取
uv run python examples/kolmogorov_flow/split_kolmogorov_les_windows.py
```

設計細節見 [`docs/superpowers/specs/2026-05-14-re1e6-les-standalone-generation.md`](docs/superpowers/specs/2026-05-14-re1e6-les-standalone-generation.md)。

---

## 兩階段訓練（Sub-project B · 設計完成，實作進行中）

針對 `3491` null result 設計的訓練 scaffold：**Stage 1** 用 dense LES 做監督回歸（無 PDE），
**Stage 2** 以 sparse sensor + PDE 在 Stage 1 權重上 warm-start，檢驗 LES prior 是否加速稀疏重建。
`train.py` 已具備 `data_constraint=="dense_les"` sampler 與 windowed transfer hooks；
尚待落地 inter-run `transfer_from_ckpt` patch。

- 設計：[`docs/superpowers/specs/2026-05-15-two-stage-pinn-training-design.md`](docs/superpowers/specs/2026-05-15-two-stage-pinn-training-design.md)
- 實作計畫（10-task）：[`docs/superpowers/plans/2026-05-15-two-stage-pinn-training.md`](docs/superpowers/plans/2026-05-15-two-stage-pinn-training.md)

---

## Slurm 工作流

伺服器計算一律經 Slurm 提交至計算節點（`r740`），head node 不跑訓練。

| 類別 | 路徑 | 範例 |
| :--- | :--- | :--- |
| 訓練 | `slurm/train/` | `train_kolmogorov_paper_repro_soap.sh`、`train_kolmogorov_re10k_soap.sh`、`train_kolmogorov_stage1_soap.sh` |
| 評估 | `slurm/eval/` | `eval_kolmogorov_paper_repro_soap.sh`、`eval_kolmogorov_re10k_sensor100_soap.sh` |
| 後處理 | `slurm/postprocess/` | `postprocess_kolmogorov_window1_checkpoint_sweep.sh`、`postprocess_kolmogorov_generate_fields.sh` |
| Sweep | `slurm/sweep/` | `sweep_kf_w1_weights.sh`（Optuna 權重 sweep） |

共用模板函式於 [`slurm/lib/common.sh`](slurm/lib/common.sh)。

---

## 評估紅線

重建研究的指標極易因評估路徑不一致而失真，以下為硬性規範：

- **只用** [`evaluate_checkpoint.py`](examples/kolmogorov_flow/evaluate_checkpoint.py) 或已對齊的新 eval driver；
  舊版 `eval.py` 已封存，避免誤用。
- **window-local time**：time-window checkpoint 評估必須使用窗口內相對時間；
  把 `t_star[si:ei]` 的 absolute time 直接餵入模型會讓 `window 2+` 指標失真。
- **domain-aware spectrum**：spectrum / Fourier 圖必須使用 dataset 實際 domain length；
  `re10k` 與 `re1e6 N512 ds4` DNS 檔 `config.L = 1.0`，不可硬寫波數軸為 `[0, 2π)`。
- **禁止靜默截尾**：`len(t_star) // num_time_windows` 若有餘數，必須由 config 顯式宣告
  `config.eval.expected_time_remainder`，禁止評估腳本默默截掉 trailing time steps。
- **batch-invariant 路徑**：window-1 checkpoint sweep 正式比較須走
  [`scripts/analysis/evaluate_window1_checkpoint_sweep.py`](scripts/analysis/evaluate_window1_checkpoint_sweep.py)
  的 direct `apply_fn` 路徑，不回退到舊 wrapper-based 路徑。

---

## 研究誠信原則

- 不把「能跑」當成成功，不把「單一 window 成功」當成整體實驗成功。
- 失敗與 null result 案例必須保留，不可隱藏。
- 缺乏證據時不推論實驗進度、最佳結果或 checkpoint lineage。
- 所有代碼變更至少提供一項可重現硬證據（`py_compile` / smoke test / checkpoint eval / 物理一致性檢查）。

---

## 文檔導覽

- [`EXPERIMENT_RECORD.md`](EXPERIMENT_RECORD.md) — 實驗外部狀態帳本（唯一追蹤來源）
- [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) — 研究結論快覽
- [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) — 評估流程說明
- [`INDEX.md`](INDEX.md) — 文檔索引
- [`server_setup_guide.md`](server_setup_guide.md) — 伺服器環境設定

---

## 授權與致謝

- 授權：Apache-2.0（見 [`LICENSE`](LICENSE)）。
- 基礎框架：[JAX-PI](https://github.com/PredictiveIntelligenceLab/JAX-PI)，作者 Sifan Wang、Shyam Sankaran、Hanwen Wang 等。
- 優化器：[SOAP-JAX](https://github.com/haydn-jones/SOAP_JAX)。
- 本分支在上述基礎上擴充稀疏感測重建、時間窗口策略、LES 工具鏈與評估對齊機制。
```
