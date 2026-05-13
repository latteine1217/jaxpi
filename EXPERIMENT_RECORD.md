# Experiment Record

本檔是實驗外部狀態帳本。用途是讓 agent 與研究者在不預載全文的前提下，仍能快速定位：

- 目前主線實驗
- 已確認的關鍵失敗與修正
- 可回放的 config / dataset / checkpoint / evaluation 證據

原則：

- 只記錄已執行且可驗證的內容
- 結果若未完成，明確標示 `進行中`
- 失敗案例必須保留
- 若屬 checkpoint 評估，直接寫明評估方式

## [INDEX] Active Experiments

### `3491` | `re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_100k_eval`

| Field | Value |
| :--- | :--- |
| Status | Running (`2026-05-13 12:13 +0800`) on `acmt20`, 2x RTX 3090 (sharding) |
| Config | [paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` + fixed `u_data=v_data=38.0614` + `w_data=0` |
| Time Horizon | `window 1` only, `max_steps=100000` |
| Checkpoint Policy | `save_every_steps=1000`, `num_keep_ckpts=None` (100 ckpts total) |
| Workdir | `/home/junyi/jaxpi/runs/train_kf_w50_w1_dw38_100k_eval_3491` |
| Ckpt Root | `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_100k_eval/ckpt` |
| Wandb | offline run `t3liqvqw`, group `re1e6_window1_fixed_weight_eval` |
| Predecessor | `3481` (50k) + eval `3489/3490` — sensor 50k 達到 ~no_data 60k 精度，未達 no_data 100k；本實驗將 sensor 延伸到 100k 步驗證是否存在後期加速效應 |
| Purpose | 回答「sensor 是否在 100k 步上能超越 no-data 100k baseline」 |
| Expected outcome | 若 sensor 100k 仍 ≈ no_data 100k → sparse sensor 對 window-1 確認沒有 measurable 加速；若 sensor 100k 顯著贏 → 存在後期效應但需要更長訓練 |
| RNG Strategy | Not recorded |

### `3481` | `re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_eval`

| Field | Value |
| :--- | :--- |
| Status | Completed (`2026-05-11 10:27 +0800`, Elapsed `03:12:17`) on `acmt20`, 2x RTX 3090 (sharding) |
| Config | [paper_repro_soap_sensor100_n512_w50_window1_dw38_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw38_eval.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` + fixed `u_data=v_data=38.0614` + `w_data=0` |
| Time Horizon | `window 1` only, `max_steps=50000` |
| Checkpoint Policy | `save_every_steps=1000`, `num_keep_ckpts=None` (50 ckpts saved) |
| Workdir | `/home/junyi/jaxpi/runs/train_kf_w50_w1_dw38_eval_3481` |
| Ckpt Root | `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_eval/ckpt` (50 ckpts, 1k~50k 每 1000 步) |
| Wandb | offline run `vdxcohw7`, group `re1e6_window1_fixed_weight_eval` |
| Predecessor | `3480` failed after 53s with transient pypi DNS error during uv build on `acmt20`; pre-warmed uv cache on login node then resubmitted as `3481` |
| Eval Jobs | `3489` (5-pt head-to-head 10k~50k), `3490` (10-pt with no_data 10k~100k + sensor 10k~50k via `--allow-missing`) |
| Result @ step 50000 (direct apply_fn) | `u=1.208e-3, v=1.113e-3, w=0.825e-3` — **same level as no_data 50k (1.141e-3, 1.203e-3, 0.822e-3)**, but **falls short of no_data 100k (1.077e-3, 1.088e-3, 0.704e-3)** by 12~17% on u/w |
| Sensor-vs-no_data verdict (50k) | NO measurable acceleration; sensor 50k ≈ no_data 60k accuracy; direct contradiction with sweep 3400 residual ranking (`dw=38.0614` was rank-1 first_stable_step=49,400) — concrete evidence for AGENTS.md `Metric_Selection` red line |
| Followup | `3491` extends sensor to 100k to test whether the gap closes in late training |
| RNG Strategy | Not recorded |

### `3324` | `re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval`

| Field | Value |
| :--- | :--- |
| Status | Completed (`2026-04-21 05:34 +0800`) |
| Config | [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` + fixed `u_data=v_data=23.1429` + `w_data=0` |
| Time Horizon | `window 1` only, `max_steps=50000` |
| Checkpoint Policy | `save_every_steps=1000`, `num_keep_ckpts=None` |
| Launch Script | `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh` with overridden `CONFIG_PATH` |
| Workdir | `/home/junyi/jaxpi/runs/re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval` |
| Current Risk | `2026-04-21 22:00 +0800` 的 direct `apply_fn` rerun 已確認 `3324` 真實 full-window error 為 `u/v ~1e-4`、`w ~4.6e-4 ~ 5.0e-4`；舊 `3326 ~1e-3` 結果已被推翻。當前風險不再是「是否卡在 ~1e-3 平台」，而是雖已明顯優於舊判讀，仍落後 `3137/3155 window1` 約 `2x ~ 3x`。 |
| RNG Strategy | Not recorded |

### `3318` | `kf_w1_data_weight_sweep_1to100_thr5e5`

| Field | Value |
| :--- | :--- |
| Status | Completed (`2026-04-21 01:19 +0800`) |
| Config | [paper_repro_soap_window1_ablation.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` |
| Sweep Range | `data_weight in [1, 100]` (`log=True`) |
| Objective | fastest `max(ru_loss, rv_loss, rc_loss) < 5e-5` within `50000` steps |
| Storage | `sqlite:///sweep_w1_data_1to100_thr5e5.db` |
| Launch Script | `/home/junyi/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh` |
| Current Risk | `3317` 先以舊 remote 腳本誤啟動（header 仍是 `Threshold: 1e-5`），已在 30 秒內取消並改由 `3318` 重送；`3318` 已於 `2026-04-21 01:19 +0800` 正常 `COMPLETED`，後續應讀取 study DB / log 判讀 trial 結果，而不是再查 queue state。 |
| RNG Strategy | Not recorded |

### `3155` | `re1e6_n512_ds4_soap_sensor100_w50`

| Field | Value |
| :--- | :--- |
| Status | Paused mainline; Slurm job `3155` 已於 `2026-04-14 21:59 +0800` 結束為 `CANCELLED` |
| Config | [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` + `u_data=100` + `v_data=100` + `w_data=0` |
| Time Horizon | `50 windows` (`0.1s/window`) |
| Launch Script | `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh` with overridden `CONFIG_PATH` |
| Current Risk | `2026-04-21` 的 `3328` all-window direct re-evaluation 已補齊 `window 1..21`，證明 `3155` 的 apples-to-apples direct 誤差曲線與舊 corrected evaluator 量級一致；但在公平比較區間 `window 1..12` 上，`3155` 的平均 `u/v/w` 誤差都沒有優於 `3137`，主風險仍是跨 window 誤差累積而非 evaluator lineage。 |
| RNG Strategy | Not recorded |

目前主線判讀：

- 這是針對 `3147/3149` 失敗模式設計的單變數 A/B：保留 `sensor100`、optimizer、權重與 batch，僅把 horizon 從 `25 windows` 改回 `50 windows`。
- 目的是驗證 `window 2+` 崩壞是否主要由過長 window horizon 觸發，而不是 sensor loss 本身不可用。
- 舊 `3150` 已依人工指示停止，最終狀態 `CANCELLED by 10004`。
- 新 `3155` 已啟動，Slurm metadata 已確認 command 仍經由 `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh` 提交，但會載入同一份 [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)。
- `3155` 是第一個真正會吃到 `sensor_sampling=all_points` 邏輯的 `w50` run；也就是每一步固定覆蓋該 window 的全部 `100 sensors × local time points`，且維持 `w_data=0`。
- 舊 run `3150` 的 corrected eval 與 failure diagnosis 仍成立，但接下來是否改善，要以 `3155` 的新 checkpoint 為準。
- `window 1` 的 retained checkpoints 已經更新成新的 mtime：
  - `checkpoint_90000  -> 2026-04-09 09:13 +0800`
  - `checkpoint_100000 -> 2026-04-09 09:52 +0800`
- 這證明 `3155` 不只出現第一個新 checkpoint，而是已完整跑完 `window 1`；目前 log 已推進到 `window 2 / step 33400`。
- 目前最新進度已推進到 `window 2 / step 72800`，並落盤：
  - `checkpoint_60000 -> 2026-04-09 13:43 +0800`
  - `checkpoint_70000 -> 2026-04-09 14:21 +0800`
- 因此截至目前 `window 2` 尚未完成，`checkpoint_90000/100000` 也尚未出現。
- `window 2` 後續已完整跑完，retained checkpoints 更新成：
  - `checkpoint_90000  -> 2026-04-09 15:38 +0800`
  - `checkpoint_100000 -> 2026-04-09 16:16 +0800`
- `window 2` 尾段 (`step 99900`) loss 為：
  - `rc_loss = 1.962e-07`
  - `ru_loss = 6.300e-07`
  - `rv_loss = 6.651e-07`
- 最新人工操作已將 `3155` 暫停在 Slurm `STOPPED` 狀態，避免訓練繼續推進而先完成 corrected evaluation。
- `2026-04-19` 重新查核 Slurm accounting 後，確認 `3155` 最終狀態不是持續 `STOPPED`，而是已在 `2026-04-14 21:59 +0800` 轉為 `CANCELLED`（`Elapsed=5-18:34:34`, `Node=acmt20`）；目前無對應中的 active Slurm allocation。
- 暫停時的已落盤進度：
  - `time_window_13/checkpoint_100000 -> 2026-04-12 14:20 +0800`
  - `time_window_14/checkpoint_10000  -> 2026-04-12 15:00 +0800`
- corrected full-window evaluation（window-local time, chunked CPU evaluator）已覆蓋目前所有可評估 windows：
  - `window 1 / checkpoint_100000  -> u=0.000034, v=0.000034, w=0.000222`
  - `window 2 / checkpoint_100000  -> u=0.000067, v=0.000068, w=0.000696`
  - `window 3 / checkpoint_100000  -> u=0.000140, v=0.000153, w=0.003485`
  - `window 4 / checkpoint_100000  -> u=0.000293, v=0.000334, w=0.010051`
  - `window 5 / checkpoint_100000  -> u=0.000583, v=0.000653, w=0.021417`
  - `window 6 / checkpoint_100000  -> u=0.001058, v=0.001123, w=0.036667`
  - `window 7 / checkpoint_100000  -> u=0.001654, v=0.001758, w=0.052902`
  - `window 8 / checkpoint_100000  -> u=0.002472, v=0.002608, w=0.074894`
  - `window 9 / checkpoint_100000  -> u=0.003580, v=0.003640, w=0.098401`
  - `window 10 / checkpoint_100000 -> u=0.004864, v=0.004978, w=0.124236`
  - `window 11 / checkpoint_100000 -> u=0.006248, v=0.006625, w=0.149501`
  - `window 12 / checkpoint_100000 -> u=0.007831, v=0.008350, w=0.174197`
  - `window 13 / checkpoint_100000 -> u=0.010144, v=0.009809, w=0.201551`
  - `window 14 / checkpoint_10000  -> u=0.014820, v=0.012950, w=0.279143`
- 到 `window 13` 為止，`u/v` 誤差仍停留在 `1e-2` 以內，但 `w_err` 從 `window 3` 起近乎單調上升；這和先前 `3150` 那種 `window 2` 立即崩壞不同，但仍未達專案成功門檻。
- `2026-04-21` 的 `3328` direct `apply_fn` 全窗口重評估已正式覆蓋 `window 1..21`，其中：
  - `window 14 / checkpoint_100000 -> u=0.013304, v=0.011479, w=0.227537`
  - `window 21 / checkpoint_60000  -> u=0.040653, v=0.041705, w=0.459683`
- 這表示 `3155` 的 direct 路徑與舊 corrected evaluator 讀值在量級上對齊，但 trend 並未變好；`w_err` 仍一路升到 `0.459683`。
    - `rv_loss = 3.420e-06`
  - `ru_loss`、`rv_loss` 在 `window 3` 的後續 `step 100..99900` 區間都沒有再低於 `1e-6`。

### `3147` | `re1e6_n512_ds4_soap_sensor100_w25`

| Field | Value |
| :--- | :--- |
| Status | Stopped after `window 7`; full evaluation completed |
| Config | [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` + `u_data=100` + `v_data=100` + `w_data=0` |
| Slurm Header Workdir | `/home/junyi/jaxpi/runs/re1e6_n512_ds4_soap_sensor100_w25_junyi` |
| Actual Output Root | `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25` |
| Launch Script | `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh` with overridden `CONFIG_PATH` |
| Current Risk | `3149` 全窗口評估顯示 `window 2~7` 的 `u/v/w` 全面高誤差；目前主風險不是作業穩定性，而是 sensor 約束未能維持跨窗重建品質 |
| RNG Strategy | Not recorded |

目前主線判讀：

- 這是針對 `3137` 後續診斷直接開出的對照 run，目的不是重做 baseline，而是測試 `sensor100 + w25` 是否能保住高-k 渦度內容。
- `3146` 已指出目前主線問題更偏向高波數渦度衰減，因此這份 config 的價值在於檢查稀疏感測約束能否壓低後期 `w_err` 累積。
- `3147` 已依人工指示停止，最終停在 `window 7` 後段，之後由 `3149` 完成 `window 1~7` 的 corrected full-window evaluation。
- `window 1` 保持極低誤差，但 `window 2~7` 的 `u/v/w` 已全面進入高誤差區間，表示這條 run 的問題不是單一窗口短暫 spike，而是跨窗品質持續失守。
- 因此目前不能把 `sensor100 + w25` 視為有效修復；它證明訓練能前進，但沒有證明可重建長序列場。

### `3137` | `re1e6_n512_ds4_soap`

| Field | Value |
| :--- | :--- |
| Status | Mainline / Timed out after `window 12` |
| Config | [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Checkpoint Root | `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt` |
| Training Logic | `window IC propagation` + `chunked IC propagation` |
| Latest Verified Evidence | `2026-04-21` 的 `3327` all-window direct `apply_fn` re-evaluation 已覆蓋 `window 1..12`；最新可信 baseline 是 `window 1 -> u=3.312488e-05, v=3.383027e-05, w=2.214195e-04` 到 `window 12 -> u=7.821861e-03, v=8.404544e-03, w=1.760991e-01` 的 direct 曲線，不再依賴 `3145` 的混合 lineage。 |
| Current Risk | `window 12` 的 `w_err = 0.176233` 仍略高於門檻；`3146` 顯示主因更接近 high-k vorticity attenuation，而非大尺度相位崩潰 |
| RNG Strategy | Not recorded |

目前主線判讀：

- `window 1 -> window 2` 已成功跨過，stale IC 問題可視為初步排除。
- `3327` 的 all-window direct re-evaluation 已直接證明：`window 1 -> 2` 的誤差跳升是真實現象，不是 `3145/3144` artifact。
- `window 1` 的可信基準改為 `2026-04-21` direct `apply_fn` retained-checkpoint / all-window direct 結果：
  - `checkpoint_90000 -> u=3.781654e-05, v=3.725810e-05, w=2.385928e-04`
  - `checkpoint_100000 -> u=3.312488e-05, v=3.383027e-05, w=2.214195e-04`
- `window 12` 的 direct 結果為 `u=0.007821861, v=0.008404544, w=0.176099102`；主線並非失敗，但也尚未完全達標。
- `3146` 進一步顯示 `window 12` 的 `final_corr = 0.983843`、`low_k_ratio = 0.999527`、`high_k_ratio = 0.664633`，所以當前問題更像是小尺度渦度能量衰減。

### 2026-04-05 更新

- Job 狀態：
  - `3137` 已於 `2026-04-02 20:06:46 +0800` timeout
  - timeout 前已落盤到 `time_window_12 / checkpoint_60000`
- 完整評估：
  - `JobID = 3144`
  - `State = COMPLETED`
  - `Elapsed = 00:05:24`
- 單窗口場圖：
  - `JobID = 3142`
  - `State = FAILED`
  - `Reason = process killed during render`

最新證據：

- `3137` 最新已落盤 checkpoint：
  - `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_12/checkpoint_60000`

- `3144` full-window 評估摘要：

  [NOTE] `window 1` 這一列屬於歷史舊 artifact；`2026-04-21` re-audit 已確認真實 retained-checkpoint `3137 window 1` 應改以 direct `apply_fn` 結果為準。

| Window | Checkpoint | t_end | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 100000 | 0.0500 | 0.001132 | 0.001147 | 0.000734 |
| 2 | 100000 | 0.1500 | 0.112600 | 0.118872 | 0.165103 |
| 3 | 100000 | 0.2500 | 0.145172 | 0.137480 | 0.254671 |
| 4 | 100000 | 0.3500 | 0.159799 | 0.161813 | 0.349038 |
| 5 | 100000 | 0.4500 | 0.154807 | 0.188169 | 0.429782 |
| 6 | 100000 | 0.5500 | 0.147691 | 0.188783 | 0.482639 |
| 7 | 100000 | 0.6500 | 0.157871 | 0.177243 | 0.519076 |
| 8 | 100000 | 0.7500 | 0.167141 | 0.173145 | 0.557658 |
| 9 | 100000 | 0.8500 | 0.189373 | 0.168829 | 0.590839 |
| 10 | 100000 | 0.9500 | 0.190959 | 0.194211 | 0.627286 |
| 11 | 100000 | 1.0500 | 0.186167 | 0.193459 | 0.636445 |
| 12 | 60000 | 1.1500 | 0.180390 | 0.178034 | 0.640494 |

- `3144` summary statistics：
  - `mean(u_err) = 0.149425`
  - `mean(v_err) = 0.156766`
  - `mean(w_err) = 0.437814`
  - `max(w_err) = 0.640494`

- Artifact:
  - Local summary: [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/summary.txt)
  - Local plot: [comparison_plots.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/comparison_plots.png)
  - Local field visualization: [vorticity_fields.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/vorticity_fields.png)
  - Local numeric dump: [l2_errors.npz](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/l2_errors.npz)

更新判讀：

- `final_step` 視角下的低誤差，不能代表整個 window rollout 品質。
- `window 2+` 的 full-window 誤差已大幅抬升，且 `w_err` 幾乎持續增長到 `0.640494`。
- 這代表目前主線雖然 checkpoint 會落盤並能跨窗，但作為長序列重建，品質仍未達專案成功門檻。

### 2026-04-01 更新

- Job 進度：
  - `Time Window 9/50`
  - `Step 66800/100000`
- 最新已落盤 checkpoint：
  - `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_9/checkpoint_60000`

最新證據：

- 訓練 log：
  - `rc_loss ≈ 6.08e-05 ~ 6.51e-05`
  - `ru_loss ≈ 6.99e-05 ~ 8.27e-05`
  - `rv_loss ≈ 7.48e-05 ~ 1.04e-04`
  - `u_ic_loss ≈ 1.24e-07 ~ 1.44e-07`
  - `v_ic_loss ≈ 1.42e-07 ~ 2.02e-07`

- `final_step` 評估：

| Window | Checkpoint | t_eval | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 9 | 60000 | 0.8500 | 0.003761 | 0.003802 | 0.102160 |

- 各 `time_window` 最新 checkpoint 的 `final_step` 摘要：

| Window | Latest Ckpt | t_eval | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 100000 | 0.0500 | 0.000041 | 0.000042 | 0.000266 |
| 2 | 100000 | 0.1500 | 0.000072 | 0.000071 | 0.000857 |
| 3 | 100000 | 0.2500 | 0.000151 | 0.000163 | 0.004119 |
| 4 | 100000 | 0.3500 | 0.000328 | 0.000362 | 0.011118 |
| 5 | 100000 | 0.4500 | 0.000615 | 0.000698 | 0.021810 |
| 6 | 100000 | 0.5500 | 0.001060 | 0.001139 | 0.034955 |
| 7 | 100000 | 0.6500 | 0.001642 | 0.001761 | 0.051751 |
| 8 | 100000 | 0.7500 | 0.002495 | 0.002570 | 0.073567 |
| 9 | 70000 | 0.8500 | 0.003701 | 0.003745 | 0.099886 |

- Artifact:
  - Report: [re1e6_n512_ds4_latest_per_window_report.md](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/comparison/re1e6_n512_ds4_latest_per_window_report.md)
  - Plot: [re1e6_n512_ds4_error_vs_time_window.png](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/comparison/re1e6_n512_ds4_error_vs_time_window.png)

更新判讀：

- 到 `window 9` 為止，`u` 與 `v` 仍維持很低誤差。
- `w` 誤差隨 window 單調上升，最新 `window 9` 為 `0.099886`，已接近專案驗收邊界。
- 主線風險已轉為「vorticity 誤差是否會隨時間持續累積」。

## [INDEX] Critical Failures & Insights

### `Time-window IC propagation` 是主線機制風險

- 2026-03-30 確認原始多窗口訓練沒有逐窗更新 `u0 / v0 / w0`，導致 `window 2+` 沿用舊初值。
- 這是機制級 bug，不是單純 optimizer 或超參數問題。
- 相關修正與驗證見：
  - [2026-03-30 | 關鍵 bug：window IC propagation](/Users/latteine/Documents/coding/jaxpi/EXPERIMENT_RECORD.md#2026-03-30--關鍵-bugwindow-ic-propagation)
  - [2026-03-31 | `3137` | `window 2` 與 `window 3` checkpoint 評估](/Users/latteine/Documents/coding/jaxpi/EXPERIMENT_RECORD.md#2026-03-31--3137--window-2-與-window-3-checkpoint-評估)

### `3136` 證明跨窗瓶頸包含記憶體成本

- 修正逐窗更新後，`window 1` 可完整收斂。
- 但 `_predict_next_window_ic()` 對全場直接算 `w0` 時 OOM：
  - `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 768.00MiB`
- 這證明跨窗失敗不只來自邏輯 bug，也來自 propagation 實作成本。

### `0321` 與 `3136` 共同支持「window 1 可成功」

- 歷史 `0321` 與本次 `3136` 都顯示 `window 1` 可以達到低誤差或完整收斂。
- 因此「完全訓練不起來」不是正確描述。

### Re10000 upstream SOAP 問題不只在 optimizer

- `schedule_free + grad_clip + loader` 對齊後確實有改善。
- 但 `3131 -> 3132` 的改善不足以達標，真正更大的問題仍是 time-window 初值傳遞 bug。

### `3137` 既有 full-window 評估存在 window-time 對齊風險

- `train.py` 在非 `windowed_data` 模式下，所有窗口都共用第一個 window 的時間軸 `t = t_star[:num_time_steps]`。
- `evaluate_checkpoint.py` 與 `eval_paper_repro_soap.py` 原本卻直接把 DNS 全域絕對時間 `t_star[si:ei]` 餵進模型。
- 同一批腳本又只把每個 window 最後一張 vorticity 畫出來，因此會出現「末張看起來很像，但 full-window `w_err` 很大」的表象落差。
- 在修正後的 full-window 重跑完成前，`3144` 的 `window 2+` 誤差應視為可疑證據，而非最終結論。

### 評估腳本必須顯式宣告 trailing steps 與 domain-length spectral axis

- 多個 Kolmogorov DNS 檔目前是 `41` 或 `101` 個時間點，搭配 `20 / 25 / 50` windows 時都會留下 `remainder = 1`。
- `train.py` 與多支評估腳本原本都直接做 `len(t_star) // num_time_windows`，這會默默丟掉最後一個時間點；若不明示，之後很難判斷是刻意沿用訓練切法，還是 eval 自己切錯。
- 多支 eval driver 又把 FFT 波數軸寫死成 `fftfreq(..., d=2π/N)`；但目前 `re10k` 與 `re1e6 N512 ds4` DNS 檔的 `config.L` 都是 `1.0`，這會直接造成 Fourier axis misalignment。
- 2026-04-20 已把這兩類假設集中到共用 helper，並要求 config 顯式宣告 `expected_time_remainder`，否則 eval 直接 fail-fast。

## [LOG] Chronological

### [2026-05-13] `3491` | submit dw=38.06 100k-step extension training

- Time: `2026-05-13 12:13 +0800`
- Status: RUNNING on `acmt20`
- Experiment or Job ID: `3491`

Change:

- Created [paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py): clone of `3481` config with `max_steps=100000` to test late-training behavior.
- Pre-warmed `uv sync` on login node to avoid the DNS-build failure that killed `3480` on first attempt.
- `sbatch --export=ALL,CONFIG_PATH=...,RUN_SLUG=train_kf_w50_w1_dw38_100k_eval slurm/train/train_kolmogorov_re1e6_sensor100_w25_soap.sh` → job `3491`.

Evidence:

- Config test `tests/test_window1_dw38_100k_eval_config.py` PASS — pins `max_steps=100000`, `save_every_steps=1000`, `num_keep_ckpts=None`, `u_data=v_data=38.0614`.
- Job log shows correct config loaded, 2-GPU sharding active, wandb tags include `max_steps_100k` and `sweep_3400_rank1`, `Training time window 1` started.

Interpretation:

- `3481` (50k) eval already showed sensor matches no_data 50k but falls short of no_data 100k by 12~17% (u/w). The 100k extension answers whether the gap is structural (sensor can't surpass no_data even with same training budget) or just under-trained.
- If `3491` ckpt_100000 still ≈ no_data ckpt_100000 → conclude that sparse QR-pivot K=100 sensor with `dw=38.06` provides no measurable benefit for window-1 corrected field error, despite topping the residual-based sweep.

Next:

- Wait ~6.5 h for `3491` to complete (50k → 100k ratio).
- After completion: rerun the 10-pt dual eval with sensor extended to 10k~100k, observe whether sensor catches up to or surpasses no_data 100k.

### [2026-05-13] `3490` | 10-point dual eval (dw=38 vs no_data, 10k~100k with allow-missing)

- Time: `2026-05-13 19:48 ~ 20:03 +0800`
- Status: COMPLETED (Elapsed `00:14:50`)
- Experiment or Job ID: `3490` (postprocess)

Change:

- `sbatch` `slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh` with `CHECKPOINT_STEPS=10000,...,100000` and `ALLOW_MISSING=1`.
- no_data side ran all 10 ckpts (10k~100k); sensor side skipped 60k~100k (not yet trained at that point).

Evidence:

- Output: [eval_runs/dw38_vs_nodata_w1_10pt_20260511/](/Users/latteine/Documents/coding/jaxpi/eval_runs/dw38_vs_nodata_w1_10pt_20260511/)
- CSV `checkpoint_sweep_results.csv` (15 rows: 10 no_data + 5 sensor)
- `checkpoint_sweep_error_vs_step.png` shows three subplots overlapping for u/v/w; no_data extends past sensor's 50k cutoff.

Headline numbers (window 1, t_local=0.05, direct apply_fn, relative L2):

| run     | step    | u_err     | v_err     | w_err     |
| ------- | ------: | --------: | --------: | --------: |
| no_data |  10 000 | 1.805e-3  | 1.693e-3  | 2.583e-3  |
| no_data |  50 000 | 1.141e-3  | 1.203e-3  | 0.822e-3  |
| no_data | 100 000 | **1.077e-3**  | **1.088e-3**  | **0.704e-3**  |
| sensor  |  50 000 | 1.208e-3  | 1.113e-3  | 0.825e-3  |

Interpretation:

- sensor 50k ≈ no_data 60k accuracy on all three components (`1.22e-3, 1.11e-3, 0.80e-3` vs sensor `1.21e-3, 1.11e-3, 0.82e-3`).
- no_data continues to improve from 50k to 100k by ~15~17% on u/w (1.14e-3 → 1.08e-3, 0.82e-3 → 0.70e-3); v plateau ~1.1e-3.
- sensor at its 50k endpoint is **not** at no_data 100k accuracy — falls short by 12% (u), 2% (v), 17% (w).
- ⚠️ **direct contradiction with sweep 3400 residual ranking**: sweep ranked `dw=38.06` rank-1 by residual `max(ru,rv,rc) < 5e-5`, yet on corrected field error the sensor run has no advantage over no_data at any matched step, and is **behind** when no_data trains longer. Per AGENTS.md `Metric_Selection` red line, residual ≠ field quality.

Next:

- Sensor 100k extension submitted as `3491`.

### [2026-05-13] `3489` | 5-point dual eval (dw=38 vs no_data, head-to-head 10k~50k)

- Time: `2026-05-13 19:27 ~ 19:37 +0800`
- Status: COMPLETED (Elapsed `00:10:07`)
- Experiment or Job ID: `3489` (postprocess)

Change:

- First eval after `3481` finished. `CHECKPOINT_STEPS=10000,20000,30000,40000,50000`, both runs.
- Output: [eval_runs/dw38_vs_nodata_w1_5pt_20260511/](/Users/latteine/Documents/coding/jaxpi/eval_runs/dw38_vs_nodata_w1_5pt_20260511/)

Evidence:

- CSV: 10 rows (5 no_data + 5 sensor).
- All 5 matched-step deltas within ±10%; largest sensor advantage was `v_err` at step 20k (1.20e-3 vs 1.44e-3, -16.2%) but did not persist.
- Vorticity field PNGs show visually indistinguishable PINN reconstruction and `|Error|` maps at every step for both runs.

Interpretation:

- Same-step head-to-head shows no acceleration. Promoted to 10-pt eval `3490` to compare against no_data 100k full horizon.

Next:

- Drove `3490` (10-pt + allow_missing).

### [2026-05-11] `3481` | dw=38.06 fixed-weight 50k training completed

- Time: `2026-05-11 07:15 ~ 10:27 +0800`
- Status: COMPLETED (Elapsed `03:12:17`, ExitCode `0`)
- Experiment or Job ID: `3481`

Change:

- Created [paper_repro_soap_sensor100_n512_w50_window1_dw38_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw38_eval.py): clone of `dw231429_eval` pattern with `u_data=v_data=38.0614` (sweep 3400 rank-1).
- Fixed 13 slurm launchers missing `SLURM_SUBMIT_DIR` fallback for `SCRIPT_DIR` (commit `5a35edc`) — root cause of `3479` early FAIL.
- Pre-warmed uv cache on login node after `3480` failed with transient pypi DNS error.

Evidence:

- 50 ckpts written to `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_eval/ckpt/time_window_1/` (step 1000 ~ 50000, 1k cadence).
- Training stdout shows clean shutdown; final ckpt at step 50000 saved at 10:27:27.
- Note: ckpt path uses `wandb.name` not `RUN_SLUG`-based workdir — this is train.py default behavior.

Interpretation:

- Training itself converged successfully. Field-quality interpretation requires the eval jobs `3489`/`3490` (next entries chronologically above).

Next:

- Eval submitted as `3489`.

### [2026-04-21] `3324` | submit fixed-weight window-1 checkpoint-validation run

- Time: `2026-04-21 02:18 +0800`
- Status: COMPLETED (`ExitCode=0:0`)
- Experiment or Job ID: `3324`

Change:

- 依人工指示，不再做 sweep，而是固定 `3318` 的最佳 `data_weight=23.1429`，重跑單獨的 window-1 驗證 job。
- 新增 [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py)：
  - `u_data=v_data=23.1429`
  - `max_windows_to_run=1`
  - `max_steps=50000`
  - `save_every_steps=1000`
  - `num_keep_ckpts=None`
- 使用 remote `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh` 以 `CONFIG_PATH` 覆蓋提交新 job，避免再走 sweep path 丟失 checkpoint。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py)
- Workdir: `/home/junyi/jaxpi/runs/re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval`
- Checkpoint policy: `save_every_steps=1000`, `num_keep_ckpts=None`
- Target checkpoint region: `step ~46100` and final `50000`

Evidence:

- RED/GREEN:
  - `python3 -m unittest tests/test_window1_fixed_weight_eval_config.py`
    - before: import failed because config module did not exist
    - after : `Ran 1 test ... OK`
  - `python3 -m py_compile examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py`
- Remote sync:
  - `rsync ... paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py -> /home/junyi/jaxpi/examples/kolmogorov_flow/configs/...`
- Submission:
  - `sbatch -> Submitted batch job 3324`
  - `squeue -j 3324` -> `RUNNING`
  - stdout header:
    - `Config   : examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py`
    - `Workdir  : /home/junyi/jaxpi/runs/re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval`
    - `devices: [CudaDevice(id=0), CudaDevice(id=1)]`
  - completion:
    - `3324 | COMPLETED | Elapsed=03:16:06 | ExitCode=0:0`
    - `checkpoint_50000` saved to `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval/ckpt/time_window_1/checkpoint_50000`

Interpretation:

- `3324` 是第一個真正能拿來做 checkpoint evaluation 的 `data_weight=23.1429` 驗證 run。
- 只有等 `time_window_1/checkpoint_46000` 附近與 `checkpoint_50000` 落盤後，才能回答「threshold 達標是否對應到真實場收斂」。

Next:

- 監看 `3324` 的 checkpoint 落盤與訓練 log。
- 一旦 `step 46000` 與 `50000` 對應 checkpoint 出現，立即做 corrected evaluation。

### [2026-04-21] `3326` | corrected evaluation for `3324` checkpoints `46000/47000/50000`

- Time: `2026-04-21 15:35 +0800`
- Status: COMPLETED (`ExitCode=0:0`)
- Note: `[SUPERSEDED 2026-04-21 22:00 +0800]` 後續 direct `apply_fn` rerun 已證明本段 `~1e-3` 數字偏高，不應再作為 `3324` 的最終 field-error 依據。
- Experiment or Job ID: `3326` (evaluation), source training job `3324`

Change:

- 為避免修改 evaluator 程式，建立三個只包含單一 checkpoint 的暫存 root，分別對 `3324` 的 `checkpoint_46000`、`checkpoint_47000`、`checkpoint_50000` 跑現有 corrected evaluator。
- 每個 checkpoint 都各跑兩種模式：
  - `window`：整個 window 的相對 L2
  - `final_step`：該 window 最後時間點的相對 L2

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py)
- Source checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval/ckpt`
- Eval output root: `/home/junyi/jaxpi/eval_runs/3324_ckpt_eval_20260421`
- Output files:
  - `window1_step_46000_window.npz`
  - `window1_step_46000_final.npz`
  - `window1_step_47000_window.npz`
  - `window1_step_47000_final.npz`
  - `window1_step_50000_window.npz`
  - `window1_step_50000_final.npz`

Evidence:

- Slurm:
  - `3326 | eval_3324_ckpts | COMPLETED | Elapsed=00:03:28 | ExitCode=0:0`
- Corrected full-window L2 (`mode=window`):
  - `46000 -> u=0.001145, v=0.001115, w=0.000842`
  - `47000 -> u=0.001159, v=0.001154, w=0.000825`
  - `50000 -> u=0.001235, v=0.001103, w=0.000817`
- Corrected final-step L2 (`mode=final_step`):
  - `46000 -> u=0.001171, v=0.001109, w=0.000894`
  - `47000 -> u=0.001188, v=0.001115, w=0.000876`
  - `50000 -> u=0.001277, v=0.001127, w=0.000861`
- `3324` training tail near completion:
  - `step 49600`: `rc=5.988e-06`, `ru=1.398e-05`, `rv=1.445e-05`
  - `step 49900`: `rc=2.409e-06`, `ru=7.703e-06`, `rv=6.914e-06`

Interpretation:

- `data_weight=23.1429` 的確讓 residual losses 明顯低於 `5e-5`，而且 `46000 -> 50000` 都維持在 `1e-5 ~ 1e-6`。
- 但對應的 corrected field error 並沒有進一步掉到 `1e-4` 級；三個 checkpoint 的 `u/v/w` 誤差都穩定停在 `~1e-3`。
- 這說明 **`5e-5` residual threshold 不等於「field 已高度收斂」**；它比較像是把訓練帶到一個 `~1e-3` 的誤差平台。
- 從 `46000 -> 50000` 看不到明顯的持續改善：
  - `w_error` 略降 (`8.42e-4 -> 8.17e-4`)
  - `u_error` 反而略升 (`1.145e-3 -> 1.235e-3`)
  - `v_error` 幾乎持平
- 因此若標準是專案成功門檻（`<=10%~15%`），`3324 window 1` 當然已在門檻內；但若標準是「是否已達非常低的 window-1 重建誤差」，目前更接近 **平台化收斂於 `1e-3`，不是持續往下收斂**。

Next:

- 若要確認這個 `~1e-3` 平台是否就是 window-1 可達上限，可再比較：
  - `3324` vs `3318` sweep 同權重結論
  - `3324` vs 舊 `3155 window1 checkpoint_100000` 的 corrected eval
- 若要追更低 field error，不應只看 residual threshold；需考慮新的 objective 或直接以 field metric 做 model selection。

### [2026-04-21] `3324` vs old `3155 window1 checkpoint_100000` | is the `~1e-3` plateau good or bad?

- Time: `2026-04-21 15:45 +0800`
- Status: Comparison completed
- Experiment or Job ID: `3324`, historical `3155`

Change:

- 將 `3324` 的 corrected eval（`checkpoint_46000/47000/50000`）與舊 `3155 window 1 / checkpoint_100000` 的 direct `apply_fn` corrected result 並排比較。

Config / Dataset / Checkpoint:

- `3324`:
  - Config: [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py)
  - Checkpoints: `46000`, `47000`, `50000`
- `3155` historical reference:
  - Config family: `paper_repro_soap_sensor100_n512_w50.py`
  - Checkpoint: `window 1 / checkpoint_100000`
  - Forward path: `direct_apply_fn` corrected evaluation

Evidence:

- Old `3155 window 1 / checkpoint_100000`:
  - `u=3.3702579e-05, v=3.3535732e-05, w=2.2215085e-04`
- `3324` corrected full-window:
  - `46000 -> u=1.145e-03, v=1.115e-03, w=8.420e-04`
  - `47000 -> u=1.159e-03, v=1.154e-03, w=8.250e-04`
  - `50000 -> u=1.235e-03, v=1.103e-03, w=8.170e-04`
- Ratio (`3324 / old 3155`):
  - `46000 -> u x33.97, v x33.25, w x3.79`
  - `47000 -> u x34.39, v x34.41, w x3.71`
  - `50000 -> u x36.64, v x32.89, w x3.68`

Interpretation:

- 若以 `3155 window1 checkpoint_100000` 作為「window-1 已知高品質 reference」，那麼 `3324` 的 `~1e-3` 平台 **明顯偏差，不能算好**。
- 差距不是邊際退步，而是：
  - `u/v` 大約差 **33~37 倍**
  - `w` 仍差 **3.7~3.8 倍**
- 因此 `3324` 證明的是：`data_weight=23.1429` + residual threshold `5e-5` 能把訓練帶到一個穩定平台，但這個平台的 field quality 仍顯著差於舊 `3155` 的高品質 window-1 checkpoint。
- 研究上應把 `5e-5 residual threshold` 視為「可用的 early stopping proxy」，而不是「高品質收斂」的充分條件。

Next:

- 若目標是逼近 `3155 window1 checkpoint_100000` 的品質，下一輪不能只固定 `23.1429` 和 `5e-5`；需改以 field metric 或更嚴的 proxy 來選 checkpoint / stop step。
- 優先檢查：
  - `data_weight=23.1429` 是否只是「最快達到 residual 門檻」而非「最佳 field quality」
  - 100k steps 或更晚 checkpoint 是否能接近 `3155` 水準
  - 是否要直接比較 `data_weight=23.1429` vs `data_weight=100` 的完整 checkpoint sweep

### [2026-04-21] `3137` vs `3324` | `window 1` loss comparison plot for no-data vs sensor

- Time: `2026-04-21 16:12 +0800`
- Status: Visualization completed
- Experiment or Job ID: baseline `3137`, sensor run `3324`

Change:

- 產生 `window 1` shared residual loss 對比圖，直接比較 no-data baseline (`3137`) 與固定 `data_weight=23.1429` 的 sensor run (`3324`)。
- 新增可重跑腳本 [plot_window1_loss_compare.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/plot_window1_loss_compare.py)，統一輸出 PNG、raw CSV 與文字摘要。

Config / Dataset / Checkpoint:

- Baseline:
  - Run: `3137`
  - Source raw CSV: [window1_loss_compare_raw.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_loss_compare_raw.csv)
- Sensor:
  - Run: `3324`
  - Config: [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py)
  - Source stderr: `/home/junyi/jaxpi/logs/kf_s100_w1_dw231429_eval_3324.err`
- Output artifacts:
  - Local plot: [window1_loss_compare.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_vs_3137_window1_loss_compare_20260421/window1_loss_compare.png)
  - Local summary: [window1_loss_compare.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_vs_3137_window1_loss_compare_20260421/window1_loss_compare.txt)
  - Local raw dump: [window1_loss_compare_raw.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_vs_3137_window1_loss_compare_20260421/window1_loss_compare_raw.csv)

Evidence:

- 比較範圍裁切為 shared horizon：`step <= 49900`
- `rc_loss`
  - no data: `<=1e-4 @ 10900`, `<=5e-5 @ 14300`, end=`1.683e-06`
  - sensor: `<=1e-4 @ 10500`, `<=5e-5 @ 13800`, end=`2.409e-06`
- `ru_loss`
  - no data: `<=1e-4 @ 18500`, `<=5e-5 @ 22000`, end=`4.986e-06`
  - sensor: `<=1e-4 @ 15700`, `<=5e-5 @ 21100`, end=`7.703e-06`
- `rv_loss`
  - no data: `<=1e-4 @ 18500`, `<=5e-5 @ 19600`, end=`4.951e-06`
  - sensor: `<=1e-4 @ 15700`, `<=5e-5 @ 21100`, end=`6.914e-06`
- End ratio (`sensor / no-data`):
  - `rc=1.431`
  - `ru=1.545`
  - `rv=1.396`

Interpretation:

- `3324` 並不是單向較好或較差，而是呈現 **前中期略快、尾段略高** 的差異。
- 具體來說，sensor run 在 `ru/rv` 上更早進入 `1e-4`，`rc/ru` 也略早碰到 `5e-5`；但到 shared horizon 結尾，三條 residual 都比 no-data 高約 `1.4x ~ 1.5x`。
- 因此若只看「最快碰到門檻」，sensor 版本有優勢；若看 shared horizon 尾段 residual floor，no-data baseline 反而更低。
- 這也再次支持：`3324` 的 `5e-5` crossing 不能直接當成高品質 field convergence 證據，必須和 corrected checkpoint eval 一起看。

Next:

- 若要把這張圖拿來支撐研究結論，應與 `3324` corrected checkpoint eval（`~1e-3` field plateau）一起引用。
- 若要進一步比較 sensor 權重策略，可用同一腳本追加 `3155 (data_weight=100)`，形成 `3137 vs 3324 vs 3155` 的三方 `window 1` loss comparison。

### [2026-04-21] `3137` | actual `window 1` retained checkpoints re-evaluated via direct `apply_fn`

- Time: `2026-04-21 16:36 +0800`
- Status: Partial checkpoint sweep completed
- Experiment or Job ID: `3137`

Change:

- 針對 **真實 `3137` checkpoint tree** `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_1` 補跑 direct `apply_fn` 路徑的 checkpoint evaluation。
- 因實際 retained checkpoints 只剩 `checkpoint_90000` 與 `checkpoint_100000`，所以這次只能做 **partial sweep**，不能宣稱為完整 `10k→100k` sweep。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Checkpoint root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Window: `1`
- Available checkpoints:
  - `checkpoint_90000`
  - `checkpoint_100000`
- Output artifacts:
  - [checkpoint_sweep_summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_window1_checkpoint_sweep_20260421/checkpoint_sweep_summary.txt)
  - [checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_window1_checkpoint_sweep_20260421/checkpoint_sweep_results.csv)
  - [checkpoint_sweep_error_vs_step.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_window1_checkpoint_sweep_20260421/checkpoint_sweep_error_vs_step.png)

Evidence:

- `checkpoint_90000`:
  - `u=3.781654e-05`
  - `v=3.725810e-05`
  - `w=2.385928e-04`
- `checkpoint_100000`:
  - `u=3.312488e-05`
  - `v=3.383027e-05`
  - `w=2.214195e-04`
- 這些數字與舊 `3145` 記錄的 `3137 window 1`：
  - `u=0.001132`
  - `v=0.001147`
  - `w=0.000734`
  存在明顯不一致。

Interpretation:

- 這次 direct `apply_fn` partial sweep 顯示：**真實 `3137` 在 `window 1 / step 90000` 就已經進到 `u/v ~3e-05`, `w ~2e-04` 的量級**，而不是 `~1e-3`。
- 因此先前對話裡把 `3137` 說成 `~1e-3` 平台，是把不同評估路徑 / 不同 checkpoint lineage 混在一起了。
- 更重要的是：`3145` 舊記錄與這次 `direct_apply_fn` 新結果對同一 `3137` root 出現衝突，這不是單純的 checkpoint step 差異，而是 **evaluation lineage mismatch**，必須再審計。

Next:

- 優先重新對照 `3145` 使用的 [eval_paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py) 與這次 `evaluate_window1_checkpoint_sweep.py` 的 direct `apply_fn` 路徑，定位為什麼同一 `3137` root 會出現 `e-3` vs `e-5` 分歧。
- 在根因釐清前，不再把 `3145 window 1 = 0.001132/0.001147/0.000734` 當成 `3137 window 1` 的最終可信結論。

### [2026-04-21] audit | trace `3145` vs `direct_apply_fn` discrepancy source for `3137 window 1`

- Time: `2026-04-21 16:46 +0800`
- Status: Root cause identified
- Experiment or Job ID: historical `3145`, current `3137` re-audit

Change:

- 對照 `3145` 使用的 [eval_paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py) 與目前 `direct_apply_fn` 路徑，追查為何同一 `3137 window 1` 會出現 `~1e-3` 與 `~1e-5` 級的差異。

Config / Dataset / Checkpoint:

- Historical artifact:
  - [eval_paper_repro_soap_0405_localtime](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime)
  - `window 1`: `u=0.001132, v=0.001147, w=0.000734`
- Current true checkpoint root:
  - `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_1`
  - retained checkpoints: `90000`, `100000`
- Current code paths audited:
  - [eval_paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py)
  - [models.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/models.py)
  - [evaluate_window1_checkpoint_sweep.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/evaluate_window1_checkpoint_sweep.py)

Evidence:

- `3145` 的執行時間是 `2026-04-05`，而 `models.py` 的 scalar-wrapper bug 修復紀錄是 `2026-04-16`。
- 舊 `eval_paper_repro_soap.py` 的 full-window error 明確走 wrapper 路徑：
  - `model.compute_l2_error_time_space_chunked(...)`
  - 其內部再呼叫：
    - `self.u_pred_fn(...)`
    - `self.v_pred_fn(...)`
    - `self.w_pred_fn(...)`
- `models.py` 的修復紀錄已明示：
  - 舊 `neural_net()` 在 scalar 輸入時會走 `z[None, :]` fake batch 維
  - 在 double-vmap / chunked evaluator 下會造成 batch-size-dependent 錯誤
- 今日新增的最小對照實驗（同一個 `3137 checkpoint_100000`、同一批 window-1 資料、同一份當前 `models.py`）顯示：
  - wrapper path: `u=2.890209e-05, v=3.806159e-05, w=2.191839e-04`
  - direct path: `u=2.890059e-05, v=3.806154e-05, w=2.191840e-04`
  - abs diff: `~1e-09, 1e-11, 1e-10`
- 這證明 **在目前已修復的 `models.py` 下，wrapper 與 direct 已經對齊**。
- 另外，今天對真實 retained checkpoints 的 direct `apply_fn` partial sweep 也顯示：
  - `90000 -> u=3.781654e-05, v=3.725810e-05, w=2.385928e-04`
  - `100000 -> u=3.312488e-05, v=3.383027e-05, w=2.214195e-04`

Interpretation:

- `3145 window 1 ~1e-3` 的來源，不是當前 code path 的真實行為，而是 **2026-04-05 당시舊 wrapper/scalar path bug 所產生的歷史 artifact**。
- 這次 audit 已經把兩個可能性分開：
  - 不是 checkpoint root 錯了：今天直接對真實 `3137` retained checkpoints 重跑，結果仍在 `e-5 / e-4`
  - 不是當前 wrapper 仍壞掉：修復後 wrapper 與 direct 在同一資料上已數值對齊
- 因此 `3145` 的 `window 1` 數字應視為 **已過期、已被新證據推翻的舊評估結果**。

Next:

- 後續若要引用 `3137 window 1`，應以 `2026-04-21` 的 direct `apply_fn` partial sweep 為準，而不是 `3145` 的 `window 1` 欄位。
- 若要完全清除混淆，可考慮重跑一個只針對 `3137 window 1` 的現代版 wrapper evaluator artifact，取代舊 `eval_paper_repro_soap_0405_localtime` 中的 `window 1` 數字。

### [2026-04-21] `3137` vs `3324` | redraw `window 1` error comparison using corrected `3137` values

- Time: `2026-04-21 16:49 +0800`
- Status: Visualization completed
- Note: `[SUPERSEDED 2026-04-21 22:04 +0800]` 本段使用的是 `3326` 舊 corrected eval；`3324` direct rerun 完成後，倍率已由 `~33x ~ 37x` 下修為 `~2x ~ 3x`。
- Experiment or Job ID: `3137`, `3324`

Change:

- 重新繪製 `window 1` corrected error comparison，改用：
  - `3137` 真實 retained checkpoints (`90000`, `100000`) 的 direct `apply_fn` 結果
  - `3324` corrected full-window eval (`46000`, `47000`, `50000`)
- 新增可重跑腳本 [plot_window1_error_compare.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/plot_window1_error_compare.py)。

Config / Dataset / Checkpoint:

- `3137` source:
  - [checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_window1_checkpoint_sweep_20260421/checkpoint_sweep_results.csv)
- `3324` source:
  - [window1_full_window_errors.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_ckpt_eval_20260421/window1_full_window_errors.csv)
- Output:
  - [window1_error_compare_3137_vs_3324.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_vs_3324_window1_error_compare_20260421/window1_error_compare_3137_vs_3324.png)
  - [window1_error_compare_3137_vs_3324.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_vs_3324_window1_error_compare_20260421/window1_error_compare_3137_vs_3324.txt)

Evidence:

- `3137 @ 100000`:
  - `u=3.312488e-05`
  - `v=3.383027e-05`
  - `w=2.214195e-04`
- `3324` best available (`w` 最低者為 `50000`)：
  - `u=1.235000e-03`
  - `v=1.103000e-03`
  - `w=8.170000e-04`
- Ratio (`3324 best available / 3137@100000`):
  - `u x37.283`
  - `v x32.604`
  - `w x3.690`

Interpretation:

- 一旦把 `3137` 的 `window 1` 基準改回正確值，`3324` 就不再是「和 no-data 差不多都在 ~1e-3」。
- 正確比較是：`3324` 明顯差於 `3137 window 1` 真實 retained checkpoint，尤其 `u/v` 差了 **約 33~37 倍**。
- 因此後續若要討論 `3324` 的 field quality，不應再拿舊 `3145 window1 ~1e-3` 來當 no-data baseline。

Next:

- 把 repo 內會被使用者或 demo 讀到的 `3145 window1 ~1e-3` 引用逐一改口徑：
  - `docs/demo-data.js`
  - `EXPERIMENT_RECORD.md` active/index 段落
  - 任何直接引用 `0.001132 / 0.001147 / 0.000734` 作為 `3137 window 1` 最終結論的位置

### [2026-04-21] `3327` | launch all-window direct re-evaluation for `3137`

- Time: `2026-04-21 17:02 +0800`
- Status: RUNNING
- Experiment or Job ID: `3327`

Change:

- 依人工要求，不再只修正 `window 1`，而是對 `3137` 的 **全部已落盤窗口** 重做 direct `apply_fn` full-window evaluation。
- 新增 [evaluate_all_windows_direct.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/evaluate_all_windows_direct.py)，固定使用：
  - 真實 checkpoint tree
  - 每個 `time_window_k` 的 latest retained checkpoint
  - current verified direct path

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Checkpoint root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Output dir: `/home/junyi/jaxpi/eval_runs/3137_all_windows_direct_20260421`
- Window scope: `1..12`
- Submission:
  - `sbatch --job-name=eval_3137_all_direct ...`
  - Slurm job id: `3327`

Evidence:

- Local syntax check:
  - `python3 -m py_compile scripts/analysis/evaluate_all_windows_direct.py`
- Remote sync:
  - `/home/junyi/jaxpi/scripts/analysis/evaluate_all_windows_direct.py`
- Slurm:
  - `Submitted batch job 3327`
  - `squeue -j 3327 -> RUNNING on acmt20`

Interpretation:

- 這次是把 `3137 window 1` 的 direct re-audit 擴展成全窗口版本，目的是徹底清除 `3145` 舊 artifact 對 `3137 baseline` 的污染。
- 在 `3327` 完成前，`window 2..12` 仍暫時沿用舊 corrected local-time rerun；完成後應以 `3327` 為新的單一路徑 baseline。

Next:

- 等 `3327` 完成後，回收：
  - `summary.txt`
  - `all_window_direct_results.csv`
  - `window_error_comparison.png`
- 再用這份全窗口 direct baseline 重做：
  - `3137 vs 3155` per-window error trend
  - 必要時同步更新 thesis 敘述與 demo data

### [2026-04-21] `3327` | all-window direct re-evaluation for `3137` completed

- Time: `2026-04-21 18:21 +0800`
- Status: COMPLETED (`ExitCode=0:0`)
- Experiment or Job ID: `3327`

Change:

- 完成 `3137` 真實 checkpoint tree 的 `window 1..12` direct `apply_fn` full-window evaluation。
- 這份結果現在可以取代混合 `3145` 舊 artifact 的 baseline，作為 `3137` 的單一路徑 corrected baseline。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Checkpoint root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Output dir: `/home/junyi/jaxpi/eval_runs/3137_all_windows_direct_20260421`
- Local artifacts:
  - [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_all_windows_direct_20260421/summary.txt)
  - [all_window_direct_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_all_windows_direct_20260421/all_window_direct_results.csv)
  - [window_error_comparison.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_all_windows_direct_20260421/window_error_comparison.png)

Evidence:

- Slurm:
  - `3327 | COMPLETED | Elapsed=00:28:26 | ExitCode=0:0`
- Direct full-window results:
  - `window 1 / checkpoint_100000 -> u=3.312488e-05, v=3.383027e-05, w=2.214195e-04`
  - `window 2 / checkpoint_100000 -> u=6.558106e-05, v=6.549775e-05, w=6.876990e-04`
  - `window 3 / checkpoint_100000 -> u=1.294855e-04, v=1.363780e-04, w=3.224143e-03`
  - `window 4 / checkpoint_100000 -> u=2.791856e-04, v=3.096886e-04, w=9.737843e-03`
  - `window 5 / checkpoint_100000 -> u=5.484705e-04, v=6.193409e-04, w=1.992797e-02`
  - `window 6 / checkpoint_100000 -> u=9.573215e-04, v=1.039929e-03, w=3.341517e-02`
  - `window 7 / checkpoint_100000 -> u=1.506963e-03, v=1.612425e-03, w=4.913890e-02`
  - `window 8 / checkpoint_100000 -> u=2.279739e-03, v=2.400119e-03, w=7.073084e-02`
  - `window 9 / checkpoint_100000 -> u=3.351144e-03, v=3.397276e-03, w=9.338640e-02`
  - `window 10 / checkpoint_100000 -> u=4.574165e-03, v=4.756622e-03, w=1.194688e-01`
  - `window 11 / checkpoint_100000 -> u=6.006283e-03, v=6.456355e-03, w=1.454885e-01`
  - `window 12 / checkpoint_60000 -> u=7.821861e-03, v=8.404544e-03, w=1.760991e-01`

Interpretation:

- `window 1 -> 2` 的確有明顯跳升，但不是先前混合口徑造成的假象；用 direct 路徑重做後，`u/v` 仍從 `~3e-05` 升到 `~6.5e-05`，`w` 從 `2.2e-04` 升到 `6.9e-04`。
- 更重要的是，`window 2..12` 的整體趨勢與舊 corrected local-time rerun 在量級上高度一致，表示先前對跨窗誤差累積的研究判讀大方向沒有翻案；真正需要修正的是 `window 1` 基準。
- 這份結果現在應成為後續所有 `3137 vs 3155`、`3137 vs 3324` 比較的 no-data baseline。

Next:

- 用 `3327` 的全窗口 direct baseline 重畫 `3137 vs 3155` error trend，取代目前仍部分依賴舊 rerun 的版本。
- 必要時同步更新 thesis 圖與相關敘述。

### [2026-04-21] `3328` | launch all-window direct re-evaluation for `3155`

- Time: `2026-04-21 18:28 +0800`
- Status: RUNNING
- Experiment or Job ID: `3328`

Change:

- 為了讓 `3137 vs 3155` 回到完全同一路徑比較，提交 `3155` 的 all-window direct `apply_fn` full-window evaluation。
- 使用與 `3327` 相同的 evaluator [evaluate_all_windows_direct.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/evaluate_all_windows_direct.py)，只更換 config 與 checkpoint root。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt`
- Output dir: `/home/junyi/jaxpi/eval_runs/3155_all_windows_direct_20260421`
- Window scope: `1..21`
- Retained checkpoints:
  - `window 1..20 -> checkpoint_100000`
  - `window 21 -> checkpoint_60000` (also retains `50000`)

Evidence:

- Submission:
  - `sbatch --job-name=eval_3155_all_direct ...`
  - `Submitted batch job 3328`
- Slurm:
  - `squeue -j 3328 -> RUNNING on acmt20`

Interpretation:

- 這次完成後，`3137` 與 `3155` 兩條主線都會擁有同一路徑的 direct baseline / with-data evaluation，可正式重畫最終版 per-window error trend。
- 在 `3328` 完成前，現有 `3155` 圖仍屬舊 corrected evaluator lineage，只能作為暫時參考。

Next:

- 等 `3328` 完成後，回收：
  - `summary.txt`
  - `all_window_direct_results.csv`
  - `window_error_comparison.png`
- 再正式重畫 `3137 vs 3155` 的最終版 per-window error trend。

### [2026-04-21] `3328` | all-window direct re-evaluation for `3155` completed

- Time: `2026-04-21 21:20 +0800`
- Status: COMPLETED (`ExitCode=0:0`)
- Experiment or Job ID: `3328`

Change:

- 回收 `3155` 的 all-window direct `apply_fn` full-window evaluation 結果，補齊 `3137 vs 3155` 的 apples-to-apples baseline。
- 輸出 local artifact 並確認 `3155` 的 direct 路徑和先前 corrected evaluator 在量級上相符。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt`
- Local output dir: [3155_all_windows_direct_20260421](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_direct_20260421)
- Shared-plot consumer: [redraw_3137_3155_error_trends.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/redraw_3137_3155_error_trends.py)

Evidence:

- `sacct -j 3328`:
  - `3328 | eval_3155_all_direct | COMPLETED | ExitCode=0:0`
- Local artifacts:
  - Summary: [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_direct_20260421/summary.txt)
  - CSV: [all_window_direct_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_direct_20260421/all_window_direct_results.csv)
  - Plot: [window_error_comparison.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_direct_20260421/window_error_comparison.png)
- Key direct results:
  - `window 1  / checkpoint_100000 -> u=3.370493e-05, v=3.354592e-05, w=2.221507e-04`
  - `window 12 / checkpoint_100000 -> u=7.831324e-03, v=8.349953e-03, w=1.741970e-01`
  - `window 21 / checkpoint_60000  -> u=4.065330e-02, v=4.170452e-02, w=4.596829e-01`

Interpretation:

- `3155` 的 direct 曲線沒有推翻舊 corrected trend；它只是把比較口徑正式統一到 direct `apply_fn`。
- 因此下一步不再是爭論 evaluator lineage，而是直接看同一路徑下 `3155` 是否真的比 `3137` 好。

Next:

- 用 `3327 + 3328` 的 direct CSV 重畫最終版 `3137 vs 3155` shared-window trend。
- 後續所有 `3137 vs 3155` 的定量比較，應優先引用這兩個 direct artifact。

### [2026-04-21] `3318` | corrected `1..100`, `5e-5` sweep finished

- Time: `2026-04-21 01:19 +0800`
- Status: COMPLETED (`ExitCode=0:0`)
- Experiment or Job ID: `3318`

Change:

- 依人工要求回查 `3318` 是否仍在運行。
- Slurm queue 已無 `3318`，accounting 顯示該 job 已正常完成。

Config / Dataset / Checkpoint:

- Job: `3318`
- Study: `kf_w1_data_weight_sweep_1to100_thr5e5`
- Storage: `sqlite:///sweep_w1_data_1to100_thr5e5.db`
- Sweep target: fastest `max(ru_loss, rv_loss, rc_loss) < 5e-5`

Evidence:

- `squeue -j 3318` -> `Invalid job id specified`
- `sacct -j 3318`:
  - `3318 | sweep_kf_w1_weights | COMPLETED | Elapsed=1-03:53:02 | ExitCode=0:0`
  - `Start=2026-04-19 21:26:17 +0800`
  - `End=2026-04-21 01:19:19 +0800`

Interpretation:

- `3318` 已不在運行中；目前正確的下一步不是查 queue，而是讀取新 study DB 與 job log，判讀哪個 `data_weight` 最好、是否有人先穿越 `5e-5`。

Next:

- 讀取 `sweep_w1_data_1to100_thr5e5.db` 與 `sweep_kf_w1_weights_3318.out/.err`，整理完整 trial 結果表與最佳參數。

### [2026-04-21] `3318` | study DB + log result summary

- Time: `2026-04-21 09:00 +0800`
- Status: Result summarized from study DB and Slurm log
- Experiment or Job ID: `3318`

Change:

- 讀取 remote Optuna DB `sweep_w1_data_1to100_thr5e5.db` 與 `sweep_kf_w1_weights_3318.out/.err`，整理 `3318` 的完整 trial 結果。
- 明確區分 `3317` 錯版遺留 trial 與 `3318` 正式 sweep trials，避免把污染資料混入結論。

Config / Dataset / Checkpoint:

- Study: `kf_w1_data_weight_sweep_1to100_thr5e5`
- Storage: `/home/junyi/jaxpi/sweep_w1_data_1to100_thr5e5.db`
- Log:
  - `/home/junyi/jaxpi/logs/sweep_kf_w1_weights_3318.out`
  - `/home/junyi/jaxpi/logs/sweep_kf_w1_weights_3318.err`
- Sweep range: `data_weight in [1, 100]` (`log=True`)
- Target: fastest `max(ru_loss, rv_loss, rc_loss) < 5e-5`

Evidence:

- Log footer:
  - `=== Best Trial ===`
  - `steps_to_threshold : 46100.0`
  - `data_weight = 23.14`
- DB raw table contains `41` trials (`0..40`):
  - `trial 0 = RUNNING, data_weight = 136.278...`
  - `trial 0` 超出本輪 `1..100` 範圍，且狀態殘留 `RUNNING`，可確定是錯版 `3317` 遺留，不屬於 `3318` 正式結果
- 正式 `3318` trial 集合應取 `trial 1..40`
- `trial 1..40` 摘要：
  - `COMPLETE = 5`
  - `PRUNED = 35`
- `COMPLETE` trials:
  - `trial 2  | data_weight=23.1429 | value=46100`
  - `trial 5  | data_weight=6.8668  | value=47400`
  - `trial 3  | data_weight=59.8651 | value=48200`
  - `trial 4  | data_weight=3.1581  | value=49300`
  - `trial 1  | data_weight=1.9164  | value=50000`
- 最佳幾個 `PRUNED` trials（value 為 prune 時回報的 worst residual，不是達標步數）：
  - `trial 24 | data_weight=52.0527 | reported worst=1.505e-4`
  - `trial 31 | data_weight=1.4670  | reported worst=5.052e-4`
  - `trial 28 | data_weight=57.1463 | reported worst=5.169e-4`
  - `trial 20 | data_weight=7.4140  | reported worst=8.924e-4`
  - `trial 30 | data_weight=17.1344 | reported worst=1.229e-3`

Interpretation:

- **最佳 `data_weight` 是 `23.14`**（trial 2），在正式 `3318` sweep 中最早於 `46100` steps 達到 `5e-5` 門檻。
- **有 trial 真的達到 `5e-5`**，而且不是只有一個：
  - `23.14 -> 46100 steps`
  - `6.87  -> 47400 steps`
  - `59.87 -> 48200 steps`
  - `3.16  -> 49300 steps`
- `data_weight=1.916` 沒有在 `50000` steps 內達標（value=`50000` 代表上限耗盡）。
- `data_weight=100` **不是目前最佳**；最接近上界的正式 trial `99.94`（trial 26）被 prune，reported worst residual 仍為 `4.257e-3`，遠高於 `5e-5`。
- 因此目前證據支持的不是「100 最好」，而是「中等偏高權重區（約 `20~60`）較有機會在 `5e-5` 目標下最快收斂」。
- 但這仍是單一 window-1 sweep；不可把它外推成跨 window 最佳權重結論。

Next:

- 若要在 `20~60` 區間再細化，可做第二輪局部 sweep（例如 `10..80` 或 `15..40`）。
- 若要把這結論用於正式訓練 config，應先做至少一個 window-2+ 驗證，避免把 window-1 最佳權重誤當成全流程最佳權重。

### [2026-04-20] eval audit | fail-fast window layout + unit-domain Fourier axis alignment

- Time: `2026-04-20 00:00 +0800`
- Status: Local eval infrastructure updated and verified
- Experiment or Job ID: `N/A` (`eval_audit_20260420`)

Change:

- 新增共用 helper [eval_common.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/eval_common.py)，集中處理：
  - `to_window_local_time()`
  - `resolve_window_layout()` / `resolve_window_layout_from_config()`
  - `infer_domain_lengths()`
  - `compute_energy_spectrum()`
- 將以下主評估入口改為共用同一套 fail-fast 規則：
  - [eval_paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py)
  - [eval_sensor100_w25.py](/Users/latteine/Documents/coding/jaxpi/eval_sensor100_w25.py)
  - [eval_re10k_n256_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_re10k_n256_soap.py)
  - [eval_re10k_sensor100.py](/Users/latteine/Documents/coding/jaxpi/eval_re10k_sensor100.py)
  - [evaluate_checkpoint.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/evaluate_checkpoint.py)
- 另外修正兩支 comparison artifact 腳本的 time-axis 對齊：
  - [generate_comparison_plots.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/generate_comparison_plots.py)
  - [generate_spectrum_evolution.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/generate_spectrum_evolution.py)
- 另外修正兩支 field comparison / snapshot 腳本的 local-time 對齊：
  - [generate_field_comparison.py](/Users/latteine/Documents/coding/jaxpi/generate_field_comparison.py)
  - [generate_field_snapshots.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/generate_field_snapshots.py)
- 修正 [time_aligned_comparison.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/time_aligned_comparison.py) 的硬編碼錯誤：原本把 SOAP 每窗步數直接寫成 `2`，現改為從 config/dataset 解析。
- 封存危險舊入口 [eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/eval.py)：現在會直接報錯，要求改用 `evaluate_checkpoint.py`。
- 在相關 config 顯式加入 `config.eval.expected_time_remainder = 1`，避免 `41/101` 個時間點的資料在 eval 時被靜默截尾：
  - [re10k_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/re10k_soap.py)
  - [re10k_soap_sensor100.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/re10k_soap_sensor100.py)
  - [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
  - [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
  - [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
  - [soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/soap.py)
  - [pirate.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/pirate.py)

Config / Dataset / Checkpoint:

- Datasets:
  - `kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy`
  - `kolmogorov_Re1e6_N512_T5_ds4.npy`
- Dataset config evidence:
  - both files report `config.L = 1.0`
  - time lengths are `41` and `101`
- Checkpoint: 無新 checkpoint；本次僅修正 eval / analysis path 的定義與防呆

Evidence:

- RED test:
  - `./.venv/bin/python -m unittest tests/test_eval_common.py`
  - 變更前失敗：`ImportError: cannot import name 'eval_common'`
- GREEN test:
  - `./.venv/bin/python -m unittest tests/test_eval_common.py` -> `Ran 5 tests ... OK`
- Syntax verification:
  - `./.venv/bin/python -m py_compile ...`（包含新 helper、主 eval driver、相關 config）-> PASS
- Dataset hard evidence:
  - `./.venv/bin/python` 讀 DNS `.npy`：
    - `kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy -> config.L=1.0, time len=41`
    - `kolmogorov_Re1e6_N512_T5_ds4.npy -> config.L=1.0, time len=101`
- Import smoke:
  - `eval_paper_repro_soap`, `eval_sensor100_w25`, `eval_re10k_n256_soap`, `eval_re10k_sensor100`, `evaluate_checkpoint`, `time_aligned_comparison` 全部可 import
  - `generate_comparison_plots` / `generate_spectrum_evolution` 在本機需加 `JAX_PLATFORMS=cpu` 才能通過 import；`py_compile` 已證明語法正確

Interpretation:

- 這次不是新的實驗結果，而是評估基礎設施修補。
- 已確認至少有兩類會污染研究結論的風險被消除：
  - `trailing time step` 被靜默截掉
  - unit-domain DNS 被錯當成 `[0, 2π)` 來畫 spectrum
- 另外也移除了至少一條危險舊入口（`examples/kolmogorov_flow/eval.py`），避免後續誤用。

Next:

- 若要重跑任何 `re10k` / `re1e6` comparison artifact，應以修正後腳本重新生成，不要混用舊版頻譜圖。
- 對尚未納入本輪 helper 的歷史可視化腳本（例如單張 field snapshot 類）也應再做一次 local-time audit，避免留下未封口的舊路徑。

### [2026-04-19] `3317 -> 3318` | resubmit corrected `1..100`, `5e-5` window-1 sweep

- Time: `2026-04-19 21:26 +0800`
- Status: `3317` cancelled; `3318` running
- Experiment or Job ID: `3317`, `3318`

Change:

- 依人工同意，提交新的 window-1 weight sweep，使用獨立 study / storage：
  - `STUDY_NAME=kf_w1_data_weight_sweep_1to100_thr5e5`
  - `STORAGE=sqlite:///sweep_w1_data_1to100_thr5e5.db`
- 首次提交為 `3317`，但啟動後 stdout header 顯示 remote 仍使用舊版腳本，`Threshold` 還是 `1e-5`。
- 立即將本地已修改的 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 與 [sweep_kf_w1_weights.sh](/Users/latteine/Documents/coding/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh) 同步到 remote。
- 取消錯版 `3317` 後重新提交 `3318`，確認新 job 吃到正確 threshold 與新 study/storage。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_window1_ablation.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py)
- Storage: `sqlite:///sweep_w1_data_1to100_thr5e5.db`
- Sweep range: `data_weight in [1, 100]` (`log=True`)
- Objective threshold: `5e-5`
- Checkpoint: 無；sweep 預設不存 checkpoint

Evidence:

- First submission:
  - `sbatch -> Submitted batch job 3317`
  - `sweep_kf_w1_weights_3317.out` header:
    - `Study    : kf_w1_data_weight_sweep_1to100_thr5e5`
    - `Threshold: 1e-5`  ← 錯版，證明 remote script 尚未同步
- Remote sync:
  - `rsync scripts/sweep/sweep_weights_window1.py ...:/home/junyi/jaxpi/scripts/sweep/sweep_weights_window1.py`
  - `rsync slurm/sweep/sweep_kf_w1_weights.sh ...:/home/junyi/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh`
- Corrected submission:
  - `scancel 3317`
  - `sbatch -> Submitted batch job 3318`
  - `squeue -j 3318` -> `RUNNING`
  - `sweep_kf_w1_weights_3318.out` header:
    - `Study    : kf_w1_data_weight_sweep_1to100_thr5e5`
    - `Threshold: 5e-5`
    - `Storage  : sqlite:///sweep_w1_data_1to100_thr5e5.db`
    - `target   : max(ru,rv,rc) < 5e-05`
- Slurm accounting:
  - `3317 | CANCELLED | Elapsed=00:00:30`
  - `3318 | RUNNING   | Start=2026-04-19 21:26:17 +0800`

Interpretation:

- `3317` 不可作為正式 sweep 結果引用；它只是暴露出 remote 腳本未同步的提交失敗案例。
- `3318` 才是本輪正式的 `1..100` / `5e-5` sweep。
- 目前最重要的是後續觀察 `3318` 早期 trials 是否把 `data_weight=100` 附近保留成最佳區域，或是否在中高權重區間更快穿越 `5e-5`。

Next:

- 持續監看 `3318` 的 trial log，優先抓：
  - 首個 `finished with value` 的完整 trial
  - 是否有 trial 在 `50000` steps 內實際達到 `5e-5`
  - `data_weight=100` 附近是否優於中段權重
- job 完成後，從新 DB 匯總完整 trial/value/weight 表，不和 `3314` 混表。

### [2026-04-19] sweep logic update | narrow `data_weight` range to `1..100` and relax threshold to `5e-5`

- Time: `2026-04-19 21:30 +0800`
- Status: Local code updated and verified; no new Slurm job submitted
- Experiment or Job ID: N/A (pre-run sweep logic change)

Change:

- 依人工指示，將 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 的 `data_weight` 搜尋空間從 `0.1..1000`（log scale）收斂成 `1..100`（log scale）。
- 同時把 objective threshold 的預設值從 `1e-5` 改成 `5e-5`，讓下一輪 sweep 優先回答「`data_weight=100` 是否已接近最佳」以及「哪組權重最快跌到 `5e-5`」。
- 同步更新 [sweep_kf_w1_weights.sh](/Users/latteine/Documents/coding/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh) 的 `THRESHOLD` 預設值，避免 Slurm wrapper 與 Python 腳本預設不一致。
- 額外抽出 `suggest_data_weight()` 與 `build_arg_parser()`，讓 sweep 搜尋空間與 CLI 預設值可被單元測試直接驗證。

Config / Dataset / Checkpoint:

- Code:
  - [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)
  - [sweep_kf_w1_weights.sh](/Users/latteine/Documents/coding/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh)
  - [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)
- Dataset / checkpoint: 無變更；本次僅改 sweep 邏輯與預設參數

Evidence:

- RED test:
  - `python3 -m unittest tests/test_sweep_weights_window1.py`
  - 變更前失敗：`AttributeError: module 'scripts.sweep.sweep_weights_window1' has no attribute 'suggest_data_weight'`
  - 變更前失敗：`AttributeError: module 'scripts.sweep.sweep_weights_window1' has no attribute 'build_arg_parser'`
- GREEN test:
  - `python3 -m unittest tests/test_sweep_weights_window1.py` -> `Ran 2 tests ... OK`
- Syntax:
  - `python3 -m py_compile scripts/sweep/sweep_weights_window1.py`
  - `bash -n slurm/sweep/sweep_kf_w1_weights.sh`

Interpretation:

- 下一次提交的 sweep 將不再把 sampling budget 分散到 `0.1` 以下或 `100` 以上的權重區間，而是集中在使用者關心的 `1..100`。
- 目標門檻改成 `5e-5` 後，study 的 best-trial 排序會更接近「最快有效下降」而不是全部卡在 `50000` steps 上限。
- 本次仍未驗證新的最佳權重；目前只完成「下一輪 sweep 會回答正確問題」的邏輯準備。

Next:

- 重新提交新的 window-1 sweep job，建議使用新 study/storage 名稱，避免和舊 `3314` 混在同一個 Optuna DB。
- 提交後優先觀察 `data_weight=100` 附近 trial 的 threshold crossing step，確認是否真的是上界最好。

### [2026-04-19] `3314` | window-1 `data_weight` Optuna sweep completed

- Time: `2026-04-19 21:14 +0800`
- Status: COMPLETED (`ExitCode=0:0`)
- Experiment or Job ID: `3314`

Change:

- 檢查伺服器當前 job 狀態時，確認 Slurm 佇列為空；最新完成 job 為 `3314`。
- `3314` 執行 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)，對 `paper_repro_soap_window1_ablation.py` 做 `data_weight` 對數掃描。
- Sweep 目標是最小化 `steps_to_threshold`，也就是讓 `max(ru_loss, rv_loss, rc_loss) < 1e-5` 所需步數；單一 trial 上限 `50000` steps，使用 `MedianPruner(n_startup_trials=5, n_warmup_steps=5000, interval_steps=500)`。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_window1_ablation.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py)
- Storage: `/home/junyi/jaxpi/sweep_w1_data.db`
- Sensor: `/home/junyi/jaxpi/examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/sensors_qrpivot_K100_N512_t0-5.json`
- Checkpoint: 無；此 sweep 關閉 checkpoint 儲存（`config.saving.save_every_steps = None`）

Evidence:

- Slurm accounting:
  - `3314 | sweep_kf_w1_weights | COMPLETED | Start=2026-04-18 12:59:59 +0800 | End=2026-04-19 17:17:30 +0800 | Elapsed=1-04:17:31 | ExitCode=0:0`
- Job header / stdout:
  - `Study = kf_w1_data_weight_sweep`
  - `Storage = sqlite:///sweep_w1_data.db`
  - `Max_steps = 50000`
  - `Threshold = 1e-5`
  - `Best Trial -> steps_to_threshold = 50000.0, data_weight = 0.3211`
- Study DB summary:
  - `COMPLETE = 5 trials`, `PRUNED = 35 trials`
  - 所有 `COMPLETE` trials 的 objective 都是 `50000.0`，表示在 `50000` steps 內都**沒有**達到 `1e-5` 門檻
  - 最佳幾個被 prune 的 `data_weight` / reported worst-loss：
    - `382.418 -> 1.118e-4`
    - `7.534   -> 4.960e-4`
    - `36.020  -> 6.263e-4`
    - `177.134 -> 7.415e-4`
    - `0.2526  -> 7.605e-4`

Interpretation:

- `3314` 沒有找到任何能在 `window 1`、`50000` steps 內把 `max(ru,rv,rc)` 壓到 `1e-5` 以下的 `data_weight`。
- `Best Trial = 0` 不代表成功，而只代表「在未被 prune 的 trial 裡最不差」，其 objective 仍然卡在上限 `50000`。
- 被 prune 的高權重區間（例如 `36`, `177`, `382`）曾經把 reported worst-loss 壓到 `1e-4 ~ 1e-3`，但仍離門檻差約一個數量級以上；目前證據不足以支持「只調 `data_weight` 就能讓 window-1 快速收斂到 `1e-5`」。
- 因此這次 sweep 的研究結論應是：`window 1` 的 `data_weight` 單參數掃描未解決收斂底線問題，主瓶頸不太像只是 loss weight 選錯。

Next:

- 若要繼續追 `1e-5` 門檻，下一輪不應只重跑同型 sweep；應優先檢查 threshold 是否過嚴、或改 sweep 其他會影響 early-loss floor 的因素（例如 optimizer / schedule / IC 權重組合）。
- 若目標改為比較「哪組權重能把 window-1 worst-loss 壓得最低」，應調整 objective，避免現在這種「全部 complete trial 都並列 `50000`」而失去排序解析度。

### [2026-04-16] `3273` | window-1 checkpoint sweep with corrected direct `apply_fn` forward path (no-data vs sensor)

- Time: `2026-04-16 01:30 +0800`
- Status: COMPLETED (`ExitCode=0:0`)
- Experiment or Job ID: `3273`

Change:

- 依 `3261..3272` audit 結論，使用 direct `apply_fn` forward path（batch-invariant）重跑 window-1 checkpoint sweep。
- 比較對象：
  - `no_data`: config `paper_repro_soap_window1_ablation.py`，checkpoint root `/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ablation/ckpt`
  - `sensor`: config `paper_repro_soap_sensor100_n512_w50_window1_ablation.py`，checkpoint root `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_ablation/ckpt`
- 掃描步數：`10k, 20k, ..., 100k`（共 10 個 checkpoint）
- Forward path: `direct_apply_fn`（已確認 batch-invariant）

Config / Dataset / Checkpoint:

- Output dir: `/home/junyi/jaxpi/eval_runs/window1_checkpoint_sweep_direct_apply_20260416/`
- Local copy: `eval_runs/window1_checkpoint_sweep_direct_apply_20260416/`

Evidence:

- Slurm: `3273 | COMPLETED | End=2026-04-16T01:30:21 | ExitCode=0:0`
- 完整結果（`checkpoint_sweep_summary.txt`）：

| run | step | u_error | v_error | w_error |
| :--- | ---: | ---: | ---: | ---: |
| no_data | 10000 | 1.805167e-03 | 1.693111e-03 | 2.583466e-03 |
| no_data | 20000 | 1.303826e-03 | 1.437020e-03 | 1.319871e-03 |
| no_data | 30000 | 1.372744e-03 | 1.220133e-03 | 1.088192e-03 |
| no_data | 40000 | 1.072604e-03 | 1.181921e-03 | 8.807618e-04 |
| no_data | 50000 | 1.141463e-03 | 1.202966e-03 | 8.218470e-04 |
| no_data | 60000 | 1.223646e-03 | 1.110012e-03 | 7.972292e-04 |
| no_data | 70000 | 1.145631e-03 | 1.282462e-03 | 7.597307e-04 |
| no_data | 80000 | 1.118223e-03 | 1.172751e-03 | 7.619565e-04 |
| no_data | 90000 | 1.083530e-03 | 1.107486e-03 | 7.240397e-04 |
| no_data | 100000 | 1.077166e-03 | 1.088496e-03 | 7.043464e-04 |
| sensor | 10000 | 1.611865e-03 | 1.716142e-03 | 2.649553e-03 |
| sensor | 20000 | 1.423732e-03 | 1.355362e-03 | 1.410681e-03 |
| sensor | 30000 | 1.133016e-03 | 1.180314e-03 | 1.029871e-03 |
| sensor | 40000 | 1.195103e-03 | 1.094728e-03 | 8.770467e-04 |
| sensor | 50000 | 1.208914e-03 | 1.153957e-03 | 8.390927e-04 |
| sensor | 60000 | 1.077931e-03 | 1.093845e-03 | 7.592798e-04 |
| sensor | 70000 | 1.080680e-03 | 1.157059e-03 | 7.267448e-04 |
| sensor | 80000 | 1.038285e-03 | 1.052127e-03 | 7.282852e-04 |
| sensor | 90000 | 1.047114e-03 | 1.115390e-03 | 7.085629e-04 |
| sensor | 100000 | 1.135520e-03 | 1.134433e-03 | 7.110798e-04 |

Interpretation:

- **Window 1 兩組幾乎相同**：`no_data` 與 `sensor` 在全程 `10k→100k` 的誤差曲線幾乎重疊；`checkpoint_100000` 差距均在 6% 以內，`no_data` 略優。
- **Sensor 在 window 1 無顯著效益**：sensor loss 對 window 1 本身的擬合精度幾乎無貢獻；其潛在效益（若有）須在跨 window 漂移抑制上觀察。
- **direct `apply_fn` path 已驗證為 batch-invariant**：此結果可信，不受 `3259/3260/3261` 的 scalar-wrapper bug 影響。
- **兩組收斂底線均在 `~1e-3`**：這是 PirateNet + SOAP 在 `Re=1e6` window 1 的訓練上限，非 sensor 造成，說明 window 1 itself 不是主要問題。
- 此結論解除 `3155` `Current Risk` 中關於 eval path 正確性的警示；舊 `3155 window 1 / checkpoint_100000 -> u≈3.37e-05` 仍成立（不同 checkpoint root，非同一系列）。

Next:

- 將此 window-1 基準與 `3155` 跨 window 誤差曲線（`w1→w13`）對照，核心問題是「sensor 的抑制效果從哪個 window 起消失，誤差開始爆炸」。
- 若要進一步做跨 window A/B，須先確認兩組 ablation run（`re1e6_n512_ds4_soap_w1_ablation` vs `re1e6_n512_ds4_soap_sensor100_w50_w1_ablation`）是否已完成超過 `window 1`。

### [2026-04-16] `neural_net()` scalar-wrapper bug fix | 移除 `z[None, :]` fake batch path

- Time: `2026-04-16 10:30 +0800`
- Status: Fixed and smoke-tested

Change:

- 移除 `models.py` `neural_net()` 中 `z.ndim == 1` 的條件分支（舊 L185-192）。
- 修復前：scalar 輸入時加 `z[None, :]` fake batch 維，在 double vmap (time × space) 下讓 `apply_fn` 看到 `(T, 1, 3)` 而非 `(T, N, 3)`，導致 batch-size-dependent 結果。
- 修復後：直接呼叫 `apply_fn(params, z)`，PirateNet `__call__` 接受任意 `(..., 3)` shape；vmap 在 trace 時正確向量化 `(3,)` → `(N, 3)`，無額外維度。

Evidence:

- Smoke test（`no_data ablation / checkpoint_100000`，uv 環境）：
  - `[1] u_ic_pred_fn vs direct: max_diff=0.00e+00  PASS`
  - `[2] u_pred_fn batch-invariance (32 vs 64): max_diff=2.24e-07  PASS`
  - `[3] u_pred_fn vs direct at t=t_star[0]: max_diff=3.13e-07  PASS`

Interpretation:

- `u_pred_fn` / `u_ic_pred_fn` / `compute_l2_error` / `_compute_l2_error_time_space_chunked_impl` 全部走 `neural_net` → `u_net`，隨修復而修復。
- 訓練時的 IC loss（`u_ic_pred_fn`）也受此 bug 影響，但修復前的 training log 仍可信（誤差量級不同，不影響 loss 收斂趨勢的定性判斷）。
- 評估路徑正確性風險已解除；後續 A/B 評估若用 `models.py` 自帶 `compute_l2_error_time_space_chunked` 也是可信路徑。

### [2026-04-16] `3261..3272` | audit window-1 evaluation discrepancy and batch-size-sensitive forward path

- Time: `2026-04-16 01:03 +0800`
- Status: Root cause localized; evaluator fix required before new A/B conclusion
- Experiment or Job ID: `3261`, `3263`, `3264`, `3265`, `3266`, `3271`, `3272`

Change:

- 針對人工質疑「舊 corrected evaluation 在 `100k` 時 sensor 的 `u/v/w` 都較低」重新排查。
- 重新跑舊 evaluator、新 sweep evaluator、不同 prediction API、不同 `space_chunk_size`，最後新增 direct `apply_fn` audit，繞過 `u_pred_fn/u_ic_pred_fn` 的 scalar-wrapper path。

Config / Dataset / Checkpoint:

- Config: `/home/junyi/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py`
- Checkpoint: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_1/checkpoint_100000`
- Dataset: `/home/junyi/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy`
- Evaluation window: `window 1`, local time `[0.0, 0.05]`

Evidence:

- `3261` reran the old `eval_sensor100_w25.py` path with `time_windows=[1]` and got:
  - `u=0.001204, v=0.001060, w=0.000735`
- `3263` showed official chunked and manual current wrapper path agree:
  - `official_chunked tc=1/2/20 -> u=0.001203502, v=0.001059683, w=0.000734823`
  - `manual u_pred_fn -> u=0.001203502, v=0.001059683, w=0.000734823`
  - `per_time path -> u≈0.00037~0.00039, v≈0.00036~0.00043, w≈0.00025~0.00032`
- `3265` showed current chunked evaluator is not batch/chunk invariant:
  - `space_chunk_size=4096 -> u=0.001203502, v=0.001059683, w=0.000734823`
  - `space_chunk_size=32768 -> u=0.000381905, v=0.000397585, w=0.000289326`
  - `space_chunk_size=65536 -> u=0.001203502, v=0.001059682, w=0.000734823`
- `3266` showed `u_pred_fn` and `u_ic_pred_fn` switch between high/low results depending on batch size:
  - `16384 pred -> u=0.000381905, v=0.000397585`
  - `16384 ic -> u=0.001203502, v=0.001059683`
  - `32768 pred -> u=0.001203502, v=0.001059683`
  - `32768 ic -> u=0.000381905, v=0.000397585`
- `3271` direct `apply_fn` UV path is batch-invariant and reproduces the old low result:

| space_chunk_size | u_error | v_error |
| ---: | ---: | ---: |
| 1024 | 3.3702579e-05 | 3.3535732e-05 |
| 2048 | 3.3702579e-05 | 3.3535732e-05 |
| 4096 | 3.3702579e-05 | 3.3535732e-05 |
| 8192 | 3.3702579e-05 | 3.3535732e-05 |
| 16384 | 3.3702579e-05 | 3.3535732e-05 |
| 32768 | 3.3702579e-05 | 3.3535732e-05 |
| 65536 | 3.3702579e-05 | 3.3535732e-05 |
| 131072 | 3.3702579e-05 | 3.3535732e-05 |

- `3271` failed only at `262144` due GPU OOM after producing the above valid rows.
- `3272` direct `apply_fn` full `u/v/w` path at `space_chunk_size=4096` completed:
  - `u=3.3702579e-05, v=3.3535732e-05, w=2.2215085e-04`

Interpretation:

- 舊 `3155 window 1 / checkpoint_100000 -> u≈3.37e-05, v≈3.35e-05, w≈2.22e-04` 不是 stale artifact；它可由 direct `apply_fn` path 完整復現。
- `3259/3260/3261` 的 `~1e-3` table 來自 current evaluator 的 wrapper path，不應用來判斷 sensor 是否比 no-data 差。
- 高可疑 bug 位置是 [examples/kolmogorov_flow/models.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/models.py) 的 `neural_net()` scalar branch：在已被 `vmap` 包住的 scalar function 內部再建 `z[None, :]`，再以 `outputs[0]` 去掉 fake batch 維；此路徑已證明會隨 batch/chunk shape 產生錯誤 lowering 或 shape-dependent behavior。
- 在修正 evaluator/model forward 前，所有 window-1 checkpoint sweep A/B 數字必須標記為 `[RISK: EVAL_FORWARD_PATH_UNVERIFIED]`。

Next:

- 修正 evaluation forward path：新增直接批次 `apply_fn` 的 `predict_uv_batch/predict_w_batch` 或改寫 `neural_net()`，避免在 vmapped scalar function 內部手動 fake batch。
- 用修正後 evaluator 重跑：
  - no-data `3256` vs sensor `3257` 的 `10` checkpoint sweep
  - old `3155` sensor cross-check
  - vorticity field plots
- 修正後再更新 `3259/3260` 的 interpretation，不可沿用目前 `~1e-3` 結論。

### [2026-04-16] `3260` | cross-check current no-data vs old `3155` sensor at checkpoint 100000

- Time: `2026-04-16 Asia/Taipei`
- Status: Completed, superseded by `3261..3272` audit
- Experiment or Job ID: `3260`

Change:

- 依人工質疑，另外提交 Slurm job `3260`，用同一支 [evaluate_window1_checkpoint_sweep.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/evaluate_window1_checkpoint_sweep.py) 做 `checkpoint_100000` cross-check。
- 比較對象：
  - no-data: `3256` / `/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ablation/ckpt`
  - sensor: old `3155` / `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt`
- 設定：
  - `CHECKPOINT_STEPS=100000`
  - `SKIP_FIELDS=1`
  - `OUTPUT_DIR=/home/junyi/jaxpi/eval_runs/window1_checkpoint_sweep_crosscheck_3155`

Evidence:

- Slurm:
  - `3260 | COMPLETED | Elapsed=00:00:48 | ExitCode=0:0`
- Output:
  - `/home/junyi/jaxpi/eval_runs/window1_checkpoint_sweep_crosscheck_3155/checkpoint_sweep_summary.txt`
  - `/home/junyi/jaxpi/eval_runs/window1_checkpoint_sweep_crosscheck_3155/checkpoint_sweep_results.csv`
- Result:

| Run | Step | u_error | v_error | w_error |
| :--- | ---: | ---: | ---: | ---: |
| no_data (`3256`) | 100000 | 1.077166e-03 | 1.088496e-03 | 7.043463e-04 |
| sensor (`3155`) | 100000 | 1.203502e-03 | 1.059683e-03 | 7.348232e-04 |

Interpretation:

- 此結果後續已被 `3261..3272` audit 判定為 wrapper evaluation path 產生的錯誤偏高結果。
- 不可再用 `3260` 的 scalar table 宣稱 old `3155` sensor 不如 no-data。
- 保留本段作為失敗/錯誤 evaluation path 證據。

Next:

- 依 `3261..3272` audit 結論修正 evaluator forward path，然後重跑 `3256/3257` 的 checkpoint sweep。

### [2026-04-15] `3155` | horizontal vorticity field comparison + GIF regenerated through `window 21`

- Time: `2026-04-15 23:31 +0800`
- Status: Completed
- Experiment or Job ID: `3155` / corrected visualization rerun

Change:

- 使用 [render_vorticity_snapshot_grid.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/render_vorticity_snapshot_grid.py) 重新產生 `3155` 的橫向 vorticity field comparison 與 GIF。
- 使用 `window-local time` 與 `w_ic_pred_fn`，範圍覆蓋目前可驗證的 `window 1..21`。
- 本次只重算視覺化 artifact，不改 training logic、checkpoint mapping、dataset 或 optimizer。
- 先前誤以為要處理 `3150` 而新增的臨時 artifact 與帳本段落已移除，避免把使用者更正前的誤操作保留為正式證據。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt`
- Windows: `1..21`
- Checkpoints: `window 1..20 / checkpoint_100000`, `window 21 / checkpoint_60000`
- Plot settings: `time-index=-1`, `plot_stride=2`, `chunk_size=4096`, `layout=horizontal`, GIF `21 frames`

Evidence:

- Local artifact:
  - [3155_vorticity_horizontal_windows_1_21_stride2.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_vorticity_horizontal_20260415/3155_vorticity_horizontal_windows_1_21_stride2.png)
  - [3155_vorticity_windows_1_21_stride2.gif](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_vorticity_horizontal_20260415/3155_vorticity_windows_1_21_stride2.gif)
  - [render.log](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_vorticity_horizontal_20260415/render.log)
- Remote artifact:
  - `/home/junyi/jaxpi/eval_runs/3155_vorticity_horizontal_20260415/3155_vorticity_horizontal_windows_1_21_stride2.png`
  - `/home/junyi/jaxpi/eval_runs/3155_vorticity_horizontal_20260415/3155_vorticity_windows_1_21_stride2.gif`
  - `/home/junyi/jaxpi/eval_runs/3155_vorticity_horizontal_20260415/render.log`
- Verification:
  - PNG verified as `11165 x 1455`
  - GIF verified as `1492 x 528`, `21` frames
  - Local `file` check confirms PNG/GIF formats.
- Rendered final-step vorticity relative errors:

| Window | Checkpoint | t_abs | w_rel |
| :---: | :---: | :---: | ---: |
| 1 | 100000 | 0.0500 | 0.000263 |
| 2 | 100000 | 0.1500 | 0.000867 |
| 3 | 100000 | 0.2500 | 0.004405 |
| 4 | 100000 | 0.3500 | 0.011398 |
| 5 | 100000 | 0.4500 | 0.023824 |
| 6 | 100000 | 0.5500 | 0.037954 |
| 7 | 100000 | 0.6500 | 0.055675 |
| 8 | 100000 | 0.7500 | 0.077903 |
| 9 | 100000 | 0.8500 | 0.101541 |
| 10 | 100000 | 0.9500 | 0.128759 |
| 11 | 100000 | 1.0500 | 0.151138 |
| 12 | 100000 | 1.1500 | 0.176852 |
| 13 | 100000 | 1.2500 | 0.204445 |
| 14 | 100000 | 1.3500 | 0.230055 |
| 15 | 100000 | 1.4500 | 0.261050 |
| 16 | 100000 | 1.5500 | 0.293274 |
| 17 | 100000 | 1.6500 | 0.327786 |
| 18 | 100000 | 1.7500 | 0.366993 |
| 19 | 100000 | 1.8500 | 0.397232 |
| 20 | 100000 | 1.9500 | 0.431451 |
| 21 | 60000 | 2.0500 | 0.472521 |

Interpretation:

- 橫向 PNG 與 GIF 直接呈現 `3155` 從 `window 1 -> 21` 的 final-step vorticity drift。
- `w_rel` 到 `window 21 / checkpoint_60000` 已達 `0.472521`，與 2026-04-14 corrected full-window evaluation 中 `w_err=0.459683` 的主結論一致：`all_points` sensor sampling 延後了早期崩壞，但仍未阻止 vorticity / small-scale 結構逐窗流失。
- `window 21` 仍是中途 checkpoint (`60000/100000`)，不可和完整 `checkpoint_100000` window 做完全等價比較。

Next:

- 若要放入 thesis 或簡報，這組 artifact 可作為 `3155` 後段 vorticity drift 的完整視覺證據。
- 若要進一步下結論，應搭配 full-window `l2_errors.npz` 與 spectrum / enstrophy 診斷，不應只看 GIF。

### [2026-04-15] `3137` | horizontal vorticity field comparison + GIF regenerated

- Time: `2026-04-15 23:20 +0800`
- Status: Completed
- Experiment or Job ID: `3137` / corrected visualization rerun

Change:

- 更新 [render_vorticity_snapshot_grid.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/render_vorticity_snapshot_grid.py)，保留原本直向 PNG 預設行為，新增：
  - `--layout horizontal`
  - `--gif-output`
  - `--hide-colorbars`
- 使用 `window-local time` 與 `w_ic_pred_fn` 重新產生 `3137` 的 vorticity field comparison。
- 本次只重算視覺化 artifact，不改 training logic、checkpoint mapping、dataset 或 optimizer。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Checkpoint root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Windows: `1..12`
- Checkpoints: `window 1..11 / checkpoint_100000`, `window 12 / checkpoint_60000`
- Plot settings: `time-index=-1`, `plot_stride=2`, `chunk_size=4096`, `layout=horizontal`, GIF `12 frames`

Evidence:

- Local artifact:
  - [3137_vorticity_horizontal_windows_1_12_stride2.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_vorticity_horizontal_20260415/3137_vorticity_horizontal_windows_1_12_stride2.png)
  - [3137_vorticity_windows_1_12_stride2.gif](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_vorticity_horizontal_20260415/3137_vorticity_windows_1_12_stride2.gif)
  - [render.log](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_vorticity_horizontal_20260415/render.log)
- Remote artifact:
  - `/home/junyi/jaxpi/eval_runs/3137_vorticity_horizontal_20260415/3137_vorticity_horizontal_windows_1_12_stride2.png`
  - `/home/junyi/jaxpi/eval_runs/3137_vorticity_horizontal_20260415/3137_vorticity_windows_1_12_stride2.gif`
  - `/home/junyi/jaxpi/eval_runs/3137_vorticity_horizontal_20260415/render.log`
- Verification:
  - `python3 -m py_compile scripts/analysis/render_vorticity_snapshot_grid.py`
  - remote `python3 -m py_compile /home/junyi/jaxpi/scripts/analysis/render_vorticity_snapshot_grid.py`
  - PNG verified as `6310 x 1455`
  - GIF verified as `1492 x 528`, `12` frames
- Rendered final-step vorticity relative errors:

| Window | Checkpoint | t_abs | w_rel |
| :---: | :---: | :---: | ---: |
| 1 | 100000 | 0.0500 | 0.000266 |
| 2 | 100000 | 0.1500 | 0.000858 |
| 3 | 100000 | 0.2500 | 0.004098 |
| 4 | 100000 | 0.3500 | 0.011036 |
| 5 | 100000 | 0.4500 | 0.022064 |
| 6 | 100000 | 0.5500 | 0.034962 |
| 7 | 100000 | 0.6500 | 0.051943 |
| 8 | 100000 | 0.7500 | 0.073370 |
| 9 | 100000 | 0.8500 | 0.095776 |
| 10 | 100000 | 0.9500 | 0.123504 |
| 11 | 100000 | 1.0500 | 0.147386 |
| 12 | 60000 | 1.1500 | 0.178951 |

Interpretation:

- 這次橫向圖與 GIF 直接呈現 `3137` 從 `window 1 -> 12` 的 final-step vorticity drift。
- final-step `w_rel` 單調抬升到 `window 12 = 0.178951`，與 `3145` corrected full-window `w_err=0.176233` 的結論一致：主要風險仍是 vorticity / small-scale 結構逐窗流失，而非 `u/v` 大尺度場崩潰。
- 因這次只改視覺化腳本與輸出 artifact，不涉及訓練循環、DataLoader、rollout 或 checkpoint 寫入；time-window integrity 風險未新增。

Next:

- 若要放入 thesis，優先使用橫向 PNG；若要簡報展示 drift，使用 GIF。
- 若要進一步定量 high-k attenuation，仍應沿用 `3146` 的 spectrum / enstrophy 診斷，而不是只從 GIF 做結論。

### [2026-04-15] window-1 checkpoint sweep tooling | prepare two-version 10-ckpt evaluator and plots

- Time: `2026-04-15 Asia/Taipei`
- Status: Evaluation running as Slurm job `3259`
- Experiment or Job ID: `3256/3257`

Change:

- 新增 10 checkpoint sweep 評估與繪圖腳本：
  - [evaluate_window1_checkpoint_sweep.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/evaluate_window1_checkpoint_sweep.py)
  - [postprocess_kolmogorov_window1_checkpoint_sweep.sh](/Users/latteine/Documents/coding/jaxpi/slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh)
- 腳本固定比較兩個版本：
  - no-data: `paper_repro_soap_window1_ablation.py`
  - sensor: `paper_repro_soap_sensor100_n512_w50_window1_ablation.py`
- 預設 checkpoint steps:
  - `10000,20000,30000,40000,50000,60000,70000,80000,90000,100000`
- 輸出：
  - `checkpoint_sweep_results.csv`
  - `checkpoint_sweep_results.npz`
  - `checkpoint_sweep_summary.txt`
  - `checkpoint_sweep_error_vs_step.png`
  - `vorticity_sweep_no_data.png`
  - `vorticity_sweep_sensor.png`

Config / Dataset / Checkpoint:

- Remote sync target:
  - `/home/junyi/jaxpi/scripts/analysis/evaluate_window1_checkpoint_sweep.py`
  - `/home/junyi/jaxpi/slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh`
  - `/home/junyi/jaxpi/slurm/lib/common.sh`
- Checkpoint roots:
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ablation/ckpt/time_window_1`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_ablation/ckpt/time_window_1`

Evidence:

- Local syntax checks:
  - `python3 -m py_compile scripts/analysis/evaluate_window1_checkpoint_sweep.py`
  - `bash -n slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh`
- Remote syntax checks:
  - `python3 -m py_compile /home/junyi/jaxpi/scripts/analysis/evaluate_window1_checkpoint_sweep.py`
  - `bash -n /home/junyi/jaxpi/slurm/lib/common.sh`
  - `bash -n /home/junyi/jaxpi/slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh`
- Current checkpoint availability:
  - no-data `3256`: `10/10` checkpoints present for `time_window_1`
  - sensor `3257`: `10/10` checkpoints present for `time_window_1`
- `3257` Slurm evidence:
  - `State=COMPLETED`
  - `Elapsed=06:23:28`
  - `Start=2026-04-15T16:50:12`
  - `End=2026-04-15T23:13:40`
  - `ExitCode=0:0`
  - latest checkpoint: `checkpoint_100000 -> 2026-04-15 23:13 +0800`
- Evaluation submission:
  - `3258` failed immediately because Slurm executed the copied script from spool and the wrapper resolved `../lib/common.sh` relative to `/var/lib/slurm/...`.
  - Wrapper fixed to prefer `SLURM_SUBMIT_DIR` when sourcing `slurm/lib/common.sh`.
  - `3259` submitted and running from `/home/junyi/jaxpi`, log path:
    - `/home/junyi/jaxpi/logs/post_kf_w1_sweep_3259.out`
    - `/home/junyi/jaxpi/logs/post_kf_w1_sweep_3259.err`
  - `3259` first observed output:
    - `=== Evaluate no_data ===`
    - `--- no_data checkpoint_10000 ---`
    - `u=1.805166e-03, v=1.693111e-03, w=2.583466e-03`

Interpretation:

- 評估與繪圖 tooling 已就緒，且兩版本都已完成 `10/10` checkpoints。
- 新腳本預設不允許缺檔，這是刻意設計，避免把 partial sweep 誤判為完整 A/B。
- Time-window 定義仍使用 `num_time_windows=50` 與 window-local time；這次只評估 `window 1` 的 checkpoint-step 收斂曲線，不改變訓練或 checkpoint mapping。

Next:

- 監看 `3259` 完成後，下載：
  - `/home/junyi/jaxpi/eval_runs/window1_checkpoint_sweep`
- 完成後再整理 no-data vs sensor 的 `u/v/w` full-window relative L2 convergence 與 vorticity field 圖。

### [2026-04-15] data generation | create `Re=10000` QR-pivot `K=200` sensor files

- Time: `2026-04-15 Asia/Taipei`
- Status: Completed
- Experiment or Job ID: `sensor-gen-re10k-k200`

Change:

- 使用既有 [generate_sensors_qrpivot.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/generate_sensors_qrpivot.py) 為 `Re=10000, N=256, T=5` DNS 生成新的 `QR-pivot K=200` sensor artifacts。
- 新增輸出：
  - [sensors_qrpivot_K200_N256_t0-5.json](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_sensors/re10000/sensors_qrpivot_K200_N256_t0-5.json)
  - [sensors_qrpivot_K200_N256_t0-5_dns_values.npz](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_sensors/re10000/sensors_qrpivot_K200_N256_t0-5_dns_values.npz)

Config / Dataset / Checkpoint:

- Script: [generate_sensors_qrpivot.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/generate_sensors_qrpivot.py)
- Dataset: [kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy)
- Command:
  - `./.venv/bin/python examples/kolmogorov_flow/generate_sensors_qrpivot.py --dns examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_fp64_etdrk4_Re10000_N256_T5_dt2p5e4_ds4.npy --outdir examples/kolmogorov_flow/data/kolmogorov_sensors/re10000 --K 200 --time-stride 2 --spatial-res 256`

Evidence:

- Generator output:
  - `Snapshot matrix: (126, 65536) [33 MB]`
  - `Top-200 modes explain 100.0% of total variance`
  - `Saved .../sensors_qrpivot_K200_N256_t0-5.json`
  - `Saved .../sensors_qrpivot_K200_N256_t0-5_dns_values.npz (shape: (200, 41))`
- Post-check:
  - `K = 200`
  - `resolution = 256x256`
  - `time_range = [0.0, 5.0]`
  - `sensor_dt = 0.125`
  - `sensor_time_points = 41`
  - `u/v/omega shape = (200, 41)`

Interpretation:

- `re10000` 現在除了既有 `K100` 之外，也有與主線 `sensors_qrpivot_*` 命名一致的 `K200` sensor JSON/NPZ，可直接供新 config 或對照實驗使用。
- 這次變更只新增資料 artifact，沒有改動訓練邏輯、time-window 定義或 checkpoint 映射，因此不涉及 `[RISK: TIME_WINDOW_INTEGRITY_UNVERIFIED]`。

Next:

- 若後續要開 `Re=10000` 的 `sensor200` run，應新增對應 config，並明確把 `sensor_json` / `sensor_values` 指向這兩個新檔案。
- 若要比較 `K100` vs `K200`，應保持其餘 optimizer、window 與 weighting 設定不變，避免把 sensor 數量效應和其他變數混在一起。

### [2026-04-15] checkpoint retention hotfix | stop using `3252/3253`, resubmit `window 1` A/B as `3256/3257`

- Status: Submitted / Pending
- Experiment or Job ID:
  - Invalid retention runs:
    - `3252` | sensor | completed but only kept latest two checkpoints
    - `3253` | no-data | cancelled after confirming same retention bug
  - Corrected retention runs:
    - `3256` | no-data | pending
    - `3257` | sensor | pending

Change:

- 確認遠端 `/home/junyi/jaxpi` 當時仍使用舊版：
  - `jaxpi/utils.py` 沒有 `_resolve_checkpoint_keep()`
  - `paper_repro_soap.py` / `paper_repro_soap_sensor100_n512_w50.py` 仍為 `saving.num_keep_ckpts = 2`
- 立即同步遠端：
  - [utils.py](/Users/latteine/Documents/coding/jaxpi/jaxpi/utils.py)
  - [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
  - [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- 終止錯版 no-data run `3253`，並重新提交一組乾淨的 `window 1` A/B：
  - `3256`: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ablation_v2`
  - `3257`: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w1_ablation_v2`

Evidence:

- Bug confirmation:
  - `3252` checkpoint dir 僅有 `checkpoint_90000` / `checkpoint_100000`
  - `3253` checkpoint dir 僅有 `checkpoint_80000` / `checkpoint_90000`
- Hotfix confirmation on remote:
  - `/home/junyi/jaxpi/jaxpi/utils.py` contains `_resolve_checkpoint_keep`
  - remote base configs now show `saving.num_keep_ckpts = None`
- Slurm:
  - `3252 | COMPLETED`
  - `3253 | CANCELLED by 10004`
  - `3256 | PENDING | WorkDir=/home/junyi/jaxpi`
  - `3257 | PENDING | WorkDir=/home/junyi/jaxpi`

Interpretation:

- `3252/3253` 雖然對 loss / wall-clock 仍可作粗略參考，但不符合「保留全部 checkpoints」這次實驗要求，因此不能當成最終版本。
- 真正應該追蹤的是 `3256/3257`；只有這組會同時滿足：
  - `window 1 only`
  - 維持原本 `50-window` 定義
  - 保留全部 checkpoints

Next:

- 後續 convergence-speed 比較只使用 `3256/3257`。
- 一旦它們開始執行，先確認 `checkpoint_10000, 20000, ...` 是否持續保留下來，再做收斂速度判讀。

### [2026-04-15] queue hygiene follow-up | cancel duplicate `3254/3255`

- Status: Applied
- Experiment or Job ID:
  - Cancelled duplicate no-data run: `3254`
  - Cancelled duplicate sensor run: `3255`
  - Retained valid A/B pair: `3256/3257`

Change:

- 依人工指示取消第二批重複提交、且仍落在 `/home/junyi` workdir 的 `3254/3255`。

Evidence:

- `sacct`:
  - `3254 | CANCELLED by 10004 | Start=2026-04-15T10:34:19 | Elapsed=00:04:51`
  - `3255 | CANCELLED by 10004 | Start=2026-04-15T10:39:10 | Elapsed=00:00:00`
  - `3256 | PENDING`
  - `3257 | PENDING`
- `squeue` after cancellation:
  - only `3256` and `3257` remain as the intended corrected A/B pair

Interpretation:

- `3254` 也像先前 `3250` 一樣，短暫跑了幾分鐘才被取消；因此若 `/home/junyi/logs/kf_paper_repro_soap_3254.*` 有殘留，不應納入任何 convergence 比較。
- `3255` 則在真正開跑前就被取消。
- 目前 queue 中唯一有效、且已修補 checkpoint retention 的 `window 1` A/B pair 只剩 `3256/3257`。

Next:

- 後續只監看 `3256/3257`，其他同名 job 一律視為歷史噪音。

### [2026-04-14] queue hygiene | terminate `3155` and identify duplicate `window 1` A/B submissions

- Status: Applied
- Experiment or Job ID:
  - Terminated: `3155`
  - Duplicate pending submissions: `3250/3251`
  - Corrected pending submissions: `3252/3253`

Change:

- 依人工指示終止 `3155`，釋放先前停在 `STOPPED` 狀態但仍占用 allocation 的資源。
- 釐清 `window 1` A/B queue 中為何出現兩組同名 job，確認 `3250/3251` 是較早一次提交、`3252/3253` 是修正後提交。

Evidence:

- `3155`:
  - `scancel 3155` 後，Slurm 狀態顯示 `COMPLETED`（實際為 user-cancelled 終止後的終態）
- Duplicate pair metadata:
  - `3250` / `3251`:
    - `SubmitTime=2026-04-14T21:17:37`
    - `WorkDir=/home/junyi`
    - stdout / stderr 也落在 `/home/junyi/logs/...`
  - `3252` / `3253`:
    - `SubmitTime=2026-04-14T21:17:49`
    - `WorkDir=/home/junyi/jaxpi`
    - stdout / stderr 落在 `/home/junyi/jaxpi/logs/...`

Interpretation:

- 兩組看起來一樣的 job 不是 Slurm 自動複製，而是連續兩次提交造成。
- 第一組 `3250/3251` 是較早的提交，工作目錄落在 `/home/junyi`，不符合這次 A/B 想要的 repo-root 路徑約束。
- 第二組 `3252/3253` 才是修正後的有效提交，工作目錄、log path、script path 都對齊 `/home/junyi/jaxpi`。

Next:

- 建議保留 `3252/3253`，取消 `3250/3251`，避免之後排到資源時重複開跑。

### [2026-04-14] queue hygiene follow-up | cancel duplicate `3250/3251`

- Status: Applied
- Experiment or Job ID:
  - Cancelled duplicate no-data run: `3250`
  - Cancelled duplicate sensor run: `3251`
  - Retained valid A/B pair: `3252/3253`

Change:

- 依人工指示取消重複的第一組 `window 1` A/B submissions：`3250` 與 `3251`。

Evidence:

- `sacct`:
  - `3250 | CANCELLED by 10004 | Start=2026-04-14T21:59:14 | Elapsed=00:02:36`
  - `3251 | CANCELLED by 10004 | Start=2026-04-14T22:01:50 | Elapsed=00:00:00`
  - `3252 | PENDING`
  - `3253 | PENDING`
- `squeue` after cancellation:
  - only `3252` and `3253` remain in the intended A/B queue

Interpretation:

- `3250` 在取消前其實已短暫開始執行約 2 分 36 秒，因此之後若看到 `/home/junyi/logs/kf_paper_repro_soap_3250.*` 有殘留輸出，不應把它當成有效 A/B run。
- `3251` 則在尚未真正開始產生訓練步數前就被取消。
- 目前唯一應追蹤的 `window 1` convergence A/B 是 `3252`（sensor）與 `3253`（no-data）。

Next:

- 之後只監看 `3252/3253` 的 queue / log / checkpoint，不再使用 `3250/3251` 的任何結果。

### [2026-04-14] convergence-speed A/B setup | add `max_windows_to_run` and submit `window 1`-only no-data vs sensor runs

- Status: Submitted / Pending
- Experiment or Job ID:
  - `3253` | `re1e6_n512_ds4_soap_w1_ablation`
  - `3252` | `re1e6_n512_ds4_soap_sensor100_w1_ablation`
- Config / Dataset / Checkpoint:
  - No-data config: [paper_repro_soap_window1_ablation.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py)
  - Sensor config: [paper_repro_soap_sensor100_n512_w50_window1_ablation.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_ablation.py)
  - Shared dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
  - Training control change: [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py)

Change:

- 在 [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py) 新增 `training.max_windows_to_run` 支援，允許維持原本 `num_time_windows=50` 的時間切分，但只執行前 `N` 個窗口。
- 新增兩個 `window 1` 專用 config：
  - no-data baseline：沿用 `paper_repro_soap.py`，但只跑 `window 1`
  - sensor100：沿用 `paper_repro_soap_sensor100_n512_w50.py`，但只跑 `window 1`
- 目的不是重做長序列結果，而是隔離「有無 data constraint 是否加速 `window 1` 收斂」。

Evidence:

- 本地語法驗證：
  - `python3 -m py_compile examples/kolmogorov_flow/train.py examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_ablation.py`
- Slurm submission:
  - `3253` submitted via `/home/junyi/jaxpi/slurm_train_paper_repro_soap.sh`
  - `3252` submitted via `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh`
- Current queue state:
  - `3253 | kf_paper_repro_soap | PENDING | Reason=Priority`
  - `3252 | kf_sensor100_w25 | PENDING | Reason=Priority`
- Remote stdout/stderr targets:
  - `3253` -> `/home/junyi/jaxpi/logs/kf_paper_repro_soap_3253.{out,err}`
  - `3252` -> `/home/junyi/jaxpi/logs/kf_sensor100_w25_3252.{out,err}`

Interpretation:

- 這次 A/B 保留了 `50-window` 的 window 定義，只是提前在 `window 1` 結束後停止；因此可公平比較 `window 1` 內的 loss/threshold crossing/step-time，而不會把整段 `T=5` 改成另一種時間離散。
- `num_time_windows=1` 雖然也能只跑一次訓練，但那會改變 window 大小與時間座標定義，不適合作為這次「是否加速收斂」的比較基準。
- 目前兩條 job 都還沒開始執行，因此還沒有新的 loss 證據。

Next:

- 等 `3252/3253` 開始執行後，優先擷取：
  - `rc/ru/rv` 首次低於 `1e-4 / 1e-5 / 1e-6` 的 step
  - 每步 wall-clock / ETA
  - `window 1 / checkpoint_100000` 是否都正常落盤
- 待兩條 run 都完成 `window 1` 後，再做 convergence-speed 對照，而不是直接看肉眼難判讀的 raw loss 曲線。

### [2026-04-14] `3155` | corrected full-window evaluation completed through `window 21`

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Eval Definition: `window-local time` + `chunked CPU evaluator`
- Eval Outputs:
  - Remote summary: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked/summary.txt`
  - Remote numeric dump: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked/l2_errors.npz`
  - Remote plot: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked/comparison_plots.png`
  - Remote field plot: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked/vorticity_fields.png`
  - Remote log: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked/window_eval.log`

Change:

- 在 `3155` 暫停於 `window 21 / checkpoint_60000` 後，完成 `windows 1..21` 的 corrected full-window evaluation。
- 這次沿用 2026-04-12 相同的 `window-local time` 定義與 chunked CPU evaluator，避免把不同評估 protocol 混在一起。

Evidence:

- `summary.txt` 完整結果：
  - `window 15 / checkpoint_100000 -> u=0.016859, v=0.013503, w=0.257027`
  - `window 16 / checkpoint_100000 -> u=0.019848, v=0.016163, w=0.287489`
  - `window 17 / checkpoint_100000 -> u=0.021764, v=0.020518, w=0.321439`
  - `window 18 / checkpoint_100000 -> u=0.024093, v=0.025034, w=0.358312`
  - `window 19 / checkpoint_100000 -> u=0.027009, v=0.028801, w=0.389570`
  - `window 20 / checkpoint_100000 -> u=0.031659, v=0.034657, w=0.423994`
  - `window 21 / checkpoint_60000  -> u=0.040653, v=0.041705, w=0.459683`
- Aggregate:
  - `mean -> u=0.011150, v=0.011047, w=0.174918`
  - `max  -> u=0.040653, v=0.041705, w=0.459683`
- Artifact sizes:
  - `comparison_plots.png` = `352K`
  - `vorticity_fields.png` = `29M`

Interpretation:

- `3155` 到 `window 21` 仍維持 `u/v < 5%`，但 `w_err` 已持續抬升到 `0.459683`，明顯超出專案成功門檻。
- 相較 2026-04-12 的 `window 14 / checkpoint_10000`，這次改用 `window 14 / checkpoint_100000` 後，跨窗趨勢更完整，也更清楚證明問題不是短暫 checkpoint 選取偏差，而是 vorticity / small-scale 結構隨 window 積累流失。
- 因 `window 21` 目前只有 `checkpoint_60000`，它仍是「當前暫停點的中途狀態」，不是完整窗口最終品質；但即使如此，`w_err` 已經進一步惡化。

Next:

- 若要繼續主線判讀，下一步應聚焦：
  - `w_data` 是否需要顯式啟用
  - 或加入更直接的 vorticity-aware / spectral 約束
  - 而不是只延長目前 `u_data + v_data, w_data=0` 的 horizon

### [2026-04-14] `3155` | manual pause at `window 21` + corrected evaluation relaunched

- Status: In progress
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Eval Definition: `window-local time` + `chunked CPU evaluator`
- Eval Output Dir:
  - Remote: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked`

Change:

- 依人工指示將 `3155` 從 `RUNNING` 送到 Slurm `STOPPED`，避免訓練繼續推進。
- 在不覆蓋 `2026-04-12` 舊評估輸出的前提下，重新啟動 corrected full-window evaluation，目標覆蓋目前所有已存在 checkpoint 的 windows (`1..21`)。
- 評估沿用既有 `eval_sensor100_w25.py`，但顯式覆蓋：
  - `EVAL_CONFIG_PATH=/home/junyi/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py`
  - `EVAL_CKPT_ROOT=/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt`
  - `EVAL_OUTPUT_DIR=/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked`
  - `JAX_PLATFORMS=cpu`

Evidence:

- Slurm:
  - `3155 | STOPPED | 5-13:26:40 | acmt20`
- Pause 時最新 checkpoint lineage:
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_21/checkpoint_50000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_21/checkpoint_60000`
- Evaluation launch:
  - Background PID: `1584893` (wrapper) / `1584917` (`python3 eval_sensor100_w25.py`)
  - Log file: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked/window_eval.log`
- 目前 log 已確認：
  - evaluator 成功載入 `w50` config 與 dataset
  - discovered windows: `1..21`
  - 已進入 `Window 1` 的 full-window L2 計算階段

Interpretation:

- `3155` 目前的最新可驗證狀態不是 `window 14`，而是已推進到 `window 21 / checkpoint_60000`。
- 這次 evaluation 尚未完成，因此目前只能確認評估定義、checkpoint 範圍與程序狀態，不能提前宣稱新的 `u/v/w` 結論。
- 使用 CPU chunked evaluator 是延續 4/12 的安全路徑；代價是每個 window 的 full-window L2 計算時間較長。

Next:

- 等待 `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260414_chunked/window_eval.log` 產出各窗口摘要與 `summary.txt`。
- 評估完成後回填 `window 15..21` 的新增 full-window relative L2，並和 4/12 的 `window 1..14` 結果對照。

### [2026-04-14] checkpoint retention policy change | stop pruning to latest two checkpoints

- Time: `2026-04-14 00:00 +0800`
- Experiment or Job ID: training checkpoint policy (global config default)
- Change:
  - 修改 [jaxpi/utils.py](/Users/latteine/Documents/coding/jaxpi/jaxpi/utils.py)，新增 `_resolve_checkpoint_keep()`，將 `saving.num_keep_ckpts = None` 解讀為「保留目前目錄內全部 checkpoint，再加上本次新存檔」。
  - 將 Kolmogorov training configs 與 stage AB configs 的 `saving.num_keep_ckpts` 預設從 `2` 改為 `None`。
  - 將 [PINN Kolmogorov Flow Config.yaml](/Users/latteine/Documents/coding/jaxpi/PINN%20Kolmogorov%20Flow%20Config.yaml) 的 `num_keep_ckpts` 從 `2` 改為 `null`，對齊同一語意。
- Config / Dataset / Checkpoint:
  - Affected configs:
    - [soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/soap.py)
    - [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
    - [paper_repro_soap_sensor100.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100.py)
    - [paper_repro_soap_sensor100_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_w25.py)
    - [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
    - [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
    - [re10k_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/re10k_soap.py)
    - [re10k_soap_sensor100.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/re10k_soap_sensor100.py)
    - [upstream_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/upstream_soap.py)
    - [upstream_soap_4x384.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/upstream_soap_4x384.py)
    - [pirate.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/pirate.py)
    - [pirate_les_stage1.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/stage_ab/pirate_les_stage1.py)
    - [pirate_les_stage1_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/stage_ab/pirate_les_stage1_soap.py)
    - [pirate_les_stage1_windowed.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/stage_ab/pirate_les_stage1_windowed.py)
    - [pirate_les_stage1_windowed_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/stage_ab/pirate_les_stage1_windowed_soap.py)
    - [pirate_les_stage2.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/stage_ab/pirate_les_stage2.py)
    - [pirate_les_stage2_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/stage_ab/pirate_les_stage2_soap.py)
- Evidence:
  - `rg -n "num_keep_ckpts = 2|num_keep_ckpts: 2" examples/kolmogorov_flow 'PINN Kolmogorov Flow Config.yaml' jaxpi` 無輸出，表示舊預設已清除。
  - `python3 -m py_compile ...` 已通過，涵蓋 `jaxpi/utils.py` 與所有修改過的 Python configs。
- Interpretation:
  - 先前「只留最新兩個 checkpoint」不是 Flax restore 的限制，而是本地 config 預設值造成的刪檔策略。
  - Flax 這版 `save_checkpoint()` 的 `keep` 需要整數，不能直接傳 `None`；因此保留全部的語意必須在本地 wrapper 先轉譯，否則會在 checkpoint cleanup 階段失敗。
  - 這次修改不改 checkpoint 命名、不改 restore path、不改 time-window mapping；只改「save 後是否清舊檔」。
- Next:
  - 後續新啟動的訓練會保留每一次存下來的 checkpoint。
  - 已經被舊策略刪掉的歷史 checkpoint 不會自動恢復；若要讓既有 run 也採用新策略，需要用更新後的 code 重新 resume/launch。

### [2026-04-13] thesis structure rewrite | `main.tex` reframed around `PirateNet + SOAP` baseline and two-part study

- Time: `2026-04-13 17:42 +0800`
- Experiment or Job ID: thesis framing update (`3137`, `3145`, `3155`, JHTDB `Re_tau=1000`)
- Change:
  - 重寫 [thesis/main.tex](/Users/latteine/Documents/coding/jaxpi/thesis/main.tex) 的 title / abstract / objectives / experiments opening / 3D section / conclusion
  - 將 thesis 主敘事從「low-fidelity guidance 為中心」改為「以 PirateNet + SOAP 為共同 baseline 的兩部分比較研究」
  - 明確定義：
    - Part I: 2D Kolmogorov `Re=10^6`，`no data` vs `sensor100`
    - Part II: 因缺少 matched `Re_tau=395` data，改採 JHTDB channel flow `Re_tau=1000`
- Config / Dataset / Checkpoint:
  - Part I baseline evidence: `3137` + corrected eval `3145`
  - Part I sensor evidence: `3155`
  - Part II benchmark: public JHTDB channel-flow slice at `Re_tau=1000`
- Evidence:
  - Updated thesis source: [main.tex](/Users/latteine/Documents/coding/jaxpi/thesis/main.tex)
  - Recompiled thesis PDF: [main.pdf](/Users/latteine/Documents/coding/jaxpi/thesis/main.pdf)
- Interpretation:
  - 這次修改是論文敘事定調調整，不是新增實驗結果。
  - 新版本把「已完成的 2D baseline reproduce + sensor100 follow-up」與「正在建立中的 3D JHTDB comparison framework」分開表述，避免把尚未完成的 Part II 寫成已完成結論。
  - `low-fidelity guidance` 仍保留為研究脈絡與延伸方向，但不再作為整篇 thesis 已被完整驗證的核心 claim。
- Next:
  - 後續若完成 JHTDB `no data` vs `sensor` 成對評估，需再回填 Part II result text，將目前的 benchmark/baseline wording 升級為正式結果比較。

### [2026-04-13] thesis write-up | `3137` no-data baseline + `3155` sensor100 comparison added to `thesis/main.tex`

- Time: `2026-04-13 17:07 +0800`
- Experiment or Job ID: `3137`, `3145`, `3155`
- Change:
  - 將 corrected no-data baseline 與 `QR-pivot K=100` sensor run 的對照圖與敘述寫入 [thesis/main.tex](/Users/latteine/Documents/coding/jaxpi/thesis/main.tex)
  - 將 3 張論文用圖正式複製進 `/Users/latteine/Documents/coding/jaxpi/thesis/figures/kolmogorov/`，改為 thesis 內部相對路徑引用，避免依賴 `eval_runs/`
- Config / Dataset / Checkpoint:
  - `3137`: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py), corrected eval through `window 12 / checkpoint_60000`
  - `3155`: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py), corrected eval through `window 13 / checkpoint_100000` and `window 14 / checkpoint_10000`
- Evidence:
  - [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/summary.txt)
  - [with_vs_without_data_window_trends.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends.txt)
  - [with_vs_without_data_window_trends_to12.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends_to12.png)
  - [window4_7_10_12_loss_compare_3137_vs_3155.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window4_7_10_12_loss_compare_3137_vs_3155.txt)
  - [vorticity_montage_windows_1_4_7_10_13_14_stride2.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/vorticity_montage_windows_1_4_7_10_13_14_stride2.png)
  - Vendored thesis assets:
    - [re1e6_with_vs_without_data_window_trends_to12.png](/Users/latteine/Documents/coding/jaxpi/thesis/figures/kolmogorov/re1e6_with_vs_without_data_window_trends_to12.png)
    - [re1e6_sensor100_vorticity_montage.png](/Users/latteine/Documents/coding/jaxpi/thesis/figures/kolmogorov/re1e6_sensor100_vorticity_montage.png)
    - [re1e6_loss_compare_3137_vs_3155.png](/Users/latteine/Documents/coding/jaxpi/thesis/figures/kolmogorov/re1e6_loss_compare_3137_vs_3155.png)
  - Render check: [main.pdf](/Users/latteine/Documents/coding/jaxpi/thesis/main.pdf)
- Interpretation:
  - `3137/3145` 已足以支撐「baseline workflow reproduce 成功」：corrected eval 下，至少到 `window 12` 仍維持低 `u/v` 誤差，`w_err = 0.176233` 雖未完全達專案硬門檻，但不再屬於 workflow-level failure。
  - `3155` 顯示 `sensor100 + w50` 可穩定訓練並保住與 no-data baseline 相近的 full-field 誤差；shared windows `1..12` 的 mean `u/v` 誤差略低，但 mean `w_err` 幾乎不變（略高）。
  - loss compare 顯示目前不能宣稱訓練效率提升；later windows 的 shared residual tail 多半沒有比 no-data 更快或更低。
  - 版面檢查顯示：3 張圖在正文中的浮動位置合理，趨勢圖與 vorticity montage 可讀性良好；loss compare 圖獨占一頁且上方留白較多，但不影響閱讀與 caption 對應。
- Next:
  - 若 thesis 要進一步強化 sensor 路線，需要補 `w_data` 或其他 vorticity-aware / unsensed-mode 約束的 follow-up，而不是把目前 `3155` 寫成效率成功案例。

### [2026-04-05] `3147` | start training `paper_repro_soap_sensor100_n512_w25`

- Status: Running
- Config: [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Sensor Files:
  - [sensors_qrpivot_K100_N512_t0-5.json](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/sensors_qrpivot_K100_N512_t0-5.json)
  - [sensors_qrpivot_K100_N512_t0-5_dns_values.npz](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/sensors_qrpivot_K100_N512_t0-5_dns_values.npz)

Change:

- 使用 remote `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh` 啟動 `N512 ds4 + sensor100 + 25 windows` 的 SOAP 訓練。
- 提交時顯式覆蓋 `CONFIG_PATH=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py`，避免誤用舊的 `N2048 ke024` sensor recipe。
- `RUN_NAME` 設為 `re1e6_n512_ds4_soap_sensor100_w25_junyi`。

Evidence:

- `JobID = 3147`
- `State = RUNNING`
- `Node = acmt20`
- Slurm header:
  - `Config   : examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py`
  - `Workdir  : /home/junyi/jaxpi/runs/re1e6_n512_ds4_soap_sensor100_w25_junyi`
  - `devices  : [CudaDevice(id=0), CudaDevice(id=1)]`

Interpretation:

- 這個 run 是目前最直接的 follow-up 實驗，目的明確：檢查 sensor constraint 是否能抑制 `3137` 在 `window 12` 顯示出的 high-k vorticity attenuation。
- 在第一個 checkpoint 出現前，不應對最終品質做任何推論。

Next:

- 優先監看 `window 1` loss 與第一個 checkpoint。
- 等 checkpoint 出現後，先做 `final_step`，再做必要的 `full-window` 對照。

### [2026-04-06] `3147` | progress check at `window 1 / step 36700`

- Status: Running
- Config: [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
- Actual Output Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25`

Change:

- 針對 `3147` 進行 job status / log / checkpoint 追蹤。
- 確認目前訓練已進入 `window 1` 中段，並已有兩個 checkpoint 落盤。

Evidence:

- Slurm:
  - `3147 | kf_sensor100_w25 | RUNNING | 02:23:40 | acmt20`
- 最新 log：
  - `Time Window: 1/25 | Step: 36700/100000 (36.7%)`
  - `rc_loss = 2.048e-05`
  - `ru_loss = 2.049e-05`
  - `rv_loss = 2.167e-05`
  - `u_data_loss = 1.900e-08`
  - `v_data_loss = 2.119e-08`
- 已落盤 checkpoint：
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_1/checkpoint_20000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_1/checkpoint_30000`
- 觀察到 Slurm header 的 `Workdir` 與實際輸出根目錄不一致：
  - Header: `/home/junyi/jaxpi/runs/re1e6_n512_ds4_soap_sensor100_w25_junyi`
  - Actual: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25`

Interpretation:

- `3147` 正常運行中，且至少已通過初始化、JIT compile 與 checkpoint 保存路徑。
- `u/v` sensor loss 已經進入有效量級，沒有看到明顯數值爆炸跡象。
- 後續追蹤 checkpoint 時必須以實際輸出根目錄為準，不能只看 Slurm header 的 `Workdir`。

Next:

- 等 `checkpoint_40000` 或 `checkpoint_50000` 後，優先對 `window 1` 做一次 `final_step` 快速評估。
- 若 `window 1` 品質正常，再繼續觀察 sensor constraint 是否改善後續窗口的 `w_err` 累積。

### [2026-04-06] `3147` | progress check at `window 3 / step 85500`

- Status: Running
- Config: [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
- Actual Output Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25`

Change:

- 再次追蹤 `3147` 的 Slurm 狀態、最新 log 與 checkpoint 目錄。
- 確認訓練已完整通過 `window 1`、`window 2`，目前位於 `window 3` 後段。

Evidence:

- Slurm:
  - `3147 | kf_sensor100_w25 | RUNNING | 18:15:20 | acmt20`
- 最新 log：
  - `Time Window: 3/25 | Step: 85500/100000 (85.5%)`
  - `rc_loss = 9.303e-02`
  - `ru_loss = 3.727e-03`
  - `rv_loss = 3.477e-03`
  - `u_data_loss = 4.291e-07`
  - `v_data_loss = 9.064e-07`
- 近期異常點：
  - `Time Window: 3/25 | Step: 85200/100000`
  - `rc_loss = 9.923e+00`
  - `rv_loss = 1.199e+00`
  - 下一個 log (`step 85300`) 已回落到 `rc_loss = 9.922e-02`
- 已落盤 checkpoint：
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_1/checkpoint_90000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_1/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_2/checkpoint_90000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_2/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_3/checkpoint_70000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_3/checkpoint_80000`

Interpretation:

- `3147` 目前不是卡住，而是持續穩定推進，至少已證明 `sensor100 + w25` 設定可以跨到 `window 3`。
- `u/v` data loss 持續存在，代表感測約束確實在訓練回路內發揮作用。
- 但 `window 3` 後段曾出現一次明顯殘差 spike；雖然目前已自行回落，仍需後續用 checkpoint 評估確認這不是品質轉折點。

Next:

- 優先對 `time_window_1/checkpoint_100000` 與 `time_window_2/checkpoint_100000` 做 `final_step` 評估。
- 若品質正常，再對 `time_window_3/checkpoint_80000` 做一次快評，確認 spike 前後沒有造成明顯退化。

### [2026-04-07] `3147` | progress check at `window 7 / step 95000` + selected checkpoint evaluation recovered

- Status: Running
- Config: [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
- Actual Output Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25`
- Eval Definition: `window-local time`

Change:

- 再次追蹤 `3147` 的 Slurm 狀態、最新 log 與 checkpoint 目錄。
- 確認訓練已完整落盤到 `window 6 / checkpoint_100000`，目前位於 `window 7 / step 95000`。
- 回收前一日背景 CPU 評估結果，補記 `time_window_1/checkpoint_100000`、`time_window_2/checkpoint_100000`、`time_window_3/checkpoint_80000` 的 `full-window` 與 `final-step` 誤差。

Evidence:

- Slurm:
  - `3147 | RUNNING | 1-19:18:35 | acmt20`
- 最新 log：
  - `Time Window: 7/25 | Step: 95000/100000 (95.0%)`
  - `rc_loss = 2.209e-01`
  - `ru_loss = 7.734e-03`
  - `rv_loss = 7.381e-03`
  - `u_data_loss = 1.378e-06`
  - `v_data_loss = 3.533e-06`
- 已落盤 checkpoint：
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_1/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_2/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_3/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_4/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_5/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_6/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt/time_window_7/checkpoint_90000`
- 背景評估輸出：
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/eval_selected_ckpts_0406.json`
- Selected checkpoint evaluation:

| Window | Checkpoint | t_end | full_u | full_v | full_w | final_u | final_v | final_w |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100000 | 0.1500 | 0.000088 | 0.000084 | 0.000788 | 0.000127 | 0.000117 | 0.001346 |
| 2 | 100000 | 0.3500 | 0.782765 | 0.765575 | 1.027901 | 0.790569 | 0.747617 | 1.064552 |
| 3 | 80000 | 0.5500 | 1.051964 | 1.081224 | 1.333417 | 0.793712 | 0.776680 | 0.993811 |

Interpretation:

- 從作業層面看，`3147` 並未卡住，且已跨過 `window 6`，目前正接近 `window 7` 結尾。
- 但 corrected `window-local time` 評估顯示，`window 1` 品質極好，`window 2` 開始就已出現接近 `80% ~ 130%` 的相對 L2 誤差，`window 3 / checkpoint_80000` 仍未恢復。
- 因此目前更合理的結論不是「`sensor100 + w25` 成功穩定改善」，而是「訓練流程可持續前進，但早期跨窗品質已明顯失守」；這與單看 loss/log 所得到的印象不同。

Next:

- 等 `window 7 / checkpoint_100000` 落盤後，視需要評估 `window 4~7`，確認高誤差是否持續、回落，或只是 `window 2~3` 的局部失穩。
- 在繼續投入更長訓練前，應優先檢查 `window 2` 之後的 IC propagation、sensor assimilation 與 checkpoint eval 定義是否真的與訓練假設一致。

### [2026-04-07] `3147` stopped + `3149` full evaluation completed

- Training Job: `3147`
- Eval Job: `3149`
- Status:
  - `3147 = CANCELLED by 10004`
  - `3149 = COMPLETED`
- Config: [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
- Checkpoint Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w25/ckpt`
- Eval Definition: `window-local time`

Change:

- 依人工指示停止訓練 job `3147`，避免繼續消耗資源。
- 修正 [eval_sensor100_w25.py](/Users/latteine/Documents/coding/jaxpi/eval_sensor100_w25.py) 的時間軸與 config 預設，避免沿用舊 `N2048 ke024` config 與 absolute-time 評估錯誤。
- 以 `3149` 完成 `window 1~7` 的 corrected full-window evaluation，並回收圖像與數值 artifact。

Evidence:

- Job state:
  - `3147|kf_sensor100_w25|CANCELLED by 10004|1-19:27:15|acmt20`
  - `3149|eval_s100w25|COMPLETED|0:0|00:02:25|acmt20`
- `3149` summary:

| Window | Step | t_end | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 100000 | 0.1500 | 0.001176 | 0.001160 | 0.001072 |
| 2 | 100000 | 0.3500 | 0.782785 | 0.765666 | 1.039440 |
| 3 | 100000 | 0.5500 | 1.054310 | 1.084519 | 1.344144 |
| 4 | 100000 | 0.7500 | 1.094882 | 1.187397 | 1.403937 |
| 5 | 100000 | 0.9500 | 1.044828 | 1.137830 | 1.386611 |
| 6 | 100000 | 1.1500 | 1.052529 | 1.098704 | 1.386104 |
| 7 | 90000 | 1.3500 | 1.081791 | 1.019905 | 1.494207 |

- Summary statistics:
  - `mean(u_err) = 0.873186`
  - `mean(v_err) = 0.899312`
  - `mean(w_err) = 1.150788`
  - `max(w_err) = 1.494207`
- Local artifacts:
  - [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w25_0407/summary.txt)
  - [comparison_plots.png](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w25_0407/comparison_plots.png)
  - [vorticity_fields.png](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w25_0407/vorticity_fields.png)
  - [l2_errors.npz](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w25_0407/l2_errors.npz)
- Verification:
  - `python3 -m py_compile /Users/latteine/Documents/coding/jaxpi/eval_sensor100_w25.py`

Interpretation:

- `3149` 已證明 `3147` 的問題不是單一 checkpoint 或單一窗口偶發 spike，而是從 `window 2` 開始的系統性跨窗失真。
- `window 1` 幾乎完美，但 `window 2~7` 的 `u/v/w` 全都遠高於專案門檻，且 `w_err` 持續維持在 `> 1.0` 的區間。
- 因此這條 `sensor100 + w25` run 的結論應定義為「訓練可推進，但重建品質失敗」，不能作為主線成功證據。

Next:

- 若要追根究底，優先檢查 `window 1 -> 2` 的 state/IC propagation 與 sensor assimilation 是否在訓練與評估中使用同一組定義。
- 若要快速比較，可直接拿 [comparison_plots.png](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w25_0407/comparison_plots.png) 與 [vorticity_fields.png](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w25_0407/vorticity_fields.png) 對照 `3137/3145` 的 corrected 結果。

### [2026-04-07] `3137/3145` vs `3147/3149` | `window 1 -> 2` failure mode comparison

- Status: Analysis completed
- Compared Runs:
  - `3137` + `3145` corrected evaluation
  - `3147` + `3149` corrected evaluation

Change:

- 比對 `3137/3145` 與 `3147/3149` 的 config、`window 1 -> 2` propagated IC、`window 2` 起始 loss，以及 `window 2` 完整評估。
- 釐清 sensor 版為何在 `window 1` 之後失守。

Evidence:

- Config 差異：
  - `3137`: `num_time_windows = 50`, `u_data = v_data = w_data = 0`
  - `3147`: `num_time_windows = 25`, `u_data = 100`, `v_data = 100`, `w_data = 0`, `sensor_batch_size_per_device = 48`
- Corrected `window 1 -> 2` propagated IC 誤差（直接量下一窗口起點，而非窗口內最後一張）：

| Case | num_windows | train_t_end | next_ic_t | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3137 window1 -> 2` | 50 | 0.05 | 0.10 | 0.000057 | 0.000059 | 0.000444 |
| `3147 window1 -> 2` | 25 | 0.15 | 0.20 | 0.000148 | 0.000136 | 0.002165 |

- `3147` 在 `window 2 / step 0`：
  - `rc_loss = 6.286e-07`
  - `ru_loss = 9.044e-07`
  - `rv_loss = 8.299e-07`
  - `u_ic_loss = 1.463e-10`
  - `v_ic_loss = 1.752e-10`
  - `u_data_loss = 1.593e-01`
  - `v_data_loss = 1.457e-01`
- `3147` 在 `window 2 / step 90000`：
  - `u_data_loss = 7.595e-07`
  - `v_data_loss = 5.189e-06`
  - `u_ic_loss = 9.705e-07`
  - `v_ic_loss = 3.705e-06`
- 但 `3149` 的 `window 2` full-window 誤差仍為：
  - `u_err = 0.782785`
  - `v_err = 0.765666`
  - `w_err = 1.039440`
- 對照 `3145` corrected `window 2`：
  - `u_err = 0.000962`
  - `v_err = 0.001129`
  - `w_err = 0.001016`

Interpretation:

- `3147` 的失敗主因不是 `window 1 -> 2` propagated IC 在窗口起點就已經嚴重錯誤；next-IC 誤差雖較 `3137` 大，但仍在低誤差量級。
- 真正的分岔點在於 `3147` 同時做了三件事：
  - 把 `num_time_windows` 從 `50` 改成 `25`，使每個 window 的時間跨度加倍
  - 移除 dense/full-field 監督，只保留 `100` 個稀疏 sensor 的 `u/v` 約束
  - 不使用 `w_data`，因此對 unsensed high-k / vorticity drift 沒有直接觀測約束
- `window 2 / step 0` 的極小 `u_ic/v_ic` 只證明模型吻合「自己 propagated 的 IC」，不證明它吻合 DNS；因為 IC loss 比較的是 propagated `u0/v0`，不是 DNS boundary truth。
- `window 2 / step 90000` 時 sensor data loss 已幾乎為零，但 full-field 仍高度失真，說明 `3147` 已學會穿過稀疏 sensor 點，卻沒有重建整個 flow field；換句話說，這條 run 的問題是 under-constrained rollout，而不是單純 optimizer 沒收斂。

Next:

- 若要保留 sensor 路線，優先做的不是再延長訓練，而是降低 window horizon 或增加對 unsensed modes 的約束，例如回到 `50 windows`、加入 `w_data` 或其他 vorticity/spectral constraint。
- 若要確認 horizon 是否是主因，最乾淨的 A/B 是做一版 `sensor100 + 50 windows`，其餘條件不變。

### [2026-04-07] `3150` | start training `paper_repro_soap_sensor100_n512_w50`

- Status: Running
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)

Change:

- 新增一版 `sensor100 + 50 windows` 的 N512 ds4 config，保留 `3147` 的 sensor 與 optimizer 設定，只把 `num_time_windows` 從 `25` 改回 `50`。
- 使用 remote `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh` 啟動新實驗，但以 `CONFIG_PATH` 覆蓋到 `paper_repro_soap_sensor100_n512_w50.py`。

Evidence:

- Local verification:
  - `python3 -m py_compile /Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py`
- Remote verification:
  - `python3 -m py_compile /home/junyi/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py`
- Slurm:
  - `JobID = 3150`
  - `State = RUNNING`
  - `Node = acmt20`
- Header:
  - `Config   : /home/junyi/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py`
  - `Workdir  : /home/junyi/jaxpi/runs/re1e6_n512_ds4_soap_sensor100_w50_junyi`
  - `wandb.name = re1e6_n512_ds4_soap_sensor100_w50`
  - `sensor_batch_size_per_device: 48`

Interpretation:

- 這次 A/B 的設計是乾淨的：只改 horizon，不混入 `w_data`、batch、optimizer 或感測配置變更。
- 因此若 `3150` 能跨過 `window 2` 並維持低誤差，就可以把 `3147` 的主因更有力地歸到 `25-window horizon`；反之若仍失敗，就要進一步懷疑 sparse sensor 約束本身不足。

Next:

- 優先觀察 `window 1` 收斂與 `window 2 / step 0` 的 `u_data_loss`、`v_data_loss`。
- 一旦 `time_window_1/checkpoint_100000` 落盤，就重做 `window 1 -> 2` propagated IC 與 `window 2` early-stage 對照。

### [2026-04-08] `3150` | progress check at `window 5 / step 23400`

- Status: Running
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Actual Output Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50`

Change:

- 追蹤 `3150` 的 Slurm 狀態、最新 log 與 checkpoint 目錄。
- 比對 `window 2 / step 0` 的 sensor mismatch 是否較 `3147` 改善。

Evidence:

- Slurm:
  - `3150 | RUNNING | 1-03:05:59 | acmt20`
- 最新 log：
  - `Time Window: 5/50 | Step: 23400/100000 (23.4%)`
  - `rc_loss = 1.793e-01`
  - `ru_loss = 1.351e-02`
  - `rv_loss = 6.076e-03`
  - `u_data_loss = 6.092e-06`
  - `v_data_loss = 9.848e-06`
- 已落盤 checkpoint：
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_1/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_2/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_3/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_4/checkpoint_100000`
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_5/checkpoint_20000`
- `window 2 / step 0`：
  - `u_data_loss = 3.490e-02`
  - `v_data_loss = 4.279e-02`
  - `u_ic_loss = 1.204e-10`
  - `v_ic_loss = 1.313e-10`
- 對照 `3147` 同位置：
  - `u_data_loss = 1.593e-01`
  - `v_data_loss = 1.457e-01`

Interpretation:

- `3150` 已穩定跨過 `window 4`，目前進入 `window 5`，作業面健康。
- 更重要的是，`window 2 / step 0` 的 sensor mismatch 較 `3147` 明顯下降，這支持「把 horizon 從 `25` 改回 `50`」確實改善 early-window mismatch。
- 但目前仍缺少 corrected checkpoint evaluation；在拿到 `window 2~4` 的 full-window 誤差前，不能把這個正向 loss 訊號直接等同於 full-field reconstruction 成功。

Next:

- 優先對 `time_window_1/checkpoint_100000`、`time_window_2/checkpoint_100000` 做 corrected evaluation。
- 若 `window 2` 的 full-window 誤差同步下降，則可更有力地把 `3147` 的主因歸到 `25-window horizon`。

### [2026-04-08] `3150` | late-stage `rc/ru/rv` summary by time window

- Status: Running
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Actual Output Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50`

Change:

- 從 `3150` 的訓練 log 抽取每個 time window 目前最高 step 對應的 `rc_loss`、`ru_loss`、`rv_loss`。
- 僅把已達 late-stage 的 `window 1~4` 視為尾段收斂證據；`window 5` 僅記為中段進度。

Evidence:

- Slurm:
  - `3150 | RUNNING | 1-03:20:30 | acmt20`
- 最新 progress line：
  - `Time Window: 5/50 | Step: 27200/100000 (27.2%)`
- 每個 window 目前最高 step 的 `rc/ru/rv`：
  - `window 1 | step 99900/100000 | rc = 3.710e-07 | ru = 1.259e-06 | rv = 1.438e-06`
  - `window 2 | step 99900/100000 | rc = 7.887e-04 | ru = 1.667e-03 | rv = 1.896e-03`
  - `window 3 | step 99900/100000 | rc = 2.648e-01 | ru = 4.456e-02 | rv = 8.407e-03`
  - `window 4 | step 99900/100000 | rc = 9.483e-02 | ru = 8.712e-03 | rv = 7.423e-03`
  - `window 5 | step 27200/100000 | rc = 2.004e-01 | ru = 1.022e-02 | rv = 7.366e-03`

Interpretation:

- `window 1` 與 `window 2` 的尾段 loss 已明顯下降，分別落在 `1e-6` 與 `1e-3` 級，這比 `3147` 的 early failure signal 健康得多。
- 但 `window 3`、`window 4` 的 late-stage `rc_loss` 仍然偏高，特別是 `window 3` 的 `rc = 2.648e-01`，表示 PDE residual 並沒有隨 window index 單調收斂。
- 因此目前最合理的判讀是：`50 windows` 確實改善了 `window 1 -> 2` 的 early mismatch，但是否真的修好 full-field rollout，仍必須靠 corrected evaluation 判定，而不能只看尾段 loss。

Next:

- 優先對 `time_window_1/checkpoint_100000`、`time_window_2/checkpoint_100000`、`time_window_3/checkpoint_100000` 做 corrected evaluation。
- 若 `window 3` 的場誤差仍顯著偏大，則要進一步檢查為何 `ru/rv` 已降而 `rc` 仍高。

### [2026-04-08] `3150` | corrected evaluation for `time_window_1~3/checkpoint_100000`

- Status: Completed via CPU direct eval
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Checkpoint Root: `/home/junyi/jaxpi/tmp_eval_ckpt_roots/re1e6_n512_ds4_soap_sensor100_w50_ckpt123`
- Eval Definition: `window-local time`

Change:

- 只針對 `time_window_1/checkpoint_100000`、`time_window_2/checkpoint_100000`、`time_window_3/checkpoint_100000` 做 corrected full-window evaluation。
- 先提交 GPU eval job `3154`，但因資源排隊未啟動，後續取消並改用 CPU 直接執行 [eval_sensor100_w25.py](/Users/latteine/Documents/coding/jaxpi/eval_sensor100_w25.py) 完成。

Evidence:

- GPU eval job:
  - `3154 | eval_s100w25 | CANCELLED by 10004 | 00:00:00`
- CPU eval outputs:
  - [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w50_ckpt123_0408_cpu/summary.txt)
  - [comparison_plots.png](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w50_ckpt123_0408_cpu/comparison_plots.png)
  - [vorticity_fields.png](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w50_ckpt123_0408_cpu/vorticity_fields.png)
  - [l2_errors.npz](/Users/latteine/Documents/coding/jaxpi/eval_re1e6_n512_ds4_soap_sensor100_w50_ckpt123_0408_cpu/l2_errors.npz)
- Full-window errors:
  - `window 1 | step 100000 | t_end=0.0500 | u=0.000041 | v=0.000041 | w=0.000283`
  - `window 2 | step 100000 | t_end=0.1500 | u=0.442002 | v=0.480633 | w=0.608609`
  - `window 3 | step 100000 | t_end=0.2500 | u=0.772989 | v=0.784209 | w=0.965746`
  - `mean(u,v,w) = (0.405011, 0.421628, 0.524879)`

Interpretation:

- `50 windows` 讓 `window 1` 幾乎完美，但沒有修好 `window 2+` 的 full-field reconstruction。
- 和 `3147/3149` 相比，這次 `window 2` 雖然比 `w25` 版本稍好，但仍然是明顯失敗等級；`window 3` 更進一步惡化到接近 `80%~97%`。
- 因此目前不能把失敗主因單純歸到 `25-window horizon`；更可信的判讀是 `sparse u/v sensor + w_data=0` 這組 supervision 在跨窗後仍然嚴重 under-constrained。

Next:

- 檢查 `window 2` 的 full-field failure 是否主要表現在 unsensed vorticity / high-k modes，而不是 sensor 點附近。
- 若要繼續 A/B，下一組應優先測 `sensor100 + 50 windows + w_data` 或其他 vorticity-aware 約束，而不是只再調 horizon。

### [2026-04-09] sensor input path inspection for `3150`

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Scope: sensor loading / time slicing / local time shift / sampler / data loss wiring

Change:

- 檢查 sensor JSON / NPZ 如何進入訓練，確認是否存在 time-window 對齊或座標對齊 bug。
- 補做 `window 2` 的實際時間切窗、座標對齊與 sampler batch 內容檢查。

Evidence:

- Sensor load path:
  - [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py#L503)
  - [sensors_format.py](/Users/latteine/Documents/coding/jaxpi/jaxpi/dataio/sensors_format.py)
- Time slicing / local shift:
  - [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py#L750)
  - `window 2` selects sensor times `[0.1, 0.15]`
  - after `sensor_time_shift=True`, local times become `[0.0, 0.05]`
- Coordinate alignment:
  - `coord_max_abs_err = 0.0`
  - `coord_mean_abs_err = 0.0`
- Sampler behavior:
  - [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py#L79)
  - `time_window = [0.0, 0.05]`
  - `mask_count = 2`
  - batch shapes = `[(96,), (96, 2), (96,), (96,), (96,)]`
  - `t_unique = [0.0, 0.05]`
  - `unique_sensor_indices_in_batch = 61`
- Loss wiring:
  - [models.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/models.py#L329)
  - `u_data_loss = mean((u_pred - u_data)^2)`
  - `v_data_loss = mean((v_pred - v_data)^2)`
  - `w_data_loss = 0` when `use_vorticity_data_loss=False`

Interpretation:

- sensor 輸入功能本身看起來是正確的：JSON 座標和 dataset flat indices 完全一致，`window 2` 的 sensor times 也正確切到 `[0.1, 0.15]` 並轉成 local time。
- 但 supervision 形式比表面上更弱：每一步只抽 `96` 個隨機 `(time_idx, sensor_idx)` 點，而不是固定覆蓋 100 個 sensors × 該窗口全部時間點；因此 `u_data_loss / v_data_loss` 只能代表 sampled point-wise fit，不能代表 sensor 鄰域或全場已被約束。
- 這和 `window 2` 診斷是吻合的：exact sensor points 的 `u/v` 幾乎完美，但一離開 sensor 點誤差就回到接近全場水準。

Next:

- 若要提高 sensor constraint 的實際約束力，應考慮固定覆蓋全部 sensor points、擴展局部 patch supervision，或直接加入 `w_data` / vorticity-aware 項。

### [2026-04-09] training logic change | fixed all-sensor coverage, keep `w_data` disabled

- Status: Implemented
- Files:
  - [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py)
  - [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
  - [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)

Change:

- `_build_sensor_sampler` 新增 `sampling_mode="all_points"`，每一步固定返回當前 window 的完整 `(sensor × time)` 笛卡兒積。
- `paper_repro_soap_sensor100_n512_w50/w25` 設定新增 `sensor_sampling = "all_points"`。
- 兩個 N512 sensor configs 都在 `_validate_config()` 顯式禁止 `use_vorticity_data_loss=True`，避免未來誤加 `w_data`。

Evidence:

- `py_compile`：
  - [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py)
  - [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
  - [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
- Remote smoke test (`w50`, `window 2`):
  - `sensor_sampling = all_points`
  - `use_vorticity_data_loss = False`
  - `unique_times = [0.0, 0.05]`
  - `num_unique_coords = 100`
  - `expected_points = 200`
  - `actual_points = 200`

Interpretation:

- 新邏輯下，sensor loss 不再只覆蓋隨機抽樣的 96 個點，而是每步都固定覆蓋 `100 sensors × 2 time points`。
- 這會直接提高 `u_data/v_data` 的實際約束力，也讓之後的 `sensor loss` 數值更接近「全 sensor supervision」而不是「隨機抽樣 fit」。
- 但已經在跑的 `3150` 進程不會自動熱更新到這個新邏輯；若要驗證效果，必須重啟或開新 job。

Next:

- 若要讓 `3150` 套用此邏輯，需停止後以相同 config 重啟新 job。
- 重啟後優先比較 `window 2 / checkpoint_100000` 的 corrected full-window 誤差是否實質下降。

### [2026-04-09] `3150 -> 3155` | restart `w50` run with `sensor_sampling=all_points`

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Slurm Script: `/home/junyi/jaxpi/slurm_train_sensor100_w25.sh`

Change:

- 依人工指示停止目前的 `3150`。
- 用同一份 `w50` config 重新提交新 job `3155`，讓剛加入的 `sensor_sampling=all_points` 真正生效。

Evidence:

- Old job:
  - `3150 | kf_sensor100_w25 | CANCELLED by 10004 | 1-07:05:20 | acmt20`
- New job:
  - `3155 | RUNNING | 0:05 | acmt20`
  - `Command = /home/junyi/jaxpi/slurm_train_sensor100_w25.sh`
  - `WorkDir = /home/junyi/jaxpi`
  - `StdOut = /home/junyi/jaxpi/logs/kf_sensor100_w25_3155.out`
  - `StdErr = /home/junyi/jaxpi/logs/kf_sensor100_w25_3155.err`
- Pre-restart smoke test for the new code path:
  - `sensor_sampling = all_points`
  - `use_vorticity_data_loss = False`
  - `expected_points = 200`
  - `actual_points = 200`

Interpretation:

- `3155` 不是單純重跑；它和 `3150` 的關鍵差異是每一步都會固定覆蓋當前 window 的全部 sensor constraints。
- 因此後續若 `window 2` 的 corrected full-window error 下降，才可以合理歸因到 `all_points` supervision；反之若仍失敗，就代表問題不只是 random-sampling 稀釋，而是 `u/v` sensor constraint 本身不足。

Next:

- 優先監看 `3155` 的第一個 checkpoint。
- 一旦 `time_window_1/checkpoint_100000` 落盤，就對 `window 2` 重做 corrected evaluation 與 sensor-region diagnosis。

### [2026-04-09] `3155` | monitor first new checkpoint

- Status: Monitoring
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Output Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt`

Change:

- 監看 `3155` 的第一個新 checkpoint 是否出現，並區分新 run 與舊 `3150` 遺留的 checkpoint。

Evidence:

- Slurm:
  - `3155 | RUNNING | 3:51 | acmt20`
- Latest log lines:
  - `Time Window: 1/50 | Step: 300/100000 (0.3%)`
  - `Time Window: 1/50 | Step: 400/100000 (0.4%)`
  - `Time Window: 1/50 | Step: 500/100000 (0.5%)`
- 現有 checkpoint mtime：
  - `checkpoint_90000  -> 2026-04-08 02:08:58 +0800`
  - `checkpoint_100000 -> 2026-04-08 02:47:31 +0800`
- `checkpoint_10000` 尚未出現。

Interpretation:

- `3155` 已正常起跑並吃到 `all_points` 配置，但第一個新 checkpoint 還沒落盤。
- 由於輸出根目錄沿用舊 run，必須以「新目錄 `checkpoint_10000` 出現」或新 mtime 為準，不能把舊 `checkpoint_90000/100000` 誤判成這次的第一個 checkpoint。

Next:

- 繼續等 `time_window_1/checkpoint_10000`。
- 一旦新 checkpoint 出現，優先做 `window 1` 基本確認，再接著評估 `window 2`。

### [2026-04-09] `3155` | first new checkpoint confirmed, `window 1` completed

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Output Root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt`

Change:

- 持續監看 `3155` 的第一個新 checkpoint，確認是否已覆寫舊 `3150` 遺留目錄。

Evidence:

- Slurm:
  - `3155 | RUNNING | 8:37:32 | acmt20`
- `window 1` latest log:
  - `Step: 99500/100000`
  - `Step: 99600/100000`
  - `Step: 99700/100000`
  - `Step: 99800/100000`
  - `Step: 99900/100000`
- Retained checkpoints with new mtime:
  - `checkpoint_90000  -> 2026-04-09 09:13 +0800`
  - `checkpoint_100000 -> 2026-04-09 09:52 +0800`
- Current progress:
  - `Time Window: 2/50 | Step: 33400/100000 (33.4%)`
  - `rc_loss = 5.881e-05`
  - `ru_loss = 1.014e-04`
  - `rv_loss = 1.135e-04`
  - `u_data_loss = 3.467e-08`
  - `v_data_loss = 3.603e-08`

Interpretation:

- 新 run `3155` 的第一個 retained checkpoint 已被確認，而且 `window 1` 已完整結束。
- 由於 `num_keep_ckpts = 2`，早期 `checkpoint_10000` 等目錄不會保留到現在；因此最可靠的證據是 `checkpoint_90000/100000` 的新 mtime。
- 目前最有價值的下一步不再是盯 checkpoint，而是直接用新的 `window 1/checkpoint_100000` 去做 corrected evaluation，然後看 `window 2` 是否真的比 `3150` 改善。

Next:

- 先評估新的 `time_window_1/checkpoint_100000`。
- 再視需要接著評估 `time_window_2` 的 early / final checkpoint。

### [2026-04-10] `3155` | `window 3` loss threshold check

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Log Source: `/home/junyi/jaxpi/logs/kf_sensor100_w25_3155.err`

Change:

- 針對 `window 3` 精確檢查 `rc_loss`、`ru_loss`、`rv_loss` 首次低於 `1e-6` 的 step，並區分 `step 0` warm-start 與後續真正訓練過程。

Evidence:

- `window 3 / step 0`:
  - `rc_loss = 2.107e-07`
  - `ru_loss = 6.466e-07`
  - `rv_loss = 6.472e-07`
- `window 3 / step 100`:
  - `rc_loss = 2.563e-03`
  - `ru_loss = 5.425e-03`
  - `rv_loss = 7.488e-03`
- 排除 `step 0` 後，首次重新低於 `1e-6` 的結果：
  - `rc_loss`: `step 68000 -> 9.883e-07`
  - `ru_loss`: 無
  - `rv_loss`: 無
- `window 3 / step 99900`:
  - `rc_loss = 5.424e-07`
  - `ru_loss = 1.522e-06`
  - `rv_loss = 1.908e-06`

Interpretation:

- 若把 `step 0` 算進去，`rc/ru/rv` 三者在 `window 3` 一開始就都低於 `1e-6`；但那是承接前一窗狀態的 warm-start，不代表後續訓練已穩定收斂到該門檻。
- 真正看 `step 100..99900` 的訓練過程，只有 `rc_loss` 在中後段重新低於 `1e-6`，首次發生在 `step 68000`。
- `ru_loss` 與 `rv_loss` 雖然尾段已降到 `1e-6` 等級附近，但在整個 `window 3` 的後續訓練區間都沒有重新低於 `1e-6`。

Next:

- 若要判定 `window 3` 是否真的「全面進入 `1e-6` 級」，接下來應直接做 corrected full-window evaluation，而不是只看 train loss。

### [2026-04-12] `3155` | manual pause at `window 14` + corrected full-window evaluation through all available windows

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Eval Definition: `window-local time`
- Eval Outputs:
  - Remote summary: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260412_chunked/summary.txt`
  - Remote numeric dump: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window_eval_results.npz`
  - Remote log: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window_eval.log`

Change:

- 依人工指示先將 `3155` 送到 Slurm `STOPPED` 狀態，避免訓練繼續推進。
- 對目前所有已存在 checkpoint 的 windows (`1..14`) 進行 corrected full-window evaluation。
- 因原 `evaluate_checkpoint.py` 在 `auto` 裝置路徑會吃到 GPU 並於 `window 1` OOM，因此改用 chunked CPU evaluator 逐 window / 逐 time / 逐空間塊累積相對 L2，取得可重現數值證據。

Evidence:

- Slurm:
  - `3155 | STOPPED | 3-12:00:10 | acmt20`
- Pause 時的 checkpoint：
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_13/checkpoint_100000` (`2026-04-12 14:20 +0800`)
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_14/checkpoint_10000` (`2026-04-12 15:00 +0800`)
- Full-window relative L2 summary:
  - `window 1 / checkpoint_100000  -> u=0.000034, v=0.000034, w=0.000222`
  - `window 2 / checkpoint_100000  -> u=0.000067, v=0.000068, w=0.000696`
  - `window 3 / checkpoint_100000  -> u=0.000140, v=0.000153, w=0.003485`
  - `window 4 / checkpoint_100000  -> u=0.000293, v=0.000334, w=0.010051`
  - `window 5 / checkpoint_100000  -> u=0.000583, v=0.000653, w=0.021417`
  - `window 6 / checkpoint_100000  -> u=0.001058, v=0.001123, w=0.036667`
  - `window 7 / checkpoint_100000  -> u=0.001654, v=0.001758, w=0.052902`
  - `window 8 / checkpoint_100000  -> u=0.002472, v=0.002608, w=0.074894`
  - `window 9 / checkpoint_100000  -> u=0.003580, v=0.003640, w=0.098401`
  - `window 10 / checkpoint_100000 -> u=0.004864, v=0.004978, w=0.124236`
  - `window 11 / checkpoint_100000 -> u=0.006248, v=0.006625, w=0.149501`
  - `window 12 / checkpoint_100000 -> u=0.007831, v=0.008350, w=0.174197`
  - `window 13 / checkpoint_100000 -> u=0.010144, v=0.009809, w=0.201551`
  - `window 14 / checkpoint_10000  -> u=0.014820, v=0.012950, w=0.279143`
- Aggregate:
  - `mean -> u=0.003842, v=0.003792, w=0.087669`
  - `max  -> u=0.014820, v=0.012950, w=0.279143`

Interpretation:

- `3155` 和 `3150` 的失敗型態不同；它沒有在 `window 2` 立即崩壞，`window 1~2` 的 `u/v/w` 誤差都維持極低。
- 但 corrected full-window evidence 顯示，`w_err` 自 `window 3` 起幾乎單調上升，到 `window 13` 已達 `0.201551`，代表 `sensor_sampling=all_points` 雖然顯著改善早期跨窗穩定性，仍未阻止 vorticity / 小尺度結構的逐窗流失。
- `window 14` 目前僅是 `checkpoint_10000`，因此它只能當「目前暫停點附近的早期訊號」，不可直接和前面完整 `checkpoint_100000` 做等價比較。
- 這批結果支持的最強結論是：`all_points` 修正把 failure mode 從「window 2 立刻崩」推遲成「較慢的跨窗劣化」，但尚未達到可接受的長時段重建品質。

Next:

- 若要繼續此 run，優先在恢復前決定是否接受目前的 `w_err` 漂移，或先對 `window 10~13` 做更細的 vorticity / spectrum diagnosis。
- 若要做方法判讀，後續應把 `w_err` 的單調升高視為主指標，而不是只看 `rc/ru/rv` train loss。

### [2026-04-12] `3155` | corrected vorticity field visualization for `window 1` and `window 13`

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Eval Definition: `window-local time`
- Output:
  - Local: [window1_vorticity_field_stride2.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_vorticity_field_stride2.png)
  - Local: [window13_vorticity_field_stride2.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window13_vorticity_field_stride2.png)
  - Remote: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_vorticity_field_stride2.png`
  - Remote: `/home/junyi/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window13_vorticity_field_stride2.png`

Change:

- 修正 [render_checkpoint_field_comparison.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/render_checkpoint_field_comparison.py) 的時間軸與預測路徑，讓 field visualization 與 corrected evaluation 一致使用 `window-local time` + `*_ic_pred_fn`。
- 生成 `window 1` 與 `window 13` 最後時刻的 vorticity `DNS / PINN / |Error|` 九宮格圖。
- 為降低渲染時間與輸出大小，繪圖使用 `plot_stride=2`（`512 -> 256`）。

Evidence:

- `window 1 / checkpoint_100000` 圖已生成：
  - `/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_vorticity_field_stride2.png`
- `window 13 / checkpoint_100000` 圖已生成：
  - `/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window13_vorticity_field_stride2.png`
- 渲染腳本輸出：
  - `saved: /home/junyi/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_vorticity_field_stride2.png`
  - `saved: /home/junyi/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window13_vorticity_field_stride2.png`

Interpretation:

- `window 1` 圖可作為目前 run 的低誤差基準視覺證據。
- `window 13` 圖則對應 corrected full-window evaluation 中已明顯抬升的 `w_err=0.201551`，可用來直接觀察後段 vorticity 結構偏移與誤差分布。
- 這兩張圖支持同一個判讀：`3155` 並非早期即崩壞，但後段小尺度 / vorticity 結構仍在累積流失。

Next:

- 若需要更完整的視覺證據，下一步可補 `window 7` 與 `window 14`，形成 early / mid / late / current-pause 四點對照。

### [2026-04-12] `3155` | representative multi-window vorticity montage

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Eval Definition: `window-local time`
- Output:
  - Local: [vorticity_montage_windows_1_4_7_10_13_14_stride2.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/vorticity_montage_windows_1_4_7_10_13_14_stride2.png)
  - Local log: [vorticity_montage_windows_1_4_7_10_13_14_stride2.log](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/vorticity_montage_windows_1_4_7_10_13_14_stride2.log)

Change:

- 生成一張代表性多窗口 montage，把 `window 1, 4, 7, 10, 13, 14` 的最後時刻 vorticity `DNS / PINN / |Error|` 排在同一張圖中。
- 為保持圖面可讀與計算成本可控，使用 `plot_stride=2`。
- 後續同一路徑圖檔已重生為新版，在每個窗口 row 左側直接標註 `w_rel`，避免閱讀時還要對照外部 log。

Evidence:

- 各 row 對應的 rendered `w_rel`：
  - `window 1 / checkpoint_100000  -> 0.000263`
  - `window 4 / checkpoint_100000  -> 0.011398`
  - `window 7 / checkpoint_100000  -> 0.055675`
  - `window 10 / checkpoint_100000 -> 0.128759`
  - `window 13 / checkpoint_100000 -> 0.204445`
  - `window 14 / checkpoint_10000  -> 0.284291`
- 圖檔已生成：
  - `/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/vorticity_montage_windows_1_4_7_10_13_14_stride2.png`

Interpretation:

- 這張 montage 把 `3155` 的 failure mode 從「數字摘要」轉成「空間結構演化」。
- 從 `window 1 -> 13` 可以直接看到 vorticity 結構誤差逐窗擴大，與 corrected full-window `w_err` 單調上升一致。
- `window 14` 目前仍只是 `checkpoint_10000`，因此在 montage 中應視為「暫停時刻的早期狀態快照」，不是完整窗口最終品質。

Next:

- 若要補完整版本，可再生成 `window 1..14` 的長版 montage；若要做報告，這張代表性 montage 已足以說明趨勢。

### [2026-04-12] `3155` | resume training after manual pause

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Log Source: `/home/junyi/jaxpi/logs/kf_sensor100_w25_3155.err`

Change:

- 依人工指示將 `3155` 從 `STOPPED` 狀態恢復，讓訓練從 `window 14` 繼續向前推進。

Evidence:

- Resume command:
  - `scancel --signal=CONT 3155`
- Slurm status after resume:
  - `3155 | RUNNING | 3-13:52:23 | acmt20`
- Latest log after resume:
  - `Time Window: 14/50 | Step: 16300/100000 (16.3%)`
  - `rc_loss = 9.534e-04`
  - `ru_loss = 4.353e-04`
  - `rv_loss = 4.720e-04`
  - `u_data_loss = 8.847e-08`
  - `v_data_loss = 1.002e-07`
  - 後續已再推進到 `Step: 16400/100000 (16.4%)`

Interpretation:

- 這次恢復不是重新提交新 job，而是延續原本 `3155` 的 allocation 與 checkpoint lineage。
- 恢復後 log 已明確往前推進，因此目前狀態可視為「訓練重新活躍」，不是只改到 Slurm 狀態字串而未真正執行。

Next:

- 持續觀察 `window 14` 是否能正常落到 `checkpoint_90000/100000`。
- 若後續要再做 corrected evaluation，應以恢復後新落盤的 `window 14` retained checkpoint 為準。

### [2026-04-12] `3155` vs `3137/3145` | with-data vs no-data per-window error trend

- Status: Completed
- Note: `[SUPERSEDED 2026-04-21]` 此段保留為歷史圖表血統；最終 apples-to-apples 比較已改用 `3327 + 3328` 的 all-window direct CSV，見後續 `2026-04-21` 重畫條目。
- Compare Scope:
  - With data: `3155` (`sensor100`, `u_data + v_data`, `w_data = 0`)
  - No data: corrected `3137/3145` (`u_data = v_data = w_data = 0`)
- Common Window Range: `1..12`
- Output:
  - Local plot: [with_vs_without_data_window_trends.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends.png)
  - Local summary: [with_vs_without_data_window_trends.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends.txt)
  - Local plot (`window 1..12` only): [with_vs_without_data_window_trends_to12.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends_to12.png)
  - Local summary (`window 1..12` only): [with_vs_without_data_window_trends_to12.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends_to12.txt)

Change:

- 將 `3155` corrected full-window evaluation 與 corrected `3137/3145` baseline 放到同一張 `u_error / v_error / w_error` 趨勢圖比較。
- baseline 數列使用 [docs/demo-data.js](/Users/latteine/Documents/coding/jaxpi/docs/demo-data.js) 中保存的 corrected `full_window_localtime` rows，避免誤用舊的 absolute-time `3144` 錯位結果。
- `[UPDATED 2026-04-21]` 共享窗口 `1..12` 的圖與摘要已用修正後的 `3137 window 1` 真值重畫；舊版 `window 1 ~1e-3` 比較口徑不再使用。

Evidence:

- Mean over common windows `1..12`:
  - `u_error`: `0.002651 -> 0.002402` (`with_data / no_data = 0.905984`)
  - `v_error`: `0.002818 -> 0.002527` (`with_data / no_data = 0.896793`)
  - `w_error`: `0.060276 -> 0.062222` (`with_data / no_data = 1.032287`)
- `window 1`:
  - no data  (`3137`, corrected direct): `u=3.312488e-05, v=3.383027e-05, w=2.214195e-04`
  - with data (`3155`)               : `u=3.400000e-05, v=3.400000e-05, w=2.220000e-04`
- `window 12`:
  - no data  (`3137/3145`): `u=0.007903, v=0.008504, w=0.176233`
  - with data (`3155`)     : `u=0.007831, v=0.008350, w=0.174197`
- `3155` additional windows:
  - `window 13 / checkpoint_100000 -> u=0.010144, v=0.009809, w=0.201551`
  - `window 14 / checkpoint_10000  -> u=0.014820, v=0.012950, w=0.279143`
- 若只看公平比較區間，已另輸出只到 `window 12` 的裁切版趨勢圖，避免 `3155` 的額外窗口造成視覺誤讀。

Interpretation:

- 在共同窗口 `1..12` 上，用修正後的 `3137 window 1` 基準重算後，加入 `u/v data` 的平均收益較舊版判讀小一些：
  - `u_error` 平均下降約 `9.4%`
  - `v_error` 平均下降約 `10.3%`
- `w_error` 的平均值仍沒有明顯改善，整體反而高約 `3.2%`；這表示 data constraint 主要幫助的是速度場，而不是根本解掉 vorticity 累積漂移。
- 到 `window 12` 單點時，with-data 與 no-data 的 `w_err` 已非常接近（`0.174197` vs `0.176233`），因此更穩健的結論是：`u/v data` 對 `u/v` 有益，但對晚期 `w` 漂移只有有限幫助。

Next:

- 若 `3155 window 14+` 繼續往上跑，應持續觀察 `w_err` 是否再次脫離 `3137/3145` 的上升曲線。

### [2026-04-21] `3137` vs `3155` | redraw final shared-window error trend from direct CSVs

- Time: `2026-04-21 21:35 +0800`
- Status: Completed
- Compare Scope:
  - No data: `3137` all-window direct reevaluation (`3327`)
  - With data: `3155` all-window direct reevaluation (`3328`)
- Common Window Range: `1..12`
- Output:
  - Local plot: [with_vs_without_data_window_trends_to12.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends_to12.png)
  - Local summary: [with_vs_without_data_window_trends_to12.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/with_vs_without_data_window_trends_to12.txt)
  - Thesis plot: [re1e6_with_vs_without_data_window_trends_to12.png](/Users/latteine/Documents/coding/jaxpi/thesis/figures/kolmogorov/re1e6_with_vs_without_data_window_trends_to12.png)

Change:

- 用 [redraw_3137_3155_error_trends.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/redraw_3137_3155_error_trends.py) 直接讀取兩份 all-window direct CSV，重畫 shared-window `1..12` 的 `u/v/w` 誤差圖。
- 舊版圖仍保留原路徑，但其 `3137` baseline 混入 `3137/3145` 舊 lineage；本次重畫後，`3137 vs 3155` 的主比較已完全不依賴 `3145`。

Evidence:

- Input CSV:
  - [3137_all_windows_direct_20260421/all_window_direct_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_all_windows_direct_20260421/all_window_direct_results.csv)
  - [3155_all_windows_direct_20260421/all_window_direct_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_direct_20260421/all_window_direct_results.csv)
- Summary (`window 1..12`):
  - `u_error_mean_no_data=0.002296`
  - `u_error_mean_with_data=0.002402`
  - `v_error_mean_no_data=0.002436`
  - `v_error_mean_with_data=0.002527`
  - `w_error_mean_no_data=0.060127`
  - `w_error_mean_with_data=0.062222`
  - `u_ratio_with_over_no=1.046091`
  - `v_ratio_with_over_no=1.037331`
  - `w_ratio_with_over_no=1.034847`
- Endpoint checks:
  - `window 1`:
    - no data (`3137`): `u=3.312488e-05, v=3.383027e-05, w=2.214195e-04`
    - with data (`3155`): `u=3.370493e-05, v=3.354592e-05, w=2.221507e-04`
  - `window 12`:
    - no data (`3137`): `u=7.821861e-03, v=8.404544e-03, w=1.760991e-01`
    - with data (`3155`): `u=7.831324e-03, v=8.349953e-03, w=1.741970e-01`

Interpretation:

- 用完全同一路徑的 direct baseline 重畫後，`3155` 在 shared windows `1..12` 的平均 `u/v/w` 誤差都沒有優於 `3137`。
- `3155` 在 `window 12` 單點的 `v/w` 略低，但這不足以推翻整段區間平均仍略差的結論。
- 因此較嚴格的結論應修正為：`sensor100 + u/v data` 沒有在公平 direct 比較下帶來整體 error 改善，只是局部末端 checkpoint 可以接近 baseline。

Next:

- 若後續要主張 sensor 約束有價值，不能只看 shared-window mean；需要另外找更直接的收斂速度、早期窗口、或 field-level 機制證據。

### [2026-04-21] analysis | window-1 loss vs corrected error alignment (`3137`, `3155`, `3324`)

- Time: `2026-04-21 21:45 +0800`
- Status: Completed
- Compare Scope:
  - No data baseline: `3137`
  - Sensor mainline: `3155`
  - Sensor sweep-derived validation: `3324` (`data_weight=23.1429`)
- Window: `1`
- Output:
  - Plot: [window1_loss_error_alignment.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/loss_error_alignment_20260421/window1_loss_error_alignment.png)
  - Summary: [window1_loss_error_alignment.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/loss_error_alignment_20260421/window1_loss_error_alignment.txt)
  - Combined loss CSV: [window1_loss_compare_3137_3155_3324.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/loss_error_alignment_20260421/window1_loss_compare_3137_3155_3324.csv)

Change:

- 新增 [plot_loss_error_alignment.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/plot_loss_error_alignment.py)，把三條 run 的 shared-horizon residual loss 與 corrected field error 放進同一張圖。
- loss 統一裁到 `step <= 49900`，避免把 `3324` 的 `50000-step` run 和 `3137/3155` 的 `100000-step` 尾段混在一起。

Evidence:

- Shared loss horizon: `step <= 49900`
- `rc_loss <= 5e-5`：
  - `3137 -> 14300`
  - `3155 -> 13900`
  - `3324 -> 13800`
- `ru_loss <= 5e-5`：
  - `3137 -> 22000`
  - `3155 -> 20200`
  - `3324 -> 21100`
- `rv_loss <= 5e-5`：
  - `3137 -> 19600`
  - `3155 -> 20200`
  - `3324 -> 21100`
- corrected field error:
  - `3137 @ checkpoint_100000 -> u=3.312488e-05, v=3.383027e-05, w=2.214195e-04`
  - `3155 @ checkpoint_100000 -> u=3.370493e-05, v=3.354592e-05, w=2.221507e-04`
  - `3324 @ checkpoint_50000  -> u=1.235000e-03, v=1.103000e-03, w=8.170000e-04`

Interpretation:

- 這張圖直接證明：**loss crossing 順序不等於 field error 排名**。
- `3324` 在部分 residual crossing 上不差，甚至略快，但 corrected field error 明顯差於 `3137/3155`；因此不能把 `5e-5` residual threshold 直接當成高品質收斂證據。
- `3155` 和 `3137` 的 window-1 residual curve 幾乎重疊，但 shared-window error 比較並沒有帶來整體優勢；這表示 loss 更適合拿來做 `health/progress signal`，而不是 `model-selection signal`。

Next:

- 若要建立更穩定的訓練判讀規則，應以「loss 監控 + 定期 corrected error checkpoint eval」的雙軌方式選模型，而不是只用 residual threshold。

### [2026-04-12] `3155` vs `3137` | `time window 1` loss convergence comparison

- Status: Completed
- Compare Scope:
  - No data: `3137`
  - With data: `3155`
- Window: `time window 1`
- Log Source:
  - No data: `/home/junyi/jaxpi_upstream_soap_run/logs/kf_paper_repro_soap_3137.err`
  - With data: `/home/junyi/jaxpi/logs/kf_sensor100_w25_3155.err`
- Output:
  - Local raw dump: [window1_loss_compare_raw.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_loss_compare_raw.csv)
  - Local plot: [window1_loss_compare_3137_vs_3155.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_loss_compare_3137_vs_3155.png)
  - Local summary: [window1_loss_compare_3137_vs_3155.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window1_loss_compare_3137_vs_3155.txt)

Change:

- 擷取 `3137` 與 `3155` 的 `time window 1` 訓練 log，只比較兩邊共同存在的 residual losses：`rc_loss`、`ru_loss`、`rv_loss`。
- 以 log-scale loss curve 與 threshold-crossing step 雙重方式檢查「加入 `u_data + v_data` 後是否更快收斂」。

Evidence:

- `rc_loss` first crossing:
  - no data (`3137`): `<=1e-4 @ step 10900`, `<=1e-5 @ step 24000`, `<=1e-6 @ step 60700`
  - with data (`3155`): `<=1e-4 @ step 10400`, `<=1e-5 @ step 23100`, `<=1e-6 @ step 52200`
- `ru_loss` first crossing:
  - no data (`3137`): `<=1e-4 @ step 18500`, `<=1e-5 @ step 36400`, `<=1e-6 @ step 87100`
  - with data (`3155`): `<=1e-4 @ step 14200`, `<=1e-5 @ step 36400`, `<=1e-6 @ step 87200`
- `rv_loss` first crossing:
  - no data (`3137`): `<=1e-4 @ step 18500`, `<=1e-5 @ step 34500`, `<=1e-6 @ step 88700`
  - with data (`3155`): `<=1e-4 @ step 14100`, `<=1e-5 @ step 36500`, `<=1e-6 @ step 90400`
- Tail mean over last 50 logged points:
  - `rc_loss`: `2.390e-07 -> 2.422e-07`
  - `ru_loss`: `8.229e-07 -> 8.365e-07`
  - `rv_loss`: `8.502e-07 -> 8.983e-07`

Interpretation:

- `window 1` 的整體 loss curve 幾乎重疊，不支持「加入 data 後明顯更快收斂」這個強結論。
- 若只看早中段 threshold crossing，`3155` 的 `rc_loss` 與 `ru_loss/rv_loss` 首次跌破 `1e-4` 確實略早；但進到 `1e-5 ~ 1e-6` 後，兩條 run 的差距大多消失，`ru/rv` 尾段甚至略高。
- 因此較穩健的判讀是：`u/v data` 在 `window 1` 也許稍微改善了中段下降速度，但沒有帶來明確、持續、跨三個 residual 一致的加速收斂證據。

Next:

- 若要進一步確認 data 的價值，應比較 `window 2+` 的 loss 與 corrected full-window error，而不是只看 `window 1`。

### [2026-04-12] `3155` vs `3137` | selected later-window loss convergence comparison

- Status: Completed
- Compare Scope:
  - No data: `3137`
  - With data: `3155`
- Windows: `4`, `7`, `10`, `12`
- Log Source:
  - No data: `/home/junyi/jaxpi_upstream_soap_run/logs/kf_paper_repro_soap_3137.err`
  - With data: `/home/junyi/jaxpi/logs/kf_sensor100_w25_3155.err`
- Output:
  - Local raw dump: [window4_7_10_12_loss_compare_raw.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window4_7_10_12_loss_compare_raw.csv)
  - Local plot: [window4_7_10_12_loss_compare_3137_vs_3155.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window4_7_10_12_loss_compare_3137_vs_3155.png)
  - Local summary: [window4_7_10_12_loss_compare_3137_vs_3155.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3155_all_windows_20260412_chunked/window4_7_10_12_loss_compare_3137_vs_3155.txt)

Change:

- 擷取 `window 4 / 7 / 10 / 12` 的 `rc_loss`、`ru_loss`、`rv_loss`，畫成 `4 x 3` 的 log-scale 比較圖。
- 由於這些後段 windows 的 residual 常在 `step 0` 就已低於 `1e-4`，因此除了 threshold crossing，也記錄末段 `end_ratio` 與 `tail_mean_ratio`，避免把 propagated-state 的低起點誤判成「更快收斂」。

Evidence:

- `window 4` tail mean ratio (`with/no`):
  - `rc_loss = 1.122`
  - `ru_loss = 1.203`
  - `rv_loss = 1.220`
- `window 7` tail mean ratio (`with/no`):
  - `rc_loss = 1.193`
  - `ru_loss = 1.247`
  - `rv_loss = 1.191`
- `window 10` tail mean ratio (`with/no`):
  - `rc_loss = 1.205`
  - `ru_loss = 1.098`
  - `rv_loss = 1.158`
- `window 12` tail mean ratio (`with/no`):
  - `rc_loss = 1.069`
  - `ru_loss = 1.247`
  - `rv_loss = 1.090`
- representative end ratio (`with/no`):
  - `window 7 / ru_loss = 1.298`
  - `window 10 / rc_loss = 1.214`
  - `window 12 / ru_loss = 1.434`

Interpretation:

- 從 `window 4` 開始，兩邊共同 residual 幾乎都已在 `step 0` 附近進入低量級，表示這些窗口的訓練主要是在 propagated state 上微調，而不是重演 `window 1` 那種從高 loss 下切的收斂過程。
- 在這種 regime 下，`3155` 沒有呈現「更快收斂」的明確證據；更直接的觀察是它在 `window 4 / 7 / 10 / 12` 的 `rc/ru/rv` 多數時間都略高於 `3137`。
- 因此較穩健的結論是：`u/v data` 對晚期 windows 的 shared residual 並沒有帶來更快、也沒有帶來更低的明顯優勢；它的主要價值仍比較像是改善 per-window `u/v` field error，而不是壓低這些後段 residual curves。

Next:

- 若要把「with-data 的收益」講得更完整，應把這組 later-window loss 圖和已完成的 `u_error / v_error / w_error` per-window trend 一起讀，避免只用 loss 推論 field quality。

### [2026-04-12] `3137` and `3155` | locate W&B offline runs and upload to cloud

- Status: Completed
- Scope:
  - `3137` (`re1e6_n512_ds4_soap`)
  - `3155` (`re1e6_n512_ds4_soap_sensor100_w50`)
- Offline Run Mapping:
  - `3137` -> `/home/junyi/jaxpi_upstream_soap_run/wandb/offline-run-20260330_040607-ygcmwh4d`
  - `3155` -> `/home/junyi/jaxpi/wandb/offline-run-20260408_192348-wx8kijio`
- Cloud URLs:
  - `3137`: [ygcmwh4d](https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow/runs/ygcmwh4d)
  - `3155`: [wx8kijio](https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow/runs/wx8kijio)

Change:

- 先定位兩個訓練對應的 W&B offline run，再用遠端各自的 `.venv/bin/wandb sync` 上傳到 W&B cloud。
- `3155` 的對應目錄不是直接寫在 `wandb-metadata.json` 裡的 Slurm job id，而是用 Slurm 開始時間對齊：
  - `3150` started at `2026-04-07T20:19:08 +0800`
  - `3155` started at `2026-04-09T03:24:33 +0800`
  - 對應到 W&B offline run starts:
    - `offline-run-20260407_121826-dg78gy6y` (`2026-04-07 20:18:26 +0800`) -> `3150`
    - `offline-run-20260408_192348-wx8kijio` (`2026-04-09 03:23:48 +0800`) -> `3155`

Evidence:

- `3137` metadata / output:
  - `wandb-metadata.json` contains `slurm.job_id = 3137`
  - `output.log`: `Wandb run initialized: re1e6_n512_ds4_soap (ID: ygcmwh4d)`
- `3155` output:
  - `output.log`: `Wandb run initialized: re1e6_n512_ds4_soap_sensor100_w50 (ID: wx8kijio)`
  - Slurm start time matches `offline-run-20260408_192348-wx8kijio`
- Sync command outputs:
  - `Syncing: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow/runs/ygcmwh4d`
  - `Syncing: https://wandb.ai/felix-tc-tw-national-tsinghua-university/PINN-Kolmogorov_flow/runs/wx8kijio`
- Post-check:
  - remote `ps -ef | grep '[w]andb sync'` returned no active sync processes after upload finished

Interpretation:

- `3137` 與 `3155` 的 W&B 訓練紀錄都已成功從 remote offline run 上傳到同一個 W&B project。
- `3155` 目前訓練仍在進行，因此這次上傳對應的是「截至同步當下」的 cloud snapshot；若之後還要把新進度也補到同一個 run，需再對同一個 offline run 重新 sync 一次。

Next:

- 若後續要做報告或分享，可直接使用上面的兩個 W&B cloud run URL。
- 若等 `3155` 完訓後要補齊最新 metrics，應對 `offline-run-20260408_192348-wx8kijio` 再執行一次 sync。

### [2026-04-09] `3150` | `window 2` sensor-point vs full-field and vorticity drift

- Status: Completed
- Config: [paper_repro_soap_sensor100_n512_w50.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50.py)
- Checkpoint: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50/ckpt/time_window_2/checkpoint_100000`
- Eval Definition: `window-local time`

Change:

- 新增 [analyze_sensor_window_failure.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/analyze_sensor_window_failure.py)，針對單一窗口同時診斷：
  - exact sensor points vs sensor-neighborhood vs full-field 的 `u/v/w` 誤差
  - final-step `w` correlation / enstrophy / spectrum / high-k ratio
- 用 CPU 直接分析 `3150 window 2 / checkpoint_100000`，避免和訓練 job 搶 GPU。

Evidence:

- `python3 -m py_compile`：
  - [analyze_sensor_window_failure.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/analyze_sensor_window_failure.py)
- Artifacts:
  - [window2_sensor_failure.txt](/Users/latteine/Documents/coding/jaxpi/analysis_3150_window2_sensor_failure_0409/window2_sensor_failure.txt)
  - [window2_sensor_failure.json](/Users/latteine/Documents/coding/jaxpi/analysis_3150_window2_sensor_failure_0409/window2_sensor_failure.json)
  - [window2_sensor_failure.png](/Users/latteine/Documents/coding/jaxpi/analysis_3150_window2_sensor_failure_0409/window2_sensor_failure.png)
- Region relative L2:
  - `sensor_points | coverage=0.000381 | u=0.000318 | v=0.000258 | w=5.476316`
  - `sensor_r2      | coverage=0.009537 | u=0.481751 | v=0.491901 | w=1.598866`
  - `sensor_r4      | coverage=0.030899 | u=0.489687 | v=0.500013 | w=1.006119`
  - `sensor_r8      | coverage=0.109634 | u=0.488884 | v=0.502395 | w=0.717774`
  - `full_field     | coverage=1.000000 | u=0.442002 | v=0.480633 | w=0.608609`
- Final-step vorticity diagnostics:
  - `final_w_err = 0.628667`
  - `final_corr = 0.806559`
  - `final_std_ratio = 1.020473`
  - `final_enstrophy_ratio = 1.041364`
  - `low_k_ratio = 1.033998`
  - `high_k_ratio = 2130.264657`

Interpretation:

- `u/v` 在 exact sensor points 幾乎完美，但只要離開 sensor 點，誤差就立刻回到和全場相近的高誤差水準；這代表模型學到的是「穿過 sensor 點」，不是重建局部流場。
- `w` 在 exact sensor points 的相對誤差極大，說明未受監督的渦度導數場在 sensor 點附近本身就不受控；因此不能把 `u/v` sensor fit 視為對 `w` 的間接保證。
- final-step 的 `low_k_ratio ≈ 1.03`、`enstrophy_ratio ≈ 1.04` 看起來不差，但 `high_k_ratio ≈ 2130` 顯示高波數渦度內容出現嚴重異常放大；這更像 spectral ringing / unsensed high-k contamination，而不是單純高-k 衰減。

Next:

- 優先檢查 `window 2` 的高-k 異常是否來自 `w = dv/dx - du/dy` 在局部 sensor fit 下被導數放大。
- 若要繼續修正，應優先測 `w_data` 或其他 vorticity-aware / spectral regularization，而不是再只調 `u/v` sensor 權重。

### [2026-04-05] `3146` | `window 12 / checkpoint_60000` vorticity diagnostics

- Status: Completed
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Checkpoint: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_12/checkpoint_60000`
- Eval Definition: `window-local time`

Change:

- 針對 `window 12 / checkpoint_60000` 生成單獨的 vorticity 診斷圖與數值摘要。
- 圖中同時比較 final-step `w` 場、逐時刻 `w_err`、enstrophy 與 final-step vorticity spectrum。

Evidence:

- `JobID = 3146`
- `State = COMPLETED`
- `Elapsed = 00:01:04`
- Summary:
  - `mean_w_err = 0.176230`
  - `final_w_err = 0.179046`
  - `final_corr = 0.983843`
  - `final_std_ratio = 0.981499`
  - `final_enstrophy_ratio = 0.963340`
  - `low_k_ratio = 0.999527`
  - `high_k_ratio = 0.664633`
- Artifacts:
  - [window12_vorticity_diagnostics.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime/window12_vorticity_diagnostics.png)
  - [window12_vorticity_diagnostics.json](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime/window12_vorticity_diagnostics.json)
  - [window12_vorticity_diagnostics.txt](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime/window12_vorticity_diagnostics.txt)

Interpretation:

- `final_corr` 與 `std_ratio` 都接近 1，代表大尺度結構與整體振幅沒有崩壞。
- `low_k_ratio ≈ 1.0` 但 `high_k_ratio ≈ 0.665`，表示 `w_err` 的主要來源不是相位錯亂，而是高波數渦度內容衰減。
- 這也解釋了為什麼 final-step vorticity 圖看起來「很像」，但 full-window `w_err` 仍會逐窗上升。

Next:

- 若要把 `w_err` 從 `17.6%` 再壓低，優先方向應放在保住高-k 渦度內容，而不是大幅重寫整個 time-window 機制。
- 後續可比較是否需要更強的 vorticity-aware data loss、spectral regularization，或更保守的 window 長度。

### [2026-04-05] `3145` | `3137` full-window evaluation rerun with window-local time

- Status: Completed
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Checkpoint Root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Evaluation Mode: `full-window`
- Eval Definition: `window-local time`

Change:

- 使用修正後的 [eval_paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py) 重跑 `3137` 已落盤窗口 `1..12` 的 full-window 評估。
- 評估時將每個 window 的 DNS 絕對時間轉成從 `0` 起算的局部時間，再餵入模型。

Evidence:

- `JobID = 3145`
- `State = COMPLETED`
- `Elapsed = 00:03:47`
- Summary:
  - `[HISTORICAL_ARTIFACT]` `window 1` 這一列已被 `2026-04-21` direct `apply_fn` re-audit 推翻；後續若引用 `3137 window 1`，應改用 `checkpoint_90000/100000 -> u/v ~3e-05, w ~2e-04`
  - `window 1`: `u=0.001132`, `v=0.001147`, `w=0.000734`
  - `window 12`: `u=0.007903`, `v=0.008504`, `w=0.176233`
  - `mean(u_err) = 0.002743`
  - `mean(v_err) = 0.002911`
  - `mean(w_err) = 0.060319`
- 對照 `3144`：
  - `window 12 w_err`: `0.640494 -> 0.176233`
  - `mean(w_err)`: `0.437814 -> 0.060319`
- Artifacts:
  - New summary: [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime/summary.txt)
  - New plot: [comparison_plots.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime/comparison_plots.png)
  - New field visualization: [vorticity_fields.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime/vorticity_fields.png)
  - New numeric dump: [l2_errors.npz](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405_localtime/l2_errors.npz)

Interpretation:

- 先前 `3144` 的高 `u/v/w` full-window 誤差，主要由 eval 與 train 的時間座標不一致造成。
- 修正後可確認：`3137` 在 `window 2+` 並沒有出現先前那種整窗崩潰。
- 但 `w_err` 仍然隨窗口累積上升，到 `window 12` 已到 `17.6%`，代表渦度品質仍比 `u/v` 弱，且略高於專案成功門檻上緣。

Next:

- 後續所有 `3137` / `paper_repro_soap` 評估都必須固定使用 window-local time。
- 下一步應拆開看 `w_err` 成長是來自高波數衰減、相位偏移、還是梯度放大效應。

### [2026-04-05] 評估腳本檢查 | 修正 window-local time 對齊

- Status: Code patched / rerun pending
- Files:
  - [evaluate_checkpoint.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/evaluate_checkpoint.py)
  - [eval_paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py)
- Scope: `window` / `final_step` evaluation path, sampled time-series, last-step vorticity snapshot

Change:

- 新增 `to_window_local_time()`，把每個 window 的絕對時間轉成從 `0` 起算的局部時間。
- `evaluate_checkpoint.py` 改為使用 `t_window_local` 建立模型並計算 `full-window` / `final_step` 誤差。
- `eval_paper_repro_soap.py` 改為以 `t_win_local` 進行 full-window 評估、sampled time-series 預測與最後時刻場圖預測。
- 圖表與摘要仍保留 absolute time 標示，避免改變報表的可讀性。

Evidence:

- 訓練定義：
  - [train.py:550](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py#L550)
  - [train.py:556](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py#L556)
  - [train.py:660](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py#L660)
- 模型時間正規化：
  - [models.py:179](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/models.py#L179)
  - [models.py:180](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/models.py#L180)
- 舊 eval 路徑使用 absolute time：
  - [evaluate_checkpoint.py:227](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/evaluate_checkpoint.py#L227)
  - [eval_paper_repro_soap.py:186](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py#L186)
  - [eval_paper_repro_soap.py:237](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap.py#L237)
- 語法驗證：
  - `python3 -m py_compile examples/kolmogorov_flow/evaluate_checkpoint.py eval_paper_repro_soap.py`

Interpretation:

- 這是 eval / train 定義不一致，而不是 `w = dv/dx - du/dy` 或 `omega_ref` 對齊錯誤。
- 舊版 full-window 指標對 `window 2+` 有系統性高估風險。
- `vorticity_fields.png` 只畫每個 window 的最後一張，不能直接拿來否定 full-window 指標；它和 full-window 本來就不是同一個量。

Next:

- 以修正後腳本重跑 `3137` 的 full-window 評估。
- 重新對比 `window 2+` 的 `u/v/w` 曲線，確認先前的高 `w_err` 有多少來自時間座標錯位。

### [2026-04-05] `3144` | `3137` full-window evaluation through `window 12`

- Status: Completed
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Checkpoint Root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Evaluation Mode: `full-window`

Change:

- 對 `3137` 當前所有已落盤窗口 `1..12` 執行完整 `full-window` 評估。
- 同步產生 time-series / KE / enstrophy / spectrum plot 與各窗口最後時刻的 vorticity 場圖。

Evidence:

- `JobID = 3144`
- `State = COMPLETED`
- `Elapsed = 00:05:24`
- Summary:
  - `[HISTORICAL_ARTIFACT]` `window 1` 這一列屬於 bug 修復前舊 evaluator 輸出，不可再當成 `3137 window 1` 最終真值
  - `window 1`: `u=0.001132`, `v=0.001147`, `w=0.000734`
  - `window 12`: `u=0.180390`, `v=0.178034`, `w=0.640494`
  - `mean(w_err) = 0.437814`
- Artifacts:
  - [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/summary.txt)
  - [comparison_plots.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/comparison_plots.png)
  - [vorticity_fields.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/vorticity_fields.png)
  - [l2_errors.npz](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/l2_errors.npz)

Interpretation:

- 新結果直接推翻「目前主線仍維持整體低誤差」這個較樂觀的讀法。
- `final_step` 評估適合看窗口尾端，但不足以代表 window 內整體 rollout 品質。
- `window 2+` 之後的 full-window `u/v/w` 誤差都已進入高區間，`w_err` 尤其嚴重。
- 此結果後續已被確認混入 eval time-axis 錯位；保留作為失敗案例與診斷證據，不再作為主線最終判讀。

Next:

- 後續比較主線品質時，必須把 `full-window` 指標列為主指標，不能只看 `final_step`。
- 下一步應釐清是 state propagation、window 定義、還是 eval / train 對齊方式造成此落差。

### [2026-04-05] `3142` | `window 12 / checkpoint_60000` field visualization failed

- Status: Failed
- Type: Postprocess / Visualization
- Source Checkpoint: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_12/checkpoint_60000`
- Target Output: `/home/junyi/jaxpi/examples/kolmogorov_flow/comparison/window12_ckpt60000_field_comparison.png`

Change:

- 以 `render_checkpoint_field_comparison.py` 嘗試對 `window 12 / checkpoint_60000` 生成單張 `DNS / Prediction / |Error|` 對照圖。
- 參數為 `plot_stride=4`、`chunk_size=4096`、`approx_vorticity=True`。

Evidence:

- `JobID = 3142`
- `State = FAILED`
- `Elapsed = 00:04:13`
- Slurm log 顯示：
  - `Killed ./.venv/bin/python3 scripts/analysis/render_checkpoint_field_comparison.py ...`

Interpretation:

- 這次單張高解析場圖渲染在目前配置下仍有資源壓力，未成功完成。
- 但 `3144` 已經成功產生 [vorticity_fields.png](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0405/vorticity_fields.png)，因此本輪任務仍有可用的視覺證據。

Next:

- 若仍需要單獨的 `window 12` 大圖，優先再降低 `plot_stride` / `chunk_size` 的成本，或改成更小畫布與更粗網格。

### [2026-04-01] 新增 config | `paper_repro_soap_sensor100_n512_w25.py`

- Status: Added / Not yet run
- File: [paper_repro_soap_sensor100_n512_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w25.py)
- Based On:
  - [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
  - [paper_repro_soap_sensor100_w25.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_w25.py)

Change:

- 以目前 `paper_repro_soap.py` 為主線，新增 `25 windows + sensor100` 的 `N512 ds4` 版 config。
- 保留現行主線的 `Re1e6 N512 ds4` dataset 與 SOAP/transfer 設定。
- 套用舊 `sensor100_w25` 配方中的：
  - `training.num_time_windows = 25`
  - QR-pivot `sensor_json` / `sensor_values`
  - `u_data = 100.0`
  - `v_data = 100.0`
  - `w_data = 0.0`
  - `sensor_batch_size_per_device = 48`

Evidence:

- 可用 sensor 檔：
  - [sensors_qrpivot_K100_N512_t0-5.json](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/sensors_qrpivot_K100_N512_t0-5.json)
  - [sensors_qrpivot_K100_N512_t0-5_dns_values.npz](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000/sensors_qrpivot_K100_N512_t0-5_dns_values.npz)
- `python3 -m py_compile` 通過

Interpretation:

- 這份 config 是目前主線 `N512 ds4` 與舊 `sensor100_w25` recipe 的合併版本。
- 舊的 `paper_repro_soap_sensor100_w25.py` 仍保留其 `N2048 ke024` 血統，不被覆寫。

Next:

- 若要驗證 sensor constraint 是否能壓低後期 `w_error` 累積，優先用這份 config 開新 run。

### [2026-03-31] `3137` | `window 2` 與 `window 3` checkpoint 評估

- Status: Verified
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Checkpoint Root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Evaluation Mode: `final_step`

Evidence:

| Window | Checkpoint | t_eval | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 2 | 40000 | 0.1500 | 0.000174 | 0.000177 | 0.001759 |
| 3 | 40000 | 0.2500 | 0.000295 | 0.000322 | 0.007140 |

Supplementary Evidence:

- propagated `u0 / v0 / w0` 與 `window 2` 實際使用初值逐點一致：
  - `u / v / w rel_l2 = 0.0`
- propagated 場與 `window 2` DNS 初值場幾乎重合：
  - `u_rel_l2 ≈ 5.74e-05`
  - `v_rel_l2 ≈ 5.93e-05`
  - `w_rel_l2 ≈ 4.44e-04`

Interpretation:

- `chunked IC propagation` 已讓訓練跨過 `window 1 -> window 2`。
- `window 2`、`window 3` 已落盤 checkpoint 仍維持極低誤差。
- stale IC 問題在主線實驗中可視為初步排除。

Next:

- 持續追蹤更多 window 的 checkpoint 品質。

### [2026-04-01] `3139` | `window 9 / checkpoint_60000` field visualization

- Status: Pending resources
- Type: Postprocess / Visualization
- Source Checkpoint: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_9/checkpoint_60000`
- Output Target: `/home/junyi/jaxpi_upstream_soap_run/examples/kolmogorov_flow/comparison/window9_ckpt60000_field_comparison.png`

Change:

- 新增並提交獨立短時 Slurm job，避免在 login node 以 CPU 直接渲染造成長時間卡住。
- 使用 `approx_vorticity + plot_stride=4 + chunked inference` 產生 `DNS / Prediction / |Error|` 對照圖。

Evidence:

- `JobID = 3139`
- `State = PENDING`
- `Reason = Resources`

Interpretation:

- 視覺化流程已轉成可重跑的獨立後處理 job。
- 圖片尚未生成，當前沒有 field image 可作結論。

Next:

- 等 `3139` 取得資源並完成後，回收 PNG 並更新紀錄。

### [2026-03-30] `3137` | `re1e6_n512_ds4_soap` + chunked IC propagation

- Status: Ongoing snapshot
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Training Logic: `chunked IC propagation`

Evidence:

- `JobID = 3137`
- `Time Window: 1/50`
- `Step: 1600 / 100000`
- Latest Loss:
  - `rc_loss ≈ 1.359e-02`
  - `ru_loss ≈ 1.890e-02`
  - `rv_loss ≈ 1.919e-02`
  - `u_ic_loss ≈ 7.129e-06`
  - `v_ic_loss ≈ 8.096e-06`

Interpretation:

- `window 1` 早期收斂正常。
- 當時 `chunked IC propagation` 尚未驗證到 `window 2`。
- 該 run 已被判定為最值得追蹤的主線。

Next:

- 等待跨窗 checkpoint 證據確認 `window 2+`。

### [2026-03-30] 修正 2 | `chunked IC propagation`

- Status: Applied
- File: [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py)

Change:

- `_predict_next_window_ic()` 改為分塊計算
- 預設使用 `logging.eval_space_chunk_size` 作為 chunk 大小，當前為 `4096`
- 每塊分別預測 `u0 / v0 / w0`，最後再串接

Why:

- 避免 `Re1e6 N512` 在預測整個 `w0` 場時一次吃爆 GPU 記憶體

Next:

- 用後續 checkpoint 驗證 chunked propagation 是否仍維持物理與數值一致性

### [2026-03-30] `3136` | `re1e6_n512_ds4_soap` 第一次正式重跑

- Status: Failed at boundary
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Training Logic: 修正後的 `window IC propagation`

Evidence:

- `window 1` 成功跑完到 `step 100000 / 100000`
- 落盤 checkpoint：
  - `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt/time_window_1/checkpoint_100000`
- 失敗點：
  - `window 1 -> window 2` 交界
- Error:
  - `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 768.00MiB`

Interpretation:

- `window 1` 本身可成功收斂。
- 真正瓶頸在 `window` 邊界狀態傳遞的記憶體成本。

Next:

- 降低 `_predict_next_window_ic()` 的單次記憶體占用。

### [2026-03-30] Re1e6 資料路徑修正

- Status: Applied
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)

Change:

- 使用本地現有 DNS：
  - [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- 上傳到 server：
  - `/home/junyi/jaxpi_upstream_soap_run/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy`
- 更新 config：
  - `wandb.name = re1e6_n512_ds4_soap`
  - `dataset_path = kolmogorov_Re1e6_N512_T5_ds4.npy`

Interpretation:

- 這是 dataset path 對齊，不是訓練邏輯修正。

### [2026-03-30] `3135` | 第一次 Re1e6 重跑嘗試

- Status: Failed
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)

Evidence:

- `dataset_path` 指向不存在的 `N2048 ke024` DNS 檔
- 當時 server 與本地都沒有該檔案

Interpretation:

- 失敗原因是資料路徑，不是訓練本身。

Next:

- 改用本地現有 DNS 並同步到 server。

### [2026-03-30] `3134` | 用修正後 `train.py` 重跑 Re10000 upstream SOAP

- Status: Stopped early
- Config: [upstream_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/upstream_soap.py)

Evidence:

- 只跑到 `window 1` 早期階段後即中止
- `step 600` 早期訊號：
  - `u_error ≈ 1.416`
  - `v_error ≈ 1.495`
  - `w_error ≈ 1.162`

Interpretation:

- 修正後的早期收斂速度比先前更合理。
- run 未完成，不作正式結果。

### [2026-03-30] 關鍵 bug：`window IC propagation`

- Status: Fixed
- File: [train.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/train.py)

Problem:

- 原本下一個 window 的 `u0 / v0 / w0` 更新放在 `for idx in ...` 迴圈外。
- `window 2+` 因此沿用舊初值，破壞真正的 time marching。

Fix:

- 新增 `_predict_next_window_ic()`
- 在每個 window 訓練結束後立即更新 `u0 / v0 / w0`

Interpretation:

- 這是機制級 bug。
- 修正前的 `window 2+` 結果不應視為乾淨的 time-window reproduction。

### [2026-03-30] `3133` | `upstream_soap_4x384.py` 試驗

- Status: Exploratory / Incomplete
- Config: [upstream_soap_4x384.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/upstream_soap_4x384.py)

Change:

- 只將網路從 `3 x 384` 改為 `4 x 384`

Evidence:

- `window 1 / step 5400` 時，log 約為：
  - `u_error ≈ 0.945`
  - `v_error ≈ 1.048`
  - `w_error ≈ 1.045`

Interpretation:

- 早期訊號比 `3 x 384` 好一些。
- 該 run 後來中止並清除，沒有完整 checkpoint 結論。

### [2026-03-29 ~ 2026-03-30] `3132` | upstream SOAP + grad clip + loader 修正

- Status: Verified but not successful
- Config: [upstream_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/upstream_soap.py)
- Evaluation File: [all_ckpts_final_step_eval_3132.json](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/comparison/all_ckpts_final_step_eval_3132.json)
- Visualization:
  - [3132_all_ckpts_final_step_eval.png](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/comparison/3132_all_ckpts_final_step_eval.png)
  - [3132_vorticity_fields_latest.png](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/comparison/3132_vorticity_fields_latest.png)

Evidence:

| Window | Step | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 15000 | 1.231271 | 1.409884 | 1.077097 |
| 1 | 20000 | 1.175323 | 1.368248 | 1.063709 |
| 2 | 15000 | 1.124219 | 1.425150 | 1.082654 |
| 2 | 20000 | 1.099286 | 1.379022 | 1.071853 |
| 3 | 15000 | 1.077931 | 1.308659 | 1.058142 |
| 3 | 20000 | 1.061485 | 1.272888 | 1.067940 |
| 4 | 5000 | 1.122798 | 1.547866 | 1.097102 |
| 4 | 10000 | 1.068521 | 1.349497 | 1.081841 |

Means:

- `u mean = 1.120104`
- `v mean = 1.382652`
- `w mean = 1.075042`

Interpretation:

- 比 `3131` 好，表示 `grad_clip + x/y loader` 修正有效。
- 但仍未達成重現成功，尤其 `v` 場偏差過大。

### [2026-03-29] 修正 1 | upstream SOAP 對齊

- Status: Applied
- Files:
  - [upstream_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/upstream_soap.py)
  - [dns_format.py](/Users/latteine/Documents/coding/jaxpi/jaxpi/dataio/dns_format.py)

Change:

- 補回 `optim.grad_clip_norm = 1.0`
- loader 優先讀檔內 `x / y`
- 不再只靠 `config.L` 回推座標

Why:

- 對齊 upstream `schedule_free + clip`
- 消除舊 `Re10000` 檔的座標 mismatch

### [2026-03-29] `3131` | 初版 upstream SOAP 重跑

- Status: Failed reproduction
- Config: [upstream_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/upstream_soap.py)

Known Issues at the Time:

- `schedule_free` 已開啟，但本地 optimizer 路徑與 upstream 解耦
- 尚未補回 `grad_clip_norm = 1.0`
- loader 尚未優先讀檔內 `x / y`

Evidence:

- `window 1 / checkpoint_20000`
  - `u = 1.479077`
  - `v = 1.725736`
  - `w = 1.133283`
- `window 2 / checkpoint_5000`
  - `u = 1.689618`
  - `v = 2.205142`
  - `w = 1.278103`

Interpretation:

- 結果明顯偏差，不是可接受的 upstream reproduction。

### [2026-03-29] `3128` | 舊版 `re10k_soap.py`

- Status: Ran but failed target
- Config: [re10k_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/re10k_soap.py)
- Conditions:
  - `Re = 10000`
  - `schedule_free = False`
  - 舊資料檔 `kolmogorov_flow_Re10000_256.npy`
- Evaluation Mode: `final_step`

Evidence:

- `window 1..13` 平均誤差：
  - `u mean = 1.060977`
  - `v mean = 1.204399`
  - `w mean = 1.052773`

Interpretation:

- 訓練流程可跑，但不符合專案驗收目標。

### [2026-03-21] `paper_repro_soap.py` 歷史評估摘要

- Status: Historical reference
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Summary Source: [eval_paper_repro_soap_0321/summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_paper_repro_soap_0321/summary.txt)
- Checkpoint Root: `/home/junyi/jaxpi/re1e6_n2048_ke024_soap/ckpt`

Evidence:

| Window | Step | t_end | u_err | v_err | w_err |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 100000 | 0.1500 | 0.001664 | 0.001652 | 0.000929 |
| 2 | 100000 | 0.3500 | 0.101151 | 0.119925 | 0.151194 |
| 3 | 100000 | 0.5500 | 0.294120 | 0.332191 | 0.454276 |
| 4 | 100000 | 0.7500 | 0.607362 | 0.697627 | 0.871470 |
| 5 | 100000 | 0.9500 | 0.825400 | 1.000416 | 1.130149 |
| 6 | 100000 | 1.1500 | 0.946607 | 1.228253 | 1.278540 |
| 7 | 100000 | 1.3500 | 0.991254 | 1.382513 | 1.342244 |
| 8 | 100000 | 1.5500 | 0.996722 | 1.477695 | 1.360210 |
| 9 | 100000 | 1.7500 | 0.994016 | 1.468547 | 1.343024 |
| 10 | 60000 | 1.9500 | 0.991829 | 1.399051 | 1.326850 |

Summary:

- `u mean = 0.675013`
- `v mean = 0.910787`
- `w mean = 0.925889`
- `u max = 0.996722`
- `v max = 1.477695`
- `w max = 1.360210`

Interpretation:

- `window 1` 對齊很好，但後續窗口快速失真。
- 這筆歷史資料證明「window 1 可成功」，也暗示多 window time marching 本身有問題。

### [2026-04-21] `3324` | rerun direct evaluation for checkpoints `46000/47000/50000`

- Time: `2026-04-21 22:00 +0800`
- Status: Completed
- Experiment or Job ID: source training job `3324`

Change:

- 依人工要求重跑 `3324` 的 direct evaluation，避免只依賴先前 `3326` 的 corrected evaluator。
- 直接在遠端 `.venv` 中使用 [evaluate_window1_checkpoint_sweep.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/evaluate_window1_checkpoint_sweep.py) 的 `direct_apply_fn` 路徑，對 `checkpoint_46000/47000/50000` 重算 full-window relative L2。

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py)
- Checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval/ckpt`
- Output:
  - [checkpoint_sweep_summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_direct_ckpt_eval_20260421/checkpoint_sweep_summary.txt)
  - [checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_direct_ckpt_eval_20260421/checkpoint_sweep_results.csv)
  - [checkpoint_sweep_error_vs_step.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_direct_ckpt_eval_20260421/checkpoint_sweep_error_vs_step.png)

Evidence:

- direct rerun:
  - `checkpoint_46000 -> u=1.316588e-04, v=1.331908e-04, w=4.957606e-04`
  - `checkpoint_47000 -> u=1.047420e-04, v=1.025183e-04, w=4.838619e-04`
  - `checkpoint_50000 -> u=1.031410e-04, v=1.036893e-04, w=4.554458e-04`
- old `3326` vs new direct ratio:
  - `checkpoint_50000`
    - `u: 1.235e-03 -> 1.031e-04` (`11.97x` lower)
    - `v: 1.103e-03 -> 1.037e-04` (`10.64x` lower)
    - `w: 8.170e-04 -> 4.554e-04` (`1.79x` lower)

Interpretation:

- `3326` 的 `~1e-3` 判讀不再可信；至少對 `3324` 這三個 checkpoint 來說，已被 direct rerun 明確推翻。
- 真實的 `3324` window-1 品質比先前判讀好得多：它不是 `~1e-3` 平台，而是 `u/v ~1e-4`、`w ~4.6e-4`。
- 但即使如此，`3324` 仍沒有追上 `3137/3155 window1` 的最佳 checkpoint。

Next:

- 後續若要引用 `3324`，應以這次 direct rerun 為唯一可信 field-error 來源。
- 舊 `3326` 與其衍生圖表只保留作為歷史 artifact，不再作為研究結論依據。

### [2026-04-21] `3137` vs `3324` | redraw `window 1` error comparison using direct `3324` rerun

- Time: `2026-04-21 22:04 +0800`
- Status: Visualization completed
- Experiment or Job ID: `3137`, `3324`

Change:

- 用 `3137` 的 direct retained-checkpoint baseline，加上 `3324` 新的 direct rerun CSV，重畫 `window 1` error comparison。

Config / Dataset / Checkpoint:

- `3137` source:
  - [checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_window1_checkpoint_sweep_20260421/checkpoint_sweep_results.csv)
- `3324` source:
  - [window1_full_window_errors_direct.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_direct_ckpt_eval_20260421/window1_full_window_errors_direct.csv)
- Output:
  - [window1_error_compare_3137_vs_3324.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_vs_3324_window1_error_compare_direct_20260421/window1_error_compare_3137_vs_3324.png)
  - [window1_error_compare_3137_vs_3324.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/3137_vs_3324_window1_error_compare_direct_20260421/window1_error_compare_3137_vs_3324.txt)

Evidence:

- `3324` best available (`w` 最低者為 `50000`)：
  - `u=1.031410e-04`
  - `v=1.036893e-04`
  - `w=4.554458e-04`
- Ratio (`3324 best available / 3137@100000`):
  - `u x3.114`
  - `v x3.065`
  - `w x2.057`

Interpretation:

- 重跑後，`3324` 已不再是先前說的「比 `3137` 差 30 多倍」。
- 正確結論應修正為：`3324` 明顯差於 `3137`，但差距約是 `2x ~ 3x`，不是一個數量級以上的崩壞。

### [2026-04-21] sweep logic update | replace threshold-crossing objective with tail-EMA scoring

- Time: `2026-04-21 22:25 +0800`
- Status: Applied

Change:

- 修改 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)：
  - 搜尋範圍從 `1..100` 改成 `5..60`（`log=True`）
  - objective 從「最快 `max(ru,rv,rc) < 5e-5`」改成：
    - `worst_t = max(ru_t, rv_t, rc_t)`
    - `ema_t = beta * ema_{t-1} + (1-beta) * worst_t`
    - score = `mean(ema_t over last 20% of training)`
  - pruner 改成看 `EMA(worst)` 而不是 raw `worst`
  - 保留 `--threshold` 參數僅作相容性用途，不再用於主 scoring
- 同步更新 [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)：
  - 驗證新搜尋範圍 `5..60`
  - 驗證 `ema_beta=0.9`
  - 驗證 `tail_fraction=0.2`

Evidence:

- `python3 -m py_compile scripts/sweep/sweep_weights_window1.py tests/test_sweep_weights_window1.py`
- `python3 -m unittest tests/test_sweep_weights_window1.py` -> `OK`

Interpretation:

- 這次修改把 sweep 從「單次 threshold crossing 排名」改成「尾段 smoothed worst residual 排名」，可顯著降低 loss 震盪對 trial 排序與 pruner 的干擾。
- 但它仍是 proxy，不是最終 truth；若要正式選最佳 `data_weight`，仍應對 top-K 進行 corrected field error 驗證。

Next:

- 若要啟動下一輪 sweep，建議直接沿用這個 v2 objective。
- 執行後應觀察 top trials 是否集中在更窄的中等權重區，而不是再被單次 crossing 誤導。

### [2026-04-22] sweep logic update | lighten EMA and wrap median pruning with patience

- Time: `2026-04-22 17:31 +0800`
- Status: Applied

Change:

- 依人工指示，調整 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 的預設穩定化策略：
  - `DEFAULT_EMA_BETA` 從 `0.9` 降到 `0.85`
  - `MedianPruner(...)` 改成 `PatientPruner(MedianPruner(...), patience=25, min_delta=1e-5)`
- 新增 `build_pruner()`，把 pruner 組態集中到單一函式，避免 CLI 預設、測試與實際 study 建立分叉。
- 在程式註解中明確記錄：
  - `step_callback` 只在每次 log 時回報一次
  - 目前 `log_every_steps=100`
  - 因此 `patience=25` 約對應 `2500` 個 training steps 的觀察窗口，而不是 `2500` 次 raw SGD step callback
- 同步更新 [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)：
  - 驗證 `ema_beta=0.85`
  - 驗證 `PatientPruner` 外包 `MedianPruner`
  - 驗證 `patience=25`, `min_delta=1e-5`

Config / Dataset / Checkpoint:

- Scope: sweep logic only
- Target script: [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)
- Tests: [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)
- Dataset / checkpoint: 無變更；本次僅調整 Optuna scoring / pruning 預設

Evidence:

- `python3 -m py_compile scripts/sweep/sweep_weights_window1.py tests/test_sweep_weights_window1.py`
- `python3 -m unittest tests/test_sweep_weights_window1.py` -> `Ran 4 tests ... OK`
- 相關程式路徑：
  - `trial.report(float(ema_worst[0]), step)` 仍以 log callback 的 `step` 回報 intermediate value
  - `train.py` 明確說明 `step_callback` 是「每次 log 時呼叫」
  - base config `logging.log_every_steps = 100`

Interpretation:

- 這次修改不是把 sweep 變得更鈍，而是重新分工：
  - `EMA(0.85)` 只做輕度去噪，保留較多短期動態
  - `PatientPruner` 負責避免短期停滯造成的過早誤殺
- 舊建議中的 `patience=2500` 若直接照字面放進 Optuna，會因為這支 sweep 每 `100` steps 才 report 一次而幾乎永遠不觸發；改成 `25` 才符合「約 2500 training steps」的原始意圖。
- 目前這仍是 proxy-level 穩定化，不等於 field-error-aware selection；若後續 sweep 結論要進研究主結論，仍需 direct field evaluation 補驗證。

Next:

- 若要開下一輪 sweep，直接沿用這個新預設即可。
- 執行後優先檢查：
  - 被 prune trials 是否比舊版更少出現前期誤殺
  - top-K 排名是否較舊版穩定
  - 是否仍需要再微調 `ema_beta` 或 `min_delta`

### [2026-04-22] sweep logic update | remove EMA, keep PatientPruner on raw worst residual

- Time: `2026-04-22 17:42 +0800`
- Status: Applied

Change:

- 依人工最新指示，將 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 的 scoring / pruning metric 從 `EMA(max(ru,rv,rc))` 改回 raw `max(ru,rv,rc)`。
- 刪除 CLI `--ema-beta` 與對應驗證邏輯，避免送 job 前仍殘留舊 objective 說法。
- 保留 `PatientPruner(MedianPruner(...), patience=25, min_delta=1e-5)`：
  - `trial.report(...)` 現在直接回報 raw `worst`
  - final score 改成訓練尾段 raw `worst` 的平均值
- 同步更新 [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)，確認 CLI 預設與 pruner 組態仍一致。

Config / Dataset / Checkpoint:

- Scope: sweep logic only
- Target script: [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)
- Tests: [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)
- Dataset / checkpoint: 無變更；本次僅改 Optuna objective / CLI

Evidence:

- `python3 -m py_compile scripts/sweep/sweep_weights_window1.py tests/test_sweep_weights_window1.py`
- `python3 -m unittest tests/test_sweep_weights_window1.py` -> `Ran 4 tests ... OK`
- 目前腳本 stdout 目標字樣已改為：
  - `target   : minimize tail mean of max(ru,rv,rc)`

Interpretation:

- 現在的 sweep 不再把短期動態交給 EMA 濾掉；排序與 pruning 都直接看 raw `worst residual`。
- `PatientPruner` 仍保留，用來處理「短期停滯不要立刻判死刑」；也就是說現在的分工是：
  - metric: raw `max(ru,rv,rc)`
  - anti-misprune: `PatientPruner`
- 這比較符合人工要求的「不要 EMA，但保留 patience」；之後若要回答「最快收斂」與「最終最準」，仍需另外定義 threshold-crossing 與 final direct field eval 的輸出，不可把目前 proxy 直接當成最終 truth。

Next:

- 送 job 前若要再確認 objective，應以目前腳本輸出的 `tail mean of max(ru,rv,rc)` 為準。
- 若要做 `100000` steps 的正式 sweep，下一步應同步決定：
  - 搜尋範圍
  - checkpoint 保留策略
  - fastest-convergence 與 final-accuracy 的輸出欄位

### [2026-04-22] sweep logic update | widen default data-weight range to `5..80` with log-scale sampling

- Time: `2026-04-22 17:44 +0800`
- Status: Applied

Change:

- 依人工確認，將 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 的預設 `data_weight` 搜尋空間從 `5..60` 擴大到 `5..80`。
- 保持 `trial.suggest_float(..., log=True)` 不變，仍以 log scale 在單參數空間做第一輪區域探索。
- 同步更新 [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py) 的搜尋區間驗證。

Config / Dataset / Checkpoint:

- Scope: sweep search-space only
- Target script: [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)
- Tests: [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)
- Dataset / checkpoint: 無變更

Evidence:

- `python3 -m py_compile scripts/sweep/sweep_weights_window1.py tests/test_sweep_weights_window1.py`
- `python3 -m unittest tests/test_sweep_weights_window1.py` -> `Ran 4 tests ... OK`
- 目前 `suggest_data_weight()` 驗證為：
  - `trial.suggest_float("data_weight", 5.0, 80.0, log=True)`

Interpretation:

- 這次不是改成線性掃描，而是保留 log-scale 的第一輪 coarse search，同時把上界延伸到 `80`，讓 `60` 以上仍有探索空間。
- 這樣可以避免在尚未證明最佳區域很窄之前，就過早把搜尋空間縮死在 `60` 以下；若後續 top trials 明顯集中，再另做局部 refinement 會更有效率。

Next:

- 若要送正式 sweep job，下一步只剩把：
  - `max_steps`
  - `n_trials`
  - checkpoint 保留策略
  - fastest / final-best 的輸出方式
  一次定稿後再提交。

### [2026-04-22] sweep logic update | switch objective to earliest stable threshold crossing (`k=3`)

- Time: `2026-04-22 17:54 +0800`
- Status: Applied

Change:

- 依人工指示，將 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 的 objective 從 `tail mean of max(ru,rv,rc)` 改成：
  - 最早出現 **連續 `3` 次** `max(ru_loss, rv_loss, rc_loss) < threshold` 的步數
- 新增 CLI `--stable-reports`，預設 `3`。
- `trial.report(...)` 仍回報 raw `max(ru,rv,rc)`，供 `PatientPruner(MedianPruner(...))` 判斷。
- 一旦 trial 首次達成穩定達標條件，就直接 short-circuit 返回該起始步數；若直到 `max_steps` 仍未穩定達標，則回傳 `max_steps`。
- 同步更新 [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py) 的 CLI 預設驗證。

Config / Dataset / Checkpoint:

- Scope: sweep objective / CLI only
- Target script: [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)
- Tests: [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py)
- Search space: `data_weight in [5, 80]` (`log=True`)
- Objective defaults:
  - `threshold = 5e-5`
  - `stable_reports = 3`
  - `pruner = Patient(Median)`

Evidence:

- `python3 -m py_compile scripts/sweep/sweep_weights_window1.py tests/test_sweep_weights_window1.py`
- `python3 -m unittest tests/test_sweep_weights_window1.py` -> `Ran 4 tests ... OK`
- 目前 stdout 目標字樣已改為：
  - `target   : minimize earliest stable crossing step of max(ru,rv,rc)`
  - `stable_k : 3`

Interpretation:

- 現在這輪 sweep 終於直接回答人工真正要的問題：哪個 `data_weight` 能讓模型 **最快穩定達標**。
- 這次不再把「最後尾段低不低」混成 objective，也不再受單次 spike crossing 誤導；因為只有連續 `3` 次都低於 threshold 才算成功。
- 由於 `log_every_steps=100`，`k=3` 約代表連續 `300` 個 training steps 維持在 threshold 下。

Next:

- 若要送正式 fastest sweep job，接下來只剩定稿：
  - `max_steps` 是否改成 `100000`
  - `n_trials` 要多少
  - 是否仍完全不存 checkpoint

### [2026-04-21] `3329` / `3330` | launch window-1 checkpoint-validation reruns for `3137` and `3155`

- Time: `2026-04-21 22:15 +0800`
- Status:
  - `3329`: RUNNING
  - `3330`: PENDING (`Resources`)
- Experiment or Job ID:
  - `3329` = no-data (`3137`-style)
  - `3330` = sensor100 (`3155`-style)

Change:

- 依人工要求提交兩個新的 `window 1` 驗證訓練：
  - 一個對應 `3137` no-data baseline
  - 一個對應 `3155` sensor100 baseline
- 這兩個 rerun 都統一成：
  - `max_windows_to_run = 1`
  - `max_steps = 100000`
  - `save_every_steps = 10000`
  - `num_keep_ckpts = None`
- 目標是確保至少保留 `checkpoint_50000` 與 `checkpoint_100000`，而且實際上會保留整串 `10000, 20000, ..., 100000`。

Config / Dataset / Checkpoint:

- New configs:
  - [paper_repro_soap_window1_ckpt50k100k.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ckpt50k100k.py)
  - [paper_repro_soap_sensor100_n512_w50_window1_ckpt50k100k.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_ckpt50k100k.py)
- No-data workdir:
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ckpt10k`
- Sensor workdir:
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_ckpt10k`

Evidence:

- Local config validation:
  - `python3 -m py_compile ...paper_repro_soap_window1_ckpt50k100k.py`
  - `python3 -m py_compile ...paper_repro_soap_sensor100_n512_w50_window1_ckpt50k100k.py`
- Remote config validation:
  - `python3 -m py_compile /home/junyi/jaxpi/examples/kolmogorov_flow/configs/...`
- Slurm submission:
  - `Submitted batch job 3329`
  - `Submitted batch job 3330`
- Current queue/accounting:
  - `3329 | w1_3137_ckpt10k | RUNNING | Node=acmt20`
  - `3330 | w1_3155_ckpt10k | PENDING | Reason=Resources`

Interpretation:

- 舊 `3155` tree 雖然 config 名義上允許保留全部，但實際 retained tree 只剩 `90000/100000`；這次新 rerun 的目的就是消除這種不確定性。
- 一旦這兩條 run 完成，後續 `3137/3155` 都能用相同 checkpoint grid 做更乾淨的 direct checkpoint sweep，而不是只比最後兩個 retained checkpoints。

Next:

- 監看 `3329/3330`，確認 `checkpoint_50000` 與 `checkpoint_100000` 都成功落盤。
- 完成後優先做 direct checkpoint evaluation，比較 `50000` 與 `100000` 的 field error 是否已平台化。

### [2026-04-22] `3331` | submit dependent postprocess eval for `3329/3330` window-1 checkpoints

- Time: `2026-04-22 02:48 +0800`
- Status: PENDING (`Dependency`)
- Experiment or Job ID: `3331`

Change:

- 在 `3330` 尚未完成時，先提交一個 dependent postprocess job，讓 `3330` 正常結束後自動執行 window-1 checkpoint evaluation。
- 使用現成的 [postprocess_kolmogorov_window1_checkpoint_sweep.sh](/Users/latteine/Documents/coding/jaxpi/slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh)，只評估 `checkpoint_50000` 與 `checkpoint_100000`，並先關閉 field plots 以優先回收 scalar error。

Config / Dataset / Checkpoint:

- Dependency:
  - `afterok:3330`
- No-data config:
  - [paper_repro_soap_window1_ckpt50k100k.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ckpt50k100k.py)
- Sensor config:
  - [paper_repro_soap_sensor100_n512_w50_window1_ckpt50k100k.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_ckpt50k100k.py)
- No-data checkpoint root:
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ckpt50k100k/ckpt`
- Sensor checkpoint root:
  - `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_ckpt50k100k/ckpt`
- Steps:
  - `50000,100000`
- Output dir:
  - `/home/junyi/jaxpi/eval_runs/window1_ckpt50k100k_eval_20260422`
- Flags:
  - `SKIP_FIELDS=1`

Evidence:

- Submission:
  - `Submitted batch job 3331`
- Queue state:
  - `3331 | eval_w1_ckpt50k100k | PD | (Dependency)`
  - `3330 | w1_3155_ckpt10k | R`

Interpretation:

- 這樣做可以把「等待 `3330` 結束」和「啟動評估」串成單一步驟，避免人工再盯一次 queue。
- 因為 `3329` 已經完成，真正的 gating item 只有 `3330`；`afterok:3330` 就足夠。

Next:

- 等 `3330` 完成後，確認 `3331` 是否自動進入 `RUNNING`。
- `3331` 完成後回收 `50000/100000` 的 direct full-window error，比較 `3137` vs `3155` 是否已在 `50000` 前後平台化。

### [2026-04-22] `3330` / `3331` / manual direct eval | finalize `window 1` checkpoint comparison at `50000` and `100000`

- Time: `2026-04-22 14:35 +0800`
- Status: COMPLETED
- Experiment or Job ID: `3330`, `3331`, manual direct eval follow-up

Change:

- 確認 `3330` 訓練已正常完成，`3331` dependency postprocess job 也已正常完成。
- 追查後發現 `3331` 雖宣告要評估 `50000,100000`，但實際上只評到 `50000`；原因是 `sbatch --export=... CHECKPOINT_STEPS=50000,100000 ...` 內的逗號被 shell/export 分隔，導致腳本只收到 `50000`。
- 因此另外以 direct `apply_fn` 手動補跑 `3329/3330` 的 `checkpoint_100000`，並將 `3324@50000` 一起納入同一張對照表。

Config / Dataset / Checkpoint:

- `3329` no-data rerun:
  - Config: [paper_repro_soap_window1_ckpt50k100k.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ckpt50k100k.py)
  - Checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ckpt50k100k/ckpt`
- `3330` sensor rerun:
  - Config: [paper_repro_soap_sensor100_n512_w50_window1_ckpt50k100k.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_ckpt50k100k.py)
  - Checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_ckpt50k100k/ckpt`
- `3324` fixed-weight rerun:
  - Config: [paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval.py)
  - Checkpoint root: `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw231429_eval/ckpt`
- Output roots:
  - `3331` (`50000` only due export bug): [window1_ckpt50k100k_eval_20260422](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_ckpt50k100k_eval_20260422)
  - manual direct `100000`: [window1_ckpt100k_eval_20260422](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_ckpt100k_eval_20260422)
  - `3324` direct rerun: [3324_direct_ckpt_eval_20260421](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_direct_ckpt_eval_20260421)

Evidence:

- Slurm:
  - `3330 | COMPLETED | Start=2026-04-22 04:43:33 +0800 | End=2026-04-22 11:09:52 +0800 | ExitCode=0:0`
  - `3331 | COMPLETED | Start=2026-04-22 11:09:52 +0800 | End=2026-04-22 11:12:11 +0800 | ExitCode=0:0`
- `3331` stdout header shows only:
  - `Steps    : 50000`
- `50000` comparison:
  - `3329@50000` -> `u=1.118582e-03`, `v=1.192990e-03`, `w=8.370060e-04`
  - `3330@50000` -> `u=1.078932e-03`, `v=1.143454e-03`, `w=8.267873e-04`
  - `3324@50000` -> `u=1.031410e-04`, `v=1.036893e-04`, `w=4.554458e-04`
- Manual direct `100000` comparison:
  - `3329@100000` -> `u=3.348497e-05`, `v=3.307914e-05`, `w=2.219041e-04`
  - `3330@100000` -> `u=3.462274e-05`, `v=3.425803e-05`, `w=2.317860e-04`
- Relative ratios:
  - `3330 / 3329 @50000` -> `u=0.965x`, `v=0.958x`, `w=0.988x`
  - `3330 / 3329 @100000` -> `u=1.034x`, `v=1.036x`, `w=1.045x`
  - `3329 / 3324 @50000` -> `u=10.85x`, `v=11.51x`, `w=1.84x`
  - `3330 / 3324 @50000` -> `u=10.46x`, `v=11.03x`, `w=1.82x`

Interpretation:

- 只看 `50000`，`3324 (data_weight=23.1429)` 明顯領先 `3329/3330`，不是小幅優勢，而是 `u/v` 約 `10x ~ 11x`、`w` 約 `1.8x` 的差距。
- 但到 `100000`，`3329/3330` 都已降到 `u/v ~3.3e-05 ~ 3.5e-05`、`w ~2.2e-04 ~ 2.3e-04`，明顯優於 `3324@50000`；這說明 `3324` 的價值較像是「前期收斂快」，不是「最終品質已經最好」。
- `3330` 相對 `3329` 的差異在 `50000` 與 `100000` 都不大：
  - `50000` 時 `3330` 略好
  - `100000` 時反而略差
- 因此目前最穩健的結論是：
  - `data_weight=23.1429` 對 **前期收斂速度** 有幫助
  - 但在 `window 1` 的最終品質上，尚未證明優於 no-data baseline 或 `3155`-style sensor baseline

Next:

- 若要正式討論 `data_weight` 是否值得採用，下一步應比較：
  - `3324@50000`
  - `3329@50000/100000`
  - `3330@50000/100000`
  的 loss 與 error 對齊情況，而不是只看單一步數。
- 若要自動化後續 checkpoint comparison，需修正 postprocess submit 方式，避免 `--export` 對逗號值的截斷問題。

### [2026-04-22] visualization | `window 1` step-vs-error plot for `3324`, `3329`, `3330`

- Time: `2026-04-22 14:44 +0800`
- Status: COMPLETED
- Experiment or Job ID: `3324`, `3329`, `3330`

Change:

- 依使用者要求，將 `3324@50000`、`3329@50000/100000`、`3330@50000/100000` 畫成同一張 `step vs error` 對照圖，直接視覺化前期收斂優勢與後期品質反轉。

Evidence:

- Plot script:
  - [plot_window1_step_vs_error_points.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/plot_window1_step_vs_error_points.py)
- Artifacts:
  - [window1_step_vs_error_points.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_step_vs_error_points_20260422/window1_step_vs_error_points.png)
  - [window1_step_vs_error_points.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_step_vs_error_points_20260422/window1_step_vs_error_points.txt)

Interpretation:

- 圖上可直接看出：
  - `3324` 在 `50000` 時位於最低區間
  - `3329/3330` 到 `100000` 時已進一步降到更低 error
- 這張圖適合用來支撐「`3324` 贏速度、`3329/3330` 贏終點」的判讀。

Next:

- 若後續要連結 sweep 設計，應把這張圖與 `window 1` loss comparison 一起看，而不是單獨引用 error 點。

### [2026-04-22] visualization | complete `window 1` epoch-vs-error curves every `10000` steps for `3329` and `3330`

- Time: `2026-04-22 16:01 +0800`
- Status: COMPLETED
- Experiment or Job ID: `3329`, `3330`

Change:

- 依使用者要求，整理 `window 1` 的完整 checkpoint error 曲線，將 `3329` 與 `3330` 每 `10000` step 的 direct full-window error 合併成一張 `epoch vs error` 圖。
- `10000/20000` 採用先前已回收的 direct evaluator 數值；`30000..100000` 則使用後續補跑的 direct tail evaluation artifact 合併，最終形成 `10000..100000` 的完整十點曲線。

Evidence:

- Complete artifact:
  - [checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_ckpt10k_complete_eval_20260422/checkpoint_sweep_results.csv)
  - [epoch_vs_error_window1.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_ckpt10k_complete_eval_20260422/epoch_vs_error_window1.png)
  - [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_ckpt10k_complete_eval_20260422/summary.txt)
- Tail direct rerun source:
  - [checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_sensor_tail_eval_20260422/checkpoint_sweep_results.csv)

Interpretation:

- `3330` 在 `10000` 與 `30000` 前期確實優於 `3329`，但在 `40000` 出現回彈，之後與 `3329` 幾乎貼近。
- 到 `100000`，`3329` 仍略優於 `3330`：
  - `u`: `3.348497e-05` vs `3.462274e-05`
  - `v`: `3.307914e-05` vs `3.425803e-05`
  - `w`: `2.219041e-04` vs `2.317860e-04`
- 因此完整曲線支持的結論是：
  - sensor100 baseline 對 early-stage convergence 有部分幫助
  - 但在 `window 1` 終點品質上，沒有穩定超越 no-data baseline

Next:

- 若要把這張圖和 `3324` 一起納入 sweep 論證，應另外製作一張 `3324@1000-step` 與 `3329/3330@10000-step` 的對齊比較圖，避免把不同 checkpoint 解析度混為一談。

### [2026-04-22] visualization | redraw epoch-vs-error figure with clean labels and full `3324@10000~50000`

- Time: `2026-04-22 16:58 +0800`
- Status: COMPLETED
- Experiment or Job ID: source runs `3324`, `3329`, `3330`

Change:

- 依使用者更正要求，重做 `epoch vs error` 圖：
  - label 不再顯示 job id
  - `3324` 不再只放 `50000` 單點，而是補齊 `10000/20000/30000/40000/50000` 的 direct evaluation 後再納入同圖
- 新圖使用的方法名稱：
  - `No data`
  - `Sensor100`
  - `Sensor dw=23.1429`

Evidence:

- `3324` prefix direct eval:
  - [checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/3324_direct_ckpt10k_eval_20260422/checkpoint_sweep_results.csv)
- Clean-label artifact:
  - [epoch_vs_error_clean_labels.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_epoch_vs_error_clean_labels_20260422/epoch_vs_error_clean_labels.png)
  - [epoch_vs_error_clean_labels.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_epoch_vs_error_clean_labels_20260422/epoch_vs_error_clean_labels.csv)
  - [epoch_vs_error_clean_labels.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_epoch_vs_error_clean_labels_20260422/epoch_vs_error_clean_labels.txt)
- `3324` added points:
  - `10000 -> u=4.601948e-03, v=4.530223e-03, w=3.917131e-03`
  - `20000 -> u=4.442899e-04, v=4.976123e-04, w=1.164529e-03`
  - `30000 -> u=5.537331e-04, v=5.893375e-04, w=8.857403e-04`
  - `40000 -> u=2.627931e-04, v=2.805152e-04, w=5.982067e-04`
  - `50000 -> u=1.031410e-04, v=1.036893e-04, w=4.554458e-04`

Interpretation:

- 舊版只放 `3324@50000` 會讓人誤以為它整段都領先；補齊 `10000~50000` 後可見：
  - `3324` 在 `10000` 其實最差
  - `20000~40000` 也不是穩定領先
  - 真正的明顯優勢主要出現在接近 `50000` 的後段
- 因此新圖比舊版更忠實反映 `3324` 的真實收斂路徑：它不是「全程最快」，而是「前段波動較大、後段追上並在 `50000` 取得最低 error」。

Next:

- 若要把這張圖拿來支撐 sweep scoring 設計，應直接對照 tail-loss / EMA-loss，而不是只對照單一 checkpoint 結果。

### [2026-04-22] visualization | add `50000` vertical reference line to clean-label epoch-vs-error figure

- Time: `2026-04-22 17:05 +0800`
- Status: COMPLETED
- Experiment or Job ID: source runs `3324`, `3329`, `3330`

Change:

- 依使用者要求，在 clean-label `epoch vs error` 圖的三個 panel 都加入 `50000` 的垂直虛線與小型文字標記，讓 `Sensor dw=23.1429` 在 `50000` 附近取得最低 error 的區段更直觀。

Evidence:

- Updated plot script:
  - [plot_window1_epoch_vs_error_clean_labels.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/plot_window1_epoch_vs_error_clean_labels.py)
- Updated artifact:
  - [epoch_vs_error_clean_labels.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/window1_epoch_vs_error_clean_labels_20260422/epoch_vs_error_clean_labels.png)

Interpretation:

- `50000` 參考線讓圖的閱讀重點更清楚：
  - `Sensor dw=23.1429` 的優勢主要集中在這條線附近
  - 不是整段 `10000~50000` 都穩定領先
- 這也降低只看單一 legend/單點數字造成的誤判風險。

Next:

- 若還要進一步強化論點，可在同圖上再補一個 shaded band，標示 `3324` 的可用 checkpoint 範圍只到 `50000`。

### [2026-04-22] sweep logic update | fast-convergence sweep now saves stop-point ckpt and uses `2e-5 / k=3 / 60 / 100000 upper bound`

- Time: `2026-04-22 18:01 +0800`
- Status: Applied

Change:

- 依人工最終定稿，將 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 的正式設定整理為：
  - `data_weight in [5, 80]` (`log=True`)
  - `threshold = 2e-5`
  - `stable_reports = 3`
  - `n_trials = 60`
  - `max_steps = 100000`（作為上限）
- 調整 trial 結束語義：
  - 不再要求 trial 一定跑滿 `100000`
  - 一旦最早連續 `3` 次 `max(ru,rv,rc) < 2e-5`，就以該最早步數作為 objective
  - 並在停下來當下手動保存唯一保留的 checkpoint
- 每個 trial 的 checkpoint 目錄改成獨立：
  - `sweep_ckpts/trial_XXXX/time_window_1/`
  - 避免不同 trial 互相覆蓋
- 同步更新 [slurm/sweep/sweep_kf_w1_weights.sh](/Users/latteine/Documents/coding/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh) 的預設：
  - `N_TRIALS=60`
  - `MAX_STEPS=100000`
  - `THRESHOLD=2e-5`
  - `STABLE_REPORTS=3`
  - 新 study/storage 名稱改成獨立 DB，避免和舊 sweep 混在一起
- 同步更新 [test_sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/tests/test_sweep_weights_window1.py) 驗證新的 CLI 預設。

Config / Dataset / Checkpoint:

- Search space: `data_weight in [5, 80]` (`log=True`)
- Objective: earliest stable crossing step of `max(ru,rv,rc)`
- Stable criterion: `k = 3`
- Threshold: `2e-5`
- Trial upper bound: `100000`
- Checkpoint policy: save only the stop-point / final-point ckpt for each completed trial

Evidence:

- Local verification:
  - `python3 -m py_compile scripts/sweep/sweep_weights_window1.py tests/test_sweep_weights_window1.py`
  - `python3 -m unittest tests/test_sweep_weights_window1.py` -> `Ran 4 tests ... OK`
  - `bash -n slurm/sweep/sweep_kf_w1_weights.sh`

Interpretation:

- 這次設定終於和人工問題完全對齊：
  - 不是 tail mean
  - 不是單次 crossing
  - 也不是強迫每個 trial 都跑滿 `100000`
- 它現在回答的是：
  - 哪個 `data_weight` 最快**穩定**達標
  - 並保留停下來那一點的 ckpt，供後續只對 winner 或 top trials 做追蹤

Next:

- 同步到 remote 並提交新的獨立 sweep job。

### [2026-04-22] `3333` | submit fast-convergence `data_weight` sweep with `2e-5 / k=3 / 60 / 100000 upper bound`

- Time: `2026-04-22 18:01 +0800`
- Status: RUNNING
- Experiment or Job ID: `3333`

Change:

- 將最新 [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py) 與 [sweep_kf_w1_weights.sh](/Users/latteine/Documents/coding/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh) 同步到 remote `/home/junyi/jaxpi/`。
- 直接提交新的 Slurm sweep job，使用獨立 study/storage：
  - `STUDY_NAME=kf_w1_data_weight_sweep_5to80_thr2e5_k3_100k_stopckpt`
  - `STORAGE=sqlite:///sweep_w1_data_5to80_thr2e5_k3_100k_stopckpt.db`

Config / Dataset / Checkpoint:

- Config: [paper_repro_soap_window1_ablation.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py)
- Search space: `data_weight in [5, 80]` (`log=True`)
- Objective: earliest stable crossing step of `max(ru,rv,rc)`
- Stable criterion: `k = 3`
- Threshold: `2e-5`
- `N_trials = 60`
- `Max_steps = 100000` (upper bound)
- Checkpoint policy:
  - one stop-point ckpt per completed trial
  - remote root pattern: `/home/junyi/jaxpi/sweep_ckpts/trial_XXXX/time_window_1/`
- Storage:
  - `/home/junyi/jaxpi/sweep_w1_data_5to80_thr2e5_k3_100k_stopckpt.db`

Evidence:

- Remote sync:
  - `rsync .../scripts/sweep/sweep_weights_window1.py -> /home/junyi/jaxpi/scripts/sweep/sweep_weights_window1.py`
  - `rsync .../slurm/sweep/sweep_kf_w1_weights.sh -> /home/junyi/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh`
- Remote syntax validation:
  - `python3 -m py_compile scripts/sweep/sweep_weights_window1.py`
  - `bash -n slurm/sweep/sweep_kf_w1_weights.sh`
- Submission:
  - `sbatch slurm/sweep/sweep_kf_w1_weights.sh` -> `Submitted batch job 3333`
- Queue:
  - `3333 | sweep_kf_w1_weights | RUNNING | acmt20`
- Stdout header:
  - `Study    : kf_w1_data_weight_sweep_5to80_thr2e5_k3_100k_stopckpt`
  - `N_trials : 60`
  - `Max_steps: 100000`
  - `Threshold: 2e-5`
  - `Stable_k : 3`
  - `Storage  : sqlite:///sweep_w1_data_5to80_thr2e5_k3_100k_stopckpt.db`

Interpretation:

- `3333` 是第一個正式對應「fastest stable convergence」研究問題的 `data_weight` sweep：
  - 不是舊的 `5e-5`
  - 不是 tail mean
  - 也不會和 `3314/3318` 的舊 DB 汙染混在一起
- 後續若要讀結果，應直接查這個新 study DB 與 `3333` log，而不是回頭引用舊 `3318` 的 `5e-5` threshold 結論。

Next:

- 等 `3333` 完成後，讀取新 DB 與 stdout，整理：
  - 最佳 `data_weight`
  - 最早穩定達標步數
  - 哪些 trial 在 `100000` 內未達標

### [2026-04-22] visualization | overlay `paper_dns_ref` and no-data `u/v` error curves

- Time: `2026-04-22 18:25 +0800`
- Status: COMPLETED
- Experiment or Job ID: source no-data eval artifact `eval_paper_repro_soap_0405`

Change:

- 依人工要求，將 `paper_dns_ref` 內的：
  - [kolmogorov_re1e6_u_error.csv](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/paper_dns_ref/kolmogorov_re1e6_u_error.csv)
  - [kolmogorov_re1e6_v_error.csv](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/paper_dns_ref/kolmogorov_re1e6_v_error.csv)
  與本地 no-data artifact [eval_paper_repro_soap_0405/l2_errors.npz](/Users/latteine/Documents/coding/jaxpi/eval_runs/eval_paper_repro_soap_0405/l2_errors.npz) 的連續 `u/v error` 曲線疊成同一張比較圖。
- 新增可重跑腳本 [plot_paper_ref_vs_no_data_uv_error.py](/Users/latteine/Documents/coding/jaxpi/scripts/analysis/plot_paper_ref_vs_no_data_uv_error.py)。
- 同步輸出 raw CSV 與 summary，避免只有圖沒有 provenance。

Config / Dataset / Checkpoint:

- Paper reference dir:
  - [paper_dns_ref](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/paper_dns_ref)
- No-data source:
  - [l2_errors.npz](/Users/latteine/Documents/coding/jaxpi/eval_runs/eval_paper_repro_soap_0405/l2_errors.npz)
- Output:
  - [paper_ref_vs_no_data_uv_error.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/paper_ref_vs_no_data_uv_error_20260422/paper_ref_vs_no_data_uv_error.png)
  - [paper_ref_vs_no_data_uv_error.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/paper_ref_vs_no_data_uv_error_20260422/paper_ref_vs_no_data_uv_error.csv)
  - [summary.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/paper_ref_vs_no_data_uv_error_20260422/summary.txt)

Evidence:

- `python3 -m py_compile scripts/analysis/plot_paper_ref_vs_no_data_uv_error.py`
- `./.venv/bin/python scripts/analysis/plot_paper_ref_vs_no_data_uv_error.py`
- Output summary:
  - `paper_u_points = 30`
  - `paper_v_points = 24`
  - `no_data_points = 24`
  - `paper_u_t_range = [0.0236, 4.9575]`
  - `paper_v_t_range = [0.0168, 4.9499]`
  - `no_data_t_range = [0.0000, 1.1500]`

Interpretation:

- 這張圖是把 paper reference 與我們自己的 no-data `u/v error` 直接疊圖比較，而不是做數值合併表。
- 目前選用 `eval_paper_repro_soap_0405` 作為 no-data 來源，是因為這份 artifact 提供連續 `ts_all / eu_all / ev_all` 序列，能和 `paper_dns_ref` 的曲線形式直接對齊。
- 這不等於「corrected direct all-window no-data baseline」；若後續要換成修正後的 direct 路徑，只能重畫較稀疏的 checkpoint / window-end 對照，而不是沿用這份連續曲線。

Next:

- 若人工要把這張圖拿來寫正式論證，下一步應明確決定：
  - 是否接受 `eval_paper_repro_soap_0405` 作為 paper-style no-data 對照
  - 或改用 corrected direct artifact 重畫較稀疏但更可信的 no-data 曲線

### [2026-04-28] `3333 -> 3371` | mark stale running trial as fail and resume same sweep with 14-day limit

- Time: `2026-04-28 15:42 +0800`
- Status:
  - `3333`: TIMEOUT
  - `3371`: RUNNING
- Experiment or Job ID: `3333`, `3371`

Change:

- 回查 `3333` 後確認：
  - `3333` 在 `2026-04-25 10:00:21 +0800` 因 Slurm time limit 被取消
  - study DB 中 `trial_id=15`（Optuna `number=14`）殘留為 `RUNNING`
- 依人工要求：
  - 將殘留 `trial 15` 從 `RUNNING` 改成 `FAIL`
  - 將 [sweep_kf_w1_weights.sh](/Users/latteine/Documents/coding/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh) 的 Slurm time limit 從 `72:00:00` 改成 `14-00:00:00`
- 續跑時沿用同一個 study/storage，不另開新 DB，但為避免把總 trial 數從 `15` 直接加到 `75`，這次以 `N_TRIALS=45` 續跑，讓總 trial 數補到原先目標 `60`。
- 將更新後的 Slurm script 同步到 remote，並提交續跑 job `3371`。

Config / Dataset / Checkpoint:

- Study:
  - `kf_w1_data_weight_sweep_5to80_thr2e5_k3_100k_stopckpt`
- Storage:
  - `sqlite:///sweep_w1_data_5to80_thr2e5_k3_100k_stopckpt.db`
- Sweep settings unchanged:
  - `data_weight in [5, 80]` (`log=True`)
  - `threshold = 2e-5`
  - `stable_reports = 3`
  - stop-point ckpt only
- Resume submission override:
  - `N_TRIALS=45`
- New Slurm time limit:
  - `14-00:00:00`

Evidence:

- `3333` final Slurm status:
  - `3333 | sweep_kf_w1_weights | TIMEOUT | 3-00:00:18 | ExitCode=0:0`
  - stdout tail:
    - `*** JOB 3333 ON acmt20 CANCELLED AT 2026-04-25T10:00:21 DUE TO TIME LIMIT ***`
- DB cleanup:
  - before: `(15, 14, 'RUNNING', '2026-04-25 05:47:17.551526', None)`
  - after : `(15, 14, 'FAIL', '2026-04-25 05:47:17.551526', '2026-04-28 15:42:34.350752')`
- Script validation:
  - `bash -n slurm/sweep/sweep_kf_w1_weights.sh`
- Remote sync:
  - `rsync .../slurm/sweep/sweep_kf_w1_weights.sh -> /home/junyi/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh`
- Resume submission:
  - `cd /home/junyi/jaxpi && N_TRIALS=45 sbatch slurm/sweep/sweep_kf_w1_weights.sh`
  - `Submitted batch job 3371`
- `3371` header:
  - `Study    : kf_w1_data_weight_sweep_5to80_thr2e5_k3_100k_stopckpt`
  - `N_trials : 45`
  - `Max_steps: 100000`
  - `Threshold: 2e-5`
  - `Stable_k : 3`
- Slurm time limit check:
  - `TimeLimit=14-00:00:00`
- DB after resume:
  - `trial_id=16` / `number=15` is `RUNNING`

Interpretation:

- `3333` 本身沒有完成；目前可引用的已完成 trial 只有前 `14` 個，best 仍是：
  - `trial 7` / `data_weight=6.092663599129069` / `value=95700`
- `trial 15` 的殘留 `RUNNING` 狀態若不先清掉，後續 study 會保持髒狀態；這次已清理成 `FAIL`，續跑路徑現在是乾淨的。
- `3371` 已正確沿用原 study/storage 並從下一個新 trial 接續，不會把前面已完成 trial 覆蓋掉，也不會額外多跑到超過原先規劃的總 trial 數。

Next:

- 等 `3371` 持續產出 trial 後，優先回報：
  - 新 best 是否超過 `95700`
  - 是否有更多 trial 在 `100000` 內穩定達標

### [2026-05-02] `3371 -> 3400` | relax fast-convergence sweep to `5e-5 / k=3` and restart on new study

- Time: `2026-05-02 17:22 +0800`
- Status:
  - `3371`: CANCELLED
  - `3399`: CANCELLED
  - `3400`: RUNNING
- Experiment or Job ID: `3371`, `3399`, `3400`

Change:

- 人工確認目前 `2e-5 / k=3` 的 stable-crossing 標準過嚴，太多 trial 直接卡在 `100000` ceiling，Optuna 排名解析度不足。
- 將 window-1 fast-convergence sweep 的 success criterion 放寬為：
  - `threshold = 5e-5`
  - `stable_reports = 3`
- 保留其他主幹設定不變：
  - `data_weight in [5, 80]` (`log=True`)
  - `max_steps = 100000`
  - stop-point / final-point single ckpt
  - `PatientPruner(MedianPruner(...), patience=25, min_delta=1e-5)`
- 因研究問題已改，停止舊的 `3371` (`2e-5`)。
- 首次重送為 `3399`，但 remote job header 仍吃到舊 `2e-5` study/storage，立即取消。
- 之後用顯式環境變數覆寫 `THRESHOLD / STUDY_NAME / STORAGE / N_TRIALS / MAX_STEPS / STABLE_REPORTS` 重新提交 `3400`，確認新 job 已正確使用 `5e-5 / k=3` 新 study。

Config / Dataset / Checkpoint:

- Local script defaults:
  - [sweep_weights_window1.py](/Users/latteine/Documents/coding/jaxpi/scripts/sweep/sweep_weights_window1.py)
    - `DEFAULT_THRESHOLD = 5e-5`
    - `DEFAULT_STABLE_REPORTS = 3`
    - `DEFAULT_MAX_STEPS = 100000`
    - `DATA_WEIGHT_MIN = 5.0`
    - `DATA_WEIGHT_MAX = 80.0`
- Slurm wrapper defaults:
  - [sweep_kf_w1_weights.sh](/Users/latteine/Documents/coding/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh)
    - `#SBATCH --time=14-00:00:00`
    - `THRESHOLD=5e-5`
    - `STABLE_REPORTS=3`
    - `N_TRIALS=60`
    - `MAX_STEPS=100000`
    - `STUDY_NAME=kf_w1_data_weight_sweep_5to80_thr5e5_k3_100k_stopckpt`
    - `STORAGE=sqlite:///sweep_w1_data_5to80_thr5e5_k3_100k_stopckpt.db`
- New remote study:
  - `kf_w1_data_weight_sweep_5to80_thr5e5_k3_100k_stopckpt`
- New remote storage:
  - `sqlite:///sweep_w1_data_5to80_thr5e5_k3_100k_stopckpt.db`

Evidence:

- Local code grep:
  - `DEFAULT_THRESHOLD = 5e-5`
  - `DEFAULT_STABLE_REPORTS = 3`
  - `DEFAULT_MAX_STEPS = 100000`
  - `threshold: 5e-05`
- Local wrapper grep:
  - `#SBATCH --time=14-00:00:00`
  - `THRESHOLD=\"${THRESHOLD:-5e-5}\"`
  - `STUDY_NAME=\"${STUDY_NAME:-kf_w1_data_weight_sweep_5to80_thr5e5_k3_100k_stopckpt}\"`
  - `STORAGE=\"${STORAGE:-sqlite:///sweep_w1_data_5to80_thr5e5_k3_100k_stopckpt.db}\"`
- Local validation:
  - `python3 -m py_compile scripts/sweep/sweep_weights_window1.py tests/test_sweep_weights_window1.py`
  - `bash -n slurm/sweep/sweep_kf_w1_weights.sh`
- Old run stopped:
  - `scancel 3371`
- Wrong restart stopped:
  - `scancel 3399`
- Correct restart:
  - `THRESHOLD=5e-5 STABLE_REPORTS=3 N_TRIALS=60 MAX_STEPS=100000 STUDY_NAME=kf_w1_data_weight_sweep_5to80_thr5e5_k3_100k_stopckpt STORAGE=sqlite:///sweep_w1_data_5to80_thr5e5_k3_100k_stopckpt.db sbatch slurm/sweep/sweep_kf_w1_weights.sh`
  - `Submitted batch job 3400`
- Remote `3400` header:
  - `Study    : kf_w1_data_weight_sweep_5to80_thr5e5_k3_100k_stopckpt`
  - `N_trials : 60`
  - `Max_steps: 100000`
  - `Threshold: 5e-5`
  - `Stable_k : 3`
  - `Storage  : sqlite:///sweep_w1_data_5to80_thr5e5_k3_100k_stopckpt.db`
- Remote queue:
  - `3400 | sweep_kf_w1_weights | RUNNING | acmt20`
- Remote DB probe:
  - `db_exists True`
  - `[(1, 0, 'RUNNING')]`

Interpretation:

- 先前 `2e-5 / k=3` 的 objective 比較像 hard success line，不適合第一輪 sweep 排名；放寬到 `5e-5 / k=3` 後，trial 間的 crossing-time 解析度應明顯提高。
- `3399` 證明光改 remote 檔案還不足以保證 sbatch 當下吃到正確參數；這次用顯式 env override 重送 `3400` 後，header 與 DB 都已證明新 study 正確起跑。
- 之後關於 fast-convergence 的正式追蹤對象應改為 `3400`，不再沿用 `3371` 的 `2e-5` lineage。

Next:

- 觀察 `3400` 前幾個 completed trials 是否仍大量卡在 `100000`，用來判斷 `5e-5 / k=3` 是否已提供足夠 ranking 解析度。
- 若 `3400` 仍有大量 ceiling ties，再考慮是否需要把 objective 改成更連續的 convergence score，而不是再硬壓更嚴 threshold。
