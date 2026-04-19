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

### `3318` | `kf_w1_data_weight_sweep_1to100_thr5e5`

| Field | Value |
| :--- | :--- |
| Status | Running |
| Config | [paper_repro_soap_window1_ablation.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` |
| Sweep Range | `data_weight in [1, 100]` (`log=True`) |
| Objective | fastest `max(ru_loss, rv_loss, rc_loss) < 5e-5` within `50000` steps |
| Storage | `sqlite:///sweep_w1_data_1to100_thr5e5.db` |
| Launch Script | `/home/junyi/jaxpi/slurm/sweep/sweep_kf_w1_weights.sh` |
| Current Risk | `3317` 先以舊 remote 腳本誤啟動（header 仍是 `Threshold: 1e-5`），已在 30 秒內取消並改由 `3318` 重送；`3318` header 已確認吃到正確 `5e-5` threshold 與 `1..100` range 所對應的新版 Python / Slurm 腳本。 |
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
| Current Risk | `2026-04-16` audit（`3261..3272`）確認舊 corrected full-window result 可由 direct `apply_fn` path 復現。`neural_net()` 的 `z[None, :] + outputs[0]` scalar-wrapper bug 已定位並修復（`models.py` L185-192 移除 `z.ndim == 1` 分支），修復後 smoke test 三項全 PASS（`u_ic_pred_fn` / `u_pred_fn` batch-invariance / vs direct diff < 1e-4）。`3259/3260` 的 `~1e-3` window-1 table 仍屬已廢棄的錯誤評估路徑，不可用於比較結論。`3273` direct `apply_fn` sweep 確認 window 1 兩組收斂底線均在 `~1e-3`；主風險移回跨 window 誤差累積行為。 |
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
| Latest Verified Evidence | `3145` full-window evaluation rerun completed through `window 12 / checkpoint_60000`; `2026-04-15` horizontal vorticity PNG + GIF regenerated for windows `1..12` |
| Current Risk | `window 12` 的 `w_err = 0.176233` 仍略高於門檻；`3146` 顯示主因更接近 high-k vorticity attenuation，而非大尺度相位崩潰 |
| RNG Strategy | Not recorded |

目前主線判讀：

- `window 1 -> window 2` 已成功跨過，stale IC 問題可視為初步排除。
- 修正後的 `3145` full-window rerun 顯示，`window 2+` 的 `u/v` 誤差維持在低區間，先前 `3144` 的高誤差主要來自 eval time-axis 錯位。
- 但 `w_err` 仍隨窗口上升，到 `window 12` 達 `0.176233`；主線並非失敗，但也尚未完全達標。
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

Evidence:

- Mean over common windows `1..12`:
  - `u_error`: `0.002743 -> 0.002402` (`with_data / no_data = 0.875715`)
  - `v_error`: `0.002911 -> 0.002527` (`with_data / no_data = 0.868190`)
  - `w_error`: `0.060319 -> 0.062222` (`with_data / no_data = 1.031557`)
- `window 12`:
  - no data  (`3137/3145`): `u=0.007903, v=0.008504, w=0.176233`
  - with data (`3155`)     : `u=0.007831, v=0.008350, w=0.174197`
- `3155` additional windows:
  - `window 13 / checkpoint_100000 -> u=0.010144, v=0.009809, w=0.201551`
  - `window 14 / checkpoint_10000  -> u=0.014820, v=0.012950, w=0.279143`
- 若只看公平比較區間，已另輸出只到 `window 12` 的裁切版趨勢圖，避免 `3155` 的額外窗口造成視覺誤讀。

Interpretation:

- 在共同窗口 `1..12` 上，加入 `u/v data` 後，`u_error` 與 `v_error` 平均約再下降 `12% ~ 13%`。
- 但 `w_error` 的平均值沒有明顯改善，整體甚至略高約 `3%`；這表示 data constraint 主要幫助的是速度場，而不是根本解掉 vorticity 累積漂移。
- 到 `window 12` 單點時，with-data 與 no-data 的 `w_err` 已非常接近（`0.174197` vs `0.176233`），因此更穩健的結論是：`u/v data` 對 `u/v` 有益，但對晚期 `w` 漂移只有有限幫助。

Next:

- 若 `3155 window 14+` 繼續往上跑，應持續觀察 `w_err` 是否再次脫離 `3137/3145` 的上升曲線。

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
