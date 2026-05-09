# AGENTS.md [v3.0 Protocol: Research Partner]

本檔定義 agent 在本專案中的操作協議。目標是以最少上下文維持高可驗證性，避免對實驗狀態、checkpoint 血統與研究結論做無根據推論。

<LANGUAGE_POLICY>
- 回覆、分析、程式註解：中文
- 圖表標題與 label：English
</LANGUAGE_POLICY>

<IDENTITY_&_CONTEXT>
- Project: Sparse-Data Turbulence Reconstruction
- Stack: JAX / PirateNet / PINN / SOAP / AdamW -> L-BFGS
- Domain: 2D Kolmogorov (`Re = 1e2 ~ 1e5`) | 3D Channel (`Re_tau = 1000`)
- Reference: DNS full field vs sparse sensors
- Sensor Budget: `K <= 100`, prefer QR-pivot
</IDENTITY_&_CONTEXT>

<SUCCESS_THRESHOLD>
- Relative L2 error: `<= 10% ~ 15%`
- Improvement vs RANS baseline: `>= 30%`
- Convergence speedup: `>= 30%`
- Hard Rule: 不可把「能跑」當成成功；不可把「單一 window 成功」當成整體實驗成功
</SUCCESS_THRESHOLD>

<DECISION_ORDER>
1. 理論完整度
2. 可驗證性與可重現性
3. 數值穩定性與收斂性
4. 簡潔性與可解釋性
5. 效能與可擴展性
</DECISION_ORDER>

<STATE_INTERFACE_PROTOCOL>
`EXPERIMENT_RECORD.md` 是唯一外部狀態帳本；預設不預載全文。

- Read_First:
  - 涉及 experiment status、config、dataset、checkpoint、eval、job state、loss、optimizer、loader、time-window logic 時，先讀本地 [EXPERIMENT_RECORD.md](/Users/latteine/Documents/coding/jaxpi/EXPERIMENT_RECORD.md)
- Targeted_Retrieval:
  - 優先讀取 `## [INDEX] Active`、`## [INDEX] Critical Failures`、`## [LOG] Chronological`
  - 僅抓與當前任務相關的 `Job_ID`、config 名稱、checkpoint 或時間戳段落
- Escalation:
  - 若記錄不存在、內容衝突、或無法唯一識別，回報 `[STATUS: CONTEXT_MISSING]`
- Hard_Constraint:
  - 無記錄證據時，嚴禁推論實驗進度、最佳結果、checkpoint lineage、RNG strategy
</STATE_INTERFACE_PROTOCOL>

<EXPERIMENT_RECORD_WRITEBACK>
若任務涉及以下任一情況，完成後必須同步更新 [EXPERIMENT_RECORD.md](/Users/latteine/Documents/coding/jaxpi/EXPERIMENT_RECORD.md)：

- Config / dataset / checkpoint 變更
- Job 狀態遷移：start / stop / resume / rerun / terminate
- Training logic 變更：loss / optimizer / loader / time-window / forcing / domain / periodic embedding
- 重要評估、checkpoint 判讀、field visualization
- 會影響研究結論的 bug

每筆紀錄至少包含：

- Time
- Experiment or Job ID
- Change
- Config / Dataset / Checkpoint
- Evidence
- Interpretation
- Next

建議：

- `## [INDEX] Active` 內若為 distributed run，應顯式記錄 `RNG_Strategy`
- 失敗案例必須保留，不可只記成功
</EXPERIMENT_RECORD_WRITEBACK>

<WORKFLOW_STATE_MACHINE>
1. 先判斷問題屬性：theory / data / config / implementation / observability
2. 改動前評估：是否影響既有 workflow、checkpoint 相容性、time-window 定義
3. 實作時遵守：
   - 禁止暫時性 hack 冒充完成
   - 未完成事項用 `TODO:` 標示目標狀態
   - 優先做 local reasoning，降低副作用
4. 改動後至少提供一項可重現硬證據
5. 若屬實驗相關修改，寫回 `EXPERIMENT_RECORD.md`
</WORKFLOW_STATE_MACHINE>

<CORE_RESEARCH_INTEGRITY>
- Physical_Consistency:
  - 任何模型或 loss 修改都必須回答：是否破壞 periodic boundary、forcing term、domain 定義
- Loader_Consistency:
  - loader 必須優先尊重 dataset 內座標與 metadata，不可任意回推
- Baseline_Alignment:
  - schedule-free / grad clip / adaptive weighting 必須和目標 baseline 對齊
- Eval_Consistency:
  - eval mode 必須與訓練 checkpoint 的 window / state 定義一致
- Metric_Selection:
  - `loss` 只能作為 `health/progress signal`：用來判斷是否發散、停滯、進入平台或值得 early-stop 候選
  - `corrected field error` 才能作為 `selection signal`：用來判斷 checkpoint / run 是否真的更好
  - 禁止把 residual threshold crossing、tail loss 或 loss rank 直接當成 field quality 證據
  - 若 `loss` 與 `error` 趨勢衝突，研究結論必須以 `corrected field error` 為準
</CORE_RESEARCH_INTEGRITY>

<TIME_WINDOW_INTEGRITY_CHECK>
任何涉及訓練循環、DataLoader、rollout、checkpoint、evaluation path 的變更，必須檢核：

- State_Propagation: 狀態傳遞來源是否顯式
- Continuity: window boundary 連續性是否保留
- Checkpoint_Mapping: 物理步數、window index、checkpoint 命名是否對齊
- Eval_Definition: eval 使用的初值與訓練時定義是否一致

若未完成驗證，必須明示：

- `[RISK: TIME_WINDOW_INTEGRITY_UNVERIFIED]`
</TIME_WINDOW_INTEGRITY_CHECK>

<MULTI_GPU_PROTOCOL>
Trigger:

- 使用 `pmap` / `pjit` / `shard_map`
- 使用 data parallel / distributed checkpoint
- 使用 2 張以上 GPU

Required_Checks:

- `global_batch = per_device_batch * num_devices`
- RNG policy 在 devices 間可重現，且 split/fold-in 策略明確
- checkpoint 含 `world_size`、mesh 或 sharding metadata
- 單卡 eval 載入多卡權重時，處理邏輯必須顯式說明

Hard_Rule:

- 若單卡 correctness 尚未驗證，不得優先提議多 GPU 最佳化
</MULTI_GPU_PROTOCOL>

<VERIFICATION_HARD_RULE>
所有代碼變更必須提供至少一項可重現硬證據，優先順序如下：

1. `py_compile`
2. smoke test
3. checkpoint evaluation
4. log 對照
5. field visualization
6. tensor shape trace
7. physical consistency check

若任務會影響研究結論，優先使用 3/4/5/7 類證據，而非只做語意檢查。
</VERIFICATION_HARD_RULE>

<SERVER_ENV>
- SSH: `ssh junyi@140.114.120.128`
- Partition: `r740`
- Time Limit: `14-00:00:00`
- Memory: `100G`
- GPU: `2x RTX 3090 Turbo`
- Server Python: `python3`
</SERVER_ENV>

<CLI_POLICY>
- 搜尋內容：`rg`
- 搜尋檔案：`fd`
- 查看結構：`tree`
- Python 環境與套件：`uv`
- shell 腳本中的 Python：`uv run python`
</CLI_POLICY>

<OUTPUT_SCHEMA_HYBRID>
- 模式 A: 實驗判讀 / 數據對比 -> 優先表格化
- 模式 B: 代碼修改 / 邏輯除錯 -> 使用四段式

模式 B 固定順序：

1. `⚡️ Current State`
2. `📊 Evidence`
3. `🧠 Critique/Interpretation`
4. `🚀 Action`

結尾必須包含：

- `Check: [Protocol_Adhered] | Record_Update: [Required/Not_Required]`
</OUTPUT_SCHEMA_HYBRID>

<FINAL_RED_LINES>
- 不可把直覺當成結論
- 不可把單次成功當成穩定趨勢
- 不可隱藏失敗案例
- 不可在缺乏證據時回憶或猜測歷史實驗
</FINAL_RED_LINES>
