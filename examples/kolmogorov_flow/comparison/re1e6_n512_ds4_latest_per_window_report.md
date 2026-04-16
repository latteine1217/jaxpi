# Re1e6 N512 ds4 Latest-Checkpoint Window Summary

## Scope

- Date: `2026-04-01`
- Job: `3137`
- Config: [paper_repro_soap.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap.py)
- Dataset: [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy)
- Checkpoint Root: `/home/junyi/jaxpi_upstream_soap_run/re1e6_n512_ds4_soap/ckpt`
- Evaluation Mode: `final_step`

## Results

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

## Summary Statistics

| Metric | Mean | Min | Max |
| :--- | ---: | ---: | ---: |
| `u_err` | 0.001123 | 0.000041 | 0.003701 |
| `v_err` | 0.001172 | 0.000042 | 0.003745 |
| `w_err` | 0.033148 | 0.000266 | 0.099886 |

## Interpretation

- `u` 與 `v` 在 `window 1..9` 都維持很低誤差，目前主線沒有速度場失控跡象。
- `w` 幾乎隨著 time window 單調上升，表示長序列 rollout 的主要風險已集中在 vorticity error accumulation。
- 到 `window 9` 時，`w_err = 0.099886`，已逼近專案驗收邊界；下一步應優先追蹤 `window 10+` 是否繼續上升。

## Artifact

- Plot: [re1e6_n512_ds4_error_vs_time_window.png](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/comparison/re1e6_n512_ds4_error_vs_time_window.png)
