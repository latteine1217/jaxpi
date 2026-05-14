# Two-stage PINN Training (Stage 1 LES warmup + Stage 2 sensor+PDE) — Design Spec

## Overview

Sub-project B of the two-stage training rework. Validate the hypothesis that
a dense-LES regression prior (Stage 1) accelerates sparse-sensor + PDE
training (Stage 2) on window-1 of the Re=1e6 Kolmogorov problem. The Stage 2
training mirrors `3491` (sensor+PDE, 100k steps, dw=38.0614, window-1 only)
exactly except for the starting weights, giving an apples-to-apples test of
the prior's effect.

## Why

Three findings motivate the experiment:

1. **`3491` null result** — 100k-step sensor+PDE training with dw=38.0614
   (sweep `3400` rank-1) ended within ±2% of the no-data baseline at every
   compared checkpoint step. Sparse-sensor + PDE alone provides **no
   measurable acceleration** over pure-PDE training on window-1.
2. **Sub-project A delivered a stand-alone Re=1e6 LES dataset** —
   `kolmogorov_les_Re1e6_N512_T5.npy` (404 MB, 101 frames at Δt=0.05) is on
   `home-gpu` and PASSes revised validation (no-decay + bounded + divergence
   1e-13). It is sub-equilibrium (see caveat) but structurally fit for
   Stage-1 supervised regression.
3. **Existing `train.py` already plumbs the required hooks**:
   `data_constraint == "dense_les"` (line 857) wires the dense LES sampler;
   `transfer_learning` (line 796) handles window-to-window weight transfer.
   The only missing piece is **inter-run stage-to-stage transfer**, which
   needs a ~15-line patch.

## Goals

- Produce **two new configs** (Stage 1 LES regression + Stage 2 sensor+PDE
  warmstart) that drop into the existing PINN training pipeline.
- Land a **minimal `train.py` patch** (`transfer_from_ckpt`) enabling
  Stage 2 window-0 to start from Stage 1's final ckpt — without breaking
  the existing intra-run windowed transfer mechanism.
- Run **Stage 1 (50k steps) then Stage 2 (100k steps)** on lab-server's
  2x RTX 3090, producing 100 Stage 2 checkpoints at 1k cadence
  (matching `3491`).
- Run **two evaluations** with the existing `evaluate_window1_checkpoint_sweep`
  tool: Stage 2 vs `3491` (primary research question), Stage 2 vs no-data
  baseline (sanity check).
- Total wallclock ≤ 18 hours including all training + eval + record writeback.

## Non-goals

- **No new evaluator code.** Reuse `scripts/analysis/evaluate_window1_checkpoint_sweep.py`
  from sub-project A's lineage. It already supports 10-pt dual eval and ran
  cleanly on `3491`.
- **No multi-window Stage 2.** Stage 2 trains only window-1 (apples-to-apples
  with `3491`). Cross-window drift effects are out of scope; reserve for
  future sub-project if Stage 2 succeeds.
- **No new LES dataset.** Use sub-project A's existing `kolmogorov_les_Re1e6_N512_T5.npy`
  exclusively, with its documented sub-equilibrium caveat.
- **No re-sweep on `dw`.** Stage 2 uses `dw=38.0614` (sweep `3400` rank-1) to
  remain apples-to-apples with `3491`. Re-sweeping with Stage 1 prior is a
  follow-up experiment, not part of this spec.
- **No PINN architecture change.** Stage 1 + Stage 2 use the same PirateNet
  (2 layers, hidden_dim=768, Fourier embed 768) as `3491` so warmstart is
  weights-compatible without surgery.

## Architecture

```
[Mac jaxpi/]                                          ← source of truth
├── examples/kolmogorov_flow/configs/
│   ├── paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py    NEW
│   └── paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py   NEW
├── examples/kolmogorov_flow/train.py                                MODIFIED (~15 lines)
└── tests/
    ├── test_window1_stage1_les_config.py    NEW
    ├── test_window1_stage2_warm_config.py   NEW
    └── test_transfer_from_ckpt.py           NEW (regression test for train.py patch)

[home-gpu ~/les-gen/]                                 ← LES dataset source
└── output/kolmogorov_les_Re1e6_N512_T5.npy          (sub-project A artefact)

           ↓ rsync ~10 min over Tailscale

[lab-server ~/jaxpi/]                                 ← training host
├── examples/kolmogorov_flow/data/kolmogorov_les/
│   └── kolmogorov_les_re1000000.npy                 NEW (404 MB)
└── runs/
    ├── train_kf_w50_w1_stage1_les_<JOB1>/
    │   └── ckpt/time_window_1/checkpoint_50000/     ← Stage 1 final
    └── train_kf_w50_w1_stage2_warm_<JOB2>/
        └── ckpt/time_window_1/checkpoint_*          ← 100 Stage 2 ckpts

[lab-server ~/jaxpi/eval_runs/]
├── stage2_warm_vs_3491_<DATE>/                      ← primary question
└── stage2_warm_vs_nodata_<DATE>/                    ← sanity check
```

The Stage 1 → Stage 2 handoff happens via `config.transfer_from_ckpt`, a
new string field pointing to Stage 1's `time_window_1/checkpoint_50000`
directory. `train.py` resolves this at the start of Stage 2 window-0 and
loads `params` + `weights` (but **not** `opt_state`), then training proceeds
as usual with sensor + PDE losses.

## Component 1: Stage 1 config

`paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py` is a clone of
`paper_repro_soap_sensor100_n512_w50_window1_ablation.py` with these
differences:

| field | value | reason |
|---|---|---|
| `data_constraint` | `"dense_les"` | enables `train.py:857` dense LES sampler |
| `dataset_path` | `"examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy"` | sub-project A artefact rsynced from home-gpu |
| `sensor_json` | `None` | disable sparse sensor JSON loader |
| `weighting.init_weights.u_data` | 1.0 | regression loss enabled |
| `weighting.init_weights.v_data` | 1.0 | regression loss enabled |
| `weighting.init_weights.w_data` | 1.0 | regression loss enabled (LES has omega) |
| `weighting.init_weights.ru` | 0.0 | PDE residual u disabled |
| `weighting.init_weights.rv` | 0.0 | PDE residual v disabled |
| `weighting.init_weights.rc` | 0.0 | continuity disabled |
| `weighting.init_weights.u_ic` | 0.0 | IC loss disabled (dense LES covers t=0) |
| `weighting.init_weights.v_ic` | 0.0 | IC loss disabled |
| `training.max_windows_to_run` | 1 | single window covering t∈[0, 5] |
| `training.max_steps` | 50000 | regression converges fast |
| `saving.save_every_steps` | 5000 | 10 ckpts for trajectory inspection |
| `saving.num_keep_ckpts` | `None` | retain all |
| `wandb.name` | `"re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les"` | identity |
| `wandb.group` | `"re1e6_window1_two_stage"` | NEW group for sub-project B |
| `wandb.tags` | base + `["stage1", "dense_les", "no_pde", "Re=1e6", "warmup_only"]` | discoverability |

The architecture (`PirateNet`, num_layers=2, hidden_dim=768, Fourier embed
768, periodicity 2π) is inherited from the base config and **must remain
unchanged**: Stage 2 warmstart requires weight-shape compatibility.

The SOAP optimizer is also inherited. SOAP for pure regression is unusual
but precedented by `stage_ab/pirate_les_stage1_soap.py` and lets Stage 1 +
Stage 2 share the same optimizer class.

## Component 2: train.py minimal patch (`transfer_from_ckpt`)

Insertion point: immediately before the existing `if config.transfer_learning: if idx > 0:` block
inside the per-window loop in `main()`.

```python
# === NEW: inter-run stage transfer (Stage 1 → Stage 2 warm start) ===
if idx == 0 and config.get("transfer_from_ckpt"):
    src_ckpt = config.transfer_from_ckpt
    logging.info(f"Stage transfer: loading params from {src_ckpt}")
    src_state = restore_checkpoint(model.state, src_ckpt)
    if parallel_state["num_devices"] > 1:
        src_state = jax.device_get(src_state)
    model.state = _create_train_state(
        config,
        params=src_state.params,
        weights=src_state.weights,
        replicate=False,
    )
    logging.info(
        "Stage transfer complete — params + weights loaded, opt_state fresh"
    )
```

Properties:

- **Triggers only at `idx == 0`** (first window of the run). This is exactly
  where the existing `transfer_learning` mechanism is silent
  (`if idx > 0:`), so the two paths never collide.
- **Triggers only if the config provides `transfer_from_ckpt`**. Absent that
  field, behavior is identical to current `train.py`.
- **Transfers `params` and `weights`, NOT `opt_state`.** Stage 1's SOAP
  state was tuned for regression-only loss landscape; reusing it for the
  PDE-augmented Stage 2 loss landscape risks first-step explosion. Fresh
  SOAP state is safer.
- **`weights` here refers to** the adaptive loss-weighting state, not the
  network params (jaxpi confusingly names both `weights`). Carrying it over
  means Stage 2 starts with the loss-balance that Stage 1 converged to —
  reasonable since the data terms are still present in Stage 2.

Regression-safe: existing `transfer_learning=True` window-to-window flow
remains untouched. The new field is opt-in and orthogonal.

## Component 3: Stage 2 config

`paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py` is a clone of
`paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py` (i.e. the
`3491` config) with these additions:

| field | value | reason |
|---|---|---|
| `transfer_from_ckpt` | `"/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les/ckpt/time_window_1/checkpoint_50000"` | path to Stage 1 final ckpt on lab-server |
| `transfer_optimizer_state` | `False` | Stage 1 SOAP state not applicable; fresh state |
| `wandb.name` | `"re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm"` | identity |
| `wandb.group` | `"re1e6_window1_two_stage"` | match Stage 1 group |
| `wandb.tags` | base + `["stage2", "warmstart", "from_stage1", "dw=38.0614"]` | discoverability |

Everything else (sensor K=100 QR-pivot, `dw=38.0614`, max_steps=100000,
save_every_steps=1000, num_keep_ckpts=None, SOAP, PirateNet arch) is
identical to `3491`. The only research-relevant difference is the starting
weights.

## Component 4: Tests

### `test_window1_stage1_les_config.py`

Pins Stage 1 surface to prevent silent regression:

```python
def test_stage1_uses_dense_les_and_disables_pde():
    config = stage1_module.get_config()
    assert config.data_constraint == "dense_les"
    assert config.dataset_path.endswith("kolmogorov_les_re1000000.npy")
    assert config.sensor_json is None
    # PDE off
    assert config.weighting.init_weights.ru == 0.0
    assert config.weighting.init_weights.rv == 0.0
    assert config.weighting.init_weights.rc == 0.0
    # IC off
    assert config.weighting.init_weights.u_ic == 0.0
    assert config.weighting.init_weights.v_ic == 0.0
    # Data on
    assert config.weighting.init_weights.u_data == 1.0
    assert config.weighting.init_weights.v_data == 1.0
    assert config.weighting.init_weights.w_data == 1.0
    # Training shape
    assert config.training.max_windows_to_run == 1
    assert config.training.max_steps == 50000
    assert config.saving.save_every_steps == 5000
    assert config.saving.num_keep_ckpts is None
    # Wandb identity
    assert config.wandb.name == "re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les"
    assert config.wandb.group == "re1e6_window1_two_stage"
    for tag in ("stage1", "dense_les", "no_pde"):
        assert tag in config.wandb.tags
```

### `test_window1_stage2_warm_config.py`

Pins Stage 2 surface:

```python
def test_stage2_uses_warmstart_and_mirrors_3491():
    config = stage2_module.get_config()
    # warmstart
    assert isinstance(config.transfer_from_ckpt, str)
    assert "stage1_les" in config.transfer_from_ckpt
    assert config.transfer_from_ckpt.endswith("checkpoint_50000")
    assert config.transfer_optimizer_state is False
    # mirror 3491
    assert config.weighting.init_weights.u_data == 38.0614
    assert config.weighting.init_weights.v_data == 38.0614
    assert config.training.max_steps == 100000
    assert config.training.max_windows_to_run == 1
    assert config.saving.save_every_steps == 1000
    assert config.saving.num_keep_ckpts is None
    # Wandb identity
    assert config.wandb.name == "re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm"
    for tag in ("stage2", "warmstart", "from_stage1"):
        assert tag in config.wandb.tags
```

### `test_transfer_from_ckpt.py`

Regression test for the `train.py` patch. Strategy: monkey-patch
`restore_checkpoint` and `_create_train_state`, build a minimal config with
`transfer_from_ckpt="/tmp/fake-ckpt"`, drive `main()` until the patch
block fires, assert that:

1. `restore_checkpoint` is called with `"/tmp/fake-ckpt"`.
2. `_create_train_state` is called with `replicate=False` and the
   monkeypatched `params` / `weights` arguments.
3. The patch is **not** triggered when `idx > 0` (i.e. doesn't shadow the
   existing windowed transfer path).
4. The patch is **not** triggered when `transfer_from_ckpt` is absent
   (backward compatibility).

This test does not run actual training; it intercepts at the model-state
creation boundary.

## Data flow

```
Phase 0 (one-time, ~10 min):
  rsync home-gpu:~/les-gen/output/kolmogorov_les_Re1e6_N512_T5.npy \
        lab-server:~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy

Phase 1 (Stage 1, ~3–5 h on 2x RTX 3090):
  sbatch slurm/train/train_kolmogorov_re1e6_sensor100_w25_soap.sh
    CONFIG_PATH=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py
    RUN_SLUG=train_kf_w50_w1_stage1_les
  → produces /home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les/ckpt/time_window_1/checkpoint_50000/

Phase 2 (Stage 2, ~6–10 h):
  sbatch slurm/train/train_kolmogorov_re1e6_sensor100_w25_soap.sh
    CONFIG_PATH=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py
    RUN_SLUG=train_kf_w50_w1_stage2_warm
  → produces /home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm/ckpt/time_window_1/checkpoint_{1000..100000}/

Phase 3 (Eval, ~30 min total):
  Eval A — Stage 2 vs 3491 (primary research question):
    sbatch slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh with
      NO_DATA_CONFIG = examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py  (3491 config; "NO_DATA_" is launcher naming, here it holds the comparison config)
      NO_DATA_CKPT_ROOT = /home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_100k_eval/ckpt
      SENSOR_CONFIG = examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py
      SENSOR_CKPT_ROOT = /home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm/ckpt
      CHECKPOINT_STEPS = 10000,20000,...,100000
      OUTPUT_DIR = /home/junyi/jaxpi/eval_runs/stage2_warm_vs_3491_<DATE>

  Eval B — Stage 2 vs no_data (sanity check):
    Same launcher with
      NO_DATA_CONFIG = examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py
      NO_DATA_CKPT_ROOT = /home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ablation/ckpt
      SENSOR_CONFIG / SENSOR_CKPT_ROOT = stage2 (same as Eval A)
      OUTPUT_DIR = /home/junyi/jaxpi/eval_runs/stage2_warm_vs_nodata_<DATE>

Phase 4 (Record writeback):
  Pull artefacts to Mac.
  Update EXPERIMENT_RECORD.md [INDEX] Active + [LOG] Chronological.
  Commit + push.
```

## Validation / Success criteria

For each research question, the primary metric is **direct apply_fn
corrected u/v/w field error** at matching checkpoint steps.

**Q1 (Stage 1 prior helps over 3491?):**

- **Win**: Stage 2 has lower u, v, and w errors than 3491 at every matched
  step from 10k onwards.
- **Partial win**: Stage 2 has ≥10% improvement on average across the
  10-point eval, even if some steps are tied.
- **Null result**: Stage 2 within ±10% of 3491 at every step (matches the
  3491 conclusion that sensor+PDE prior provides no measurable
  acceleration).
- **Loss**: Stage 2 worse than 3491 (warmstart from sub-equilibrium prior
  actively hurt PINN learning) — would force re-examining the LES
  calibration.

**Q2 (Stage 2 reaches no_data baseline?):**

- **Acceptable**: Stage 2 within ±10% of no_data 100k baseline
  `(u=1.077e-3, v=1.088e-3, w=0.704e-3)`.
- **Concerning**: Stage 2 worse than no_data — would suggest the LES prior
  is misleading the PDE training.

Null result is a valid outcome and gets written into EXPERIMENT_RECORD with
the same care as a positive result.

## Effort

| Phase | Time |
|---|---|
| Dataset rsync (home-gpu → lab-server) | 5~15 min |
| Stage 1 config + test | ~30 min |
| Stage 2 config + test | ~30 min |
| train.py patch + regression test | ~1 h |
| Commit + push + lab-server pull | ~10 min |
| Stage 1 training (50k steps) | ~3~5 h wallclock |
| Stage 2 training (100k steps) | ~6~10 h wallclock |
| Eval A + Eval B | ~30 min |
| Pull artefacts + EXPERIMENT_RECORD update | ~30 min |
| **Total** | **~12~18 h** (9~15 h of which is GPU training) |

## Risks

1. **R1 — Sub-equilibrium LES prior hurts Stage 2.** Stage 1 learns a spectrum
   with slope -3.67 (not -5/3), which encodes too-rapid energy decay at
   high-k. Stage 2 PDE may struggle to correct this baked-in bias.
   *Mitigation*: caveat carried into EXPERIMENT_RECORD; Eval A directly
   measures this.

2. **R2 — `train.py` patch introduces regression.** Modifying the model-init
   flow could break existing windowed transfer.
   *Mitigation*: TDD with `test_transfer_from_ckpt.py` covering 4 paths
   (patch fires correctly, doesn't fire when `idx>0`, doesn't fire when
   field absent, monkey-patched ckpt restore is invoked).

3. **R3 — Stage 1 50k steps over-converges.** SOAP on pure regression for
   50k iterations may push training MSE below noise floor, locking in
   numerical artefacts.
   *Mitigation*: inspect Stage 1 final loss; if `data_loss < 1e-8` consider
   reducing to 20k for next iteration. Save 10 ckpts so an earlier
   checkpoint is selectable as Stage 2 source.

4. **R4 — SOAP unstable on dense regression.** SOAP is a second-order
   optimizer; on pure data fit (Hessian dominated by quadratic regression
   loss) it may exhibit instability.
   *Mitigation*: precedent in `stage_ab/pirate_les_stage1_soap.py` shows
   SOAP+dense_les works for Re=1e5. If instability hits, fallback option
   is AdamW for Stage 1 (forces `transfer_optimizer_state=False` which
   we already set anyway).

5. **R5 — Stage 2 first step explosion.** Stage 1 params land at a point
   where PDE residual is large (Stage 1 never saw PDE). The first Stage 2
   step's gradient may be huge.
   *Mitigation*: SOAP base config has `warmup_steps` (typically 5000–22000)
   so LR ramps up from 0. Combined with fresh `opt_state`, the first
   actual update is small.

6. **R6 — Loss-scale mismatch between stages.** Stage 1 has 262144
   "sensor" points per frame; Stage 2 has 100. The adaptive `weights`
   inherited from Stage 1 are tuned for dense-data magnitudes.
   *Mitigation*: jaxpi adaptive weighting recomputes per-step; within
   the first few hundred steps it should rebalance. Watch for early
   wandb logs showing weights drifting.

7. **R7 — GPU queue contention.** Lab-server may be busy.
   *Mitigation*: nohup background launch; no time pressure since the
   research question is well-defined and reproducible later.

## Open questions

- None blocking. The `transfer_from_ckpt` mechanism is the only new code
  path; if it interacts poorly with existing transfer_learning windowed
  flow, the `test_transfer_from_ckpt.py` regression test catches it
  before Stage 2 launch.

## Deliverables

1. Two new config files + their unit tests, committed to the jaxpi
   `pirate` branch.
2. `train.py` patched with `transfer_from_ckpt` support + regression
   test, committed to the same branch.
3. Re=1e6 LES dataset rsynced to `lab-server:~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy`.
4. Stage 1 + Stage 2 training runs on lab-server with full ckpt
   trajectories saved.
5. Two `eval_runs/` artefact sets (Stage 2 vs 3491, Stage 2 vs no_data),
   each with `checkpoint_sweep_results.csv` + `*.png`.
6. EXPERIMENT_RECORD.md `[INDEX] Active` + `[LOG] Chronological`
   entries covering the two-stage training run with results table,
   verdict on Q1/Q2, and follow-up direction.
