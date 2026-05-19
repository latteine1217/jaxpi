# Two-stage PINN Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train PINN with Stage 1 (dense LES regression on Re=1e6, no PDE) → Stage 2 (sparse sensor + PDE, warmstart from Stage 1, mirror 3491 hyperparams). Validate whether the LES prior accelerates window-1 corrected field error.

**Architecture:** Two new configs + one train.py patch (~15 lines) wired into the existing PINN training pipeline. Stage 1 produces a single ckpt at step 50000; Stage 2's new `transfer_from_ckpt` field loads it at `idx==0` (orthogonal to existing windowed `transfer_learning`). Existing `evaluate_window1_checkpoint_sweep.py` runs two dual-evals (Stage 2 vs 3491 + Stage 2 vs no_data).

**Tech Stack:** Python 3.13, JAX + Flax + Optax (SOAP), pytest, jaxpi pipeline, slurm on lab-server (2x RTX 3090), Tailscale rsync home-gpu↔lab-server.

---

## File Structure

| Path | Role |
| --- | --- |
| `examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py` | Stage 1 config: dense LES regression, PDE off, single window t∈[0,5] |
| `examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py` | Stage 2 config: mirror 3491 + `transfer_from_ckpt` |
| `examples/kolmogorov_flow/train.py` (modify ~15 lines at L796) | Add `transfer_from_ckpt` inter-run stage transfer |
| `tests/test_window1_stage1_les_config.py` | Pin Stage 1 surface (PDE off, dense LES on, training shape) |
| `tests/test_window1_stage2_warm_config.py` | Pin Stage 2 surface (mirror 3491 + transfer_from_ckpt set) |
| `tests/test_transfer_from_ckpt.py` | Regression test for the train.py patch |
| `EXPERIMENT_RECORD.md` | New `[INDEX] Active` + `[LOG] Chronological` entries (after results) |
| `lab-server:~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy` | LES dataset (404 MB rsynced from home-gpu) |
| `lab-server:~/jaxpi/runs/train_kf_w50_w1_stage1_les_<JOB>/` | Stage 1 ckpt + wandb |
| `lab-server:~/jaxpi/runs/train_kf_w50_w1_stage2_warm_<JOB>/` | Stage 2 ckpts (100 at 1k cadence) |
| `lab-server:~/jaxpi/eval_runs/stage2_warm_vs_3491_<DATE>/` | Eval A artefacts |
| `lab-server:~/jaxpi/eval_runs/stage2_warm_vs_nodata_<DATE>/` | Eval B artefacts |
| `eval_runs/stage2_warm_vs_3491_<DATE>/` (local Mac) | rsync'd back for review |
| `eval_runs/stage2_warm_vs_nodata_<DATE>/` (local Mac) | rsync'd back for review |

---

## Task 1: Deploy LES dataset to lab-server

**Files:**
- Create: `lab-server:~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy` (404 MB)

- [ ] **Step 1: rsync the dataset over Tailscale**

```bash
ssh lab-server 'mkdir -p ~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les'
ssh home-gpu 'rsync -avz --progress \
    ~/les-gen/output/kolmogorov_les_Re1e6_N512_T5.npy \
    lab-server:~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy'
```

Note: ssh from Mac → trigger rsync home-gpu → lab-server via Tailscale.
Expected: ~404 MB transferred in 5-15 min depending on Tailscale path.

- [ ] **Step 2: Verify the dataset on lab-server**

```bash
ssh lab-server 'python3 -c "
import numpy as np
p = np.load(\"/home/junyi/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy\", allow_pickle=True).item()
cfg = p[\"config\"]
print(\"calibration_mode:\", cfg[\"calibration_mode\"])
print(\"N:\", cfg[\"N\"], \"nu:\", cfg[\"nu\"], \"T_end:\", cfg[\"T_end\"])
omega = np.asarray(p[\"omega\"])
print(\"omega shape:\", omega.shape, \"dtype:\", omega.dtype)
import os
print(\"size MB:\", round(os.path.getsize(\"/home/junyi/jaxpi/examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy\") / 1024**2, 1))
"'
```

Expected:
- `calibration_mode: stand_alone`
- `N: 512, nu: 1e-06, T_end: 5.0`
- `omega shape: (101, 512, 512), dtype: float32`
- `size MB: ~404`

If any check fails: STOP, report dataset transfer is corrupt.

- [ ] **Step 3: No commit needed**

Dataset is gitignored (`.gitignore` covers `examples/kolmogorov_flow/data/kolmogorov_les/`). Nothing to commit.

---

## Task 2: Stage 1 config + test (TDD)

**Files:**
- Create: `tests/test_window1_stage1_les_config.py`
- Create: `examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_window1_stage1_les_config.py`:

```python
"""
What:
    驗證 window-1 Stage 1 LES warmup config 的關鍵行為。
Why:
    Stage 1 是 sub-project B 的第一階段：dense LES regression，no PDE。
    架構必須與 Stage 2 ablation base 對齊（warmstart 相容），但 loss
    配置完全不同。任何 silent regression（PDE 被誤開、IC loss 漏掉、
    dataset_path 變動）都會破壞 stage transfer。
"""

import sys
import types
import unittest


def _install_stub_modules():
    """
    What:
        安裝最小 stub，讓 config module 可以在本地缺少 JAX/ConfigDict 依賴時匯入。
    Why:
        測試只驗證 config 輸出，不需要真實數值運算。
    """

    if "jax" not in sys.modules:
        jax_mod = types.ModuleType("jax")
        jnp_mod = types.ModuleType("jax.numpy")
        jnp_mod.pi = 3.141592653589793
        jax_mod.numpy = jnp_mod
        sys.modules["jax"] = jax_mod
        sys.modules["jax.numpy"] = jnp_mod

    if "ml_collections" not in sys.modules:
        mlc = types.ModuleType("ml_collections")

        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        mlc.ConfigDict = ConfigDict
        sys.modules["ml_collections"] = mlc


_install_stub_modules()


class Window1Stage1LesConfigTest(unittest.TestCase):
    def test_stage1_uses_dense_les_and_disables_pde(self):
        from examples.kolmogorov_flow.configs import (
            paper_repro_soap_sensor100_n512_w50_window1_stage1_les as cfg_mod,
        )

        config = cfg_mod.get_config()

        # data_constraint switched to dense_les
        self.assertEqual(config.data_constraint, "dense_les")
        self.assertTrue(
            config.dataset_path.endswith("kolmogorov_les_re1000000.npy"),
            f"dataset_path = {config.dataset_path}",
        )
        # sensor disabled
        self.assertIsNone(config.sensor_json)

        # data loss weights enabled (regression on u, v, w/omega)
        self.assertEqual(config.weighting.init_weights.u_data, 1.0)
        self.assertEqual(config.weighting.init_weights.v_data, 1.0)
        self.assertEqual(config.weighting.init_weights.w_data, 1.0)

        # PDE / IC weights zeroed (no physics in Stage 1)
        self.assertEqual(config.weighting.init_weights.ru, 0.0)
        self.assertEqual(config.weighting.init_weights.rv, 0.0)
        self.assertEqual(config.weighting.init_weights.rc, 0.0)
        self.assertEqual(config.weighting.init_weights.u_ic, 0.0)
        self.assertEqual(config.weighting.init_weights.v_ic, 0.0)

        # Training shape
        self.assertEqual(config.training.max_windows_to_run, 1)
        self.assertEqual(config.training.max_steps, 50000)
        self.assertEqual(config.saving.save_every_steps, 5000)
        self.assertIsNone(config.saving.num_keep_ckpts)

        # wandb identity
        self.assertEqual(
            config.wandb.name,
            "re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les",
        )
        self.assertEqual(config.wandb.group, "re1e6_window1_two_stage")
        for tag in ("stage1", "dense_les", "no_pde", "warmup_only"):
            self.assertIn(tag, config.wandb.tags)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, verify it FAILS**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/test_window1_stage1_les_config.py -v`

Expected: `ModuleNotFoundError: No module named 'examples.kolmogorov_flow.configs.paper_repro_soap_sensor100_n512_w50_window1_stage1_les'` — config doesn't exist yet.

- [ ] **Step 3: Write the Stage 1 config**

Write `examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py`:

```python
import examples.kolmogorov_flow.configs.paper_repro_soap_sensor100_n512_w50_window1_ablation as base_config


def get_config():
    """
    What:
        Stage 1 of two-stage PINN training (sub-project B).
        Dense LES regression on Re=1e6 LES dataset; no PDE, no sensor.
    Why:
        3491 null result showed sparse-sensor + PDE alone provides no
        measurable acceleration over no_data baseline on window-1.
        Hypothesis: starting Stage 2 from a dense-LES regression prior
        might fix it. Stage 1 here produces that prior.

        Architecture matches Stage 2 (PirateNet 2 layers, hidden_dim=768,
        SOAP) so warmstart is weight-shape compatible.

        Caveat (carried from sub-project A): the Re=1e6 LES dataset is
        sub-equilibrium. Stage 1 learns this sub-equilibrium representation,
        not fully-developed turbulence statistics.
    """

    config = base_config.get_config()

    # === LES regression scope ===
    config.data_constraint = "dense_les"
    config.dataset_path = (
        "examples/kolmogorov_flow/data/kolmogorov_les/kolmogorov_les_re1000000.npy"
    )
    # Sparse sensor JSON disabled; dense_les mode samples the full N×N field.
    config.sensor_json = None

    # === Loss weights: data on, PDE off, IC off ===
    # data terms reweighted to 1.0 (down from base 100.0) since dense LES
    # gives 262144 points per frame vs base's 100 sparse points — keep
    # gradient magnitude reasonable.
    config.weighting.init_weights.u_data = 1.0
    config.weighting.init_weights.v_data = 1.0
    config.weighting.init_weights.w_data = 1.0
    # PDE residual losses disabled.
    config.weighting.init_weights.ru = 0.0
    config.weighting.init_weights.rv = 0.0
    config.weighting.init_weights.rc = 0.0
    # IC losses disabled: dense LES already covers t=0 explicitly.
    config.weighting.init_weights.u_ic = 0.0
    config.weighting.init_weights.v_ic = 0.0

    # === Training shape: single long window over t∈[0, 5] ===
    config.training.max_windows_to_run = 1
    config.training.max_steps = 50000

    # === Save 10 ckpts so we can pick an earlier one if 50k over-converges ===
    config.saving.save_every_steps = 5000
    config.saving.num_keep_ckpts = None

    # === wandb identity ===
    config.wandb.name = "re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les"
    config.wandb.group = "re1e6_window1_two_stage"
    config.wandb.tags = list(config.wandb.tags) + [
        "stage1",
        "dense_les",
        "no_pde",
        "Re=1e6",
        "warmup_only",
    ]
    config.wandb.notes = (
        "Stage 1 of two-stage PINN training. Dense LES regression on "
        "Re=1e6 stand-alone LES dataset; PDE and IC losses disabled. "
        "Saves 10 ckpts at 5k step cadence; Stage 2 will load the final "
        "ckpt via transfer_from_ckpt."
    )

    return config
```

- [ ] **Step 4: Run test, verify it PASSES**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/test_window1_stage1_les_config.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py \
        tests/test_window1_stage1_les_config.py
git commit -m "$(cat <<'EOF'
✨ feat(stage1): add Stage 1 LES regression config (dense_les, no PDE)

Sub-project B Stage 1: dense regression on Re=1e6 LES dataset over
single window t∈[0, 5]. PDE and IC losses zeroed. SOAP optimizer
inherited from sensor100 base for warmstart compatibility with Stage 2.

Data loss weights set to 1.0 (down from base 100.0) because dense_les
samples 262144 points/frame vs base's 100 sparse points — keeps
gradient magnitude reasonable.

Save every 5000 steps (10 ckpts total) so an earlier ckpt is
selectable as Stage 2 source if 50k over-converges.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Stage 2 config + test (TDD)

**Files:**
- Create: `tests/test_window1_stage2_warm_config.py`
- Create: `examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_window1_stage2_warm_config.py`:

```python
"""
What:
    驗證 window-1 Stage 2 warmstart config 的關鍵行為。
Why:
    Stage 2 必須與 3491 (paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval)
    完全一致，唯一差別是 warmstart from Stage 1 ckpt。任何 hyperparameter
    漂移（dw, max_steps, save cadence）都會破壞 apples-to-apples 對比。
"""

import sys
import types
import unittest


def _install_stub_modules():
    if "jax" not in sys.modules:
        jax_mod = types.ModuleType("jax")
        jnp_mod = types.ModuleType("jax.numpy")
        jnp_mod.pi = 3.141592653589793
        jax_mod.numpy = jnp_mod
        sys.modules["jax"] = jax_mod
        sys.modules["jax.numpy"] = jnp_mod

    if "ml_collections" not in sys.modules:
        mlc = types.ModuleType("ml_collections")

        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        mlc.ConfigDict = ConfigDict
        sys.modules["ml_collections"] = mlc


_install_stub_modules()


class Window1Stage2WarmConfigTest(unittest.TestCase):
    def test_stage2_warmstart_and_mirrors_3491(self):
        from examples.kolmogorov_flow.configs import (
            paper_repro_soap_sensor100_n512_w50_window1_stage2_warm as cfg_mod,
        )

        config = cfg_mod.get_config()

        # === warmstart wiring ===
        self.assertIsInstance(config.transfer_from_ckpt, str)
        self.assertIn("stage1_les", config.transfer_from_ckpt)
        self.assertTrue(
            config.transfer_from_ckpt.endswith("checkpoint_50000"),
            f"transfer_from_ckpt = {config.transfer_from_ckpt}",
        )
        self.assertFalse(config.transfer_optimizer_state)

        # === mirror 3491 exactly ===
        self.assertEqual(config.weighting.init_weights.u_data, 38.0614)
        self.assertEqual(config.weighting.init_weights.v_data, 38.0614)
        self.assertEqual(config.training.max_steps, 100000)
        self.assertEqual(config.training.max_windows_to_run, 1)
        self.assertEqual(config.saving.save_every_steps, 1000)
        self.assertIsNone(config.saving.num_keep_ckpts)

        # === wandb identity ===
        self.assertEqual(
            config.wandb.name,
            "re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm",
        )
        self.assertEqual(config.wandb.group, "re1e6_window1_two_stage")
        for tag in ("stage2", "warmstart", "from_stage1"):
            self.assertIn(tag, config.wandb.tags)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, verify it FAILS**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/test_window1_stage2_warm_config.py -v`

Expected: `ModuleNotFoundError` — config doesn't exist.

- [ ] **Step 3: Write the Stage 2 config**

Write `examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py`:

```python
import examples.kolmogorov_flow.configs.paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval as base_config


def get_config():
    """
    What:
        Stage 2 of two-stage PINN training (sub-project B).
        Mirror 3491 (dw=38.0614, 100k steps, sparse sensor K=100 + PDE);
        only difference is warmstart from Stage 1 LES regression ckpt.
    Why:
        Direct apples-to-apples test of the hypothesis "Stage 1 LES
        prior accelerates sparse-sensor + PDE training". Comparing
        Stage 2 ckpts against 3491 ckpts at matched steps isolates
        the effect of the starting weights.
    """

    config = base_config.get_config()

    # === Inter-run stage transfer (NEW field, train.py patch reads it) ===
    config.transfer_from_ckpt = (
        "/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les/"
        "ckpt/time_window_1/checkpoint_50000"
    )
    config.transfer_optimizer_state = False

    # === Mirror 3491 wandb identity (new group/tags to separate from 3491) ===
    config.wandb.name = "re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm"
    config.wandb.group = "re1e6_window1_two_stage"
    config.wandb.tags = list(config.wandb.tags) + [
        "stage2",
        "warmstart",
        "from_stage1",
    ]
    config.wandb.notes = (
        "Stage 2 of two-stage PINN training. Mirror 3491 exactly except "
        "for transfer_from_ckpt loading Stage 1 LES regression ckpt. "
        "Compare against 3491 (sensor+PDE no warmstart) + no_data baseline."
    )

    return config
```

- [ ] **Step 4: Run test, verify it PASSES**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/test_window1_stage2_warm_config.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Run all stage-related tests to confirm no regression**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/test_window1_stage1_les_config.py tests/test_window1_stage2_warm_config.py tests/test_window1_dw38_eval_config.py tests/test_window1_dw38_100k_eval_config.py -v`

Expected: 4 passed (2 new + 2 existing 3491-family configs).

- [ ] **Step 6: Commit**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py \
        tests/test_window1_stage2_warm_config.py
git commit -m "$(cat <<'EOF'
✨ feat(stage2): add Stage 2 warmstart config (mirror 3491 + transfer_from_ckpt)

Sub-project B Stage 2: clone of 3491 config with NEW field
transfer_from_ckpt pointing to Stage 1 final ckpt. All other params
identical to 3491 (dw=38.0614, 100k steps, sensor K=100, save every
1000 steps). Single-variable isolation of "does LES prior help?"

transfer_optimizer_state=False — Stage 1 SOAP state was tuned for
regression-only loss; fresh state for PDE-augmented Stage 2.

Wandb group "re1e6_window1_two_stage" separates this lineage from
3491's "re1e6_window1_fixed_weight_eval".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: train.py `transfer_from_ckpt` patch (TDD)

**Files:**
- Create: `tests/test_transfer_from_ckpt.py`
- Modify: `examples/kolmogorov_flow/train.py` (insert ~15 lines before line 796)

- [ ] **Step 1: Write the failing test**

Write `tests/test_transfer_from_ckpt.py`:

```python
"""
What:
    Regression test for the inter-run stage transfer hook in
    examples/kolmogorov_flow/train.py.
Why:
    Stage 2 of sub-project B needs to load Stage 1's final ckpt at
    `idx == 0` (first window). Existing `transfer_learning=True` flow
    only fires at `idx > 0` (intra-run window-to-window). The new
    `transfer_from_ckpt` config field must:
    1. trigger only at idx==0 AND when the field is set
    2. invoke restore_checkpoint with the configured path
    3. invoke _create_train_state with the restored params/weights
       and replicate=False
    4. NOT trigger when the field is absent (backward compatibility)
    5. NOT shadow the existing idx>0 transfer_learning path

    These properties are verified via monkeypatch of the two boundary
    functions; no actual training is run.
"""

import sys
import types
import unittest
from unittest import mock


def _install_stub_modules():
    """Same stub pattern as the config tests."""
    if "jax" not in sys.modules:
        jax_mod = types.ModuleType("jax")
        jnp_mod = types.ModuleType("jax.numpy")
        jnp_mod.pi = 3.141592653589793
        jax_mod.numpy = jnp_mod
        sys.modules["jax"] = jax_mod
        sys.modules["jax.numpy"] = jnp_mod

    if "ml_collections" not in sys.modules:
        mlc = types.ModuleType("ml_collections")

        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        mlc.ConfigDict = ConfigDict
        sys.modules["ml_collections"] = mlc


_install_stub_modules()


# We test the patch logic in isolation by importing only the small
# branching function that we will extract from train.py.
def _make_fake_config(*, transfer_from_ckpt=None, transfer_learning=False):
    """Build a minimal config object with just the fields the patch reads."""
    class Cfg:
        def __init__(self):
            self.transfer_from_ckpt = transfer_from_ckpt
            self.transfer_learning = transfer_learning

        def get(self, key, default=None):
            return getattr(self, key, default)

    return Cfg()


def _make_fake_state():
    """Build a minimal state object with .params, .weights, .opt_state."""
    return types.SimpleNamespace(
        params={"fake": "params"},
        weights={"fake": "weights"},
        opt_state={"fake": "opt_state"},
        step=0,
    )


class TransferFromCkptTest(unittest.TestCase):
    """Tests the helper function maybe_apply_stage_transfer()."""

    def test_patch_fires_when_idx_zero_and_field_set(self):
        from examples.kolmogorov_flow.train import maybe_apply_stage_transfer

        config = _make_fake_config(transfer_from_ckpt="/tmp/fake-ckpt")
        captured = {"restore_called_with": None, "create_called_with": None}

        def fake_restore(state, path):
            captured["restore_called_with"] = path
            return _make_fake_state()

        def fake_create(cfg, params=None, weights=None, replicate=None):
            captured["create_called_with"] = (params, weights, replicate)
            return "fake-new-state"

        result = maybe_apply_stage_transfer(
            idx=0,
            config=config,
            current_state=_make_fake_state(),
            restore_fn=fake_restore,
            create_state_fn=fake_create,
        )

        self.assertEqual(captured["restore_called_with"], "/tmp/fake-ckpt")
        self.assertEqual(captured["create_called_with"][0], {"fake": "params"})
        self.assertEqual(captured["create_called_with"][1], {"fake": "weights"})
        self.assertEqual(captured["create_called_with"][2], False)
        self.assertEqual(result, "fake-new-state")

    def test_patch_does_not_fire_when_idx_nonzero(self):
        from examples.kolmogorov_flow.train import maybe_apply_stage_transfer

        config = _make_fake_config(transfer_from_ckpt="/tmp/fake-ckpt")
        sentinel = _make_fake_state()

        def fake_restore(state, path):
            self.fail("restore must not be called when idx > 0")

        def fake_create(cfg, **kwargs):
            self.fail("create must not be called when idx > 0")

        result = maybe_apply_stage_transfer(
            idx=1,
            config=config,
            current_state=sentinel,
            restore_fn=fake_restore,
            create_state_fn=fake_create,
        )

        # When the patch doesn't fire, it returns the current state unchanged.
        self.assertIs(result, sentinel)

    def test_patch_does_not_fire_when_field_absent(self):
        from examples.kolmogorov_flow.train import maybe_apply_stage_transfer

        config = _make_fake_config(transfer_from_ckpt=None)
        sentinel = _make_fake_state()

        def fake_restore(state, path):
            self.fail("restore must not be called when transfer_from_ckpt is None")

        def fake_create(cfg, **kwargs):
            self.fail("create must not be called when transfer_from_ckpt is None")

        result = maybe_apply_stage_transfer(
            idx=0,
            config=config,
            current_state=sentinel,
            restore_fn=fake_restore,
            create_state_fn=fake_create,
        )

        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, verify it FAILS**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/test_transfer_from_ckpt.py -v`

Expected: `ImportError: cannot import name 'maybe_apply_stage_transfer' from 'examples.kolmogorov_flow.train'` — function doesn't exist.

- [ ] **Step 3: Add `maybe_apply_stage_transfer` helper to train.py**

In `examples/kolmogorov_flow/train.py`, find the function definitions near the top of the module (after imports, before the main training function). Insert this helper. A safe location is immediately after the imports — search for the first `def ` in the module and insert before it.

Add to `examples/kolmogorov_flow/train.py`:

```python
def maybe_apply_stage_transfer(
    idx,
    config,
    current_state,
    restore_fn,
    create_state_fn,
):
    """Inter-run stage transfer hook (Stage 1 → Stage 2 warmstart).

    What:
        At the first window (idx==0), if config.transfer_from_ckpt is
        set, restore params + weights from that external ckpt and
        rebuild model.state with fresh opt_state.
    Why:
        Existing transfer_learning flow handles intra-run window-to-
        window transfer (idx > 0). Sub-project B needs an idx==0 path
        for stage-to-stage transfer (e.g. Stage 2 warmstarts from
        Stage 1 final ckpt). The two paths are orthogonal: this hook
        does nothing when idx > 0 or when the field is absent.
    Returns:
        Either the new state (when transfer fires) or current_state
        unchanged (when transfer is skipped).
    """
    transfer_path = config.get("transfer_from_ckpt") if hasattr(config, "get") else getattr(config, "transfer_from_ckpt", None)
    if idx != 0 or not transfer_path:
        return current_state

    src_state = restore_fn(current_state, transfer_path)
    new_state = create_state_fn(
        config,
        params=src_state.params,
        weights=src_state.weights,
        replicate=False,
    )
    return new_state
```

- [ ] **Step 4: Wire the helper into the per-window loop**

Find line 796 in `examples/kolmogorov_flow/train.py`:

```python
        if config.transfer_learning:
            if idx > 0:
                # restore the checkpoint from the previous time window
```

Insert immediately BEFORE the `if config.transfer_learning:` line:

```python
        # === Inter-run stage transfer (sub-project B: Stage 1 → Stage 2) ===
        # Hook helper handles its own guards (idx==0 + field present).
        # Returns current_state unchanged if hook doesn't fire.
        model.state = maybe_apply_stage_transfer(
            idx=idx,
            config=config,
            current_state=model.state,
            restore_fn=restore_checkpoint,
            create_state_fn=_create_train_state,
        )
        if parallel_state["num_devices"] > 1 and idx == 0 and getattr(config, "transfer_from_ckpt", None):
            # Re-shard if the helper just rebuilt state on unreplicated host.
            model.state = jax.device_put(model.state, parallel_state["replicated_sharding"])
            logging.info(
                f"Stage transfer complete — params + weights loaded from {config.transfer_from_ckpt}, opt_state fresh"
            )
```

- [ ] **Step 5: Run the new tests, verify they PASS**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/test_transfer_from_ckpt.py -v`

Expected: 3 passed.

- [ ] **Step 6: Run the full test suite for regression**

Run: `cd /Users/latteine/Documents/coding/jaxpi && /Users/latteine/Documents/coding/jaxpi/.venv/bin/python -m pytest tests/ -v`

Expected: all existing tests still pass (the new tests for stage configs + transfer hook, plus all sub-project A tests, plus any other pre-existing tests).

- [ ] **Step 7: Commit**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add examples/kolmogorov_flow/train.py tests/test_transfer_from_ckpt.py
git commit -m "$(cat <<'EOF'
✨ feat(train): add transfer_from_ckpt hook for inter-run stage transfer

Sub-project B Stage 2 needs to load Stage 1's final ckpt at idx==0
(first window). Existing transfer_learning flow only fires at idx>0
(intra-run window-to-window). New helper maybe_apply_stage_transfer()
provides the orthogonal idx==0 path:

- Triggers only when idx==0 AND config.transfer_from_ckpt is set
- Transfers params + weights via restore_checkpoint + _create_train_state
- Does NOT transfer opt_state (Stage 1 SOAP state tuned for regression-
  only loss; fresh state safer for PDE-augmented Stage 2)
- Returns current_state unchanged if guard conditions fail

Helper is dependency-injected for unit testability — the test passes
fake restore_fn / create_state_fn and verifies the boundary calls
without running actual training. Three regression tests cover:
  1. Patch fires at idx==0 with field set → calls boundary functions
  2. Patch silent at idx>0 (doesn't shadow existing windowed path)
  3. Patch silent when field absent (backward compatibility)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Push + lab-server sync

**Files:** none (git remote + ssh pull only)

- [ ] **Step 1: Push commits**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git push origin pirate
```

Expected: 3 commits pushed (Stage 1, Stage 2, train.py patch).

- [ ] **Step 2: Pull on lab-server**

```bash
ssh lab-server 'cd ~/jaxpi && git fetch origin && git pull --ff-only origin pirate 2>&1 | tail -5'
```

Expected: fast-forward applied, no merge conflict.

- [ ] **Step 3: Verify tests pass on lab-server**

```bash
ssh lab-server 'cd ~/jaxpi && /home/junyi/.local/bin/uv run python -m pytest tests/test_window1_stage1_les_config.py tests/test_window1_stage2_warm_config.py tests/test_transfer_from_ckpt.py -v'
```

Expected: 5 passed (1 Stage 1 + 1 Stage 2 + 3 transfer patch).

If anything fails on lab-server: STOP, investigate before launching Stage 1.

---

## Task 6: Stage 1 training run

**Files:**
- Create: `lab-server:/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les/ckpt/time_window_1/checkpoint_50000/`
- Create: `lab-server:~/jaxpi/logs/train_kf_w50_w1_stage1_les_<JOB>.{out,err}`

- [ ] **Step 1: Pre-warm uv cache on lab-server login node (avoid DNS race)**

```bash
ssh lab-server 'cd ~/jaxpi && /home/junyi/.local/bin/uv sync 2>&1 | tail -3'
```

Expected: silent or "Audited X packages" — cache already warm.

- [ ] **Step 2: sbatch Stage 1 training**

```bash
ssh lab-server '
sbatch \
  --job-name=train_kf_w50_w1_stage1_les \
  --export=ALL,CONFIG_PATH=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py,RUN_SLUG=train_kf_w50_w1_stage1_les \
  ~/jaxpi/slurm/train/train_kolmogorov_re1e6_sensor100_w25_soap.sh
sleep 8
squeue -u $USER
'
```

Expected: `Submitted batch job <JOBID>` then `squeue` shows the job RUNNING on `acmt20` (or queued).

- [ ] **Step 3: Verify job started cleanly after 60s**

```bash
ssh lab-server '
sleep 60
JOBID=$(squeue -u $USER -h -o "%i" -n train_kf_w50_w1_stage1_les | head -1)
echo "JobID: $JOBID"
sacct -j $JOBID --format=JobID,JobName,State,Elapsed -P 2>&1 | head -3
echo "=== stdout head ==="
tail -25 ~/jaxpi/logs/train_kf_w50_w1_stage1_les_${JOBID}.out 2>&1
echo "=== stderr tail ==="
tail -10 ~/jaxpi/logs/train_kf_w50_w1_stage1_les_${JOBID}.err 2>&1
'
```

Expected: State `RUNNING`, stdout shows `Kolmogorov Re=1e6 SOAP + Sensor100 W25 Train` header + JAX device info + wandb init + `Training time window 1`. No CUDA error, no Python traceback.

If job FAILED in <2 min: read `.err`; common failure modes:
- DNS error from uv → re-run Step 1 then re-submit
- ConfigDict error → check Stage 1 config syntax
- `dataset_path not found` → confirm Task 1 dataset rsync landed

- [ ] **Step 4: Poll until completion (background Bash run_in_background)**

```bash
ssh lab-server '
JOBID=$(squeue -u $USER -h -o "%i" -n train_kf_w50_w1_stage1_les | head -1)
while squeue -j $JOBID -h >/dev/null 2>&1 && [ -n "$(squeue -j $JOBID -h)" ]; do
  echo "[$(date -Is)] Stage 1 PID still in queue: $(squeue -j $JOBID -h)"
  sleep 600
done
echo "=== Stage 1 finished at $(date -Is) ==="
sacct -j $JOBID --format=JobID,JobName,State,Elapsed -P 2>&1 | head -3
echo "=== final stdout tail ==="
tail -20 ~/jaxpi/logs/train_kf_w50_w1_stage1_les_${JOBID}.out 2>&1
'
```

Use `Bash run_in_background: true` for this; the background task notifies when complete.

Expected: completion in 3-5 hours; final state `COMPLETED`; stdout shows `Exit code : 0` + Elapsed in seconds.

- [ ] **Step 5: Verify Stage 1 ckpt landed**

```bash
ssh lab-server '
CKPT=/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les/ckpt/time_window_1
ls -la $CKPT
echo ""
echo "=== ckpts present ==="
ls $CKPT | sort
'
```

Expected: 10 ckpts at steps `5000, 10000, ..., 50000`. Final ckpt `checkpoint_50000` exists.

If `checkpoint_50000` missing: Stage 1 may have failed before final save. Check `.err` log; do NOT proceed to Stage 2.

---

## Task 7: Stage 2 training run (warmstart from Stage 1)

**Files:**
- Create: `lab-server:/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm/ckpt/time_window_1/checkpoint_{1000..100000}/` (100 ckpts)
- Create: `lab-server:~/jaxpi/logs/train_kf_w50_w1_stage2_warm_<JOB>.{out,err}`

- [ ] **Step 1: Verify Stage 1 ckpt path matches Stage 2 config's `transfer_from_ckpt`**

```bash
ssh lab-server '
EXPECTED=/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les/ckpt/time_window_1/checkpoint_50000
ls -la $EXPECTED
echo "exists: $?"
'
```

Expected: ckpt dir exists with orbax content. If not, Task 6 didn't complete properly — STOP.

- [ ] **Step 2: sbatch Stage 2 training**

```bash
ssh lab-server '
sbatch \
  --job-name=train_kf_w50_w1_stage2_warm \
  --export=ALL,CONFIG_PATH=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py,RUN_SLUG=train_kf_w50_w1_stage2_warm \
  ~/jaxpi/slurm/train/train_kolmogorov_re1e6_sensor100_w25_soap.sh
sleep 8
squeue -u $USER
'
```

Expected: Submitted, RUNNING within 60s.

- [ ] **Step 3: Verify the transfer fired in stdout**

```bash
ssh lab-server '
sleep 90
JOBID=$(squeue -u $USER -h -o "%i" -n train_kf_w50_w1_stage2_warm | head -1)
echo "JobID: $JOBID"
echo "=== stderr — look for 'Stage transfer' line ==="
grep -E "Stage transfer|transfer_from_ckpt" ~/jaxpi/logs/train_kf_w50_w1_stage2_warm_${JOBID}.err 2>&1 | head -5
echo "=== stdout head ==="
tail -25 ~/jaxpi/logs/train_kf_w50_w1_stage2_warm_${JOBID}.out 2>&1
'
```

Expected: `Stage transfer complete — params + weights loaded from /home/junyi/.../checkpoint_50000, opt_state fresh` in stderr. If this line is MISSING the warmstart did not fire — STOP and investigate.

- [ ] **Step 4: Poll until completion**

```bash
ssh lab-server '
JOBID=$(squeue -u $USER -h -o "%i" -n train_kf_w50_w1_stage2_warm | head -1)
while squeue -j $JOBID -h >/dev/null 2>&1 && [ -n "$(squeue -j $JOBID -h)" ]; do
  echo "[$(date -Is)] Stage 2 still running: $(squeue -j $JOBID -h)"
  sleep 600
done
echo "=== Stage 2 finished at $(date -Is) ==="
sacct -j $JOBID --format=JobID,JobName,State,Elapsed -P 2>&1 | head -3
echo "=== final stdout tail ==="
tail -20 ~/jaxpi/logs/train_kf_w50_w1_stage2_warm_${JOBID}.out 2>&1
'
```

Use `Bash run_in_background: true` for this; expected wallclock 6-10 hours.

Expected: `COMPLETED`, Exit code 0.

- [ ] **Step 5: Verify 100 Stage 2 ckpts landed**

```bash
ssh lab-server '
CKPT=/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm/ckpt/time_window_1
echo "=== total ckpts ==="
ls $CKPT | wc -l
echo ""
echo "=== first 3 / last 3 ==="
ls $CKPT | sort -t_ -k2 -n | head -3
echo "..."
ls $CKPT | sort -t_ -k2 -n | tail -3
'
```

Expected: 100 ckpts; first 3 are `checkpoint_1000, checkpoint_2000, checkpoint_3000`; last 3 are `checkpoint_98000, checkpoint_99000, checkpoint_100000`.

---

## Task 8: Eval A — Stage 2 vs 3491 (primary research question)

**Files:**
- Create: `lab-server:~/jaxpi/eval_runs/stage2_warm_vs_3491_<DATE>/{validation_report.txt, checkpoint_sweep_results.csv, *.png}`

- [ ] **Step 1: sbatch Eval A**

```bash
ssh lab-server '
DATE=$(date +%Y%m%d)
mkdir -p ~/jaxpi/eval_runs/stage2_warm_vs_3491_${DATE}

export NO_DATA_CONFIG=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_dw38_100k_eval.py
export NO_DATA_CKPT_ROOT=/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_dw38_100k_eval/ckpt
export SENSOR_CONFIG=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py
export SENSOR_CKPT_ROOT=/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm/ckpt
export CHECKPOINT_STEPS="10000,20000,30000,40000,50000,60000,70000,80000,90000,100000"
export OUTPUT_DIR=/home/junyi/jaxpi/eval_runs/stage2_warm_vs_3491_${DATE}

sbatch \
  --job-name=post_stage2_vs_3491 \
  --export=ALL \
  ~/jaxpi/slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh
sleep 5
squeue -u $USER
'
```

Expected: RUNNING; wallclock ~15-20 min.

- [ ] **Step 2: Poll until completion**

```bash
ssh lab-server '
JOBID=$(squeue -u $USER -h -o "%i" -n post_stage2_vs_3491 | head -1)
while squeue -j $JOBID -h >/dev/null 2>&1 && [ -n "$(squeue -j $JOBID -h)" ]; do
  echo "[$(date -Is)] Eval A still running"
  sleep 300
done
echo "=== Eval A finished ==="
sacct -j $JOBID --format=JobID,State,Elapsed -P 2>&1 | head -2
'
```

Use `Bash run_in_background: true`.

Expected: COMPLETED in 15-25 min.

- [ ] **Step 3: Inspect Eval A results**

```bash
ssh lab-server '
DATE=$(date +%Y%m%d)
DIR=~/jaxpi/eval_runs/stage2_warm_vs_3491_${DATE}
ls -la $DIR
echo ""
echo "=== CSV ==="
cat $DIR/checkpoint_sweep_results.csv
'
```

Expected: 4 files (csv + npz + summary + 3 PNGs). CSV has 20 rows (10 no_data/3491 + 10 sensor/stage2).

---

## Task 9: Eval B — Stage 2 vs no_data (sanity check)

**Files:**
- Create: `lab-server:~/jaxpi/eval_runs/stage2_warm_vs_nodata_<DATE>/{validation_report.txt, checkpoint_sweep_results.csv, *.png}`

- [ ] **Step 1: sbatch Eval B**

```bash
ssh lab-server '
DATE=$(date +%Y%m%d)
mkdir -p ~/jaxpi/eval_runs/stage2_warm_vs_nodata_${DATE}

export NO_DATA_CONFIG=examples/kolmogorov_flow/configs/paper_repro_soap_window1_ablation.py
export NO_DATA_CKPT_ROOT=/home/junyi/jaxpi/re1e6_n512_ds4_soap_w1_ablation/ckpt
export SENSOR_CONFIG=examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py
export SENSOR_CKPT_ROOT=/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm/ckpt
export CHECKPOINT_STEPS="10000,20000,30000,40000,50000,60000,70000,80000,90000,100000"
export OUTPUT_DIR=/home/junyi/jaxpi/eval_runs/stage2_warm_vs_nodata_${DATE}

sbatch \
  --job-name=post_stage2_vs_nodata \
  --export=ALL \
  ~/jaxpi/slurm/postprocess/postprocess_kolmogorov_window1_checkpoint_sweep.sh
sleep 5
squeue -u $USER
'
```

- [ ] **Step 2: Poll until completion** — same pattern as Task 8 Step 2.

- [ ] **Step 3: Inspect Eval B CSV** — same pattern as Task 8 Step 3.

- [ ] **Step 4: Pull both eval artefact sets to Mac**

```bash
DATE=$(date +%Y%m%d)
mkdir -p /Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_3491_${DATE}
mkdir -p /Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_nodata_${DATE}

rsync -avz lab-server:~/jaxpi/eval_runs/stage2_warm_vs_3491_${DATE}/ \
    /Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_3491_${DATE}/

rsync -avz lab-server:~/jaxpi/eval_runs/stage2_warm_vs_nodata_${DATE}/ \
    /Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_nodata_${DATE}/

echo "=== local artefacts ==="
ls -la /Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_3491_${DATE}/
ls -la /Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_nodata_${DATE}/
```

Expected: 4 files in each dir.

---

## Task 10: EXPERIMENT_RECORD writeback + final commit

**Files:**
- Modify: `EXPERIMENT_RECORD.md` (add `[INDEX] Active` entry for `3491_followup` series + chronological log entries)

- [ ] **Step 1: Add `[INDEX] Active` entry for the two-stage runs**

In `EXPERIMENT_RECORD.md`, find `## [INDEX] Active Experiments`. Add this entry immediately under that header:

```markdown
### `<STAGE2_JOBID>` | `re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm`

| Field | Value |
| :--- | :--- |
| Status | Completed (`<COMPLETION_TIMESTAMP>`, Elapsed `<ELAPSED>`) on `acmt20`, 2x RTX 3090 |
| Config | [paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/configs/paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py) |
| Dataset | [kolmogorov_Re1e6_N512_T5_ds4.npy](/Users/latteine/Documents/coding/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N512_T5_ds4.npy) |
| Sensor Constraint | `QR-pivot K100` + fixed `u_data=v_data=38.0614` + `w_data=0` |
| Time Horizon | `window 1` only, `max_steps=100000` |
| Checkpoint Policy | `save_every_steps=1000`, `num_keep_ckpts=None` (100 ckpts) |
| Warmstart | `transfer_from_ckpt = .../w1_stage1_les/ckpt/time_window_1/checkpoint_50000`, `transfer_optimizer_state=False` |
| Stage 1 Prerequisite | `<STAGE1_JOBID>` (3-5h SOAP regression on Re=1e6 LES, 50k steps, 10 ckpts at 5k cadence) |
| Workdir | `/home/junyi/jaxpi/runs/train_kf_w50_w1_stage2_warm_<STAGE2_JOBID>` |
| Eval Jobs | `<EVAL_A_JOBID>` (Stage 2 vs 3491, primary), `<EVAL_B_JOBID>` (Stage 2 vs no_data, sanity) |
| Result @ step 100000 (direct apply_fn) | Stage 2 `(u=<U>, v=<V>, w=<W>)` vs 3491 `(u=1.059e-3, v=1.096e-3, w=0.711e-3)` |
| Verdict (Q1: Stage 1 prior helps over 3491?) | <PASTE_VERDICT_HERE>: <one of: win / partial win / null result / loss> |
| Verdict (Q2: Stage 2 reaches no_data baseline?) | <PASTE_VERDICT_HERE>: <one of: acceptable / concerning> |
| Caveat | Stage 1 LES dataset is sub-equilibrium (see sub-project A 2026-05-15 record). Stage 2 inherits any non-physical statistics learned in Stage 1. |
| RNG Strategy | Not recorded |
```

Fill `<STAGE2_JOBID>`, `<STAGE1_JOBID>`, `<EVAL_A_JOBID>`, `<EVAL_B_JOBID>`, `<COMPLETION_TIMESTAMP>`, `<ELAPSED>`, `<U>`, `<V>`, `<W>`, and the two verdicts based on actual outputs from Tasks 6-9.

- [ ] **Step 2: Add chronological log entry for the full sub-project B**

Insert immediately under `## [LOG] Chronological` (above the existing top entry):

```markdown
### [<DATE>] sub-project B | Two-stage PINN training — Stage 1 + Stage 2 + dual eval

- Time: `<DATE>`
- Status: Stage 1 + Stage 2 both COMPLETED; dual eval done
- Experiment or Job IDs: `<STAGE1_JOBID>` (Stage 1), `<STAGE2_JOBID>` (Stage 2), `<EVAL_A_JOBID>` (vs 3491), `<EVAL_B_JOBID>` (vs no_data)

Change:

- Rsynced Re=1e6 LES dataset from home-gpu to lab-server as `kolmogorov_les_re1000000.npy` (404 MB; sub-project A artefact, sub-equilibrium caveat applies).
- Added Stage 1 config `paper_repro_soap_sensor100_n512_w50_window1_stage1_les.py`: dense LES regression, PDE / IC weights zeroed, SOAP, max_steps=50000, save_every_steps=5000.
- Added Stage 2 config `paper_repro_soap_sensor100_n512_w50_window1_stage2_warm.py`: mirror 3491 + `transfer_from_ckpt` pointing to Stage 1 final ckpt; `transfer_optimizer_state=False`.
- Modified `examples/kolmogorov_flow/train.py`: added `maybe_apply_stage_transfer` helper + per-window-loop call site. Triggers only at `idx==0` AND `transfer_from_ckpt` set; orthogonal to existing `transfer_learning` windowed flow. Regression test covers 3 paths (fires correctly, silent at idx>0, silent when field absent).
- Stage 1 training: `<STAGE1_ELAPSED>`, 10 ckpts at 5k cadence under `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les/ckpt/time_window_1/`.
- Stage 2 training: `<STAGE2_ELAPSED>`, 100 ckpts at 1k cadence under `/home/junyi/jaxpi/re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm/ckpt/time_window_1/`. Stderr confirmed `Stage transfer complete — params + weights loaded from <PATH>, opt_state fresh` line at idx==0.
- Eval A (Stage 2 vs 3491) and Eval B (Stage 2 vs no_data): both completed, artefacts under `eval_runs/stage2_warm_vs_3491_<DATE>/` and `eval_runs/stage2_warm_vs_nodata_<DATE>/`.

Evidence (10-point direct apply_fn corrected error):

| step    | Stage 2 (u,v,w) ×1e-3 | 3491 (u,v,w) ×1e-3   | no_data (u,v,w) ×1e-3 |
| ------: | --------------------- | -------------------- | --------------------- |
|  10 000 | <PASTE>               | (2.16, 1.87, 2.64)   | (1.81, 1.69, 2.58)    |
|  50 000 | <PASTE>               | (1.12, 1.06, 0.81)   | (1.14, 1.20, 0.82)    |
| 100 000 | <PASTE>               | (1.06, 1.10, 0.71)   | (1.08, 1.09, 0.70)    |

- [eval_runs/stage2_warm_vs_3491_<DATE>/checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_3491_<DATE>/checkpoint_sweep_results.csv)
- [eval_runs/stage2_warm_vs_3491_<DATE>/checkpoint_sweep_error_vs_step.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_3491_<DATE>/checkpoint_sweep_error_vs_step.png)
- [eval_runs/stage2_warm_vs_nodata_<DATE>/checkpoint_sweep_results.csv](/Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_nodata_<DATE>/checkpoint_sweep_results.csv)
- [eval_runs/stage2_warm_vs_nodata_<DATE>/checkpoint_sweep_error_vs_step.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/stage2_warm_vs_nodata_<DATE>/checkpoint_sweep_error_vs_step.png)

Interpretation:

- **Q1 (Stage 1 prior helps over 3491?): <VERDICT>.** <One-sentence narrative based on the result.>
- **Q2 (Stage 2 reaches no_data baseline?): <VERDICT>.** <One-sentence narrative.>
- Sub-project A's sub-equilibrium caveat <was / was not> a deciding factor: <observation about whether Stage 1's non-physical spectrum impaired Stage 2 in observed ways>.

Next:

- <If null result>: Confirms 3491's null result generalises — sparse-sensor + PDE + dense-LES prior all fail to accelerate window-1 corrected error. Bottleneck likely in model expressiveness, optimizer choice, or PDE conditioning. Next research direction: re-examine architecture (PirateNet variants, deeper net) or change problem framing (different Re, different window).
- <If win>: Pivot to multi-window Stage 2 to test whether the prior helps cross-window drift. Re-sweep `dw` on top of Stage 1 init to find new optimum.
- <If loss>: Investigate whether sub-equilibrium Stage 1 actively misled Stage 2. Repeat with longer T_end on the LES dataset.
```

Fill all `<...>` placeholders with actual numbers from Tasks 6-9.

- [ ] **Step 3: Commit + push**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add EXPERIMENT_RECORD.md \
    eval_runs/stage2_warm_vs_3491_*/checkpoint_sweep_results.csv \
    eval_runs/stage2_warm_vs_nodata_*/checkpoint_sweep_results.csv
git commit -m "$(cat <<'EOF'
📝 docs(record): land two-stage PINN training (sub-project B) results

Stage 1 (Re=1e6 LES regression, 50k steps, <ELAPSED1>) and Stage 2
(sensor + PDE + Stage 1 warmstart, 100k steps, <ELAPSED2>) both
completed. Dual eval against 3491 + no_data baseline produced
<COUNT> data points each.

Q1 (Stage 1 prior helps over 3491?): <VERDICT>
Q2 (Stage 2 reaches no_data baseline?): <VERDICT>

Sub-project A's sub-equilibrium caveat <was/was not> material to
the result.

EXPERIMENT_RECORD.md updates:
- [INDEX] Active: new entry for Stage 2 with full eval result
- [LOG] Chronological: cradle-to-grave narrative of sub-project B
  with PNG/CSV evidence references

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push origin pirate
```

---

## Self-Review checklist run against spec

- [x] **Spec Goals 1 (two new configs + train.py patch)** — Tasks 2, 3, 4 cover the three artefacts with TDD.
- [x] **Spec Goals 2 (minimal train.py patch)** — Task 4 inserts `maybe_apply_stage_transfer` helper + single call site, ~15 lines.
- [x] **Spec Goals 3 (Stage 1 + Stage 2 training)** — Tasks 6, 7 sbatch both with the production hyperparams from the spec.
- [x] **Spec Goals 4 (two evals reusing existing evaluator)** — Tasks 8, 9 use existing `postprocess_kolmogorov_window1_checkpoint_sweep.sh`.
- [x] **Spec Goals 5 (≤18h total)** — Tasks 6 (3-5h) + 7 (6-10h) + 8 (15-25min) + 9 (15-25min) + other tasks <2h. Within budget.
- [x] **Spec Non-goal 1 (no new evaluator code)** — confirmed by Tasks 8, 9 reusing the existing launcher.
- [x] **Spec Non-goal 2 (no multi-window Stage 2)** — Task 3's config keeps `max_windows_to_run=1`.
- [x] **Spec Non-goal 3 (no new LES dataset)** — Task 1 rsync's the existing sub-project A artefact.
- [x] **Spec Non-goal 4 (no re-sweep)** — Task 3 hardcodes `dw=38.0614`.
- [x] **Spec Non-goal 5 (no architecture change)** — Stage 1 and Stage 2 inherit from the same `paper_repro_soap_sensor100_n512_w50_window1_ablation` lineage; arch is identical.
- [x] **Component 1 (Stage 1 field table)** — fully captured by Task 2 Step 3 config + Step 1 test.
- [x] **Component 2 (train.py patch code)** — Task 4 Step 3 (helper) + Step 4 (call site) + Step 1 (3 regression tests).
- [x] **Component 3 (Stage 2 field table)** — fully captured by Task 3 Step 3 config + Step 1 test.
- [x] **Component 4 (Tests for both configs + patch)** — Tasks 2, 3, 4 each include a TDD-first test.
- [x] **Data flow (Phase 0/1/2/3/4)** — Tasks 1 (Phase 0), 6 (Phase 1), 7 (Phase 2), 8+9 (Phase 3), 10 (Phase 4).
- [x] **Validation / Success criteria** — Tasks 8, 9 produce the CSV/PNG evidence; Task 10 records the verdicts.
- [x] **Risks R1-R7** — R2 mitigation via Task 4 regression tests; R3 mitigation via Task 2 saving 10 ckpts (earlier ckpt selectable); R5 via base config's warmup_steps (inherited unchanged); rest are observation-only and recorded in Task 10's chronological entry.
- [x] **Placeholder scan** — `<JOB>`, `<DATE>`, `<U>`, etc. are template fields to fill from actual runs (not engineer guesses). All code blocks are complete; no "TBD" / "TODO" / "fill in details".
- [x] **Type / name consistency** — `transfer_from_ckpt` (field), `maybe_apply_stage_transfer` (helper), `restore_checkpoint` (existing function), `_create_train_state` (existing function), `re1e6_n512_ds4_soap_sensor100_w50_w1_stage1_les` (wandb name), `re1e6_n512_ds4_soap_sensor100_w50_w1_stage2_warm` (wandb name) — all consistent across tasks.
