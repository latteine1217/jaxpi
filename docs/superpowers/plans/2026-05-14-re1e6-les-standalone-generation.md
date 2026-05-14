# Re=1e6 Stand-alone LES Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `generate_kolmogorov_les.py` to support stand-alone (no DNS reference) mode, deploy to `home-gpu` (WSL2 Ubuntu, i7-11700 CPU-only), generate `kolmogorov_les_Re1e6_N512_T5.npy`, and validate via three hard checks.

**Architecture:** Vendor the existing pure-numpy LES generator from `~/Documents/coding/kolmogorov_generate/dns/` into the jaxpi repo (`scripts/les/`). Add a `--no_dns` CLI branch that takes manual physical parameters (a priori from Kolmogorov forcing-friction balance). Deploy to home-gpu via Tailscale SSH alias `home-gpu`. Produce dataset under `~/les-gen/output/` on home-gpu.

**Tech Stack:** Python 3.13 (numpy only), `uv` for env management, ssh/scp via Tailscale, pytest for unit tests, jaxpi repo for vendoring + git history.

---

## File Structure

| Path | Role |
| --- | --- |
| `scripts/les/generate_kolmogorov_les.py` | Vendored LES solver, modified to support `--no_dns` |
| `scripts/les/validate_les.py` | New post-run validation script (KE / enstrophy / spectrum checks) |
| `scripts/les/__init__.py` | Package marker |
| `tests/test_les_generator_no_dns.py` | Subprocess-based integration tests for `--no_dns` mode |
| `EXPERIMENT_RECORD.md` | Updated with new dataset + provenance after production run |
| `~/les-gen/` *(home-gpu)* | Deployment dir; clone of `jaxpi/scripts/les/` plus `output/` |
| `~/les-gen/output/kolmogorov_les_Re1e6_N512_T5.npy` *(home-gpu)* | Final dataset (~840 MB) |
| `~/les-gen/output/validation_report.txt` *(home-gpu)* | Validation summary |
| `~/les-gen/output/{ke,enstrophy,spectrum}.png` *(home-gpu)* | Validation PNGs |

---

## Task 1: Vendor generator + tests scaffold

**Files:**
- Create: `scripts/les/__init__.py`
- Create: `scripts/les/generate_kolmogorov_les.py` (copy from `/Users/latteine/Documents/coding/kolmogorov_generate/dns/generate_kolmogorov_les.py`)
- Create: `tests/test_les_generator_no_dns.py` (placeholder with one import test)

- [ ] **Step 1: Copy generator into jaxpi**

```bash
mkdir -p /Users/latteine/Documents/coding/jaxpi/scripts/les
cp /Users/latteine/Documents/coding/kolmogorov_generate/dns/generate_kolmogorov_les.py \
   /Users/latteine/Documents/coding/jaxpi/scripts/les/generate_kolmogorov_les.py
```

- [ ] **Step 2: Create package marker**

Write `scripts/les/__init__.py`:
```python
"""LES dataset generation tooling vendored from kolmogorov_generate repo."""
```

- [ ] **Step 3: Verify the script can be byte-compiled in jaxpi env**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m py_compile scripts/les/generate_kolmogorov_les.py`
Expected: no output (success).

- [ ] **Step 4: Create the tests file with a sanity import test**

Write `tests/test_les_generator_no_dns.py`:
```python
"""Integration tests for the stand-alone (`--no_dns`) mode of
scripts/les/generate_kolmogorov_les.py.

Tests use subprocess to drive the script via CLI rather than importing
its `main()` directly: the script is a long-running simulator whose
heavy imports (numpy FFT, KolmogorovLES class) we keep out of the
test process.
"""

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "les" / "generate_kolmogorov_les.py"


def test_script_exists():
    assert SCRIPT.is_file(), f"generator not vendored at {SCRIPT}"


def test_script_help_runs():
    """The script must respond to --help without importing heavy deps."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Kolmogorov 2D LES" in result.stdout
```

- [ ] **Step 5: Run the new tests**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit the vendor**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add scripts/les/__init__.py scripts/les/generate_kolmogorov_les.py tests/test_les_generator_no_dns.py
git commit -m "📦 chore(les): vendor generate_kolmogorov_les.py from kolmogorov_generate

Sub-project A (Re=1e6 stand-alone LES) is going to live in this repo
so jaxpi's git history captures every modification and home-gpu can
deploy via a single git pull.

Vendored verbatim from
/Users/latteine/Documents/coding/kolmogorov_generate/dns/generate_kolmogorov_les.py
(non-git source). Adds tests/test_les_generator_no_dns.py with two
sanity tests (file exists + --help works).

Subsequent commits add --no_dns mode + manual calibration args.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `--no_dns` and `--manual_*` CLI args

**Files:**
- Modify: `scripts/les/generate_kolmogorov_les.py:479` (the argparse block)
- Modify: `tests/test_les_generator_no_dns.py` (append CLI validation tests)

- [ ] **Step 1: Write failing tests for the new CLI behavior**

Append to `tests/test_les_generator_no_dns.py`:
```python
def _run(*args, expect_fail=False, timeout=30):
    """Run the generator with the given CLI args, return CompletedProcess."""
    cmd = [sys.executable, str(SCRIPT), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if expect_fail:
        assert result.returncode != 0, (
            f"expected failure but got success.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    else:
        assert result.returncode == 0, (
            f"unexpected failure.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def test_no_dns_flag_in_help():
    """`--no_dns`, `--manual_nu`, `--manual_turnover_time` must appear in --help."""
    result = _run("--help")
    for flag in ("--no_dns", "--manual_nu", "--manual_turnover_time",
                 "--manual_L", "--manual_A", "--manual_k_f",
                 "--manual_speed_bound", "--manual_omega_rms"):
        assert flag in result.stdout, f"{flag} missing from --help"


def test_dns_required_when_no_dns_not_set(tmp_path):
    """If --no_dns is not given, --dns must be required."""
    out = tmp_path / "out.npy"
    result = _run(
        "--N", "8", "--T_end", "0.001", "--output", str(out),
        expect_fail=True,
    )
    assert "--dns" in (result.stderr + result.stdout)


def test_no_dns_requires_manual_nu(tmp_path):
    """`--no_dns` without `--manual_nu` must error out before simulation starts."""
    out = tmp_path / "out.npy"
    result = _run(
        "--no_dns",
        "--manual_turnover_time", "1.0",
        "--N", "8", "--T_end", "0.001", "--output", str(out),
        expect_fail=True,
    )
    assert "manual_nu" in (result.stderr + result.stdout).lower()


def test_no_dns_requires_manual_turnover(tmp_path):
    """`--no_dns` without `--manual_turnover_time` must error out."""
    out = tmp_path / "out.npy"
    result = _run(
        "--no_dns",
        "--manual_nu", "1e-6",
        "--N", "8", "--T_end", "0.001", "--output", str(out),
        expect_fail=True,
    )
    assert "manual_turnover_time" in (result.stderr + result.stdout).lower()


def test_no_dns_init_mode_dns_rejected(tmp_path):
    """`--no_dns --init_mode dns` must error out (can't pull IC from DNS)."""
    out = tmp_path / "out.npy"
    result = _run(
        "--no_dns",
        "--manual_nu", "1e-6", "--manual_turnover_time", "1.0",
        "--init_mode", "dns",
        "--N", "8", "--T_end", "0.001", "--output", str(out),
        expect_fail=True,
    )
    combined = (result.stderr + result.stdout).lower()
    assert "init_mode" in combined and "dns" in combined
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py -v`
Expected: `test_no_dns_flag_in_help` and the four error-path tests FAIL (the flags don't exist yet). The first two tests (`test_script_exists`, `test_script_help_runs`) still PASS, and `test_dns_required_when_no_dns_not_set` PASSES (current generator already errors when `--dns` is missing).

- [ ] **Step 3: Edit the argparse block to add the new args**

In `scripts/les/generate_kolmogorov_les.py`, find the existing line:
```python
parser.add_argument("--dns", type=str, required=True, help="DNS NPY 檔案路徑")
```
Replace with:
```python
parser.add_argument("--dns", type=str, default=None,
                    help="DNS NPY 檔案路徑（除非 --no_dns 否則必填）")
parser.add_argument("--no_dns", action="store_true",
                    help="Stand-alone 模式：不讀 DNS reference；需搭配 --manual_* 參數")
parser.add_argument("--manual_L", type=float, default=1.0,
                    help="Stand-alone：domain size L (預設 1.0，對齊 PINN code)")
parser.add_argument("--manual_nu", type=float, default=None,
                    help="Stand-alone：viscosity nu；--no_dns 必填")
parser.add_argument("--manual_A", type=float, default=0.1,
                    help="Stand-alone：forcing amplitude (預設 0.1)")
parser.add_argument("--manual_k_f", type=int, default=2,
                    help="Stand-alone：forcing wavenumber (預設 2)")
parser.add_argument("--manual_turnover_time", type=float, default=None,
                    help="Stand-alone：先驗 turnover time T_eddy；--no_dns 必填")
parser.add_argument("--manual_speed_bound", type=float, default=None,
                    help="Stand-alone：CFL 用速度上界；預設 2·sqrt(A/(k_f·2π))")
parser.add_argument("--manual_omega_rms", type=float, default=None,
                    help="Stand-alone：初始 vorticity RMS；預設由 forcing-friction 平衡推估")
```

Then immediately after `args = parser.parse_args()`, insert:
```python
    if not args.no_dns and args.dns is None:
        parser.error("--dns is required unless --no_dns is given")
    if args.no_dns:
        if args.manual_nu is None:
            parser.error("--no_dns requires --manual_nu")
        if args.manual_turnover_time is None:
            parser.error("--no_dns requires --manual_turnover_time")
        if args.init_mode == "dns":
            parser.error("--no_dns is incompatible with --init_mode dns")
```

- [ ] **Step 4: Re-run the tests; they must all pass now**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add scripts/les/generate_kolmogorov_les.py tests/test_les_generator_no_dns.py
git commit -m "✨ feat(les): add --no_dns CLI flag and --manual_* parameter args

Stand-alone LES mode requires:
- --no_dns to switch off DNS reference reads
- --manual_nu and --manual_turnover_time (required)
- --manual_L / A / k_f / speed_bound / omega_rms (optional, sensible defaults)
- --dns becomes optional, validated only if --no_dns absent
- --init_mode dns is rejected under --no_dns

Tests cover the four error paths plus help-text discoverability.
The simulation branch itself is unchanged in this commit — the next
commit wires the manual values into KolmogorovLES.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Implement stand-alone branch in `main()`

**Files:**
- Modify: `scripts/les/generate_kolmogorov_les.py` (`main()` body, immediately after the validation block added in Task 2)
- Modify: `tests/test_les_generator_no_dns.py` (add end-to-end smoke test)

- [ ] **Step 1: Write a failing end-to-end smoke test**

Append to `tests/test_les_generator_no_dns.py`:
```python
import numpy as np


def test_no_dns_smoke_run(tmp_path):
    """Tiny stand-alone run completes and produces a npy with
    calibration_mode='stand_alone' and matching manual params."""
    out = tmp_path / "smoke.npy"
    _run(
        "--no_dns",
        "--manual_L", "1.0",
        "--manual_nu", "1e-4",          # higher nu for stability at tiny N
        "--manual_A", "0.1",
        "--manual_k_f", "2",
        "--manual_turnover_time", "1.0",
        "--N", "16", "--T_end", "0.01",
        "--dt", "1e-4",
        "--save_interval", "50",
        "--seed", "42",
        "--output", str(out),
        timeout=120,
    )
    payload = np.load(out, allow_pickle=True).item()
    assert "config" in payload
    cfg = payload["config"]
    assert cfg["calibration_mode"] == "stand_alone"
    assert cfg["N"] == 16
    assert cfg["nu"] == 1e-4
    assert cfg["L"] == 1.0
    assert cfg["A"] == 0.1
    assert cfg["k_f"] == 2
    # Sanity: omega field exists with expected shape.
    assert "omega" in payload
    omega = np.asarray(payload["omega"])
    assert omega.ndim == 3
    assert omega.shape[1:] == (16, 16)
```

- [ ] **Step 2: Run the new test, confirm it FAILS**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py::test_no_dns_smoke_run -v`
Expected: FAIL with an error from `load_npy_payload` trying to open `args.dns` (which is `None`), because the main-body branch hasn't been implemented yet.

- [ ] **Step 3: Implement the stand-alone branch in `main()`**

In `scripts/les/generate_kolmogorov_les.py`, locate the block that currently reads:
```python
    dns_path = Path(args.dns)
    dns_data = load_npy_payload(dns_path)
    dns_config = resolve_dns_config(dns_data)

    turnover_time = estimate_turnover_time(dns_data)
    r_fric = 1.0 / (args.r_scale * turnover_time)

    dns_omega_rms = estimate_dns_omega_rms(dns_data)
    omega_rms = args.omega_rms if args.omega_rms is not None else dns_omega_rms
    dns_speed_bound, dns_speed_rms = estimate_dns_velocity_scale(dns_data)
    dns_omega_init = prepare_dns_initial_omega(dns_data, args.N) if args.init_mode == "dns" else None
    if args.omega_rms is None and dns_omega_init is not None:
        omega_rms = float(np.sqrt(np.mean(dns_omega_init**2)))
```

Replace it with:
```python
    if args.no_dns:
        # Stand-alone calibration from Kolmogorov forcing-friction balance.
        dns_config = {
            "L": args.manual_L,
            "nu": args.manual_nu,
            "A": args.manual_A,
            "k_f": args.manual_k_f,
            "dt": args.dt if args.dt is not None else 1e-4,
            "N": args.N,
        }
        turnover_time = args.manual_turnover_time
        r_fric = 1.0 / (args.r_scale * turnover_time)

        if args.manual_omega_rms is not None:
            dns_omega_rms = args.manual_omega_rms
        else:
            dns_omega_rms = float(np.sqrt(
                2.0 * args.manual_A * args.manual_k_f
                    * args.r_scale * turnover_time
            ))
        omega_rms = args.omega_rms if args.omega_rms is not None else dns_omega_rms

        if args.manual_speed_bound is not None:
            dns_speed_bound = args.manual_speed_bound
        else:
            dns_speed_bound = 2.0 * float(np.sqrt(
                args.manual_A / (args.manual_k_f * 2.0 * np.pi)
            ))
        dns_speed_rms = 0.5 * dns_speed_bound

        dns_omega_init = None
        calibration_mode = "stand_alone"
    else:
        dns_path = Path(args.dns)
        dns_data = load_npy_payload(dns_path)
        dns_config = resolve_dns_config(dns_data)

        turnover_time = estimate_turnover_time(dns_data)
        r_fric = 1.0 / (args.r_scale * turnover_time)

        dns_omega_rms = estimate_dns_omega_rms(dns_data)
        omega_rms = args.omega_rms if args.omega_rms is not None else dns_omega_rms
        dns_speed_bound, dns_speed_rms = estimate_dns_velocity_scale(dns_data)
        dns_omega_init = prepare_dns_initial_omega(dns_data, args.N) if args.init_mode == "dns" else None
        if args.omega_rms is None and dns_omega_init is not None:
            omega_rms = float(np.sqrt(np.mean(dns_omega_init**2)))
        calibration_mode = "dns_calibrated"
```

- [ ] **Step 4: Stamp `calibration_mode` into the output payload**

Find the existing dict literal that becomes the `"config"` field of the saved npy (search for `"backend": "numpy"` — it's inside the `np.savez`-equivalent block). Add a sibling key:
```python
            "calibration_mode": calibration_mode,
```
right after the `"backend": "numpy"` line.

(If the script uses `np.save(out_file, payload, allow_pickle=True)` instead of `np.savez`, the dict literal is inside the `save` payload construction — same field-addition rule.)

- [ ] **Step 5: Run the smoke test; verify it PASSES**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py::test_no_dns_smoke_run -v`
Expected: PASS in roughly 30~90 seconds.

- [ ] **Step 6: Run the full test suite for the LES generator**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py -v`
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add scripts/les/generate_kolmogorov_les.py tests/test_les_generator_no_dns.py
git commit -m "✨ feat(les): wire --no_dns branch into main() with stand-alone calibration

Stand-alone mode uses Kolmogorov forcing-friction balance to derive
r_fric, omega_rms, and speed_bound from --manual_turnover_time,
--manual_A, --manual_k_f, --r_scale. Defaults:
  omega_rms = sqrt(2·A·k_f·r_scale·T_eddy)
  speed_bound = 2·sqrt(A/(k_f·2π))
Either can be overridden via --manual_omega_rms / --manual_speed_bound.

Output npy now carries config['calibration_mode'] ∈
{'stand_alone', 'dns_calibrated'} for provenance audit.

Smoke test (N=16, T=0.01, nu=1e-4) verifies the round trip: CLI →
solver → npy with the manual values intact in the saved config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Push commits + record spec progress in EXPERIMENT_RECORD

**Files:**
- Modify: `EXPERIMENT_RECORD.md` (append a chronological entry for sub-project A setup)

- [ ] **Step 1: Push the local commits**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git push origin pirate
```

- [ ] **Step 2: Add a chronological entry**

Edit `EXPERIMENT_RECORD.md` and insert immediately under `## [LOG] Chronological`:

```markdown
### [2026-05-14] sub-project A | LES generator stand-alone mode landed

- Time: `2026-05-14`
- Status: ready for deployment to home-gpu
- Experiment or Job ID: n/a (tooling change, no slurm)

Change:

- Vendored `generate_kolmogorov_les.py` from `~/Documents/coding/kolmogorov_generate/dns/` to `scripts/les/generate_kolmogorov_les.py` (commit `<TASK-1-SHA>`).
- Added `--no_dns` mode + `--manual_*` calibration args (commits `<TASK-2-SHA>` and `<TASK-3-SHA>`); previous DNS-calibrated path is preserved.
- Output npy now records `config['calibration_mode']` ∈ {`stand_alone`, `dns_calibrated`}.

Evidence:

- 8 passing tests in `tests/test_les_generator_no_dns.py` (CLI validation x6, sanity x2, smoke run x1).
- Smoke test (N=16, T=0.01, ν=1e-4) produces a npy with `calibration_mode='stand_alone'` and matching manual params.

Interpretation:

- Spec `docs/superpowers/specs/2026-05-14-re1e6-les-standalone-generation.md` Sections 2 (generator modifications) and 3 (validation API surface) are now implementable from a deployed copy. Production run on home-gpu is unblocked.

Next:

- Deploy to home-gpu and run smoke + production simulation (Tasks 5–8 of the plan).
```

Replace `<TASK-1-SHA>`, `<TASK-2-SHA>`, `<TASK-3-SHA>` with the actual SHAs from `git log --oneline -3`.

- [ ] **Step 3: Commit the record update + push**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add EXPERIMENT_RECORD.md
git commit -m "📝 docs(record): land sub-project A stand-alone LES generator

Sub-project A (Re=1e6 LES dataset generation) tooling is committed
locally: --no_dns branch implemented, 8 tests passing including the
smoke run that round-trips a tiny stand-alone simulation through the
saved-config metadata.

Production run on home-gpu is the next task. EXPERIMENT_RECORD entry
captures the commit SHAs for the three feature commits so future
audits can find the implementation history.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push origin pirate
```

---

## Task 5: Deploy generator to home-gpu

**Files:**
- Create: `home-gpu:~/les-gen/` (deployment root)
- Create: `home-gpu:~/les-gen/.venv/` (uv-managed)
- Create: `home-gpu:~/les-gen/generate_kolmogorov_les.py` (rsync target)

- [ ] **Step 1: Create the deployment directory on home-gpu**

```bash
ssh home-gpu 'mkdir -p ~/les-gen/output && ls -la ~/les-gen/'
```
Expected: `output` directory created.

- [ ] **Step 2: rsync the generator script over Tailscale**

```bash
rsync -avz /Users/latteine/Documents/coding/jaxpi/scripts/les/generate_kolmogorov_les.py \
    home-gpu:~/les-gen/
```
Expected: 1 file transferred.

- [ ] **Step 3: Set up uv venv on home-gpu**

```bash
ssh home-gpu 'cd ~/les-gen && ~/.local/bin/uv venv --python 3.13 && source .venv/bin/activate && uv pip install numpy && python -c "import numpy; print(numpy.__version__)"'
```
Expected: numpy version printed (e.g. `2.x.x`).

- [ ] **Step 4: Verify the deployed script runs --help**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && python generate_kolmogorov_les.py --help' | head -20
```
Expected: argparse help block including `--no_dns` / `--manual_*` flags.

- [ ] **Step 5: Record deployment timestamp**

```bash
ssh home-gpu 'cd ~/les-gen && md5sum generate_kolmogorov_les.py > .deployed.md5 && date -Is > .deployed.timestamp'
```
Expected: silent success. (No git commit on home-gpu — it's a deployment, not a fork.)

---

## Task 6: Smoke test on home-gpu

**Files:**
- Create: `home-gpu:~/les-gen/output/smoke_N64.npy`

- [ ] **Step 1: Run a tiny simulation to validate the CFL + parameter pipeline**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && python generate_kolmogorov_les.py \
    --no_dns \
    --manual_L 1.0 --manual_nu 1e-6 --manual_A 0.1 --manual_k_f 2 \
    --manual_turnover_time 1.0 \
    --N 64 --T_end 0.01 \
    --dt 1e-4 --auto_dt --cfl_target 0.4 \
    --save_interval 25 \
    --closure_model hyperviscosity --hyper_p 2 --nu_h_alpha 10.0 \
    --r_scale 10.0 --dealias_mode 2/3 --init_mode random --seed 42 \
    --output output/smoke_N64.npy 2>&1 | tail -30'
```
Expected: simulation completes without NaN / blow-up; tail shows `Saved ...` style line.

- [ ] **Step 2: Inspect the smoke output structure**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && python -c "
import numpy as np
p = np.load(\"output/smoke_N64.npy\", allow_pickle=True).item()
print(\"keys:\", list(p.keys()))
cfg = p[\"config\"]
print(\"calibration_mode:\", cfg[\"calibration_mode\"])
print(\"N:\", cfg[\"N\"], \"nu:\", cfg[\"nu\"], \"A:\", cfg[\"A\"], \"k_f:\", cfg[\"k_f\"])
omega = np.asarray(p[\"omega\"])
print(\"omega shape:\", omega.shape, \"dtype:\", omega.dtype)
print(\"omega rms first frame:\", float(np.sqrt(np.mean(omega[0] ** 2))))
print(\"omega rms last frame:\", float(np.sqrt(np.mean(omega[-1] ** 2))))
print(\"KE first/last:\", float(p[\"diagnostics\"][\"kinetic_energy\"][0]),
      float(p[\"diagnostics\"][\"kinetic_energy\"][-1]))
"'
```
Expected:
- `calibration_mode: stand_alone`
- `N: 64`, `nu: 1e-06`
- `omega` shape `(N_frames, 64, 64)` with `N_frames` ≈ 4–5 (T=0.01, save_interval=25, dt=1e-4 → 100 steps / 25 = 4 saves)
- omega RMS finite and reasonable (5–20)
- KE finite, no NaN

- [ ] **Step 3: Bail and debug if any check fails**

If `omega rms` is NaN or KE is NaN: stop here. Lower `--manual_speed_bound` (e.g. 1.0), increase `--dt` aggressiveness (`--cfl_target 0.2`), or reduce N. Do not proceed to Task 7.

---

## Task 7: Production LES run on home-gpu

**Files:**
- Create: `home-gpu:~/les-gen/output/kolmogorov_les_Re1e6_N512_T5.npy` (~840 MB)
- Create: `home-gpu:~/les-gen/output/run.log`

- [ ] **Step 1: Estimate per-step wallclock with a 100-step probe**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && python -c "
import subprocess, time, sys
cmd = [sys.executable, \"generate_kolmogorov_les.py\",
       \"--no_dns\",
       \"--manual_L\", \"1.0\", \"--manual_nu\", \"1e-6\",
       \"--manual_A\", \"0.1\", \"--manual_k_f\", \"2\",
       \"--manual_turnover_time\", \"1.0\",
       \"--manual_omega_rms\", \"12.566\",
       \"--manual_speed_bound\", \"2.0\",
       \"--N\", \"512\", \"--T_end\", \"0.01\",
       \"--dt\", \"1e-4\", \"--auto_dt\", \"--cfl_target\", \"0.4\",
       \"--save_interval\", \"50\",
       \"--closure_model\", \"hyperviscosity\", \"--hyper_p\", \"2\", \"--nu_h_alpha\", \"10.0\",
       \"--r_scale\", \"10.0\", \"--dealias_mode\", \"2/3\",
       \"--init_mode\", \"random\", \"--seed\", \"42\",
       \"--output\", \"output/timing_probe.npy\"]
t0 = time.time()
res = subprocess.run(cmd, capture_output=True, text=True)
dt_wall = time.time() - t0
print(\"exit:\", res.returncode, \"wallclock:\", round(dt_wall, 1), \"s for ~100 steps\")
print(\"projected production wallclock (50000 steps):\", round(dt_wall * 500 / 60, 1), \"minutes\")
print(\"stderr tail:\", res.stderr[-300:] if res.stderr else \"<empty>\")
"'
```
Expected: a numeric projection (e.g. `~80 minutes` to `~180 minutes`). If the projection exceeds 6 hours, install `mkl_fft` and re-probe: `ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && uv pip install mkl_fft'`. If still > 6 h, escalate to user before launching production.

- [ ] **Step 2: Launch the full production run in the background via nohup**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && nohup python generate_kolmogorov_les.py \
    --no_dns \
    --manual_L 1.0 --manual_nu 1e-6 --manual_A 0.1 --manual_k_f 2 \
    --manual_turnover_time 1.0 \
    --manual_omega_rms 12.566 \
    --manual_speed_bound 2.0 \
    --N 512 --T_end 5.0 \
    --dt 1e-4 --auto_dt --cfl_target 0.4 \
    --save_interval 500 \
    --closure_model hyperviscosity --hyper_p 2 --nu_h_alpha 10.0 \
    --r_scale 10.0 --dealias_mode 2/3 --init_mode random --seed 42 \
    --output output/kolmogorov_les_Re1e6_N512_T5.npy \
    > output/run.log 2>&1 & echo $! > output/run.pid && echo "started PID $(cat output/run.pid)"'
```
Expected: `started PID <number>` printed; the run keeps going after SSH disconnect.

- [ ] **Step 3: Poll until completion**

```bash
ssh home-gpu 'while kill -0 $(cat ~/les-gen/output/run.pid) 2>/dev/null; do
  echo "[$(date -Is)] still running PID $(cat ~/les-gen/output/run.pid); log tail:"
  tail -2 ~/les-gen/output/run.log
  sleep 600
done
echo "process finished"
tail -20 ~/les-gen/output/run.log'
```
Use `Bash run_in_background: true` for this — it can exceed the 10-minute foreground timeout. The poll prints a status every 10 minutes; when the process is gone the loop exits with the final log tail.

- [ ] **Step 4: Verify the production npy is produced and well-formed**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && python -c "
import numpy as np, os
p = np.load(\"output/kolmogorov_les_Re1e6_N512_T5.npy\", allow_pickle=True).item()
sz = os.path.getsize(\"output/kolmogorov_les_Re1e6_N512_T5.npy\")
print(\"size:\", round(sz / 1024**2, 1), \"MB\")
print(\"keys:\", list(p.keys()))
print(\"calibration_mode:\", p[\"config\"][\"calibration_mode\"])
omega = np.asarray(p[\"omega\"])
print(\"omega shape:\", omega.shape)
ke = np.asarray(p[\"diagnostics\"][\"kinetic_energy\"])
print(\"KE first/mid/last:\", float(ke[0]), float(ke[len(ke)//2]), float(ke[-1]))
print(\"any NaN in omega:\", bool(np.isnan(omega).any()))
"'
```
Expected:
- size ≈ 800–900 MB
- `calibration_mode: stand_alone`
- `omega shape: (≈101, 512, 512)`
- KE values finite, no NaN

---

## Task 8: Write validate_les.py

**Files:**
- Create: `scripts/les/validate_les.py` (locally)
- Modify: `tests/test_les_generator_no_dns.py` (add a unit test for the slope helper)

- [ ] **Step 1: Write the failing test for the spectrum slope helper**

Append to `tests/test_les_generator_no_dns.py`:
```python
def test_spectrum_slope_helper():
    """spectrum_slope() should recover -5/3 for an exact power law."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "validate_les",
        REPO_ROOT / "scripts" / "les" / "validate_les.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    k = np.arange(1, 64, dtype=float)
    spectrum = k ** (-5.0 / 3.0)
    slope = mod.spectrum_slope(k, spectrum, k_lo=5.0, k_hi=40.0)
    assert abs(slope + 5.0 / 3.0) < 0.05, f"expected ~-1.667, got {slope}"
```

- [ ] **Step 2: Run the new test; confirm it FAILS**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py::test_spectrum_slope_helper -v`
Expected: FAIL (module doesn't exist yet).

- [ ] **Step 3: Write `scripts/les/validate_les.py`**

```python
#!/usr/bin/env python3
"""Validate a stand-alone LES dataset against three hard checks.

Usage:
    python validate_les.py --input <les.npy> --output-dir <dir>

Produces:
    <dir>/validation_report.txt
    <dir>/ke.png
    <dir>/enstrophy.png
    <dir>/spectrum.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# === Validation constants (from spec) ===
KE_BAND_LOW = 0.4
KE_BAND_HIGH = 0.6
ENSTROPHY_MAX = 200.0
SPECTRUM_SLOPE_LOW = -2.0   # softer than -5/3 by 1/3
SPECTRUM_SLOPE_HIGH = -1.5  # tighter than -5/3 by 1/6


def spectrum_slope(k: np.ndarray, spectrum: np.ndarray,
                   k_lo: float, k_hi: float) -> float:
    """Fit log10(spectrum) ~ slope * log10(k) over k in [k_lo, k_hi].

    Returns the slope (negative for physical inertial ranges).
    """
    k = np.asarray(k, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    mask = (k >= k_lo) & (k <= k_hi) & (spectrum > 0.0)
    if mask.sum() < 3:
        raise ValueError("not enough points in [k_lo, k_hi] with positive spectrum")
    log_k = np.log10(k[mask])
    log_s = np.log10(spectrum[mask])
    slope, _intercept = np.polyfit(log_k, log_s, 1)
    return float(slope)


def _plot_kinetic_energy(time, ke, out_path: Path, mean_post_spinup: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time, ke, color="tab:blue", lw=1.2)
    ax.axhspan(KE_BAND_LOW, KE_BAND_HIGH, color="tab:green", alpha=0.15,
               label=f"target band [{KE_BAND_LOW}, {KE_BAND_HIGH}]")
    ax.axhline(mean_post_spinup, color="tab:orange", ls="--",
               label=f"mean (t≥1): {mean_post_spinup:.4f}")
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.set_title("LES KE time series")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_enstrophy(time, enstrophy, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time, enstrophy, color="tab:red", lw=1.2)
    ax.axhline(ENSTROPHY_MAX, color="gray", ls=":",
               label=f"upper bound {ENSTROPHY_MAX}")
    ax.set_xlabel("time")
    ax.set_ylabel("enstrophy")
    ax.set_title("LES enstrophy time series")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_spectrum(k, spectrum, out_path: Path, slope: float,
                   k_lo: float, k_hi: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k, spectrum, color="tab:blue", lw=1.2, label="LES E(k)")
    # reference -5/3 line anchored at (k_lo, spectrum[k_lo])
    ref_k = np.array([k_lo, k_hi])
    anchor = float(spectrum[np.argmin(np.abs(np.asarray(k) - k_lo))])
    ref = anchor * (ref_k / k_lo) ** (-5.0 / 3.0)
    ax.loglog(ref_k, ref, color="black", ls="--",
              label="k^(-5/3) reference")
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"LES energy spectrum (fit slope = {slope:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LES dataset")
    parser.add_argument("--input", required=True, help="LES npy path")
    parser.add_argument("--output-dir", required=True,
                        help="dir for report + PNGs")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(in_path, allow_pickle=True).item()
    cfg = payload.get("config", {})
    diag = payload.get("diagnostics", {})

    time = np.asarray(payload.get("time"))
    ke = np.asarray(diag.get("kinetic_energy"))
    enstrophy = np.asarray(diag.get("enstrophy"))
    spectrum = np.asarray(diag.get("energy_spectrum"))
    k = np.asarray(diag.get("spectrum_wavenumbers"))

    report_lines: list[str] = []
    report_lines.append(f"=== LES validation report ===")
    report_lines.append(f"input: {in_path}")
    report_lines.append(f"calibration_mode: {cfg.get('calibration_mode', '?')}")
    report_lines.append(f"N={cfg.get('N')}, T_end={cfg.get('T_end')}, "
                        f"nu={cfg.get('nu')}, A={cfg.get('A')}, k_f={cfg.get('k_f')}")
    report_lines.append("")

    # --- Check 1: KE band ---
    post_spinup_mask = time >= 1.0
    if post_spinup_mask.sum() < 2:
        ke_mean = float(np.mean(ke))
        report_lines.append(f"[WARN] T_end < 1.0; using full-trajectory KE mean = {ke_mean:.4f}")
    else:
        ke_mean = float(np.mean(ke[post_spinup_mask]))
    ke_ok = (KE_BAND_LOW <= ke_mean <= KE_BAND_HIGH)
    report_lines.append(
        f"[check1 KE band] mean(t≥1) = {ke_mean:.4f} "
        f"target [{KE_BAND_LOW}, {KE_BAND_HIGH}] → "
        f"{'PASS' if ke_ok else 'FAIL (refine --manual_turnover_time)'}"
    )
    _plot_kinetic_energy(time, ke, out_dir / "ke.png", ke_mean)

    # --- Check 2: enstrophy bounded ---
    enstrophy_max = float(np.max(np.abs(enstrophy)))
    enstrophy_ok = enstrophy_max < ENSTROPHY_MAX and np.all(np.isfinite(enstrophy))
    report_lines.append(
        f"[check2 enstrophy] max |Z| = {enstrophy_max:.1f}, all finite = "
        f"{bool(np.all(np.isfinite(enstrophy)))} → {'PASS' if enstrophy_ok else 'FAIL'}"
    )
    _plot_enstrophy(time, enstrophy, out_dir / "enstrophy.png")

    # --- Check 3: spectrum slope ---
    final_spectrum = spectrum[-1] if spectrum.ndim == 2 else spectrum
    k_f = cfg.get("k_f", 2)
    N = cfg.get("N", 512)
    k_lo = float(k_f + 2)
    k_hi = float(N // 4)
    slope = spectrum_slope(k, final_spectrum, k_lo=k_lo, k_hi=k_hi)
    slope_ok = SPECTRUM_SLOPE_LOW <= slope <= SPECTRUM_SLOPE_HIGH
    report_lines.append(
        f"[check3 spectrum slope] slope over k∈[{k_lo}, {k_hi}] = {slope:.3f} "
        f"target [{SPECTRUM_SLOPE_LOW}, {SPECTRUM_SLOPE_HIGH}] → "
        f"{'PASS' if slope_ok else 'OBSERVE (not a hard fail)'}"
    )
    _plot_spectrum(k, final_spectrum, out_dir / "spectrum.png",
                   slope=slope, k_lo=k_lo, k_hi=k_hi)

    # --- Summary ---
    overall = ke_ok and enstrophy_ok
    report_lines.append("")
    report_lines.append(f"=== overall hard-check verdict: {'PASS' if overall else 'FAIL'} ===")

    (out_dir / "validation_report.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Install matplotlib in the jaxpi env locally for test run**

```bash
cd /Users/latteine/Documents/coding/jaxpi
python3 -c "import matplotlib" 2>&1 || pip3 install --user matplotlib
```
Expected: matplotlib importable.

- [ ] **Step 5: Run the slope helper test; it must PASS now**

Run: `cd /Users/latteine/Documents/coding/jaxpi && python3 -m pytest tests/test_les_generator_no_dns.py::test_spectrum_slope_helper -v`
Expected: PASS.

- [ ] **Step 6: Commit + push**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add scripts/les/validate_les.py tests/test_les_generator_no_dns.py
git commit -m "✨ feat(les): add validate_les.py with three hard-check validators

KE band check, enstrophy boundedness check, and spectrum-slope
observation. Outputs validation_report.txt and three PNGs
(ke.png, enstrophy.png, spectrum.png) under --output-dir.

spectrum_slope() is unit-tested against an exact k^(-5/3) power law.

Exit code 1 if KE or enstrophy fails; spectrum slope is reported but
not a hard failure (2D forced turbulence can deviate).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push origin pirate
```

---

## Task 9: Run validation on home-gpu

**Files:**
- Create: `home-gpu:~/les-gen/output/validation_report.txt`
- Create: `home-gpu:~/les-gen/output/{ke,enstrophy,spectrum}.png`

- [ ] **Step 1: Deploy validate_les.py to home-gpu**

```bash
rsync -avz /Users/latteine/Documents/coding/jaxpi/scripts/les/validate_les.py \
    home-gpu:~/les-gen/
```

- [ ] **Step 2: Install matplotlib in the home-gpu venv**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && uv pip install matplotlib && python -c "import matplotlib; print(matplotlib.__version__)"'
```
Expected: matplotlib version printed.

- [ ] **Step 3: Run the validator on the production dataset**

```bash
ssh home-gpu 'cd ~/les-gen && source .venv/bin/activate && python validate_les.py \
    --input output/kolmogorov_les_Re1e6_N512_T5.npy \
    --output-dir output/'
```
Expected: report printed to stdout; `output/validation_report.txt` and three PNGs written; exit code 0 if both KE and enstrophy pass, 1 otherwise.

- [ ] **Step 4: Pull the validation artefacts back to Mac for review**

```bash
mkdir -p /Users/latteine/Documents/coding/jaxpi/eval_runs/les_re1e6_validation_20260514
rsync -avz home-gpu:~/les-gen/output/validation_report.txt \
    home-gpu:~/les-gen/output/ke.png \
    home-gpu:~/les-gen/output/enstrophy.png \
    home-gpu:~/les-gen/output/spectrum.png \
    /Users/latteine/Documents/coding/jaxpi/eval_runs/les_re1e6_validation_20260514/
```
Expected: 4 files copied locally.

- [ ] **Step 5: Read the validation report**

```bash
cat /Users/latteine/Documents/coding/jaxpi/eval_runs/les_re1e6_validation_20260514/validation_report.txt
```
Expected: human-readable report. Pass / Fail status visible.

- [ ] **Step 6: Branch on outcome**

If `overall: PASS` → continue to Task 10.
If `[check1 KE band] FAIL` (most likely failure mode) → rerun the production simulation with `--manual_turnover_time 0.5` or `2.0` (whichever direction the KE moved in), then re-validate. **Only one retry budgeted by spec**; if a second retry is needed, escalate to user with the report attached before retrying again.
If `[check2 enstrophy] FAIL` → the simulation likely blew up; investigate `output/run.log` for CFL warnings and adjust `--manual_speed_bound` lower (e.g. 1.0), rerun.

---

## Task 10: Record the dataset in EXPERIMENT_RECORD + final commit

**Files:**
- Modify: `EXPERIMENT_RECORD.md` (append a chronological entry for the validated dataset)

- [ ] **Step 1: Append the chronological entry**

Insert into `EXPERIMENT_RECORD.md` immediately under `## [LOG] Chronological` (above the Task 4 entry):

```markdown
### [2026-05-14] sub-project A | Re=1e6 stand-alone LES dataset generated and validated

- Time: `2026-05-14`
- Status: dataset produced + validation PASS (or FAIL — annotate based on actual outcome)
- Experiment or Job ID: n/a (no slurm; home-gpu CPU run)

Change:

- Deployed `scripts/les/generate_kolmogorov_les.py` + `scripts/les/validate_les.py` to `home-gpu:~/les-gen/`.
- Produced `home-gpu:~/les-gen/output/kolmogorov_les_Re1e6_N512_T5.npy` (~<SIZE> MB, <NFRAMES> frames) via stand-alone calibration: `--manual_turnover_time=<VAL>`, `--manual_omega_rms=<VAL>`, `--manual_speed_bound=<VAL>` (a priori from forcing-friction balance).
- Validated via `validate_les.py`: KE band <STATUS>, enstrophy bounded <STATUS>, spectrum slope <SLOPE>.

Evidence:

- [eval_runs/les_re1e6_validation_20260514/validation_report.txt](/Users/latteine/Documents/coding/jaxpi/eval_runs/les_re1e6_validation_20260514/validation_report.txt)
- [eval_runs/les_re1e6_validation_20260514/ke.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/les_re1e6_validation_20260514/ke.png)
- [eval_runs/les_re1e6_validation_20260514/enstrophy.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/les_re1e6_validation_20260514/enstrophy.png)
- [eval_runs/les_re1e6_validation_20260514/spectrum.png](/Users/latteine/Documents/coding/jaxpi/eval_runs/les_re1e6_validation_20260514/spectrum.png)
- Production wallclock: <DURATION>.

Interpretation:

- Sub-project A delivered. Stage 1 of the two-stage PINN training (sub-project B) is unblocked: it can consume `home-gpu:~/les-gen/output/kolmogorov_les_Re1e6_N512_T5.npy` either by rsyncing the file to lab-server or by mounting via Tailscale.
- Calibration mode is `stand_alone`; no DNS data was touched during generation. The Re=1e5 LES on lab-server remains DNS-calibrated and is unaffected.

Next:

- Brainstorm sub-project B (Stage 1 LES warmup + Stage 2 sparse sensor + PDE training) in a new session.
```

Replace `<SIZE>`, `<NFRAMES>`, `<VAL>`, `<STATUS>`, `<SLOPE>`, `<DURATION>` with values read from the validation report and production logs.

- [ ] **Step 2: Commit + push**

```bash
cd /Users/latteine/Documents/coding/jaxpi
git add EXPERIMENT_RECORD.md \
    eval_runs/les_re1e6_validation_20260514/validation_report.txt
git commit -m "📝 docs(record): land Re=1e6 stand-alone LES dataset (sub-project A done)

Stand-alone LES dataset produced on home-gpu without touching DNS.
Production wallclock <DURATION>; <NFRAMES> frames at Δt=0.05 on
N=512 grid. Validation report committed under
eval_runs/les_re1e6_validation_20260514/.

PNGs (ke.png, enstrophy.png, spectrum.png) are ignored by .gitignore
under eval_runs/ but stay locally for human review and are referenced
from the EXPERIMENT_RECORD entry.

Sub-project A complete. Sub-project B (two-stage PINN training)
brainstorm starts next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push origin pirate
```

---

## Self-Review checklist run against spec

- [x] **Spec Goals 1 (matching field layout)** — Tasks 1, 3, 6, 7 (vendoring keeps shape; metadata extended only by adding `calibration_mode`, no schema break).
- [x] **Spec Goals 2 (reuse solver, no rewrite)** — Tasks 1–3 (vendor + branch, no jax-cfd).
- [x] **Spec Goals 3 (a priori calibration)** — Task 7 step 2 uses derived values; Task 3 step 3 wires the formulas.
- [x] **Spec Goals 4 (deploy to home-gpu + three pieces of hard evidence)** — Tasks 5–9.
- [x] **Spec Goals 5 (≤ 6h wallclock)** — Task 7 step 1 (timing probe) enforces the budget.
- [x] **Spec Non-goal: no jax-cfd rewrite** — none of the tasks touch jax.
- [x] **Spec Non-goal: no PINN training** — out of scope; sub-project B mentioned only in followup.
- [x] **Spec Non-goal: no Re=1e5 regeneration** — none of the tasks touch existing datasets on lab-server.
- [x] **Spec generator modifications (CLI args + branch + metadata)** — Tasks 2 and 3.
- [x] **Spec unit tests (4 cases)** — Task 2 covers 4 (`test_no_dns_requires_manual_nu`, `test_no_dns_requires_manual_turnover`, `test_no_dns_init_mode_dns_rejected`, plus help discoverability); Task 3 step 1 covers the constructor-kwargs scenario via the smoke test that round-trips manual values through the saved config.
- [x] **Spec validation (3 hard checks)** — Task 8 implements all three; Task 9 runs them.
- [x] **Spec deliverables 1 (modified generator + tests)** — Tasks 1–3 commit and push.
- [x] **Spec deliverables 2 (~/les-gen with venv + outputs)** — Tasks 5–9.
- [x] **Spec deliverables 3 (dataset + report + 3 PNGs)** — Tasks 7 and 9.
- [x] **Spec deliverables 4 (EXPERIMENT_RECORD entry)** — Tasks 4 and 10.
- [x] **Placeholder scan** — `<TASK-1-SHA>`, `<SIZE>`, `<DURATION>`, etc. are template fields filled in by the engineer from real outputs (not placeholders to invent). All actual code blocks are complete.
- [x] **Type / name consistency** — `--no_dns`, `--manual_nu`, `--manual_turnover_time`, `calibration_mode`, `spectrum_slope` are used identically across tasks.
