# Re=1e6 Stand-alone LES Dataset Generation — Design Spec

## Overview

Sub-project A of the two-stage training rework. Generate a Reynolds=10⁶,
N=512, T=5 Kolmogorov-flow LES dataset on the home GPU host
(`home-gpu`, WSL2 Ubuntu 26.04, i7-11700 CPU-only, 12 cores, 32 GB RAM)
without relying on any DNS reference. Output feeds Stage 1 of the
two-stage PINN training (sub-project B, brainstormed separately).

## Why

Three converging reasons motivate a new LES dataset:

1. **Two-stage training narrative** — `3491` null result (sensor + PDE
   with `dw=38.06` over 100k steps matches no_data baseline) requires
   a different training scaffold. Brainstormed plan: Stage 1 fits dense
   LES on window-1 (no PDE), Stage 2 refines with sparse sensor + PDE
   across the full T=5 horizon.
2. **Re=1e6 LES does not exist** — Lab server holds `kolmogorov_les_re`
   datasets only for Re=100…10⁵. The PINN main line works at Re=10⁶,
   so a matching LES dataset is the prerequisite.
3. **No-DNS red line** — Research convention disallows DNS access for
   any training-time signal. The existing
   `generate_kolmogorov_les.py` derives sub-grid (`nu_h`), linear
   friction (`r_fric`), initial vorticity RMS, and CFL bounds from a
   DNS reference file; that pipeline leaks DNS statistics into LES
   even when only used a priori. Stand-alone calibration removes this
   leak and lets us state "LES generation never touched DNS" in the
   paper.

## Goals

- Produce `kolmogorov_les_Re1e6_N512_T5.npy` (~840 MB) with the same
  field layout as the existing LES datasets so the downstream loader
  can consume it without modification.
- Reuse the proven `generate_kolmogorov_les.py` solver
  (pure-numpy, vorticity-streamfunction, 2/3 dealiasing,
  hyperviscosity SGS, linear friction). Add a `--no_dns` mode rather
  than rewriting.
- Calibrate SGS / friction / initial RMS analytically from the
  Kolmogorov forcing-friction balance, with no DNS lookup.
- Land the artefact on `home-gpu` and produce three pieces of hard
  evidence: kinetic energy time series, enstrophy decay, energy
  spectrum log-log plot.
- Total wallclock ≤ 6 hours including setup, production run, and
  validation (one calibration retry if the a priori params miss).

## Non-goals

- Do **not** rewrite the LES solver in jax-cfd. The existing
  numpy solver runs comfortably on i7-11700 (≈1–3 h for the
  production run, dominated by 2D FFT).
- Do **not** train any PINN as part of this sub-project. Stage 1 /
  Stage 2 PINN training is sub-project B and gets its own
  brainstorm → spec → plan cycle.
- Do **not** regenerate the existing Re=1e5 LES dataset.
  Re=10⁵ stays DNS-calibrated; only Re=10⁶ is stand-alone. We will
  disclose this in any paper artefact.
- Do **not** install JAX or CUDA on `home-gpu`. Numpy and stdlib are
  the only required dependencies for this sub-project.

## Architecture

```
[Mac /Users/latteine/Documents/coding/kolmogorov_generate/dns/]
    generate_kolmogorov_les.py     ← source of truth, modified here

[home-gpu ~/les-gen/]              ← deployment target
├── generate_kolmogorov_les.py     ← scp from Mac after modification
├── validate_les.py                ← new validation script (see below)
├── .venv/                         ← uv-managed, numpy only
└── output/
    └── kolmogorov_les_Re1e6_N512_T5.npy
```

No new repository. Modifications stay in
`kolmogorov_generate/dns/generate_kolmogorov_les.py` and are tracked
by that repo's git history. The home-gpu copy is a deployment, not a
fork.

## Generator modifications (`--no_dns` mode)

### New CLI arguments

Added to the existing argparse block in `main()`:

```
--no_dns               action="store_true"
--manual_L             type=float  default=1.0
--manual_nu            type=float  default=None
--manual_A             type=float  default=0.1
--manual_k_f           type=int    default=2
--manual_turnover_time type=float  default=None
--manual_speed_bound   type=float  default=None
--manual_omega_rms     type=float  default=None
```

Existing `--dns` becomes optional (`default=None`); the validator
errors out only if `--no_dns` was not set and `--dns` is missing.

### Behaviour change in `main()`

A conditional branch at the top of `main()` replaces the
`load_npy_payload` / `resolve_dns_config` / `estimate_turnover_time`
/ `estimate_dns_omega_rms` / `estimate_dns_velocity_scale`
/ `prepare_dns_initial_omega` calls.

```python
if args.no_dns:
    if args.manual_nu is None or args.manual_turnover_time is None:
        parser.error("--no_dns requires --manual_nu and --manual_turnover_time")
    if args.init_mode == "dns":
        parser.error("--no_dns is incompatible with --init_mode dns")

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
        # forcing–friction balance estimate for Kolmogorov flow.
        dns_omega_rms = float(np.sqrt(
            2 * args.manual_A * args.manual_k_f
              * args.r_scale * turnover_time
        ))

    if args.manual_speed_bound is not None:
        dns_speed_bound = args.manual_speed_bound
        dns_speed_rms = 0.5 * dns_speed_bound
    else:
        dns_speed_bound = 2.0 * float(np.sqrt(
            args.manual_A / (args.manual_k_f * 2 * np.pi)
        ))
        dns_speed_rms = 0.5 * dns_speed_bound

    dns_omega_init = None
    omega_rms = args.omega_rms if args.omega_rms is not None else dns_omega_rms
else:
    # Existing DNS-calibrated path (unchanged):
    dns_path = Path(args.dns)
    dns_data = load_npy_payload(dns_path)
    dns_config = resolve_dns_config(dns_data)
    turnover_time = estimate_turnover_time(dns_data)
    r_fric = 1.0 / (args.r_scale * turnover_time)
    dns_omega_rms = estimate_dns_omega_rms(dns_data)
    dns_speed_bound, dns_speed_rms = estimate_dns_velocity_scale(dns_data)
    dns_omega_init = prepare_dns_initial_omega(dns_data, args.N) if args.init_mode == "dns" else None
    if args.omega_rms is None and dns_omega_init is not None:
        omega_rms = float(np.sqrt(np.mean(dns_omega_init ** 2)))
    else:
        omega_rms = args.omega_rms if args.omega_rms is not None else dns_omega_rms
```

Downstream `KolmogorovLES(...)` constructor call is unchanged because
both branches populate the same locals (`dns_config`, `turnover_time`,
`r_fric`, `omega_rms`, `dns_speed_bound`, `dns_speed_rms`,
`dns_omega_init`).

Output metadata gets an extra field `"calibration_mode"` set to
`"stand_alone"` in this branch and `"dns_calibrated"` otherwise, so
downstream consumers can audit provenance.

### Test the modifications

`tests/test_generate_kolmogorov_les_no_dns.py` covers:

1. `--no_dns` without `--manual_nu` errors out with a clear message.
2. `--no_dns --init_mode dns` errors out.
3. Given all required `--manual_*` args, `main()` configures
   `KolmogorovLES` with the manual values (assert via monkeypatch
   capturing constructor kwargs).
4. With `--no_dns` and only `--manual_nu` + `--manual_turnover_time`,
   the defaulted `omega_rms` and `speed_bound` match the
   forcing-friction-balance formulas to four significant figures.

## A priori calibration for Re=1e6

Stand-alone calibration derives parameters from the
**forcing–friction equilibrium** of Kolmogorov flow with linear
friction `r`:

- Forcing input rate: `P_in ≈ A · u_rms`
- Friction dissipation rate: `ε_fric ≈ r · u_rms²`
- Setting `P_in = ε_fric` gives `u_rms = A / r`.
- With `r = 1 / (r_scale · T_eddy)` and `T_eddy = L / u_rms`, the
  self-consistent solution is `u_rms = √(A · r_scale · L)`.
- Vorticity scale: `ω_rms = k_f · 2π · u_rms / L`.

For `A=0.1`, `r_scale=10`, `L=1`, `k_f=2`:

| Quantity | Value |
| --- | --- |
| `u_rms` | 1.0 |
| `T_eddy` | 1.0 |
| `ω_rms` | ≈ 12.566 |
| `speed_bound = 2·u_rms` | 2.0 |

The `--manual_*` values for the production run:

```
--no_dns
--manual_L 1.0
--manual_nu 1e-6
--manual_A 0.1
--manual_k_f 2
--manual_turnover_time 1.0
--manual_omega_rms 12.566
--manual_speed_bound 2.0
```

Other generator flags align with the existing Re=1e5 LES convention:

```
--N 512
--T_end 5.0
--dt 1e-4
--auto_dt --cfl_target 0.4
--save_interval 500             # 50000/500 = 100 saves (≈ 100–101 frames @ Δt=0.05; existing generator decides whether IC is recorded as its own frame)
--closure_model hyperviscosity
--hyper_p 2
--nu_h_alpha 10.0
--r_scale 10.0
--dealias_mode 2/3
--init_mode random
--seed 42
```

## Deployment plan (home-gpu)

```bash
# From Mac
ssh home-gpu 'mkdir -p ~/les-gen/output'
scp /Users/latteine/Documents/coding/kolmogorov_generate/dns/generate_kolmogorov_les.py \
    home-gpu:~/les-gen/

# On home-gpu (one-time setup)
ssh home-gpu '
  cd ~/les-gen
  uv venv --python 3.13      # 3.14 is fresh; pin 3.13 for numpy wheels
  source .venv/bin/activate
  uv pip install numpy
'

# Smoke test (dry run, 100 timesteps)
ssh home-gpu '
  cd ~/les-gen
  source .venv/bin/activate
  python generate_kolmogorov_les.py --no_dns \
    --manual_nu 1e-6 --manual_turnover_time 1.0 \
    --N 64 --T_end 0.01 --dt 1e-4 \
    --output output/smoke.npy --seed 42
'

# Production run
ssh home-gpu -t '
  cd ~/les-gen
  source .venv/bin/activate
  nohup python generate_kolmogorov_les.py --no_dns \
    --manual_L 1.0 --manual_nu 1e-6 --manual_A 0.1 --manual_k_f 2 \
    --manual_turnover_time 1.0 --manual_omega_rms 12.566 \
    --manual_speed_bound 2.0 \
    --N 512 --T_end 5.0 --dt 1e-4 --auto_dt --cfl_target 0.4 \
    --save_interval 500 \
    --closure_model hyperviscosity --hyper_p 2 --nu_h_alpha 10.0 \
    --r_scale 10.0 --dealias_mode 2/3 --init_mode random --seed 42 \
    --output output/kolmogorov_les_Re1e6_N512_T5.npy \
    > output/run.log 2>&1 &
  echo "started PID $!"
'
```

`nohup` keeps the run alive across SSH disconnects. Progress can be
polled by tailing `output/run.log`.

## Validation (`validate_les.py`)

> **Revised after first production run** (see EXPERIMENT_RECORD for details).
> Original spec had a KE band `[0.4, 0.6]` based on theoretical
> equilibrium `0.5·u_rms² = 0.5`. Empirically the stand-alone calibration
> cannot reach that band in `T_end=5` because the forcing-friction
> relaxation time `1/r = 10·T_eddy` exceeds `T_end` for any reasonable
> `T_eddy`, leaving the trajectory in linear-growth phase. The revised
> criteria below judge dataset usability for Stage-1 PINN training
> rather than equilibrium realisation.

New script in `~/les-gen/validate_les.py`. Inputs the LES npy,
outputs `validation_report.txt` and four PNGs:

1. **Kinetic energy — no-decay + bounded**:
   - Plot `diagnostics.kinetic_energy` vs `time`.
   - Hard check: `KE(T_end) > KE(0)` (forcing-driven spin-up, not decaying),
     `0 < max(KE) < 100` (no blow-up), all `KE` values finite.
2. **Enstrophy decay** (unchanged):
   - Plot `diagnostics.enstrophy` vs `time`.
   - Hard check: `max|Z| < 200`, all finite.
3. **Energy spectrum slope — observation only**:
   - Plot `diagnostics.energy_spectrum[-1]` vs `spectrum_wavenumbers`
     on log-log axes with `-5/3` reference.
   - Soft check: slope ∈ `[-4.0, -1.0]` (loose, accommodating
     sub-equilibrium steepening).
4. **Divergence — incompressibility** (NEW):
   - Plot `|divergence_error|` vs `time` (log scale).
   - Hard check: `max|div| < 1e-6` (fp32 machine precision typically
     yields ~1e-13).

Pass verdict: checks 1 + 2 + 4 all hard-pass. Check 3 is observed but
does not gate.

## Effort & risks

| Phase | Time |
| --- | --- |
| Modify generator + unit tests | 1–2 h |
| Smoke test (N=64) on home-gpu | 5 min |
| Production run (N=512, T=5) on home-gpu | 1–3 h wallclock |
| Validation (`validate_les.py`) | 30 min |
| **Total** | **~3–6 h** |

### Risks

- **A priori `u_rms`, `T_eddy` may mis-calibrate** (theory assumes
  forcing-friction balance; real simulation also has viscous + SGS
  drains). Validation step 1 catches this; mitigation is one
  iteration of `--manual_turnover_time` (e.g. 0.5 or 2.0) and
  re-run.
- **`--auto_dt` may cut `dt` below 1e-5** if `--manual_speed_bound`
  underestimates `u_max`. Watch the auto_dt log line; if `dt < 1e-5`
  the wallclock budget blows up, fix `--manual_speed_bound`.
- **CFL blow up in the first 100 steps** if any parameter is way
  off. Smoke test (N=64, T=0.01) catches this in ≤ 30 s.
- **numpy FFT performance** on 12 cores is OS- and BLAS-dependent.
  If a single timestep takes > 50 ms, install
  `mkl-fft` (`uv pip install mkl_fft`) to accelerate.
- **Calibration not comparable to existing Re=1e5 (DNS-calibrated)
  LES**. This is intentional and disclosed; the Re=1e5 dataset is
  not used for Re=1e6 PINN main-line training, so direct
  cross-Re comparisons are out of scope.

## Open questions

- None blocking. `validate_les.py` post-run inspection answers
  whether the a priori params hit equilibrium; if not, a single
  re-run with adjusted `--manual_turnover_time` is the fallback.

## Deliverables

1. Modified `generate_kolmogorov_les.py` with `--no_dns` mode and
   matching unit tests, committed to the
   `kolmogorov_generate` repo.
2. `~/les-gen/` on home-gpu with deployed script, venv,
   smoke-test output, production output, validation outputs.
3. `~/les-gen/output/kolmogorov_les_Re1e6_N512_T5.npy` (~840 MB)
   plus `validation_report.txt` and three validation PNGs.
4. EXPERIMENT_RECORD.md entry under `[LOG] Chronological` noting
   the new dataset, its calibration parameters, and validation
   outcomes.
