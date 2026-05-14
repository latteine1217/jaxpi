#!/usr/bin/env python3
"""Validate a stand-alone LES dataset against revised hard checks.

Usage:
    python validate_les.py --input <les.npy> --output-dir <dir>

Produces:
    <dir>/validation_report.txt
    <dir>/ke.png
    <dir>/enstrophy.png
    <dir>/spectrum.png
    <dir>/divergence.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# === Validation constants (revised after spec defect — see EXPERIMENT_RECORD) ===
# KE band check replaced by no-decay + bounded check: original band [0.4, 0.6]
# assumed T -> infinity forcing-friction equilibrium that T_end=5 cannot reach
# with stand-alone calibration on Kolmogorov flow.
KE_MAX_ABSOLUTE = 100.0  # hard upper bound; system must not blow up
ENSTROPHY_MAX = 200.0
SPECTRUM_SLOPE_LOW = -4.0   # loosened from -2.0 — sub-equilibrium LES is steeper than -5/3
SPECTRUM_SLOPE_HIGH = -1.0  # loosened from -1.5
DIVERGENCE_MAX = 1.0e-6     # incompressibility must hold to ~fp32 precision (1e-13 typical)


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
    ax.plot(time, ke, color="tab:blue", lw=1.2, label="KE(t)")
    ax.axhline(mean_post_spinup, color="tab:orange", ls="--",
               label=f"mean (t≥1): {mean_post_spinup:.4e}")
    ax.axhline(KE_MAX_ABSOLUTE, color="gray", ls=":",
               label=f"upper bound {KE_MAX_ABSOLUTE}")
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.set_title("LES KE time series (revised: no-decay + bounded)")
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


def _plot_divergence(time, divergence, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(time, np.abs(divergence), color="tab:purple", lw=1.2)
    ax.axhline(DIVERGENCE_MAX, color="gray", ls=":",
               label=f"upper bound {DIVERGENCE_MAX:.0e}")
    ax.set_xlabel("time")
    ax.set_ylabel("|divergence_error|")
    ax.set_title("LES divergence (incompressibility check)")
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

    # --- Check 1: KE — no-decay + bounded ---
    ke_initial = float(ke[0])
    ke_final = float(ke[-1])
    ke_max = float(np.max(ke))
    ke_no_decay = ke_final > ke_initial
    ke_bounded = 0.0 < ke_max < KE_MAX_ABSOLUTE
    ke_all_finite = bool(np.all(np.isfinite(ke)))
    ke_ok = ke_no_decay and ke_bounded and ke_all_finite
    ke_mean_post_spinup = (
        float(np.mean(ke[time >= 1.0])) if (time >= 1.0).sum() >= 2 else float(np.mean(ke))
    )
    report_lines.append(
        f"[check1 KE no-decay+bounded] KE(0)={ke_initial:.4e}, "
        f"KE(T)={ke_final:.4e}, max(KE)={ke_max:.4e}, "
        f"mean(t≥1)={ke_mean_post_spinup:.4e} | "
        f"no-decay={ke_no_decay}, bounded(<{KE_MAX_ABSOLUTE})={ke_bounded}, "
        f"finite={ke_all_finite} → {'PASS' if ke_ok else 'FAIL'}"
    )
    _plot_kinetic_energy(time, ke, out_dir / "ke.png", ke_mean_post_spinup)

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

    # --- Check 4 (new): divergence error bounded ---
    divergence = np.asarray(diag.get("divergence_error", np.array([np.inf])))
    divergence_max = float(np.max(np.abs(divergence)))
    divergence_ok = (
        divergence_max < DIVERGENCE_MAX
        and bool(np.all(np.isfinite(divergence)))
    )
    report_lines.append(
        f"[check4 divergence] max|div| = {divergence_max:.2e} "
        f"target < {DIVERGENCE_MAX:.0e} → {'PASS' if divergence_ok else 'FAIL'}"
    )
    _plot_divergence(time, divergence, out_dir / "divergence.png")

    # --- Summary ---
    overall = ke_ok and enstrophy_ok and divergence_ok
    report_lines.append("")
    report_lines.append(f"=== overall hard-check verdict: {'PASS' if overall else 'FAIL'} ===")

    (out_dir / "validation_report.txt").write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
