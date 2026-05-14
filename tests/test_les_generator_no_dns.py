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
