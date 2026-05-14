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
