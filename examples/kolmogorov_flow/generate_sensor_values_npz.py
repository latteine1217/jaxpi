"""
Generate sensor values .npz from DNS/LES source fields.

What: Build per-sensor (u,v,omega,time) arrays aligned to an existing sensor JSON.
Why: Stage A/B uses sensor constraints; this script prepares the npz needed by train.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_source(path: Path):
    data = np.load(path, allow_pickle=True).item()
    time = np.asarray(data["time"], dtype=float).flatten()
    u = np.asarray(data["u"], dtype=float)
    v = np.asarray(data["v"], dtype=float)
    omega = np.asarray(data["omega"], dtype=float)
    return time, u, v, omega


def _sample_by_indices(field: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if field.ndim == 3:
        nt, nx, ny = field.shape
        flat = field.reshape(nt, nx * ny)
    else:
        flat = field
    if flat.ndim != 2:
        raise ValueError(f"Unsupported field shape {field.shape}; expected (t, x, y) or (t, npoints)")
    return flat[:, indices]


def generate_sensor_values(sensor_json: Path, source_npy: Path, output_npz: Path) -> None:
    with sensor_json.open("r") as handle:
        payload = json.load(handle)

    indices = np.asarray(payload.get("indices", []), dtype=int)
    if indices.size == 0:
        raise ValueError("sensor_json missing indices")

    time, u, v, omega = _load_source(source_npy)

    u_vals = _sample_by_indices(u, indices).T
    v_vals = _sample_by_indices(v, indices).T
    omega_vals = _sample_by_indices(omega, indices).T

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        time=time,
        u=u_vals,
        v=v_vals,
        omega=omega_vals,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate sensor values npz from DNS/LES source")
    parser.add_argument("--sensor-json", required=True, help="Path to sensors_temporal_*.json")
    parser.add_argument("--source-npy", required=True, help="Path to kolmogorov_dns/les .npy")
    parser.add_argument("--output-npz", required=True, help="Output .npz path")
    args = parser.parse_args()

    generate_sensor_values(Path(args.sensor_json), Path(args.source_npy), Path(args.output_npz))


if __name__ == "__main__":
    main()
