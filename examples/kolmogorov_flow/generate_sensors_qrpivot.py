#!/usr/bin/env python3
"""
Generate optimal sensor positions for Kolmogorov flow DNS data via QR column pivoting.

What:
    Select K spatial locations that are maximally informative across the DNS snapshot ensemble.

Why (algorithm):
    1. Build snapshot matrix A ∈ ℝ^{n_feat_time × n_spatial} from u, v, p, omega, |∇u|, |∇v|
       sampled at a fixed time stride (e.g. every 0.1s).
       Shape: (n_features × n_time_sel, n_spatial).
    2. Truncated SVD via Gram matrix: A A^T (small square matrix) → top-K left singular vectors.
    3. U_k = A^T V_k / sigma_k ∈ ℝ^{n_spatial × K}: right singular vectors (spatial modes).
    4. QR with column pivoting on U_k^T → first K pivots = maximally separated, representative
       spatial locations.

Memory strategy (two-stage, avoids materialising A twice simultaneously):
    Stage A: build A from sel data → compute Gram matrix → free A → eigendecompose → get V_k.
    Stage B: rebuild A → compute U_k → QR pivot → free all → load full DNS → extract values.

Usage:
    uv run python generate_sensors_qrpivot.py \\
        --dns    examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_Re1e6_N2048_T5_ke024.npy \\
        --outdir examples/kolmogorov_flow/data/kolmogorov_sensors/re1000000 \\
        --K 100 \\
        --time-stride 2

Outputs:
    sensors_qrpivot_K{K}_N{res}_t{t0}-{t1}.json       — sensor positions + metadata
    sensors_qrpivot_K{K}_N{res}_t{t0}-{t1}_dns_values.npz — u, v, omega at sensors × all t
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.linalg import qr

N_FEATURES = 6  # u, v, p, omega, |∇u|, |∇v|


def spectral_grad_magnitude(field: np.ndarray) -> np.ndarray:
    """
    What: Compute |∇f| via spectral differentiation on a [0,1) periodic square domain.
    Why:  Spectral derivatives are exact for periodic fields; FD introduces phase errors.
    """
    n = field.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    f_hat = np.fft.fft2(field)
    fx = np.real(np.fft.ifft2(1j * KX * f_hat))
    fy = np.real(np.fft.ifft2(1j * KY * f_hat))
    return np.sqrt(fx**2 + fy**2).astype(np.float32)


def build_snapshot_block(i: int, fields: dict) -> np.ndarray:
    """
    What: Build a (N_FEATURES, n_spatial) block for the i-th selected time step.
    Why:  Encapsulates the 6-feature stacking so both Stage A and B can call it identically.
    """
    u = fields["u"][i]
    v = fields["v"][i]
    return np.stack(
        [
            u.reshape(-1),
            v.reshape(-1),
            fields["p"][i].reshape(-1),
            fields["omega"][i].reshape(-1),
            spectral_grad_magnitude(u).reshape(-1),
            spectral_grad_magnitude(v).reshape(-1),
        ],
        axis=0,
        dtype=np.float32,
    )


def normalize_rows(A: np.ndarray) -> None:
    """
    What: In-place row normalisation (zero mean, unit std).
    Why:  Equalises contribution of all features regardless of physical scale.
    """
    A -= A.mean(axis=1, keepdims=True)
    std = A.std(axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    A /= std


def build_snapshot_matrix(sel_fields: dict, nt: int, n_spatial: int) -> np.ndarray:
    n_rows = N_FEATURES * nt
    A = np.empty((n_rows, n_spatial), dtype=np.float32)
    for i in range(nt):
        if i % 10 == 0:
            print(f"    time step {i+1}/{nt}")
        A[i * N_FEATURES : (i + 1) * N_FEATURES] = build_snapshot_block(i, sel_fields)
    normalize_rows(A)
    return A


def main() -> None:
    parser = argparse.ArgumentParser(description="QR-pivot optimal sensor placement for Kolmogorov DNS")
    parser.add_argument("--dns",         required=True,  help="Path to DNS .npy file")
    parser.add_argument("--outdir",      required=True,  help="Output directory")
    parser.add_argument("--K",           type=int, default=100, help="Number of sensors")
    parser.add_argument("--time-stride", type=int, default=2,   help="Time stride (1=every step, 2=every 0.1s for dt=0.05)")
    args = parser.parse_args()

    dns_path  = Path(args.dns)
    out_dir   = Path(args.outdir)
    K         = args.K
    stride    = args.time_stride

    print(f"Loading DNS: {dns_path}")
    raw = np.load(dns_path, allow_pickle=True).item()
    time_all  = np.array(raw["time"])           # (nt_full,)
    x_coords  = np.array(raw["x"])             # (nx,)
    y_coords  = np.array(raw["y"])             # (ny,)

    sel = slice(None, None, stride)
    sel_fields = {
        "u":     np.array(raw["u"],     dtype=np.float32)[sel],
        "v":     np.array(raw["v"],     dtype=np.float32)[sel],
        "p":     np.array(raw["p"],     dtype=np.float32)[sel],
        "omega": np.array(raw["omega"], dtype=np.float32)[sel],
    }
    time_sel = time_all[sel]
    nt  = sel_fields["u"].shape[0]
    nx  = sel_fields["u"].shape[1]
    ny  = sel_fields["u"].shape[2]
    n_spatial = nx * ny
    n_rows    = N_FEATURES * nt

    print(f"Snapshot matrix: ({n_rows}, {n_spatial})  "
          f"[{n_rows * n_spatial * 4 / 1e9:.1f} GB if materialised]")

    # ── Stage A: build A → Gram matrix → free A → get V_k ───────────────────
    print("Stage A: building snapshot matrix ...")
    A = build_snapshot_matrix(sel_fields, nt, n_spatial)

    print("Computing Gram matrix G = A A^T ...")
    G = (A @ A.T).astype(np.float64)
    del A

    eigenvalues, V = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, V = eigenvalues[order], V[:, order]

    pos_ev = eigenvalues[eigenvalues > 0]
    explained = eigenvalues[:K].sum() / pos_ev.sum() if pos_ev.size > 0 else 0.0
    print(f"Top-{K} modes explain {explained:.1%} of total variance")

    V_k     = V[:, :K].astype(np.float32)                                  # (n_rows, K)
    sigma_k = np.sqrt(np.maximum(eigenvalues[:K], 0)).astype(np.float32)   # (K,)

    # ── Stage B: rebuild A → U_k → QR pivot → free A ────────────────────────
    print("Stage B: rebuilding snapshot matrix ...")
    A = build_snapshot_matrix(sel_fields, nt, n_spatial)
    del sel_fields

    print(f"Computing U_k ({n_spatial} × {K}) ...")
    U_k = (A.T @ V_k) / sigma_k[None, :]
    del A

    print(f"QR with column pivoting on U_k^T ({K} × {n_spatial}) ...")
    _, _, piv = qr(U_k.T.astype(np.float64), pivoting=True)
    del U_k

    sensor_flat_idx = np.sort(piv[:K])
    ix = (sensor_flat_idx // ny).astype(int)
    iy = (sensor_flat_idx % ny).astype(int)
    coords_xy = np.stack([x_coords[ix], y_coords[iy]], axis=1)  # (K, 2)

    print(f"Sensor x ∈ [{coords_xy[:, 0].min():.4f}, {coords_xy[:, 0].max():.4f}]")
    print(f"Sensor y ∈ [{coords_xy[:, 1].min():.4f}, {coords_xy[:, 1].max():.4f}]")

    # ── Save outputs ─────────────────────────────────────────────────────────
    t0_str = f"{time_sel[0]:.0f}"
    t1_str = f"{time_sel[-1]:.0f}"
    res_str = f"N{nx}"
    base_name = f"sensors_qrpivot_K{K}_{res_str}_t{t0_str}-{t1_str}"
    npz_name  = base_name + "_dns_values.npz"
    json_name = base_name + ".json"

    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "K": K,
        "resolution": f"{nx}x{ny}",
        "method": "qr_pivoting",
        "features": ["u", "v", "p", "omega", "grad_u_mag", "grad_v_mag"],
        "time_stride": stride,
        "time_range": [float(time_sel[0]), float(time_sel[-1])],
        "time_steps": int(nt),
        "selected_coordinates": coords_xy.tolist(),
        "indices": sensor_flat_idx.tolist(),
        "source_file": str(dns_path),
        "dns_values_npz": str(out_dir / npz_name),
    }
    json_path = out_dir / json_name
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved: {json_path}")

    # Extract DNS values at sensor locations for all time steps
    print("Extracting sensor values from full DNS ...")
    u_all     = np.array(raw["u"],     dtype=np.float32)
    v_all     = np.array(raw["v"],     dtype=np.float32)
    omega_all = np.array(raw["omega"], dtype=np.float32)

    np.savez(
        out_dir / npz_name,
        time=time_all,
        u=u_all[:, ix, iy].T,         # (K, nt_full)
        v=v_all[:, ix, iy].T,
        omega=omega_all[:, ix, iy].T,
    )
    print(f"Saved: {out_dir / npz_name}")
    print("Done.")


if __name__ == "__main__":
    main()
