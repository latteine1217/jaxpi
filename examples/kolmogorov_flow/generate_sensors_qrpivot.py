#!/usr/bin/env python3
"""
Generate optimal sensor positions for Kolmogorov flow DNS data via QR column pivoting.

What:
    Select K spatial locations that are maximally informative across the DNS snapshot ensemble.

Why (algorithm):
    1. Spatially downsample DNS to --spatial-res (default 512) to reduce memory.
       2048² = 4.2M points → 512² = 262k points (16× smaller snapshot matrix).
    2. Build snapshot matrix A ∈ ℝ^{n_feat_time × n_spatial_ds} from u, v, p, omega, |∇u|, |∇v|
       sampled at a fixed time stride (e.g. every 0.1s).
       Shape: (n_features × n_time_sel, 512²) ≈ (306, 262k) → ~320 MB.
    3. Truncated SVD via Gram matrix: A A^T (small square matrix) → top-K left singular vectors.
    4. U_k = A^T V_k / sigma_k ∈ ℝ^{n_spatial_ds × K}: right singular vectors (spatial modes).
    5. QR with column pivoting on U_k^T → first K pivots = maximally separated sensor positions
       on the downsampled grid.
    6. Map downsampled indices back to original grid (stride = orig_res // spatial_res).
    7. Extract DNS values from the original full-resolution DNS at those exact grid points.

Memory budget (512 downsample, K=100, 51 time steps):
    Snapshot matrix A : 306 × 262k × 4 B ≈ 320 MB
    U_k              : 262k × 100 × 4 B ≈ 100 MB
    DNS load (full)  : 4 fields × 101 × 2048² × 4 B ≈ 6.7 GB  (only at end for value extraction)

Usage:
    python3 generate_sensors_qrpivot.py \\
        --dns          examples/.../kolmogorov_Re1e6_N2048_T5_ke024.npy \\
        --outdir       examples/.../kolmogorov_sensors/re1000000 \\
        --K 100 \\
        --time-stride  2 \\
        --spatial-res  512

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
    What: Build a (N_FEATURES, n_spatial_ds) block for the i-th selected time step.
    Why:  Encapsulates the 6-feature stacking so both Stage A and B can call it identically.
          Fields are already downsampled before being stored in `fields`.
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
    parser.add_argument("--dns",          required=True,  help="Path to DNS .npy file")
    parser.add_argument("--outdir",       required=True,  help="Output directory")
    parser.add_argument("--K",            type=int,   default=100,   help="Number of sensors")
    parser.add_argument("--time-stride",  type=int,   default=2,     help="Time stride for QR snapshot matrix (2=every 0.1s)")
    parser.add_argument("--spatial-res",  type=int,   default=512,   help="Downsample spatial grid to NxN before QR")
    parser.add_argument("--sensor-dt",    type=float, default=None,
                        help="Temporal resolution of output sensor NPZ (seconds). "
                             "If finer than DNS dt, values are CubicSpline-interpolated. "
                             "Default: use DNS native dt (no interpolation).")
    args = parser.parse_args()

    dns_path    = Path(args.dns)
    out_dir     = Path(args.outdir)
    K           = args.K
    stride      = args.time_stride
    spatial_res = args.spatial_res
    sensor_dt   = args.sensor_dt

    print(f"Loading DNS: {dns_path}")
    raw = np.load(dns_path, allow_pickle=True).item()
    time_all = np.array(raw["time"])   # (nt_full,)
    x_orig   = np.array(raw["x"])     # (nx_orig,)
    y_orig   = np.array(raw["y"])     # (ny_orig,)

    nx_orig = x_orig.shape[0]
    ny_orig = y_orig.shape[0]

    # Compute spatial downsample stride so that the grid becomes spatial_res × spatial_res
    sp_stride = max(1, nx_orig // spatial_res)
    nx_ds = nx_orig // sp_stride
    ny_ds = ny_orig // sp_stride
    print(f"Spatial downsample: {nx_orig}×{ny_orig} → {nx_ds}×{ny_ds}  (stride={sp_stride})")

    x_ds = x_orig[::sp_stride]   # (nx_ds,)
    y_ds = y_orig[::sp_stride]   # (ny_ds,)

    # Load and downsample DNS fields in time (time stride) + space (spatial stride)
    t_sel = slice(None, None, stride)
    sel_fields = {
        "u":     np.array(raw["u"],     dtype=np.float32)[t_sel, ::sp_stride, ::sp_stride],
        "v":     np.array(raw["v"],     dtype=np.float32)[t_sel, ::sp_stride, ::sp_stride],
        "p":     np.array(raw["p"],     dtype=np.float32)[t_sel, ::sp_stride, ::sp_stride],
        "omega": np.array(raw["omega"], dtype=np.float32)[t_sel, ::sp_stride, ::sp_stride],
    }
    time_sel  = time_all[t_sel]
    nt        = sel_fields["u"].shape[0]
    n_spatial = nx_ds * ny_ds
    n_rows    = N_FEATURES * nt

    print(f"Snapshot matrix: ({n_rows}, {n_spatial})  "
          f"[{n_rows * n_spatial * 4 / 1e6:.0f} MB]")

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

    # Sensor positions on the downsampled grid
    sensor_flat_ds = np.sort(piv[:K])
    ix_ds = (sensor_flat_ds // ny_ds).astype(int)
    iy_ds = (sensor_flat_ds % ny_ds).astype(int)

    # Map back to original full-resolution grid indices
    ix_orig_sel = ix_ds * sp_stride
    iy_orig_sel = iy_ds * sp_stride
    sensor_flat_orig = ix_orig_sel * ny_orig + iy_orig_sel

    coords_xy = np.stack([x_orig[ix_orig_sel], y_orig[iy_orig_sel]], axis=1)  # (K, 2)
    print(f"Sensor x ∈ [{coords_xy[:, 0].min():.4f}, {coords_xy[:, 0].max():.4f}]")
    print(f"Sensor y ∈ [{coords_xy[:, 1].min():.4f}, {coords_xy[:, 1].max():.4f}]")

    # ── Save outputs ─────────────────────────────────────────────────────────
    t0_str    = f"{time_sel[0]:.0f}"
    t1_str    = f"{time_sel[-1]:.0f}"
    res_str   = f"N{nx_orig}"
    base_name = f"sensors_qrpivot_K{K}_{res_str}_t{t0_str}-{t1_str}"
    npz_name  = base_name + "_dns_values.npz"
    json_name = base_name + ".json"

    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "K": K,
        "resolution": f"{nx_orig}x{ny_orig}",
        "spatial_downsample_res": f"{nx_ds}x{ny_ds}",
        "spatial_downsample_stride": int(sp_stride),
        "method": "qr_pivoting",
        "features": ["u", "v", "p", "omega", "grad_u_mag", "grad_v_mag"],
        "time_stride": stride,
        "time_range": [float(time_sel[0]), float(time_sel[-1])],
        "time_steps": int(nt),
        "selected_coordinates": coords_xy.tolist(),
        "indices": sensor_flat_orig.tolist(),
        "source_file": str(dns_path),
        "dns_values_npz": str(out_dir / npz_name),
    }
    json_path = out_dir / json_name
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved: {json_path}")

    # Extract DNS values at sensor locations
    print("Extracting sensor values from full DNS ...")
    u_all     = np.array(raw["u"],     dtype=np.float32)   # (nt_full, nx, ny)
    v_all     = np.array(raw["v"],     dtype=np.float32)
    omega_all = np.array(raw["omega"], dtype=np.float32)

    # Raw DNS values at sensor positions: (K, nt_full)
    u_dns     = u_all[:,     ix_orig_sel, iy_orig_sel].T.astype(np.float64)
    v_dns     = v_all[:,     ix_orig_sel, iy_orig_sel].T.astype(np.float64)
    omega_dns = omega_all[:, ix_orig_sel, iy_orig_sel].T.astype(np.float64)

    dns_dt = float(time_all[1] - time_all[0])

    if sensor_dt is not None and sensor_dt < dns_dt - 1e-12:
        # Interpolate to finer time grid using CubicSpline along the time axis.
        # Why CubicSpline: provides C² continuity and is accurate for smooth signals;
        # for turbulent fields at individual sensor locations the time series is
        # smoother than the spatial field because we're tracking a single trajectory.
        from scipy.interpolate import CubicSpline

        t_fine = np.arange(time_all[0], time_all[-1] + sensor_dt * 0.5, sensor_dt)
        n_fine = len(t_fine)
        pts_per_window = round(0.1 / sensor_dt)  # approximate, for reporting

        print(f"Interpolating sensor time series: DNS dt={dns_dt:.4f}s → sensor dt={sensor_dt:.4f}s")
        print(f"  DNS time points: {len(time_all)}  →  interpolated: {n_fine}")
        print(f"  ≈ {pts_per_window} sensor time points per 0.1s window")

        # Fit CubicSpline over the full time axis for each sensor (vectorised over K)
        # CubicSpline expects axis=0 to be the interpolation axis
        cs_u     = CubicSpline(time_all, u_dns.T,     axis=0)   # (nt_full, K)
        cs_v     = CubicSpline(time_all, v_dns.T,     axis=0)
        cs_omega = CubicSpline(time_all, omega_dns.T, axis=0)

        time_out  = t_fine
        u_out     = cs_u(t_fine).T.astype(np.float32)     # (K, n_fine)
        v_out     = cs_v(t_fine).T.astype(np.float32)
        omega_out = cs_omega(t_fine).T.astype(np.float32)

        # Update JSON with interpolation metadata
        json_payload["sensor_dt"]            = float(sensor_dt)
        json_payload["sensor_time_points"]   = int(n_fine)
        json_payload["interpolation_method"] = "CubicSpline"
    else:
        # No interpolation: store native DNS time steps
        time_out  = time_all
        u_out     = u_dns.astype(np.float32)
        v_out     = v_dns.astype(np.float32)
        omega_out = omega_dns.astype(np.float32)
        json_payload["sensor_dt"]          = float(dns_dt)
        json_payload["sensor_time_points"] = int(len(time_all))

    # Re-write JSON now that interpolation metadata is known
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    np.savez(
        out_dir / npz_name,
        time=time_out,
        u=u_out,        # (K, n_time_out)
        v=v_out,
        omega=omega_out,
    )
    print(f"Saved: {out_dir / npz_name}  (shape: {u_out.shape})")
    print("Done.")


if __name__ == "__main__":
    main()
