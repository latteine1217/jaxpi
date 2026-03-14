import argparse
from pathlib import Path

import numpy as np


def convert(src_path: str, dst_path: str):
    src = Path(src_path)
    dst = Path(dst_path)
    data = np.load(src, allow_pickle=True).item()

    required = ["t", "coords", "velocity", "pressure", "vorticity", "nu"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"source file missing keys: {missing}")

    t = np.asarray(data["t"], dtype=np.float32)
    coords = np.asarray(data["coords"], dtype=np.float32)
    vel = np.asarray(data["velocity"], dtype=np.float32)
    p = np.asarray(data["pressure"], dtype=np.float32)
    omega = np.asarray(data["vorticity"], dtype=np.float32)
    nu = float(np.asarray(data["nu"]))

    if vel.ndim != 3 or vel.shape[-1] != 2:
        raise ValueError(f"velocity must have shape (nt, npoints, 2), got {vel.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (npoints, 2), got {coords.shape}")

    npoints = coords.shape[0]
    nt = t.shape[0]
    if vel.shape[:2] != (nt, npoints):
        raise ValueError(f"velocity shape {vel.shape} incompatible with nt={nt}, npoints={npoints}")
    if p.shape != (nt, npoints):
        raise ValueError(f"pressure shape {p.shape} incompatible with (nt, npoints)=({nt}, {npoints})")
    if omega.shape != (nt, npoints):
        raise ValueError(f"vorticity shape {omega.shape} incompatible with (nt, npoints)=({nt}, {npoints})")

    grid_n = int(round(np.sqrt(npoints)))
    if grid_n * grid_n != npoints:
        raise ValueError(f"coords point count {npoints} is not a square grid")

    # coords appear to be on [0,1]^2 cell centers; recover sorted unique axes.
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    if len(x_unique) != grid_n or len(y_unique) != grid_n:
        raise ValueError("unique coordinate counts do not match inferred grid size")

    out = {
        "x": x_unique.astype(np.float32),
        "y": y_unique.astype(np.float32),
        "u": vel[:, :, 0].astype(np.float32),
        "v": vel[:, :, 1].astype(np.float32),
        "p": p.astype(np.float32),
        "omega": omega.astype(np.float32),
        "time": t.astype(np.float32),
        "diagnostics": {
            "kinetic_energy": (0.5 * np.mean(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2, axis=1)).astype(np.float32),
            "enstrophy": (0.5 * np.mean(omega ** 2, axis=1)).astype(np.float32),
            "divergence_error": np.zeros((nt,), dtype=np.float32),
        },
        "config": {
            "N": grid_n,
            "L": 1.0,
            "nu": nu,
            "A": None,
            "k_f": None,
            "dt": float(t[1] - t[0]) if len(t) > 1 else None,
            "T_end": float(t[-1]) if len(t) else None,
            "source_file": str(src),
            "source_format": "kolmogorov_flow_Re10000_256",
        },
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out, allow_pickle=True)
    print(f"Converted {src} -> {dst}")
    print(f"grid={grid_n}x{grid_n}, nt={nt}, nu={nu}")
    print(f"time range=[{float(t[0])}, {float(t[-1])}] dt={float(t[1]-t[0]) if len(t)>1 else 'n/a'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert kolmogorov_flow_Re10000_256.npy into JAXPI DNS loader format")
    parser.add_argument("--src", required=True, help="source npy path")
    parser.add_argument("--dst", required=True, help="destination npy path")
    args = parser.parse_args()
    convert(args.src, args.dst)
