import argparse
from pathlib import Path

import numpy as np


TWO_PI = 2.0 * np.pi


def map_domain(src_path: str, dst_path: str):
    src = Path(src_path)
    dst = Path(dst_path)
    data = np.load(src, allow_pickle=True).item()

    required = ["x", "y", "u", "v", "p", "omega", "time", "config"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"source file missing keys: {missing}")

    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)

    x_unit = (x / TWO_PI).astype(np.float32)
    y_unit = (y / TWO_PI).astype(np.float32)

    cfg = dict(data.get("config", {}))
    old_L = cfg.get("L", None)
    old_kf = cfg.get("k_f", None)

    cfg["source_file"] = str(src)
    cfg["source_format"] = "mapped_from_[0,2pi]_to_[0,1]"
    cfg["source_L"] = old_L
    cfg["L"] = 1.0
    cfg["coordinate_mapping"] = "x_new=x_old/(2pi), y_new=y_old/(2pi)"
    cfg["mapping_note"] = (
        "This is a coordinate-mapped version only. Field values are unchanged. "
        "If the original forcing was A*sin(k_f*y) on [0,2pi], then in [0,1] coordinates "
        "it corresponds to A*sin(2*pi*k_f*y_new), i.e. Fourier mode k=k_f. "
        "This does NOT imply alignment with literature forcing [0.1*sin(4*pi*y), 0] unless k_f=2."
    )
    if old_kf is not None:
        cfg["source_k_f"] = old_kf
        cfg["mapped_domain_fourier_mode"] = int(old_kf)
        cfg["mapped_domain_forcing_expression"] = f"A*sin(2*pi*{int(old_kf)}*y)"

    out = {
        "x": x_unit,
        "y": y_unit,
        "u": np.asarray(data["u"], dtype=np.float32),
        "v": np.asarray(data["v"], dtype=np.float32),
        "p": np.asarray(data["p"], dtype=np.float32),
        "omega": np.asarray(data["omega"], dtype=np.float32),
        "time": np.asarray(data["time"], dtype=np.float32),
        "diagnostics": data.get("diagnostics", {}),
        "config": cfg,
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out, allow_pickle=True)
    print(f"Mapped {src} -> {dst}")
    print(f"x: [{float(x.min())}, {float(x.max())}] -> [{float(x_unit.min())}, {float(x_unit.max())}]")
    print(f"y: [{float(y.min())}, {float(y.max())}] -> [{float(y_unit.min())}, {float(y_unit.max())}]")
    print(f"config.L: {old_L} -> {cfg['L']}")
    if old_kf is not None:
        print(f"source k_f={old_kf}, mapped-domain Fourier mode={cfg['mapped_domain_fourier_mode']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Kolmogorov DNS coordinates from [0,2pi] to [0,1]")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    map_domain(args.src, args.dst)
