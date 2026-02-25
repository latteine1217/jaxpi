"""
Split LES data into time windows while keeping the full dataset.

What: Create per-window .npy files from a single LES .npy file.
Why: Reduce memory footprint during training by loading one time window at a time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _slice_if_time_aligned(value: np.ndarray, start: int, end: int, nt: int) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.shape and value.shape[0] == nt:
        return value[start:end]
    return value


def split_les_windows(source_path: Path, output_dir: Path, num_windows: int) -> None:
    data = np.load(source_path, allow_pickle=True).item()

    time = np.asarray(data["time"])
    nt = time.shape[0]
    steps_per_window = nt // num_windows
    if steps_per_window == 0:
        raise ValueError(f"num_windows {num_windows} is larger than time steps {nt}")

    remainder = nt - steps_per_window * num_windows
    if remainder != 0:
        print(f"Warning: dropping last {remainder} steps to fit {num_windows} windows")

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_windows):
        start = idx * steps_per_window
        end = start + steps_per_window

        window_data = {
            "x": np.asarray(data["x"]),
            "y": np.asarray(data["y"]),
            "u": np.asarray(data["u"])[start:end],
            "v": np.asarray(data["v"])[start:end],
            "omega": np.asarray(data["omega"])[start:end],
            "time": np.asarray(time)[start:end],
            "config": data.get("config", {}),
        }
        if "p" in data:
            window_data["p"] = np.asarray(data["p"])[start:end]

        diagnostics = data.get("diagnostics")
        if diagnostics is not None:
            if isinstance(diagnostics, dict):
                diag_out = {}
                for key, value in diagnostics.items():
                    if isinstance(value, np.ndarray) and value.shape and value.shape[0] == nt:
                        diag_out[key] = value[start:end]
                    else:
                        diag_out[key] = value
                window_data["diagnostics"] = diag_out
            else:
                window_data["diagnostics"] = _slice_if_time_aligned(diagnostics, start, end, nt)

        output_path = output_dir / f"window_{idx:03d}.npy"
        np.save(output_path, window_data, allow_pickle=True)
        print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Split LES data into time windows")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to kolmogorov_les .npy file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write window_*.npy files",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        required=True,
        help="Number of time windows to split",
    )
    args = parser.parse_args()

    split_les_windows(Path(args.source), Path(args.output_dir), args.num_windows)


if __name__ == "__main__":
    main()
