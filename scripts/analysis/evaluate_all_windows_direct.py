#!/usr/bin/env python3
"""
What:
    用 direct `apply_fn` 路徑重做整個 checkpoint tree 的 multi-window full-window evaluation。
Why:
    舊的 `eval_paper_repro_soap.py` / `3145` 產物混入了歷史 wrapper bug；
    這支腳本固定以當前已驗證的 direct 路徑重新計算所有窗口，避免混合 lineage。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.evaluate_window1_checkpoint_sweep import (
    RunSpec,
    compute_l2_error_direct,
    load_config,
    resolve_checkpoint_root,
    resolve_eval_chunk_sizes,
    resolve_repo_path,
    to_window_local_time,
)


def latest_step(window_dir: Path) -> int | None:
    steps = []
    for child in window_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint_"):
            continue
        try:
            steps.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return max(steps) if steps else None


def discover_windows(ckpt_root: Path) -> list[int]:
    windows = []
    for child in ckpt_root.iterdir():
        if not child.is_dir() or not child.name.startswith("time_window_"):
            continue
        try:
            windows.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    return sorted(windows)


def write_outputs(output_dir: Path, rows: list[dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_window_direct_results.csv"
    txt_path = output_dir / "summary.txt"
    fig_path = output_dir / "window_error_comparison.png"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window", "checkpoint_step", "t_end", "u_error", "v_error", "w_error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = ["=== Summary ===", "Win\tStep\tu_error\tv_error\tw_error"]
    for row in rows:
        lines.append(
            f"{int(row['window'])}\t{int(row['checkpoint_step'])}\t"
            f"{row['u_error']:.6f}\t{row['v_error']:.6f}\t{row['w_error']:.6f}\t"
            f"(t_end={row['t_end']:.4f})"
        )
    u_vals = [row["u_error"] for row in rows]
    v_vals = [row["v_error"] for row in rows]
    w_vals = [row["w_error"] for row in rows]
    lines.append(f"mean\t-\t{sum(u_vals)/len(u_vals):.6f}\t{sum(v_vals)/len(v_vals):.6f}\t{sum(w_vals)/len(w_vals):.6f}")
    lines.append(f"max\t-\t{max(u_vals):.6f}\t{max(v_vals):.6f}\t{max(w_vals):.6f}")
    txt_path.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    metrics = [("u_error", "u Relative L2"), ("v_error", "v Relative L2"), ("w_error", "vorticity Relative L2")]
    windows = [int(row["window"]) for row in rows]
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(windows, [row[key] for row in rows], marker="o", linewidth=2.0, color="#1d4ed8")
        ax.set_yscale("log")
        ax.set_xlabel("Time Window")
        ax.set_ylabel("Relative L2 error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("All-Window Direct Evaluation")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate all windows using direct apply_fn.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-max", type=int, default=None)
    parser.add_argument("--space-chunk-size", type=int, default=None)
    parser.add_argument("--time-chunk-size", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    ckpt_root = resolve_checkpoint_root(Path(args.checkpoint_root))
    output_dir = Path(args.output_dir)
    config = load_config(config_path)

    from examples.kolmogorov_flow import models
    from examples.kolmogorov_flow.utils import get_dataset
    from jaxpi.utils import restore_checkpoint

    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
        time_fraction=config.time_fraction,
        dataset_path=str(resolve_repo_path(str(config.dataset_path))),
        time_range=config.get("dns_time_range"),
        time_stride=config.get("dns_time_stride", 1),
    )

    num_windows = int(config.training.num_time_windows)
    num_time_steps = len(t_star) // num_windows
    windows = discover_windows(ckpt_root)
    if args.window_max is not None:
        windows = [w for w in windows if w <= args.window_max]

    rows: list[dict[str, float]] = []
    for window in windows:
        start_idx = (window - 1) * num_time_steps
        end_idx = window * num_time_steps
        if end_idx > len(t_star):
            break

        t_window = np.asarray(t_star[start_idx:end_idx])
        t_local = to_window_local_time(t_window)
        eval_time_chunk, eval_space_chunk = resolve_eval_chunk_sizes(
            config, t_local, args.time_chunk_size, args.space_chunk_size
        )

        model = models.NavierStokes(
            config,
            t_local,
            coords,
            u_ref[start_idx, :],
            v_ref[start_idx, :],
            w_ref[start_idx, :],
            nu,
            replicate_state=False,
        )

        window_dir = ckpt_root / f"time_window_{window}"
        step = latest_step(window_dir)
        if step is None:
            continue

        print(f"=== window {window} checkpoint_{step} ===", flush=True)
        model.state = restore_checkpoint(model.state, str(window_dir), step=step)
        u_error, v_error, w_error = compute_l2_error_direct(
            model,
            model.state.params,
            t_local,
            coords,
            np.asarray(u_ref[start_idx:end_idx, :]),
            np.asarray(v_ref[start_idx:end_idx, :]),
            np.asarray(w_ref[start_idx:end_idx, :]),
            space_chunk_size=eval_space_chunk,
        )
        row = {
            "window": float(window),
            "checkpoint_step": float(step),
            "t_end": float(t_window[-1]),
            "u_error": float(u_error),
            "v_error": float(v_error),
            "w_error": float(w_error),
        }
        rows.append(row)
        print(
            f"u={row['u_error']:.6e}, v={row['v_error']:.6e}, w={row['w_error']:.6e}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No windows evaluated.")

    write_outputs(output_dir, rows)
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
