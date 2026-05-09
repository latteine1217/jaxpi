"""
What:
    將 paper reference 的 u/v error 曲線與本地 no-data eval 曲線疊圖比較。
Why:
    使用者要直接比較 `paper_dns_ref` CSV 與我們自己的 no-data 版本，
    並保留可重跑腳本與 raw 匯出，避免只剩一張無 provenance 的圖片。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_two_col_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open() as f:
        for row in csv.reader(f):
            if not row:
                continue
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def write_long_csv(
    output_path: Path,
    paper_u: tuple[np.ndarray, np.ndarray],
    paper_v: tuple[np.ndarray, np.ndarray],
    no_data_t: np.ndarray,
    no_data_u: np.ndarray,
    no_data_v: np.ndarray,
) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "source", "t", "error"])
        for t, err in zip(*paper_u):
            writer.writerow(["u_error", "paper_dns_ref", float(t), float(err)])
        for t, err in zip(*paper_v):
            writer.writerow(["v_error", "paper_dns_ref", float(t), float(err)])
        for t, err in zip(no_data_t, no_data_u):
            writer.writerow(["u_error", "no_data_eval_0405", float(t), float(err)])
        for t, err in zip(no_data_t, no_data_v):
            writer.writerow(["v_error", "no_data_eval_0405", float(t), float(err)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay paper ref and no-data u/v error curves.")
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("examples/kolmogorov_flow/data/paper_dns_ref"),
        help="Directory containing paper reference CSV files.",
    )
    parser.add_argument(
        "--no-data-npz",
        type=Path,
        default=Path("eval_runs/eval_paper_repro_soap_0405/l2_errors.npz"),
        help="NPZ artifact for no-data evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_runs/paper_ref_vs_no_data_uv_error_20260422"),
        help="Output directory.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_u = load_two_col_csv(args.paper_dir / "kolmogorov_re1e6_u_error.csv")
    paper_v = load_two_col_csv(args.paper_dir / "kolmogorov_re1e6_v_error.csv")

    no_data = np.load(args.no_data_npz)
    ts_all = np.asarray(no_data["ts_all"], dtype=float)
    eu_all = np.asarray(no_data["eu_all"], dtype=float)
    ev_all = np.asarray(no_data["ev_all"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False)
    fig.suptitle("Paper Reference vs No-data U/V Error", fontsize=13, fontweight="bold")

    u_ax, v_ax = axes

    u_ax.semilogy(paper_u[0], paper_u[1], color="#1f77b4", linewidth=2.2, label="Paper ref")
    u_ax.semilogy(ts_all, eu_all, color="#d62728", linestyle="--", linewidth=2.0, label="No data")
    u_ax.set_title("U Error")
    u_ax.set_xlabel("t")
    u_ax.set_ylabel("Relative L2 Error")
    u_ax.grid(True, alpha=0.3, which="both")
    u_ax.legend()

    v_ax.semilogy(paper_v[0], paper_v[1], color="#1f77b4", linewidth=2.2, label="Paper ref")
    v_ax.semilogy(ts_all, ev_all, color="#d62728", linestyle="--", linewidth=2.0, label="No data")
    v_ax.set_title("V Error")
    v_ax.set_xlabel("t")
    v_ax.set_ylabel("Relative L2 Error")
    v_ax.grid(True, alpha=0.3, which="both")
    v_ax.legend()

    fig.tight_layout()
    plot_path = output_dir / "paper_ref_vs_no_data_uv_error.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / "paper_ref_vs_no_data_uv_error.csv"
    write_long_csv(csv_path, paper_u, paper_v, ts_all, eu_all, ev_all)

    summary_path = output_dir / "summary.txt"
    summary_lines = [
        "Paper Reference vs No-data U/V Error",
        f"paper_dir = {args.paper_dir}",
        f"no_data_npz = {args.no_data_npz}",
        f"paper_u_points = {len(paper_u[0])}",
        f"paper_v_points = {len(paper_v[0])}",
        f"no_data_points = {len(ts_all)}",
        f"paper_u_t_range = [{paper_u[0].min():.4f}, {paper_u[0].max():.4f}]",
        f"paper_v_t_range = [{paper_v[0].min():.4f}, {paper_v[0].max():.4f}]",
        f"no_data_t_range = [{ts_all.min():.4f}, {ts_all.max():.4f}]",
        "",
        "Note:",
        "  no-data curve uses eval_paper_repro_soap_0405/l2_errors.npz",
        "  This artifact provides continuous u/v error series needed for direct overlay.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved plot to {plot_path}")
    print(f"Saved csv to {csv_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
