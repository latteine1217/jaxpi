#!/usr/bin/env python3
"""
視覺化 DNS vs Prediction vs Error Field。
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_field_comparison_from_arrays(
    dns_u,
    dns_v,
    dns_w,
    pred_u,
    pred_v,
    pred_w,
    time_idx,
    model_name="PINN",
    output_file="field_comparison.png",
):
    """
    給定 DNS 和預測數組，繪製對比圖。
    """
    if dns_u.ndim == 1:
        grid_size = int(np.sqrt(len(dns_u)))
        dns_u = dns_u.reshape(grid_size, grid_size)
        dns_v = dns_v.reshape(grid_size, grid_size)
        dns_w = dns_w.reshape(grid_size, grid_size)
        pred_u = pred_u.reshape(grid_size, grid_size)
        pred_v = pred_v.reshape(grid_size, grid_size)
        pred_w = pred_w.reshape(grid_size, grid_size)

    error_u = np.abs(pred_u - dns_u)
    error_v = np.abs(pred_v - dns_v)
    error_w = np.abs(pred_w - dns_w)

    rel_error_u = np.linalg.norm(error_u) / np.linalg.norm(dns_u)
    rel_error_v = np.linalg.norm(error_v) / np.linalg.norm(dns_v)
    rel_error_w = np.linalg.norm(error_w) / np.linalg.norm(dns_w)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(
        f"{model_name} vs DNS - Field Comparison (Time Index {time_idx})\n"
        f"L2 Relative Errors: u={rel_error_u:.4f}, v={rel_error_v:.4f}, w={rel_error_w:.4f}",
        fontsize=16,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dns_u, cmap="RdBu_r", origin="lower", aspect="equal")
    ax1.set_title("DNS: u velocity", fontsize=13, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        pred_u,
        cmap="RdBu_r",
        origin="lower",
        aspect="equal",
        vmin=dns_u.min(),
        vmax=dns_u.max(),
    )
    ax2.set_title(f"{model_name}: u velocity", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(error_u, cmap="hot", origin="lower", aspect="equal")
    ax3.set_title(f"Absolute Error: u\n(L2 rel: {rel_error_u:.4f})", fontsize=13, fontweight="bold")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(dns_v, cmap="RdBu_r", origin="lower", aspect="equal")
    ax4.set_title("DNS: v velocity", fontsize=13, fontweight="bold")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(
        pred_v,
        cmap="RdBu_r",
        origin="lower",
        aspect="equal",
        vmin=dns_v.min(),
        vmax=dns_v.max(),
    )
    ax5.set_title(f"{model_name}: v velocity", fontsize=13, fontweight="bold")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(error_v, cmap="hot", origin="lower", aspect="equal")
    ax6.set_title(f"Absolute Error: v\n(L2 rel: {rel_error_v:.4f})", fontsize=13, fontweight="bold")
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(dns_w, cmap="RdBu_r", origin="lower", aspect="equal")
    ax7.set_title("DNS: Vorticity", fontsize=13, fontweight="bold")
    ax7.set_xlabel("x")
    ax7.set_ylabel("y")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(
        pred_w,
        cmap="RdBu_r",
        origin="lower",
        aspect="equal",
        vmin=dns_w.min(),
        vmax=dns_w.max(),
    )
    ax8.set_title(f"{model_name}: Vorticity", fontsize=13, fontweight="bold")
    ax8.set_xlabel("x")
    ax8.set_ylabel("y")
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    ax9 = fig.add_subplot(gs[2, 2])
    im9 = ax9.imshow(error_w, cmap="hot", origin="lower", aspect="equal")
    ax9.set_title(
        f"Absolute Error: Vorticity\n(L2 rel: {rel_error_w:.4f})",
        fontsize=13,
        fontweight="bold",
    )
    ax9.set_xlabel("x")
    ax9.set_ylabel("y")
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"saved: {output_file}")
    return fig


def generate_field_comparison_demo(output_prefix: str):
    print("=" * 70)
    print("生成示例場對比圖（模擬數據）")
    print("=" * 70)

    grid_size = 128
    x = np.linspace(0, 2 * np.pi, grid_size)
    y = np.linspace(0, 2 * np.pi, grid_size)
    x_grid, y_grid = np.meshgrid(x, y)

    dns_u = np.sin(y_grid) + 0.3 * np.cos(2 * x_grid) * np.sin(y_grid)
    dns_v = 0.2 * np.sin(2 * x_grid) * np.cos(y_grid)
    dns_w = np.cos(x_grid) * np.sin(y_grid) - np.sin(x_grid) * np.cos(y_grid)

    pred_u_pirate = dns_u + 0.3 * np.random.randn(*dns_u.shape) * np.std(dns_u)
    pred_v_pirate = dns_v + 0.3 * np.random.randn(*dns_v.shape) * np.std(dns_v)
    pred_w_pirate = dns_w + 0.8 * np.random.randn(*dns_w.shape) * np.std(dns_w)

    pred_u_soap = dns_u + 0.05 * np.random.randn(*dns_u.shape) * np.std(dns_u)
    pred_v_soap = dns_v + 0.05 * np.random.randn(*dns_v.shape) * np.std(dns_v)
    pred_w_soap = dns_w + 0.15 * np.random.randn(*dns_w.shape) * np.std(dns_w)

    plot_field_comparison_from_arrays(
        dns_u,
        dns_v,
        dns_w,
        pred_u_pirate,
        pred_v_pirate,
        pred_w_pirate,
        time_idx=10,
        model_name="PIRATE (Adam)",
        output_file=f"{output_prefix}_pirate_demo.png",
    )

    plot_field_comparison_from_arrays(
        dns_u,
        dns_v,
        dns_w,
        pred_u_soap,
        pred_v_soap,
        pred_w_soap,
        time_idx=10,
        model_name="SOAP",
        output_file=f"{output_prefix}_soap_demo.png",
    )


def load_arrays_from_npz(npz_path: str):
    """
    讀取可視化資料。

    必要 key:
    - dns_u, dns_v, dns_w
    - pred_u, pred_v, pred_w
    可選 key:
    - time_idx, model_name
    """
    data = np.load(npz_path, allow_pickle=True)
    required = ["dns_u", "dns_v", "dns_w", "pred_u", "pred_v", "pred_w"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"missing keys in {npz_path}: {missing}")

    time_idx = int(data["time_idx"]) if "time_idx" in data else 0
    model_name = str(data["model_name"]) if "model_name" in data else "PINN"
    return data, time_idx, model_name


def main():
    parser = argparse.ArgumentParser(description="視覺化 DNS vs Prediction 場對比")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "npz"])
    parser.add_argument("--input-npz", type=str, default=None)
    parser.add_argument("--output", type=str, default="field_comparison.png")
    parser.add_argument("--output-prefix", type=str, default="field_comparison")
    args = parser.parse_args()

    if args.mode == "demo":
        generate_field_comparison_demo(args.output_prefix)
        return

    if args.input_npz is None:
        raise ValueError("--mode npz 時必須提供 --input-npz")

    data, time_idx, model_name = load_arrays_from_npz(args.input_npz)
    plot_field_comparison_from_arrays(
        data["dns_u"],
        data["dns_v"],
        data["dns_w"],
        data["pred_u"],
        data["pred_v"],
        data["pred_w"],
        time_idx=time_idx,
        model_name=model_name,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
