#!/usr/bin/env python3
"""
檢查 Kolmogorov Flow DNS 資料並輸出基本可視化。
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def load_dns_data(data_path):
    print(f"正在載入資料: {data_path}")
    data = np.load(data_path, allow_pickle=True).item()

    config = data.get("config", {})
    nu = float(config.get("nu")) if config is not None and config.get("nu") is not None else None

    t = np.array(data["time"]).flatten()
    u = np.array(data["u"])
    v = np.array(data["v"])
    omega = np.array(data["omega"])

    x_vals = np.array(data["x"])
    y_vals = np.array(data["y"])
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
    coords = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    return {
        "t": t,
        "u": u,
        "v": v,
        "omega": omega,
        "coords": coords,
        "nu": nu,
    }


def reshape_fields(t, u, v, omega, coords):
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    nx, ny = len(unique_x), len(unique_y)

    nt = len(t)
    return (
        u.reshape(nt, nx, ny),
        v.reshape(nt, nx, ny),
        omega.reshape(nt, nx, ny),
        nx,
        ny,
    )


def generate_vorticity_gif(t, vorticity, re_value, output_path, subsample):
    frames = list(range(0, len(t), max(1, subsample)))
    vmin_global = float(vorticity.min())
    vmax_global = float(vorticity.max())

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    im = ax.imshow(
        vorticity[0].T,
        cmap="RdBu_r",
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        vmin=vmin_global,
        vmax=vmax_global,
    )
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Vorticity", fontsize=12)

    def update(frame_idx):
        frame = frames[frame_idx]
        im.set_data(vorticity[frame].T)
        ax.set_title(
            f"Vorticity Field - Re={re_value:.0f} - t={t[frame]:.4f}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="檢查 DNS 資料並輸出 vorticity GIF")
    parser.add_argument(
        "--data-path",
        type=str,
        default="examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy",
    )
    parser.add_argument("--output-dir", type=str, default="examples/kolmogorov_flow/visualization")
    parser.add_argument("--subsample", type=int, default=2)
    args = parser.parse_args()

    data = load_dns_data(args.data_path)
    t = data["t"]
    u = data["u"]
    v = data["v"]
    omega = data["omega"]
    coords = data["coords"]

    re_value = (1.0 / data["nu"]) if data["nu"] else 0.0

    print("=" * 60)
    print("DNS 數據資訊")
    print("=" * 60)
    print(f"nu: {data['nu']}")
    print(f"Re: {re_value:.1f}")
    print(f"time steps: {len(t)}")
    print(f"time range: [{t[0]:.4f}, {t[-1]:.4f}]")
    print(f"avg dt: {np.mean(np.diff(t)):.6f}")
    print(f"u shape(raw): {u.shape}")
    print(f"v shape(raw): {v.shape}")
    print(f"omega shape(raw): {omega.shape}")
    print(f"coords shape: {coords.shape}")

    u, v, omega, nx, ny = reshape_fields(t, u, v, omega, coords)
    print(f"grid: {nx} x {ny}")
    print(f"u shape(reshaped): {u.shape}")
    print(f"v shape(reshaped): {v.shape}")
    print(f"omega shape(reshaped): {omega.shape}")

    print("=" * 60)
    print("統計")
    print("=" * 60)
    print(f"u: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}, std={u.std():.4f}")
    print(f"v: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.4f}")
    print(
        f"omega: min={omega.min():.4f}, max={omega.max():.4f}, "
        f"mean={omega.mean():.4f}, std={omega.std():.4f}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    gif_path = os.path.join(args.output_dir, "vorticity_animation.gif")
    print(f"生成 GIF: {gif_path}")
    generate_vorticity_gif(t, omega, re_value, gif_path, args.subsample)
    print("完成")


if __name__ == "__main__":
    main()
