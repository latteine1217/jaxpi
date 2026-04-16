#!/usr/bin/env python3
"""
生成多個 checkpoint 的 vorticity snapshot grid。
"""

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_SCRIPT_PATH = Path(__file__).resolve()
if (_SCRIPT_PATH.parent / "examples").exists():
    REPO_ROOT = _SCRIPT_PATH.parent
elif (_SCRIPT_PATH.parent.parent / "examples").exists():
    REPO_ROOT = _SCRIPT_PATH.parent.parent
elif (Path.cwd() / "examples").exists():
    REPO_ROOT = Path.cwd()
else:
    REPO_ROOT = _SCRIPT_PATH.parents[min(2, len(_SCRIPT_PATH.parents) - 1)]
sys.path.insert(0, str(REPO_ROOT))

from examples.kolmogorov_flow import models
from examples.kolmogorov_flow.utils import get_dataset
from jaxpi.utils import restore_checkpoint


def load_config(config_path: Path):
    """What: 動態載入 config。 Why: 讓同一支腳本能重用於不同 checkpoint 設定。"""
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def latest_checkpoint_step(ckpt_dir: Path) -> int:
    """What: 取得最新 checkpoint step。 Why: 預設應該優先看目前最新可用結果。"""
    steps = [
        int(path.name.split("_")[1])
        for path in ckpt_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint_")
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoint_* found under {ckpt_dir}")
    return max(steps)


def to_window_local_time(t_window: np.ndarray) -> np.ndarray:
    """
    What: 將單一 window 的絕對時間軸轉成局部時間軸。
    Why: 訓練 checkpoint 在 `window 2+` 依賴的是 window-local time 定義。
    """
    t_window = np.asarray(t_window)
    return t_window - float(t_window[0])


def parse_windows(window_arg: str, available_windows: list[int]) -> list[int]:
    """What: 解析 windows 參數。 Why: 支援 `all` 與逗號分隔列表。"""
    if window_arg == "all":
        return available_windows
    windows = []
    for part in window_arg.split(","):
        part = part.strip()
        if not part:
            continue
        windows.append(int(part))
    return windows


def predict_vorticity_in_chunks(model, params, t_eval: float, coords: np.ndarray, chunk_size: int):
    """What: 分塊預測渦度。 Why: 避免高解析網格一次 forward 造成記憶體壓力。"""
    w_chunks = []
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        coords_chunk = coords[start:end]
        x_chunk = jnp.asarray(coords_chunk[:, 0])
        y_chunk = jnp.asarray(coords_chunk[:, 1])
        w_chunks.append(
            np.asarray(jax.device_get(model.w_ic_pred_fn(params, t_eval, x_chunk, y_chunk)))
        )
    return np.concatenate(w_chunks, axis=0)


def add_vorticity_triplet(
    axes,
    item: dict,
    show_ylabel: bool,
    show_colorbars: bool,
):
    """What: 畫 DNS / PINN / |Error| 三聯圖。 Why: PNG 與 GIF 共用同一套版面與色階邏輯。"""
    dns = item["dns"]
    pred = item["pred"]
    err = item["error"]
    vmax = max(np.abs(dns).max(), np.abs(pred).max())
    vmin = -vmax

    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(dns.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax0.set_title(f"DNS\n(t={item['t_abs']:.4f})")
    if show_colorbars:
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(pred.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax1.set_title(f"PINN\n(W{item['window']}, C{item['step']})")
    if show_colorbars:
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(err.T, origin="lower", cmap="hot", vmin=0, vmax=err.max(), aspect="equal")
    ax2.set_title(f"|Error|\nw_rel={item['w_rel']:.4f}")
    if show_colorbars:
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    if show_ylabel:
        ax0.set_ylabel(
            f"Window {item['window']}",
            fontsize=9,
            rotation=0,
            labelpad=26,
            va="center",
        )


def save_montage(rows: list[dict], output_path: Path, layout: str, show_colorbars: bool):
    """What: 儲存多 window montage。 Why: 支援原直向與論文/簡報用橫向比較。"""
    if layout == "horizontal":
        fig, axes = plt.subplots(3, len(rows), figsize=(3.0 * len(rows), 8.2))
        if len(rows) == 1:
            axes = axes[:, np.newaxis]
        fig.suptitle("Vorticity Snapshot Comparison", fontsize=14, fontweight="bold")
        for col_idx, item in enumerate(rows):
            add_vorticity_triplet(
                axes[:, col_idx],
                item,
                show_ylabel=(col_idx == 0),
                show_colorbars=show_colorbars,
            )
    else:
        fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.2 * len(rows)))
        if len(rows) == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle("Vorticity Snapshot Comparison", fontsize=14, fontweight="bold")
        for row_idx, item in enumerate(rows):
            add_vorticity_triplet(
                axes[row_idx, :],
                item,
                show_ylabel=True,
                show_colorbars=show_colorbars,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_window_gif(rows: list[dict], gif_path: Path, duration_ms: int, show_colorbars: bool):
    """What: 逐 window 產生 GIF。 Why: 讓 vorticity drift 可用同一張動圖掃描。"""
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = gif_path.parent / f".{gif_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    try:
        for frame_idx, item in enumerate(rows):
            frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
            fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
            fig.suptitle(
                f"Vorticity Snapshot Comparison | Window {item['window']} | "
                f"t={item['t_abs']:.4f} | ckpt={item['step']}",
                fontsize=13,
                fontweight="bold",
            )
            add_vorticity_triplet(
                axes,
                item,
                show_ylabel=False,
                show_colorbars=show_colorbars,
            )
            plt.tight_layout()
            fig.savefig(frame_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            frame_paths.append(frame_path)

        frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for frame in locals().get("frames", []):
            frame.close()
        shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Render vorticity snapshot grid.")
    parser.add_argument("--config", required=True, help="Absolute or repo-relative config path.")
    parser.add_argument("--checkpoint-root", required=True, help="Checkpoint root containing time_window_*.")
    parser.add_argument("--windows", default="all", help="`all` or comma-separated window list.")
    parser.add_argument("--time-index", type=int, default=-1, help="Index within window. Default: last.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Spatial chunk size.")
    parser.add_argument("--plot-stride", type=int, default=1, help="Subsample stride for plotting grid.")
    parser.add_argument(
        "--layout",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="PNG montage layout. Default keeps the legacy vertical layout.",
    )
    parser.add_argument(
        "--gif-output",
        default=None,
        help="Optional animated GIF output path. Frames advance across selected windows.",
    )
    parser.add_argument("--gif-duration-ms", type=int, default=900, help="GIF frame duration in ms.")
    parser.add_argument(
        "--hide-colorbars",
        action="store_true",
        help="Hide per-panel colorbars. Useful for dense horizontal montage and GIF output.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)

    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path

    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(
        time_fraction=config.time_fraction,
        dataset_path=str(dataset_path),
        time_range=config.get("dns_time_range"),
        time_stride=config.get("dns_time_stride", 1),
    )

    num_windows = int(config.training.num_time_windows)
    num_time_steps = len(t_star) // num_windows
    checkpoint_root = Path(args.checkpoint_root)
    available_windows = sorted(
        int(path.name.split("_")[-1])
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.startswith("time_window_")
    )
    windows = parse_windows(args.windows, available_windows)

    grid_size = int(np.sqrt(coords.shape[0]))
    if grid_size * grid_size != coords.shape[0]:
        raise ValueError(f"coords size {coords.shape[0]} is not a square grid")

    stride = max(1, int(args.plot_stride))
    coords_grid = np.asarray(coords).reshape(grid_size, grid_size, 2)
    coords_sub = coords_grid[::stride, ::stride, :].reshape(-1, 2)

    rows = []
    for window_idx in windows:
        start_idx = (window_idx - 1) * num_time_steps
        end_idx = window_idx * num_time_steps
        if start_idx >= len(t_star) or end_idx > len(t_star):
            continue

        t_window = t_star[start_idx:end_idx]
        t_window_local = to_window_local_time(t_window)

        local_time_idx = args.time_index
        if local_time_idx < 0:
            local_time_idx = len(t_window) - 1
        if local_time_idx >= len(t_window):
            raise ValueError(
                f"time-index {local_time_idx} out of range for window length {len(t_window)}"
            )

        global_time_idx = start_idx + local_time_idx
        t_eval_abs = float(t_window[local_time_idx])
        t_eval_local = float(t_window_local[local_time_idx])

        u0_full = np.asarray(u_ref[start_idx, :]).reshape(grid_size, grid_size)
        v0_full = np.asarray(v_ref[start_idx, :]).reshape(grid_size, grid_size)
        w0_full = np.asarray(w_ref[start_idx, :]).reshape(grid_size, grid_size)
        w_dns_full = np.asarray(w_ref[global_time_idx, :]).reshape(grid_size, grid_size)

        u0 = u0_full[::stride, ::stride].reshape(-1)
        v0 = v0_full[::stride, ::stride].reshape(-1)
        w0 = w0_full[::stride, ::stride].reshape(-1)
        w_dns = w_dns_full[::stride, ::stride]

        model = models.NavierStokes(
            config,
            t_window_local,
            coords_sub,
            u0,
            v0,
            w0,
            nu,
            replicate_state=False,
        )
        ckpt_dir = checkpoint_root / f"time_window_{window_idx}"
        step = latest_checkpoint_step(ckpt_dir)
        model.state = restore_checkpoint(model.state, str(ckpt_dir), step=step)

        w_pred = predict_vorticity_in_chunks(
            model,
            model.state.params,
            t_eval_local,
            np.asarray(coords_sub),
            int(args.chunk_size),
        ).reshape(coords_grid[::stride, ::stride, 0].shape)

        rows.append(
            {
                "window": window_idx,
                "step": step,
                "t_abs": t_eval_abs,
                "dns": w_dns,
                "pred": w_pred,
                "error": np.abs(w_pred - w_dns),
                "w_rel": float(np.linalg.norm(w_pred - w_dns) / np.linalg.norm(w_dns)),
            }
        )
        print(
            f"window={window_idx} step={step} t_abs={t_eval_abs:.4f} "
            f"w_rel={rows[-1]['w_rel']:.6f}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No rows rendered.")

    output_path = Path(args.output)
    save_montage(
        rows=rows,
        output_path=output_path,
        layout=args.layout,
        show_colorbars=not args.hide_colorbars,
    )
    print(f"saved: {output_path}")

    if args.gif_output:
        gif_path = Path(args.gif_output)
        save_window_gif(
            rows=rows,
            gif_path=gif_path,
            duration_ms=int(args.gif_duration_ms),
            show_colorbars=not args.hide_colorbars,
        )
        print(f"saved: {gif_path}")


if __name__ == "__main__":
    main()
