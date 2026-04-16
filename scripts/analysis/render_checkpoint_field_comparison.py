#!/usr/bin/env python3
"""
生成單一 checkpoint 的 DNS vs Prediction vs Error 九宮格比較圖。
"""

import argparse
import importlib.util
import os
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
from scripts.analysis.plot_field_comparison import plot_field_comparison_from_arrays


def load_config(config_path: Path):
    """What: 動態載入 config 檔。 Why: 讓腳本可直接重用於不同實驗。"""
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def latest_checkpoint_step(ckpt_dir: Path) -> int:
    """What: 回傳最新 checkpoint step。 Why: 預設行為應該直接看最新可用結果。"""
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
    What: 將單一 window 的絕對時間轉成以 0 起算的局部時間。
    Why: `window 2+` 的 checkpoint 是在 window-local time 定義下訓練；
        視覺化若直接使用 absolute time，會和訓練條件不一致。
    """
    t_window = np.asarray(t_window)
    return t_window - float(t_window[0])


def predict_field_in_chunks(model, params, t_eval: float, coords: np.ndarray, chunk_size: int):
    """What: 分塊預測 u/v/w。 Why: 避免一次對整個高解析網格做 forward / autodiff 造成 OOM。"""
    u_chunks = []
    v_chunks = []
    w_chunks = []

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        coords_chunk = coords[start:end]
        x_chunk = jnp.asarray(coords_chunk[:, 0])
        y_chunk = jnp.asarray(coords_chunk[:, 1])

        u_chunks.append(
            np.asarray(jax.device_get(model.u_ic_pred_fn(params, t_eval, x_chunk, y_chunk)))
        )
        v_chunks.append(
            np.asarray(jax.device_get(model.v_ic_pred_fn(params, t_eval, x_chunk, y_chunk)))
        )
        w_chunks.append(
            np.asarray(jax.device_get(model.w_ic_pred_fn(params, t_eval, x_chunk, y_chunk)))
        )

    return (
        np.concatenate(u_chunks, axis=0),
        np.concatenate(v_chunks, axis=0),
        np.concatenate(w_chunks, axis=0),
    )


def predict_velocity_in_chunks(model, params, t_eval: float, coords: np.ndarray, chunk_size: int):
    """What: 分塊預測 u/v。 Why: 視覺化時可用有限差分近似渦度，降低計算成本。"""
    u_chunks = []
    v_chunks = []

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        coords_chunk = coords[start:end]
        x_chunk = jnp.asarray(coords_chunk[:, 0])
        y_chunk = jnp.asarray(coords_chunk[:, 1])

        u_chunks.append(
            np.asarray(jax.device_get(model.u_ic_pred_fn(params, t_eval, x_chunk, y_chunk)))
        )
        v_chunks.append(
            np.asarray(jax.device_get(model.v_ic_pred_fn(params, t_eval, x_chunk, y_chunk)))
        )

    return np.concatenate(u_chunks, axis=0), np.concatenate(v_chunks, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Render DNS vs Prediction field comparison.")
    parser.add_argument("--config", required=True, help="Absolute or repo-relative config path.")
    parser.add_argument("--checkpoint-root", required=True, help="Checkpoint root containing time_window_*.")
    parser.add_argument("--window", type=int, required=True, help="1-based time window index.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Default: latest.")
    parser.add_argument("--time-index", type=int, default=-1, help="Index within window. Default: last.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Spatial chunk size.")
    parser.add_argument("--plot-stride", type=int, default=1, help="Subsample stride for plotting grid.")
    parser.add_argument(
        "--approx-vorticity",
        action="store_true",
        help="Approximate vorticity from predicted u/v using finite differences on the plotting grid.",
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
    window_idx = int(args.window)
    if window_idx < 1 or window_idx > num_windows:
        raise ValueError(f"window must be in [1, {num_windows}]")

    start_idx = (window_idx - 1) * num_time_steps
    end_idx = window_idx * num_time_steps
    t_window = t_star[start_idx:end_idx]
    t_window_local = to_window_local_time(t_window)

    local_time_idx = args.time_index
    if local_time_idx < 0:
        local_time_idx = len(t_window) - 1
    if local_time_idx >= len(t_window):
        raise ValueError(f"time-index {local_time_idx} out of range for window length {len(t_window)}")

    global_time_idx = start_idx + local_time_idx
    t_eval_abs = float(t_window[local_time_idx])
    t_eval_local = float(t_window_local[local_time_idx])

    grid_size = int(np.sqrt(coords.shape[0]))
    if grid_size * grid_size != coords.shape[0]:
        raise ValueError(f"coords size {coords.shape[0]} is not a square grid")

    stride = max(1, int(args.plot_stride))
    coords_grid = np.asarray(coords).reshape(grid_size, grid_size, 2)

    u0_full = np.asarray(u_ref[start_idx, :]).reshape(grid_size, grid_size)
    v0_full = np.asarray(v_ref[start_idx, :]).reshape(grid_size, grid_size)
    w0_full = np.asarray(w_ref[start_idx, :]).reshape(grid_size, grid_size)

    u_dns_full = np.asarray(u_ref[global_time_idx, :]).reshape(grid_size, grid_size)
    v_dns_full = np.asarray(v_ref[global_time_idx, :]).reshape(grid_size, grid_size)
    w_dns_full = np.asarray(w_ref[global_time_idx, :]).reshape(grid_size, grid_size)

    coords_sub = coords_grid[::stride, ::stride, :].reshape(-1, 2)
    u0 = u0_full[::stride, ::stride].reshape(-1)
    v0 = v0_full[::stride, ::stride].reshape(-1)
    w0 = w0_full[::stride, ::stride].reshape(-1)
    u_dns = u_dns_full[::stride, ::stride].reshape(-1)
    v_dns = v_dns_full[::stride, ::stride].reshape(-1)
    w_dns = w_dns_full[::stride, ::stride].reshape(-1)

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
    ckpt_dir = Path(args.checkpoint_root) / f"time_window_{window_idx}"
    step = args.step if args.step is not None else latest_checkpoint_step(ckpt_dir)
    model.state = restore_checkpoint(model.state, str(ckpt_dir), step=step)

    params = model.state.params
    coords_np = np.asarray(coords_sub)
    if args.approx_vorticity:
        u_pred, v_pred = predict_velocity_in_chunks(
            model, params, t_eval_local, coords_np, int(args.chunk_size)
        )
        subgrid_shape = coords_grid[::stride, ::stride, 0].shape
        u_pred_2d = u_pred.reshape(subgrid_shape)
        v_pred_2d = v_pred.reshape(subgrid_shape)
        x_sub = coords_grid[::stride, ::stride, 0]
        y_sub = coords_grid[::stride, ::stride, 1]
        dx = float(np.mean(np.diff(x_sub[:, 0]))) if x_sub.shape[0] > 1 else 1.0
        dy = float(np.mean(np.diff(y_sub[0, :]))) if y_sub.shape[1] > 1 else 1.0
        dv_dx = np.gradient(v_pred_2d, dx, axis=0, edge_order=2)
        du_dy = np.gradient(u_pred_2d, dy, axis=1, edge_order=2)
        w_pred = (dv_dx - du_dy).reshape(-1)
    else:
        u_pred, v_pred, w_pred = predict_field_in_chunks(
            model, params, t_eval_local, coords_np, int(args.chunk_size)
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_field_comparison_from_arrays(
        dns_u=u_dns,
        dns_v=v_dns,
        dns_w=w_dns,
        pred_u=u_pred,
        pred_v=v_pred,
        pred_w=w_pred,
        time_idx=(
            f"window {window_idx}, "
            f"t_abs={t_eval_abs:.4f}, t_local={t_eval_local:.4f}, "
            f"ckpt={step}, stride={stride}"
        ),
        model_name="SOAP",
        output_file=str(output_path),
    )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
