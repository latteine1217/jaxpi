"""
使用 CPU 評估 checkpoint（避免 GPU 記憶體問題）
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # 強制使用 CPU

import sys
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random

# 添加專案路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset
from examples.kolmogorov_flow import models


def evaluate_final_step_cpu(config_name, checkpoint_path):
    """
    使用 CPU 評估所有窗口的最終時間步誤差
    """

    # 載入配置
    if config_name == "soap":
        from configs import soap as config_module
    elif config_name == "pirate":
        from configs import pirate as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")

    config = config_module.get_config()

    # 載入 DNS 數據
    print("載入 DNS 參考數據...")
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(time_fraction=config.time_fraction)

    print(f"DNS 數據形狀:")
    print(f"  - 時間步數: {len(t_star)}")
    print(f"  - 空間點數: {coords.shape[0]}")
    print(f"  - 雷諾數: Re = {1 / nu:.0f}")
    print()

    # 計算時間窗口
    num_time_steps = len(t_star) // config.training.num_time_windows

    # 找到所有可用的窗口
    window_dirs = sorted([d for d in os.listdir(checkpoint_path) if d.startswith("time_window_")])
    time_windows = [int(d.split("_")[-1]) for d in window_dirs]

    print(f"找到 {len(time_windows)} 個時間窗口")
    print(f"{'=' * 80}")
    print(f"{'Window':<10} {'時間':<20} {'u_error':<15} {'v_error':<15} {'w_error':<15}")
    print(f"{'-' * 80}")

    results = []

    for window_idx in time_windows:
        try:
            # 確定當前窗口的時間範圍
            start_idx = (window_idx - 1) * num_time_steps
            end_idx = window_idx * num_time_steps

            t = t_star[start_idx:end_idx]
            t_final = t[-1]  # 只取最後一個時間步

            u_ref_final = u_ref[end_idx - 1, :]
            v_ref_final = v_ref[end_idx - 1, :]
            w_ref_final = w_ref[end_idx - 1, :]

            # 初始化模型 - 只用2個時間步以減少記憶體
            t_small = jnp.array([t[0], t[-1]])
            u0 = u_ref[start_idx, :]
            v0 = v_ref[start_idx, :]
            w0 = w_ref[start_idx, :]

            model = models.NavierStokes(config, t_small, coords, u0, v0, w0, nu)

            # 載入 checkpoint
            ckpt_dir = os.path.join(checkpoint_path, f"time_window_{window_idx}")

            if not os.path.exists(ckpt_dir):
                print(f"{window_idx:<10} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                continue

            # 查找最新的 checkpoint
            ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")]
            if not ckpt_files:
                print(f"{window_idx:<10} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                continue

            steps = [int(f.split("_")[1]) for f in ckpt_files]
            max_step = max(steps)

            model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)

            # 計算最終時間步的誤差（chunked）
            t_array = jnp.full((coords.shape[0],), t_final)
            chunk_seconds = getattr(config.logging, "eval_time_chunk_seconds", 1.0)
            u_error, v_error, w_error = model.compute_l2_error_time_chunked(
                model.state.params,
                t_array,
                coords,
                u_ref_final[None, :],
                v_ref_final[None, :],
                w_ref_final[None, :],
                chunk_seconds=chunk_seconds,
            )

            u_error = float(u_error)
            v_error = float(v_error)
            w_error = float(w_error)

            print(
                f"{window_idx:<10} {t_final:<20.4f} {u_error:<15.6f} {v_error:<15.6f} {w_error:<15.6f}"
            )

            results.append(
                {
                    "window": window_idx,
                    "time": float(t_final),
                    "u_error": u_error,
                    "v_error": v_error,
                    "w_error": w_error,
                }
            )

        except Exception as e:
            print(f"{window_idx:<10} ERROR: {str(e)[:40]}")
            continue

    print(f"{'=' * 80}")

    # 統計
    if results:
        u_errors = np.array([r["u_error"] for r in results])
        v_errors = np.array([r["v_error"] for r in results])
        w_errors = np.array([r["w_error"] for r in results])

        print()
        print("統計摘要:")
        print(
            f"  u_error: mean={u_errors.mean():.6f}, min={u_errors.min():.6f}, max={u_errors.max():.6f}"
        )
        print(
            f"  v_error: mean={v_errors.mean():.6f}, min={v_errors.min():.6f}, max={v_errors.max():.6f}"
        )
        print(
            f"  w_error: mean={w_errors.mean():.6f}, min={w_errors.min():.6f}, max={w_errors.max():.6f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="使用 CPU 評估所有窗口的最終誤差")
    parser.add_argument(
        "--config", type=str, required=True, choices=["soap", "pirate"], help="配置文件名稱"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint 根目錄路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出結果文件路徑（.npz 格式）")

    args = parser.parse_args()

    # 評估
    results = evaluate_final_step_cpu(args.config, args.checkpoint_path)

    # 保存結果
    if args.output:
        print(f"\n保存結果到: {args.output}")
        np.savez(args.output, results=results)
        print("✓ 結果已保存")


if __name__ == "__main__":
    main()
