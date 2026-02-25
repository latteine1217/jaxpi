#!/usr/bin/env python3
"""
生成 DNS vs PINN Prediction vs Error Field 視覺化
從 checkpoint 載入模型並生成真實的場對比圖
使用批次處理避免記憶體溢出
"""

# 強制使用 CPU 模式，避免 GPU cuDNN 問題導致掛起
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加專案路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, 'examples', 'kolmogorov_flow')
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

print("載入 JAX...")
import jax.numpy as jnp
from jax import random

print("載入專案模組...")
from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset
from examples.kolmogorov_flow import models


def generate_field_comparison(config_name, checkpoint_path, time_window, time_step_idx,
                              output_dir='.', batch_size=4096):
    """
    生成指定時間窗口和時間步的場對比圖

    Args:
        config_name: 'soap' 或 'pirate'
        checkpoint_path: checkpoint 根目錄
        time_window: 時間窗口編號
        time_step_idx: 窗口內的時間步索引（0 到 num_time_steps-1，-1 表示最後一步）
        output_dir: 輸出目錄
        batch_size: 批次處理大小（默認 4096）
    """

    print("=" * 80)
    print(f"生成場對比圖: {config_name.upper()} - Window {time_window}")
    print("=" * 80)
    print()

    # 載入配置
    print(f"載入配置: {config_name}")
    if config_name == 'soap':
        from configs import soap as config_module
    elif config_name == 'pirate':
        from configs import pirate as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")

    config = config_module.get_config()
    print(f"✓ 配置載入完成")
    print()

    # 載入 DNS 數據
    print("載入 DNS 參考數據...")
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(time_fraction=config.time_fraction)

    print(f"DNS 數據形狀:")
    print(f"  - 時間步數: {len(t_star)}")
    print(f"  - 空間點數: {coords.shape[0]}")
    print(f"  - 網格大小: {int(np.sqrt(coords.shape[0]))}×{int(np.sqrt(coords.shape[0]))}")
    print()

    # 計算時間窗口
    num_time_steps = len(t_star) // config.training.num_time_windows
    start_idx = (time_window - 1) * num_time_steps
    end_idx = time_window * num_time_steps

    # 處理 time_step_idx = -1 的情況
    if time_step_idx == -1:
        time_step_idx = num_time_steps - 1
        print(f"使用窗口最後一個時間步: {time_step_idx}")

    # 檢查 time_step_idx 是否有效
    if time_step_idx >= num_time_steps or time_step_idx < 0:
        raise ValueError(f"time_step_idx {time_step_idx} 超出範圍 [0, {num_time_steps-1}]")

    # 獲取當前時間步的數據
    global_time_idx = start_idx + time_step_idx
    t_current = t_star[global_time_idx]

    print(f"目標時間: t = {t_current:.6f}")
    print(f"  - Window {time_window}, Step {time_step_idx}/{num_time_steps-1}")
    print(f"  - Global time index: {global_time_idx}/{len(t_star)-1}")
    print()

    # 提取窗口數據
    t_window = t_star[start_idx:end_idx]
    u0 = u_ref[start_idx, :]
    v0 = v_ref[start_idx, :]
    w0 = w_ref[start_idx, :]

    # 初始化模型
    print("初始化模型...")
    model = models.NavierStokes(config, t_window, coords, u0, v0, w0, nu)
    print("✓ 模型初始化完成")
    print()

    # 載入 checkpoint
    ckpt_dir = os.path.join(checkpoint_path, f'time_window_{time_window}')

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_dir}")

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_')]
    if not ckpt_files:
        raise FileNotFoundError(f"沒有找到 checkpoint 文件: {ckpt_dir}")

    steps = [int(f.split('_')[1]) for f in ckpt_files]
    max_step = max(steps)

    print(f"載入 checkpoint: {ckpt_dir}/checkpoint_{max_step}")
    model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)
    print(f"✓ Checkpoint 載入成功 (step {max_step})")
    print()

    # 批次處理預測（避免記憶體溢出）
    print(f"生成預測（批次處理，batch_size={batch_size}）...")
    n_points = coords.shape[0]
    n_batches = (n_points + batch_size - 1) // batch_size

    u_pred_list = []
    v_pred_list = []
    w_pred_list = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_points)
        batch_coords = coords[start:end]

        t_array = jnp.full((batch_coords.shape[0],), t_current)

        # 預測
        u_batch = np.array(model.u_pred_fn(model.state.params, t_array,
                                           batch_coords[:, 0], batch_coords[:, 1]))
        v_batch = np.array(model.v_pred_fn(model.state.params, t_array,
                                           batch_coords[:, 0], batch_coords[:, 1]))
        w_batch = np.array(model.w_pred_fn(model.state.params, t_array,
                                           batch_coords[:, 0], batch_coords[:, 1]))

        u_pred_list.append(u_batch)
        v_pred_list.append(v_batch)
        w_pred_list.append(w_batch)

        if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
            print(f"  批次 {i+1}/{n_batches} 完成 ({100*(i+1)/n_batches:.1f}%)")

    # 合併結果
    u_pred = np.concatenate(u_pred_list, axis=0)
    v_pred = np.concatenate(v_pred_list, axis=0)
    w_pred = np.concatenate(w_pred_list, axis=0)

    print(f"✓ 預測完成: shape={u_pred.shape}")
    print()

    # 獲取 DNS 參考數據
    u_dns = np.array(u_ref[global_time_idx, :])
    v_dns = np.array(v_ref[global_time_idx, :])
    w_dns = np.array(w_ref[global_time_idx, :])

    # 計算 L2 相對誤差
    print("計算 L2 相對誤差...")
    u_error = np.linalg.norm(u_pred - u_dns) / np.linalg.norm(u_dns)
    v_error = np.linalg.norm(v_pred - v_dns) / np.linalg.norm(v_dns)
    w_error = np.linalg.norm(w_pred - w_dns) / np.linalg.norm(w_dns)

    print(f"L2 相對誤差:")
    print(f"  - u_error: {u_error:.6f}")
    print(f"  - v_error: {v_error:.6f}")
    print(f"  - w_error: {w_error:.6f}")
    print()

    # 重塑為 2D 網格
    grid_size = int(np.sqrt(coords.shape[0]))
    u_dns_2d = u_dns.reshape(grid_size, grid_size)
    v_dns_2d = v_dns.reshape(grid_size, grid_size)
    w_dns_2d = w_dns.reshape(grid_size, grid_size)

    u_pred_2d = u_pred.reshape(grid_size, grid_size)
    v_pred_2d = v_pred.reshape(grid_size, grid_size)
    w_pred_2d = w_pred.reshape(grid_size, grid_size)

    # 繪製對比圖
    print("繪製場對比圖...")
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    model_name = config_name.upper()
    fig.suptitle(f'{model_name} vs DNS - Field Comparison\n'
                 f'Time Window {time_window}, t = {t_current:.6f}\n'
                 f'L2 Relative Errors: u={u_error:.4f}, v={v_error:.4f}, w={w_error:.4f}',
                 fontsize=16, fontweight='bold')

    # === 第一行: u 速度分量 ===
    # DNS
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(u_dns_2d, cmap='RdBu_r', origin='lower', aspect='equal')
    ax1.set_title('DNS: u velocity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(u_pred_2d, cmap='RdBu_r', origin='lower', aspect='equal',
                     vmin=u_dns_2d.min(), vmax=u_dns_2d.max())
    ax2.set_title(f'{model_name}: u velocity', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Error
    ax3 = fig.add_subplot(gs[0, 2])
    error_u_2d = np.abs(u_pred_2d - u_dns_2d)
    im3 = ax3.imshow(error_u_2d, cmap='hot', origin='lower', aspect='equal')
    ax3.set_title(f'Absolute Error: u\n(L2 rel: {u_error:.4f})',
                  fontsize=13, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # === 第二行: v 速度分量 ===
    # DNS
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(v_dns_2d, cmap='RdBu_r', origin='lower', aspect='equal')
    ax4.set_title('DNS: v velocity', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Prediction
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(v_pred_2d, cmap='RdBu_r', origin='lower', aspect='equal',
                     vmin=v_dns_2d.min(), vmax=v_dns_2d.max())
    ax5.set_title(f'{model_name}: v velocity', fontsize=13, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Error
    ax6 = fig.add_subplot(gs[1, 2])
    error_v_2d = np.abs(v_pred_2d - v_dns_2d)
    im6 = ax6.imshow(error_v_2d, cmap='hot', origin='lower', aspect='equal')
    ax6.set_title(f'Absolute Error: v\n(L2 rel: {v_error:.4f})',
                  fontsize=13, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # === 第三行: 渦度 ===
    # DNS
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(w_dns_2d, cmap='RdBu_r', origin='lower', aspect='equal')
    ax7.set_title('DNS: Vorticity', fontsize=13, fontweight='bold')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # Prediction
    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(w_pred_2d, cmap='RdBu_r', origin='lower', aspect='equal',
                     vmin=w_dns_2d.min(), vmax=w_dns_2d.max())
    ax8.set_title(f'{model_name}: Vorticity', fontsize=13, fontweight='bold')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # Error
    ax9 = fig.add_subplot(gs[2, 2])
    error_w_2d = np.abs(w_pred_2d - w_dns_2d)
    im9 = ax9.imshow(error_w_2d, cmap='hot', origin='lower', aspect='equal')
    ax9.set_title(f'Absolute Error: Vorticity\n(L2 rel: {w_error:.4f})',
                  fontsize=13, fontweight='bold')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    # 保存圖表
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir,
                               f'field_comparison_{config_name}_window{time_window}_step{time_step_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 場對比圖已保存: {output_file}")
    print()

    plt.close()

    return {
        'u_error': float(u_error),
        'v_error': float(v_error),
        'w_error': float(w_error),
        'time': float(t_current),
        'output_file': output_file
    }


def main():
    parser = argparse.ArgumentParser(description='生成 DNS vs PINN 場對比圖（批次處理版本）')
    parser.add_argument('--config', type=str, required=True, choices=['soap', 'pirate'],
                        help='配置文件名稱')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Checkpoint 根目錄路徑')
    parser.add_argument('--window', type=int, required=True,
                        help='時間窗口編號')
    parser.add_argument('--time_step', type=int, default=-1,
                        help='窗口內的時間步索引（默認 -1 表示最後一步）')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='輸出目錄')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='批次處理大小（默認 4096）')

    args = parser.parse_args()

    # 生成場對比圖
    result = generate_field_comparison(
        args.config,
        args.checkpoint_path,
        args.window,
        args.time_step,
        args.output_dir,
        args.batch_size
    )

    print("=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
