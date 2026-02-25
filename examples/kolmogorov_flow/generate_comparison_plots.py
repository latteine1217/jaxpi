"""
生成 DNS vs PINN 预测的对比图
包含：Rel. L2 error, Kinetic energy, Enstrophy, Energy spectrum
优化内存使用，采样间隔 0.5 秒
"""

import os
import sys
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加专案路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset
from examples.kolmogorov_flow import models


def compute_kinetic_energy(u, v):
    """计算动能 (平均)"""
    return 0.5 * float(jnp.mean(u**2 + v**2))


def compute_enstrophy(w):
    """计算涡度拟能 (平均)"""
    return 0.5 * float(jnp.mean(w**2))


def compute_energy_spectrum_2d(field, nx, ny):
    """
    计算2D能量谱
    
    Args:
        field: 2D场 (nx, ny)
        nx, ny: 网格尺寸
    
    Returns:
        k: 波数
        E_k: 能量谱
    """
    # 2D FFT
    field_fft = np.fft.fft2(field)
    field_fft = np.fft.fftshift(field_fft)
    
    # 计算能量密度
    energy_density = np.abs(field_fft)**2
    
    # 创建波数网格 (使用实际物理波数)
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=2*np.pi/nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=2*np.pi/ny))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    
    # 将能量按波数分箱
    k_max = np.max(K)
    k_bins = np.linspace(1, k_max, min(50, int(k_max)))  # 最多50个bins
    E_k = []
    k_out = []
    
    for i in range(len(k_bins) - 1):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        n_points = np.sum(mask)
        if n_points > 0:
            # 能量谱：每个shell的总能量
            E_k.append(np.sum(energy_density[mask]))
            k_out.append((k_bins[i] + k_bins[i+1]) / 2)
    
    E_k = np.array(E_k)
    k_out = np.array(k_out)
    
    # 归一化到总能量
    if np.sum(E_k) > 0:
        E_k = E_k / np.sum(E_k)
    
    return k_out, E_k


def evaluate_and_compare(config_name, checkpoint_path, output_dir, time_interval=0.5):
    """
    评估checkpoint并生成对比图
    
    Args:
        time_interval: 采样时间间隔（秒）
    """
    
    # 载入配置
    if config_name == 'soap':
        from configs import soap as config_module
    elif config_name == 'pirate':
        from configs import pirate as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = config_module.get_config()
    
    # 载入 DNS 数据
    print("载入 DNS 参考数据...")
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(time_fraction=config.time_fraction)
    
    print(f"DNS 数据信息:")
    print(f"  - 时间步数: {len(t_star)}")
    print(f"  - 时间范围: [{t_star[0]:.4f}, {t_star[-1]:.4f}]")
    print(f"  - 空间点数: {coords.shape[0]}")
    print(f"  - 雷诺数: Re = {1/nu:.0f}")
    
    # 推断网格大小
    nx = int(np.sqrt(coords.shape[0]))
    ny = nx
    print(f"  - 网格大小: {nx} x {ny}")
    print(f"  - 采样间隔: {time_interval} 秒")
    print()
    
    # 计算时间窗口
    num_time_steps = len(t_star) // config.training.num_time_windows
    
    # 找到所有可用的窗口
    window_dirs = sorted([d for d in os.listdir(checkpoint_path) if d.startswith('time_window_')])
    time_windows = sorted([int(d.split('_')[-1]) for d in window_dirs])
    
    print(f"找到 {len(time_windows)} 个时间窗口: {time_windows}")
    print("=" * 80)
    print()
    
    # 存储所有时间步的结果
    all_times = []
    all_errors_u = []
    all_errors_v = []
    all_errors_w = []
    all_ke_ref = []
    all_ke_pred = []
    all_ens_ref = []
    all_ens_pred = []
    
    # 最后一个时间步的能量谱
    last_spectrum_k = None
    last_spectrum_ref = None
    last_spectrum_pred = None
    
    for window_idx in time_windows:
        print(f"处理 Time Window {window_idx}...")
        
        # 确定当前窗口的时间范围
        start_idx = (window_idx - 1) * num_time_steps
        end_idx = window_idx * num_time_steps
        
        t = t_star[start_idx:end_idx]
        u_ref_window = u_ref[start_idx:end_idx, :]
        v_ref_window = v_ref[start_idx:end_idx, :]
        w_ref_window = w_ref[start_idx:end_idx, :]
        
        # 初始化模型
        u0 = u_ref[start_idx, :]
        v0 = v_ref[start_idx, :]
        w0 = w_ref[start_idx, :]
        
        model = models.NavierStokes(config, t, coords, u0, v0, w0, nu)
        
        # 载入 checkpoint
        ckpt_dir = os.path.join(checkpoint_path, f'time_window_{window_idx}')
        
        if not os.path.exists(ckpt_dir):
            print(f"  ⚠️  Checkpoint 不存在，跳过")
            continue
        
        # 查找最新的 checkpoint
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_')]
        if not ckpt_files:
            print(f"  ⚠️  没有找到 checkpoint 文件，跳过")
            continue
        
        steps = [int(f.split('_')[1]) for f in ckpt_files]
        max_step = max(steps)
        
        model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)
        print(f"  ✓ 载入 checkpoint (step {max_step})")
        
        # 确定采样的时间步索引
        t_window_start = t[0]
        sample_indices = []
        for i, t_i in enumerate(t):
            if i == 0 or (t_i - t_window_start) >= len(sample_indices) * time_interval:
                sample_indices.append(i)
        
        print(f"  采样 {len(sample_indices)}/{len(t)} 个时间步")
        
        # 计算每个采样时间步的指标
        for idx, i in enumerate(sample_indices):
            t_i = t[i]
            
            # 创建时间数组
            t_array = jnp.full((coords.shape[0],), t_i)
            
            # 获取参考数据
            u_ref_i = u_ref_window[i, :]
            v_ref_i = v_ref_window[i, :]
            w_ref_i = w_ref_window[i, :]
            
            # 获取预测值
            # 模型的pred_fn需要分别传入x和y坐标
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            
            # 使用单个时间点的预测函数
            u_pred = model.u_ic_pred_fn(model.state.params, t_i, x_coords, y_coords)
            v_pred = model.v_ic_pred_fn(model.state.params, t_i, x_coords, y_coords)
            w_pred = model.w_ic_pred_fn(model.state.params, t_i, x_coords, y_coords)
            
            # 计算相对L2误差
            u_error = float(jnp.linalg.norm(u_pred - u_ref_i) / jnp.linalg.norm(u_ref_i))
            v_error = float(jnp.linalg.norm(v_pred - v_ref_i) / jnp.linalg.norm(v_ref_i))
            w_error = float(jnp.linalg.norm(w_pred - w_ref_i) / jnp.linalg.norm(w_ref_i))
            
            # 计算动能和涡度拟能
            ke_ref = compute_kinetic_energy(u_ref_i, v_ref_i)
            ke_pred = compute_kinetic_energy(u_pred, v_pred)
            ens_ref = compute_enstrophy(w_ref_i)
            ens_pred = compute_enstrophy(w_pred)
            
            # 存储结果
            all_times.append(float(t_i))
            all_errors_u.append(u_error)
            all_errors_v.append(v_error)
            all_errors_w.append(w_error)
            all_ke_ref.append(ke_ref)
            all_ke_pred.append(ke_pred)
            all_ens_ref.append(ens_ref)
            all_ens_pred.append(ens_pred)
            
            # 如果是最后一个窗口的最后一个采样点，计算能量谱
            if window_idx == time_windows[-1] and idx == len(sample_indices) - 1:
                print(f"  计算最终能量谱 (t={t_i:.4f})...")
                # 重塑为2D
                w_ref_2d = np.array(w_ref_i).reshape(nx, ny)
                w_pred_2d = np.array(w_pred).reshape(nx, ny)
                
                k_ref, E_ref = compute_energy_spectrum_2d(w_ref_2d, nx, ny)
                k_pred, E_pred = compute_energy_spectrum_2d(w_pred_2d, nx, ny)
                
                last_spectrum_k = k_ref
                last_spectrum_ref = E_ref
                last_spectrum_pred = E_pred
        
        print(f"  ✓ 完成")
    
    # 转换为 numpy 数组
    all_times = np.array(all_times)
    all_errors_u = np.array(all_errors_u)
    all_errors_v = np.array(all_errors_v)
    all_errors_w = np.array(all_errors_w)
    all_ke_ref = np.array(all_ke_ref)
    all_ke_pred = np.array(all_ke_pred)
    all_ens_ref = np.array(all_ens_ref)
    all_ens_pred = np.array(all_ens_pred)
    
    print()
    print("=" * 80)
    print("生成对比图...")
    
    # 创建2x2布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Rel. L2 error (左上)
    ax1 = axes[0, 0]
    ax1.plot(all_times, all_errors_u, 'b-', linewidth=2.5, label='u', marker='o', markersize=5)
    ax1.plot(all_times, all_errors_v, 'r--', linewidth=2.5, label='v', marker='s', markersize=5)
    ax1.set_xlabel('t', fontsize=14)
    ax1.set_ylabel('Rel. L2 error', fontsize=14)
    ax1.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Relative L2 Error', fontsize=15, fontweight='bold', pad=10)
    ax1.tick_params(labelsize=12)
    
    # 2. Kinetic energy (右上)
    ax2 = axes[0, 1]
    ax2.plot(all_times, all_ke_ref, 'b-', linewidth=2.5, label='Reference', marker='o', markersize=5)
    ax2.plot(all_times, all_ke_pred, 'r--', linewidth=2.5, label='PINN', marker='s', markersize=5)
    ax2.set_xlabel('t', fontsize=14)
    ax2.set_ylabel('Kinetic energy', fontsize=14)
    ax2.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Kinetic Energy', fontsize=15, fontweight='bold', pad=10)
    ax2.tick_params(labelsize=12)
    
    # 3. Enstrophy (左下)
    ax3 = axes[1, 0]
    ax3.plot(all_times, all_ens_ref, 'b-', linewidth=2.5, label='Reference', marker='o', markersize=5)
    ax3.plot(all_times, all_ens_pred, 'r--', linewidth=2.5, label='PINN', marker='s', markersize=5)
    ax3.set_xlabel('t', fontsize=14)
    ax3.set_ylabel('Enstrophy', fontsize=14)
    ax3.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_title('Enstrophy', fontsize=15, fontweight='bold', pad=10)
    ax3.tick_params(labelsize=12)
    
    # 4. Energy spectrum (右下)
    ax4 = axes[1, 1]
    if last_spectrum_k is not None and len(last_spectrum_k) > 0:
        # 过滤掉零值和无效值
        mask = (last_spectrum_ref > 1e-12) & (last_spectrum_pred > 1e-12) & (last_spectrum_k > 0)
        if np.sum(mask) > 0:
            k_plot = last_spectrum_k[mask]
            E_ref_plot = last_spectrum_ref[mask]
            E_pred_plot = last_spectrum_pred[mask]
            
            ax4.loglog(k_plot, E_ref_plot, 'b-', linewidth=2.5, label='Reference', 
                       marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.5)
            ax4.loglog(k_plot, E_pred_plot, 'r--', linewidth=2.5, label='PINN', 
                       marker='s', markersize=5, markerfacecolor='white', markeredgewidth=1.5)
            
            # 添加 k^{-3} 参考线
            if len(k_plot) > 5:
                k_start_idx = len(k_plot) // 4
                k_end_idx = 3 * len(k_plot) // 4
                k_ref_line = k_plot[k_start_idx:k_end_idx]
                if len(k_ref_line) > 0 and E_ref_plot[k_start_idx] > 0:
                    E_ref_line = (k_ref_line / k_plot[k_start_idx])**(-3) * E_ref_plot[k_start_idx]
                    ax4.loglog(k_ref_line, E_ref_line, 'k:', linewidth=2, label=r'$k^{-3}$', alpha=0.7)
            
            ax4.set_xlabel('Wavenumber (k)', fontsize=14)
            ax4.set_ylabel('Energy spectrum', fontsize=14)
            ax4.legend(fontsize=12, loc='best', frameon=True, shadow=True)
            ax4.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
            ax4.set_title(f'Energy Spectrum (t={all_times[-1]:.2f})', fontsize=15, fontweight='bold', pad=10)
            ax4.set_xlim([k_plot[0]*0.8, k_plot[-1]*1.2])
            ax4.tick_params(labelsize=12)
        else:
            ax4.text(0.5, 0.5, 'Invalid spectrum data', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Energy Spectrum', fontsize=15, fontweight='bold', pad=10)
    else:
        ax4.text(0.5, 0.5, 'No spectrum data', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Energy Spectrum', fontsize=15, fontweight='bold', pad=10)
    
    plt.tight_layout(pad=2.5)
    
    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'dns_vs_pinn_comparison_{config_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_path}")
    
    # 保存数据
    data_path = os.path.join(output_dir, f'comparison_data_{config_name}.npz')
    np.savez(data_path,
             times=all_times,
             errors_u=all_errors_u,
             errors_v=all_errors_v,
             errors_w=all_errors_w,
             ke_ref=all_ke_ref,
             ke_pred=all_ke_pred,
             ens_ref=all_ens_ref,
             ens_pred=all_ens_pred,
             spectrum_k=last_spectrum_k if last_spectrum_k is not None else np.array([]),
             spectrum_ref=last_spectrum_ref if last_spectrum_ref is not None else np.array([]),
             spectrum_pred=last_spectrum_pred if last_spectrum_pred is not None else np.array([]))
    print(f"✓ 数据已保存: {data_path}")
    
    plt.close()
    
    print("=" * 80)
    print("统计摘要:")
    print(f"  总采样点数: {len(all_times)}")
    print(f"  时间范围: [{all_times[0]:.4f}, {all_times[-1]:.4f}]")
    print(f"  u_error: mean={all_errors_u.mean():.6f}, min={all_errors_u.min():.6f}, max={all_errors_u.max():.6f}")
    print(f"  v_error: mean={all_errors_v.mean():.6f}, min={all_errors_v.min():.6f}, max={all_errors_v.max():.6f}")
    print(f"  w_error: mean={all_errors_w.mean():.6f}, min={all_errors_w.min():.6f}, max={all_errors_w.max():.6f}")
    print("=" * 80)
    print("完成!")


def main():
    parser = argparse.ArgumentParser(description='生成 DNS vs PINN 对比图')
    parser.add_argument('--config', type=str, required=True, choices=['soap', 'pirate'],
                        help='配置文件名称')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Checkpoint 根目录路径')
    parser.add_argument('--output_dir', type=str, default='examples/kolmogorov_flow/comparison',
                        help='输出目录')
    parser.add_argument('--time_interval', type=float, default=0.5,
                        help='采样时间间隔（秒）')
    
    args = parser.parse_args()
    
    evaluate_and_compare(args.config, args.checkpoint_path, args.output_dir, args.time_interval)


if __name__ == '__main__':
    main()
