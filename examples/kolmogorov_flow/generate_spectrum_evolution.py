"""
生成能量譜時間演化動畫
展示 DNS vs PINN 的能量譜隨時間的變化，並包含 Kolmogorov -5/3 律參考線
"""

import os
import sys
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 添加專案路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset
from examples.kolmogorov_flow import models


def compute_energy_spectrum_2d(u, v, nx, ny):
    """
    計算 2D 速度場的能量譜（使用速度場而非渦度）
    
    Args:
        u, v: 速度場分量 (nx, ny)
        nx, ny: 網格尺寸
    
    Returns:
        k: 波數
        E_k: 能量譜
    """
    # 2D FFT for both velocity components
    u_fft = np.fft.fft2(u)
    v_fft = np.fft.fft2(v)
    u_fft = np.fft.fftshift(u_fft)
    v_fft = np.fft.fftshift(v_fft)
    
    # 計算動能密度 E = 0.5 * (|u_hat|^2 + |v_hat|^2)
    energy_density = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2)
    
    # 創建波數網格
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=2*np.pi/nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=2*np.pi/ny))
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    
    # 將能量按波數分箱（使用整數波數）
    k_max = int(np.max(K))
    k_bins = np.arange(1, min(k_max, nx//2))  # 從1開始，避免零波數
    E_k = []
    k_out = []
    
    for k_i in k_bins:
        mask = (K >= k_i - 0.5) & (K < k_i + 0.5)
        n_points = np.sum(mask)
        if n_points > 0:
            # 能量譜：每個 shell 的總能量除以 shell 中的點數（使其密度化）
            E_k.append(np.sum(energy_density[mask]) / n_points)
            k_out.append(k_i)
    
    return np.array(k_out), np.array(E_k)


def generate_spectrum_evolution(config_name, checkpoint_path, output_dir, time_interval=0.5):
    """
    生成能量譜時間演化動畫
    """
    
    # 載入配置
    if config_name == 'soap':
        from configs import soap as config_module
    elif config_name == 'pirate':
        from configs import pirate as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = config_module.get_config()
    
    # 載入 DNS 數據
    print("載入 DNS 參考數據...")
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(time_fraction=config.time_fraction)
    
    print(f"DNS 數據信息:")
    print(f"  - 時間步數: {len(t_star)}")
    print(f"  - 時間範圍: [{t_star[0]:.4f}, {t_star[-1]:.4f}]")
    print(f"  - 空間點數: {coords.shape[0]}")
    print(f"  - 雷諾數: Re = {1/nu:.0f}")
    
    # 推斷網格大小
    nx = int(np.sqrt(coords.shape[0]))
    ny = nx
    print(f"  - 網格大小: {nx} x {ny}")
    print(f"  - 採樣間隔: {time_interval} 秒")
    print()
    
    # 計算時間窗口
    num_time_steps = len(t_star) // config.training.num_time_windows
    
    # 找到所有可用的窗口
    window_dirs = sorted([d for d in os.listdir(checkpoint_path) if d.startswith('time_window_')])
    time_windows = sorted([int(d.split('_')[-1]) for d in window_dirs])
    
    print(f"找到 {len(time_windows)} 個時間窗口: {time_windows}")
    print("=" * 80)
    print()
    
    # 存儲所有時間步的能量譜
    all_times = []
    all_spectrum_k = []
    all_spectrum_ref = []
    all_spectrum_pred = []
    
    for window_idx in time_windows:
        print(f"處理 Time Window {window_idx}...")
        
        # 確定當前窗口的時間範圍
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
        
        # 載入 checkpoint
        ckpt_dir = os.path.join(checkpoint_path, f'time_window_{window_idx}')
        
        if not os.path.exists(ckpt_dir):
            print(f"  ⚠️  Checkpoint 不存在，跳過")
            continue
        
        # 查找最新的 checkpoint
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_')]
        if not ckpt_files:
            print(f"  ⚠️  沒有找到 checkpoint 文件，跳過")
            continue
        
        steps = [int(f.split('_')[1]) for f in ckpt_files]
        max_step = max(steps)
        
        model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)
        print(f"  ✓ 載入 checkpoint (step {max_step})")
        
        # 確定採樣的時間步索引
        t_window_start = t[0]
        sample_indices = []
        for i, t_i in enumerate(t):
            if i == 0 or (t_i - t_window_start) >= len(sample_indices) * time_interval:
                sample_indices.append(i)
        
        print(f"  採樣 {len(sample_indices)}/{len(t)} 個時間步")
        
        # 計算每個採樣時間步的能量譜
        for idx, i in enumerate(sample_indices):
            t_i = t[i]
            
            # 獲取參考數據
            u_ref_i = u_ref_window[i, :]
            v_ref_i = v_ref_window[i, :]
            
            # 獲取預測值
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            
            u_pred = model.u_ic_pred_fn(model.state.params, t_i, x_coords, y_coords)
            v_pred = model.v_ic_pred_fn(model.state.params, t_i, x_coords, y_coords)
            
            # 重塑為2D
            u_ref_2d = np.array(u_ref_i).reshape(nx, ny)
            v_ref_2d = np.array(v_ref_i).reshape(nx, ny)
            u_pred_2d = np.array(u_pred).reshape(nx, ny)
            v_pred_2d = np.array(v_pred).reshape(nx, ny)
            
            # 計算能量譜
            k_ref, E_ref = compute_energy_spectrum_2d(u_ref_2d, v_ref_2d, nx, ny)
            k_pred, E_pred = compute_energy_spectrum_2d(u_pred_2d, v_pred_2d, nx, ny)
            
            # 存儲結果
            all_times.append(float(t_i))
            all_spectrum_k.append(k_ref)
            all_spectrum_ref.append(E_ref)
            all_spectrum_pred.append(E_pred)
            
            print(f"    t={t_i:.4f}: k範圍=[{k_ref[0]:.1f}, {k_ref[-1]:.1f}], 點數={len(k_ref)}")
        
        print(f"  ✓ 完成")
    
    print()
    print("=" * 80)
    print("生成動畫...")
    
    # 創建動畫
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 初始化plot對象
    line_ref, = ax.plot([], [], 'b-', linewidth=2.5, label='DNS Reference', 
                        marker='o', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    line_pred, = ax.plot([], [], 'r--', linewidth=2.5, label='PINN', 
                         marker='s', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    line_k53, = ax.plot([], [], 'k:', linewidth=2, label=r'$k^{-5/3}$', alpha=0.7)
    
    # 找出全局最大最小值以設置軸範圍
    all_k = np.concatenate(all_spectrum_k)
    all_E_ref = np.concatenate(all_spectrum_ref)
    all_E_pred = np.concatenate(all_spectrum_pred)
    
    k_min = np.min(all_k[all_k > 0])
    k_max = np.max(all_k)
    E_min = np.min([np.min(all_E_ref[all_E_ref > 0]), np.min(all_E_pred[all_E_pred > 0])])
    E_max = np.max([np.max(all_E_ref), np.max(all_E_pred)])
    
    ax.set_xlim([k_min * 0.8, k_max * 1.2])
    ax.set_ylim([E_min * 0.5, E_max * 2.0])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavenumber (k)', fontsize=14)
    ax.set_ylabel('Energy Spectrum E(k)', fontsize=14)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)
    
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes, 
                    ha='center', fontsize=15, fontweight='bold')
    
    def init():
        line_ref.set_data([], [])
        line_pred.set_data([], [])
        line_k53.set_data([], [])
        title.set_text('')
        return line_ref, line_pred, line_k53, title
    
    def animate(frame):
        k = all_spectrum_k[frame]
        E_ref = all_spectrum_ref[frame]
        E_pred = all_spectrum_pred[frame]
        t = all_times[frame]
        
        # 過濾有效數據
        mask = (E_ref > 1e-12) & (E_pred > 1e-12) & (k > 0)
        if np.sum(mask) > 0:
            k_plot = k[mask]
            E_ref_plot = E_ref[mask]
            E_pred_plot = E_pred[mask]
            
            line_ref.set_data(k_plot, E_ref_plot)
            line_pred.set_data(k_plot, E_pred_plot)
            
            # 添加 k^{-5/3} 參考線（在慣性次範圍）
            if len(k_plot) > 5:
                # 選擇中間範圍作為參考
                k_start_idx = len(k_plot) // 4
                k_end_idx = 3 * len(k_plot) // 4
                k_ref_line = k_plot[k_start_idx:k_end_idx]
                if len(k_ref_line) > 0 and E_ref_plot[k_start_idx] > 0:
                    E_ref_line = (k_ref_line / k_plot[k_start_idx])**(-5./3.) * E_ref_plot[k_start_idx]
                    line_k53.set_data(k_ref_line, E_ref_line)
        
        title.set_text(f'Energy Spectrum Evolution (t={t:.3f}, Re={1/nu:.0f})')
        
        return line_ref, line_pred, line_k53, title
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(all_times), 
                        interval=200, blit=True, repeat=True)
    
    # 保存動畫
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'spectrum_evolution_{config_name}.gif')
    
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer, dpi=100)
    
    print(f"✓ 動畫已保存: {output_path}")
    print(f"  - 總幀數: {len(all_times)}")
    print(f"  - 時間範圍: [{all_times[0]:.4f}, {all_times[-1]:.4f}]")
    
    plt.close()
    
    # 保存時間演化數據
    data_path = os.path.join(output_dir, f'spectrum_evolution_data_{config_name}.npz')
    np.savez(data_path,
             times=np.array(all_times),
             spectrum_k=np.array(all_spectrum_k, dtype=object),
             spectrum_ref=np.array(all_spectrum_ref, dtype=object),
             spectrum_pred=np.array(all_spectrum_pred, dtype=object))
    print(f"✓ 數據已保存: {data_path}")
    
    # 計算並保存時間平均能量譜
    print()
    print("=" * 80)
    print("計算時間平均能量譜...")
    
    # 找出所有時間步共同的波數範圍
    k_common = all_spectrum_k[0]
    for k in all_spectrum_k[1:]:
        if len(k) < len(k_common):
            k_common = k
    
    # 內插所有能量譜到共同波數
    E_ref_interp_list = []
    E_pred_interp_list = []
    
    for i in range(len(all_times)):
        k_i = all_spectrum_k[i]
        E_ref_i = all_spectrum_ref[i]
        E_pred_i = all_spectrum_pred[i]
        
        # 內插
        if len(k_i) == len(k_common) and np.allclose(k_i, k_common):
            E_ref_interp_list.append(E_ref_i)
            E_pred_interp_list.append(E_pred_i)
        else:
            E_ref_interp = np.interp(k_common, k_i, E_ref_i)
            E_pred_interp = np.interp(k_common, k_i, E_pred_i)
            E_ref_interp_list.append(E_ref_interp)
            E_pred_interp_list.append(E_pred_interp)
    
    # 計算時間平均
    E_ref_mean = np.mean(E_ref_interp_list, axis=0)
    E_ref_std = np.std(E_ref_interp_list, axis=0)
    E_pred_mean = np.mean(E_pred_interp_list, axis=0)
    E_pred_std = np.std(E_pred_interp_list, axis=0)
    
    # 繪製時間平均能量譜
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 過濾有效數據
    mask = (E_ref_mean > 1e-12) & (E_pred_mean > 1e-12) & (k_common > 0)
    k_plot = k_common[mask]
    E_ref_mean_plot = E_ref_mean[mask]
    E_ref_std_plot = E_ref_std[mask]
    E_pred_mean_plot = E_pred_mean[mask]
    E_pred_std_plot = E_pred_std[mask]
    
    # 繪製平均值和標準差
    ax.loglog(k_plot, E_ref_mean_plot, 'b-', linewidth=2.5, label='DNS (mean)', 
              marker='o', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    ax.fill_between(k_plot, E_ref_mean_plot - E_ref_std_plot, E_ref_mean_plot + E_ref_std_plot,
                     alpha=0.2, color='blue', label='DNS (±1σ)')
    
    ax.loglog(k_plot, E_pred_mean_plot, 'r--', linewidth=2.5, label='PINN (mean)', 
              marker='s', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    ax.fill_between(k_plot, E_pred_mean_plot - E_pred_std_plot, E_pred_mean_plot + E_pred_std_plot,
                     alpha=0.2, color='red', label='PINN (±1σ)')
    
    # 添加 k^{-5/3} 參考線
    if len(k_plot) > 5:
        k_start_idx = len(k_plot) // 4
        k_end_idx = 3 * len(k_plot) // 4
        k_ref_line = k_plot[k_start_idx:k_end_idx]
        if len(k_ref_line) > 0 and E_ref_mean_plot[k_start_idx] > 0:
            E_ref_line = (k_ref_line / k_plot[k_start_idx])**(-5./3.) * E_ref_mean_plot[k_start_idx]
            ax.loglog(k_ref_line, E_ref_line, 'k:', linewidth=2, label=r'$k^{-5/3}$ (Kolmogorov)', alpha=0.7)
    
    ax.set_xlabel('Wavenumber (k)', fontsize=14)
    ax.set_ylabel('Energy Spectrum E(k)', fontsize=14)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(f'Time-Averaged Energy Spectrum (Re={1/nu:.0f})', fontsize=15, fontweight='bold', pad=10)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'spectrum_time_averaged_{config_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 時間平均能量譜已保存: {output_path}")
    
    plt.close()
    
    # 保存時間平均數據
    avg_data_path = os.path.join(output_dir, f'spectrum_time_averaged_data_{config_name}.npz')
    np.savez(avg_data_path,
             k=k_common,
             E_ref_mean=E_ref_mean,
             E_ref_std=E_ref_std,
             E_pred_mean=E_pred_mean,
             E_pred_std=E_pred_std)
    print(f"✓ 時間平均數據已保存: {avg_data_path}")
    
    print("=" * 80)
    print("完成!")


def main():
    parser = argparse.ArgumentParser(description='生成能量譜時間演化動畫')
    parser.add_argument('--config', type=str, required=True, choices=['soap', 'pirate'],
                        help='配置文件名稱')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Checkpoint 根目錄路徑')
    parser.add_argument('--output_dir', type=str, default='examples/kolmogorov_flow/comparison',
                        help='輸出目錄')
    parser.add_argument('--time_interval', type=float, default=0.5,
                        help='採樣時間間隔（秒）')
    
    args = parser.parse_args()
    
    generate_spectrum_evolution(args.config, args.checkpoint_path, args.output_dir, args.time_interval)


if __name__ == '__main__':
    main()
