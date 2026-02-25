"""
生成場視覺化快照比較
選擇關鍵時刻展示 DNS vs PIRATE vs SOAP 的速度場和渦度場對比
"""

import os
import sys
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加專案路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset
from examples.kolmogorov_flow import models


def find_closest_time_index(t_star, target_time, window_idx, num_time_steps):
    """找到最接近目標時間的索引"""
    start_idx = (window_idx - 1) * num_time_steps
    end_idx = window_idx * num_time_steps
    t_window = t_star[start_idx:end_idx]
    
    # 找到最接近的時間步
    closest_idx = np.argmin(np.abs(t_window - target_time))
    return start_idx + closest_idx, t_window[closest_idx]


def load_checkpoint_for_time(config, checkpoint_path, t_target, t_star, coords, u_ref, v_ref, w_ref, nu):
    """載入指定時間的checkpoint並返回預測值"""
    
    num_time_steps = len(t_star) // config.training.num_time_windows
    
    # 找到包含目標時間的窗口
    window_idx = int(t_target // (t_star[-1] / config.training.num_time_windows)) + 1
    window_idx = max(1, min(window_idx, config.training.num_time_windows))
    
    # 確定當前窗口的時間範圍
    start_idx = (window_idx - 1) * num_time_steps
    end_idx = window_idx * num_time_steps
    
    t = t_star[start_idx:end_idx]
    
    # 初始化模型
    u0 = u_ref[start_idx, :]
    v0 = v_ref[start_idx, :]
    w0 = w_ref[start_idx, :]
    
    model = models.NavierStokes(config, t, coords, u0, v0, w0, nu)
    
    # 載入 checkpoint
    ckpt_dir = os.path.join(checkpoint_path, f'time_window_{window_idx}')
    
    if not os.path.exists(ckpt_dir):
        print(f"  ⚠️  Window {window_idx} checkpoint 不存在")
        return None, None, None, None
    
    # 查找最新的 checkpoint
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_')]
    if not ckpt_files:
        print(f"  ⚠️  Window {window_idx} 沒有找到 checkpoint 文件")
        return None, None, None, None
    
    steps = [int(f.split('_')[1]) for f in ckpt_files]
    max_step = max(steps)
    
    model.state = restore_checkpoint(model.state, ckpt_dir, step=max_step)
    
    # 找到最接近目標時間的索引
    actual_idx, actual_time = find_closest_time_index(t_star, t_target, window_idx, num_time_steps)
    
    # 獲取預測值
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    u_pred = model.u_ic_pred_fn(model.state.params, actual_time, x_coords, y_coords)
    v_pred = model.v_ic_pred_fn(model.state.params, actual_time, x_coords, y_coords)
    w_pred = model.w_ic_pred_fn(model.state.params, actual_time, x_coords, y_coords)
    
    return np.array(u_pred), np.array(v_pred), np.array(w_pred), actual_time


def generate_snapshots(config_pirate_name, config_soap_name, 
                       checkpoint_path_pirate, checkpoint_path_soap,
                       output_dir, snapshot_times=[0.5, 1.0, 1.5, 1.8]):
    """
    生成場視覺化快照比較
    """
    
    # 載入配置
    if config_pirate_name == 'pirate':
        from configs import pirate as config_pirate_module
    else:
        raise ValueError(f"Unknown pirate config: {config_pirate_name}")
    
    if config_soap_name == 'soap':
        from configs import soap as config_soap_module
    else:
        raise ValueError(f"Unknown soap config: {config_soap_name}")
    
    config_pirate = config_pirate_module.get_config()
    config_soap = config_soap_module.get_config()
    
    # 載入 DNS 數據
    print("載入 DNS 參考數據...")
    u_ref_all, v_ref_all, w_ref_all, t_star, coords, nu = get_dataset(time_fraction=1.0)
    
    print(f"DNS 數據信息:")
    print(f"  - 時間步數: {len(t_star)}")
    print(f"  - 時間範圍: [{t_star[0]:.4f}, {t_star[-1]:.4f}]")
    print(f"  - 空間點數: {coords.shape[0]}")
    print(f"  - 雷諾數: Re = {1/nu:.0f}")
    
    # 推斷網格大小
    nx = int(np.sqrt(coords.shape[0]))
    ny = nx
    print(f"  - 網格大小: {nx} x {ny}")
    print()
    
    # 為每個場類型生成對比圖
    fields = ['u', 'v', 'w']
    field_names = ['u-velocity', 'v-velocity', 'vorticity']
    
    for field_idx, (field, field_name) in enumerate(zip(fields, field_names)):
        print("=" * 80)
        print(f"處理 {field_name} 場...")
        print("=" * 80)
        
        # 創建大圖：每行一個時間，每列：DNS, PIRATE, SOAP, |DNS-PIRATE|, |DNS-SOAP|
        n_times = len(snapshot_times)
        fig = plt.figure(figsize=(20, 4*n_times))
        gs = GridSpec(n_times, 5, figure=fig, hspace=0.3, wspace=0.3)
        
        for t_idx, t_target in enumerate(snapshot_times):
            print(f"\n處理時間 t={t_target:.2f}...")
            
            # 找到最接近的DNS時間索引
            dns_idx = np.argmin(np.abs(t_star - t_target))
            t_actual = t_star[dns_idx]
            
            # 獲取DNS數據
            if field == 'u':
                field_ref = u_ref_all[dns_idx, :].reshape(nx, ny)
            elif field == 'v':
                field_ref = v_ref_all[dns_idx, :].reshape(nx, ny)
            else:  # w
                field_ref = w_ref_all[dns_idx, :].reshape(nx, ny)
            
            # 載入PIRATE預測
            print(f"  載入 PIRATE checkpoint...")
            u_pirate, v_pirate, w_pirate, t_pirate = load_checkpoint_for_time(
                config_pirate, checkpoint_path_pirate, t_target, t_star, 
                coords, u_ref_all, v_ref_all, w_ref_all, nu
            )
            
            if u_pirate is not None:
                if field == 'u':
                    field_pirate = u_pirate.reshape(nx, ny)
                elif field == 'v':
                    field_pirate = v_pirate.reshape(nx, ny)
                else:
                    field_pirate = w_pirate.reshape(nx, ny)
                print(f"    ✓ PIRATE: t={t_pirate:.4f}")
            else:
                field_pirate = np.zeros((nx, ny))
                print(f"    ✗ PIRATE: 無數據")
            
            # 載入SOAP預測
            print(f"  載入 SOAP checkpoint...")
            u_soap, v_soap, w_soap, t_soap = load_checkpoint_for_time(
                config_soap, checkpoint_path_soap, t_target, t_star,
                coords, u_ref_all, v_ref_all, w_ref_all, nu
            )
            
            if u_soap is not None:
                if field == 'u':
                    field_soap = u_soap.reshape(nx, ny)
                elif field == 'v':
                    field_soap = v_soap.reshape(nx, ny)
                else:
                    field_soap = w_soap.reshape(nx, ny)
                print(f"    ✓ SOAP: t={t_soap:.4f}")
            else:
                field_soap = np.zeros((nx, ny))
                print(f"    ✗ SOAP: 無數據")
            
            # 計算誤差
            error_pirate = np.abs(field_ref - field_pirate)
            error_soap = np.abs(field_ref - field_soap)
            
            # 計算相對L2誤差
            rel_error_pirate = np.linalg.norm(field_ref - field_pirate) / np.linalg.norm(field_ref)
            rel_error_soap = np.linalg.norm(field_ref - field_soap) / np.linalg.norm(field_ref)
            
            print(f"  相對L2誤差: PIRATE={rel_error_pirate:.4f}, SOAP={rel_error_soap:.4f}")
            
            # 設置色標範圍（對於場值）
            vmin_field = min(field_ref.min(), field_pirate.min(), field_soap.min())
            vmax_field = max(field_ref.max(), field_pirate.max(), field_soap.max())
            
            # 對於誤差，使用獨立範圍
            vmax_error_pirate = error_pirate.max()
            vmax_error_soap = error_soap.max()
            
            # 繪製DNS
            ax1 = fig.add_subplot(gs[t_idx, 0])
            im1 = ax1.imshow(field_ref.T, origin='lower', cmap='RdBu_r', 
                            vmin=vmin_field, vmax=vmax_field, aspect='auto')
            ax1.set_title(f'DNS\nt={t_actual:.3f}', fontsize=12, fontweight='bold')
            ax1.set_ylabel(f't={t_target:.2f}s', fontsize=12, fontweight='bold')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # 繪製PIRATE
            ax2 = fig.add_subplot(gs[t_idx, 1])
            im2 = ax2.imshow(field_pirate.T, origin='lower', cmap='RdBu_r', 
                            vmin=vmin_field, vmax=vmax_field, aspect='auto')
            ax2.set_title(f'PIRATE\nRel. L2={rel_error_pirate:.4f}', fontsize=12, fontweight='bold')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # 繪製SOAP
            ax3 = fig.add_subplot(gs[t_idx, 2])
            im3 = ax3.imshow(field_soap.T, origin='lower', cmap='RdBu_r', 
                            vmin=vmin_field, vmax=vmax_field, aspect='auto')
            ax3.set_title(f'SOAP\nRel. L2={rel_error_soap:.4f}', fontsize=12, fontweight='bold')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            # 繪製PIRATE誤差
            ax4 = fig.add_subplot(gs[t_idx, 3])
            im4 = ax4.imshow(error_pirate.T, origin='lower', cmap='hot', 
                            vmin=0, vmax=vmax_error_pirate, aspect='auto')
            ax4.set_title(f'|DNS - PIRATE|', fontsize=12, fontweight='bold')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            
            # 繪製SOAP誤差
            ax5 = fig.add_subplot(gs[t_idx, 4])
            im5 = ax5.imshow(error_soap.T, origin='lower', cmap='hot', 
                            vmin=0, vmax=vmax_error_soap, aspect='auto')
            ax5.set_title(f'|DNS - SOAP|', fontsize=12, fontweight='bold')
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
            
            # 移除所有subplot的刻度標籤以節省空間
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 添加總標題
        fig.suptitle(f'{field_name.upper()} Field Comparison: DNS vs PIRATE vs SOAP (Re={1/nu:.0f})', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 保存圖形
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'field_snapshot_{field}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ {field_name} 快照已保存: {output_path}")
        
        plt.close()
    
    print()
    print("=" * 80)
    print("完成!")


def main():
    parser = argparse.ArgumentParser(description='生成場視覺化快照比較')
    parser.add_argument('--config_pirate', type=str, default='pirate',
                        help='PIRATE 配置文件名稱')
    parser.add_argument('--config_soap', type=str, default='soap',
                        help='SOAP 配置文件名稱')
    parser.add_argument('--checkpoint_path_pirate', type=str, required=True,
                        help='PIRATE Checkpoint 根目錄路徑')
    parser.add_argument('--checkpoint_path_soap', type=str, required=True,
                        help='SOAP Checkpoint 根目錄路徑')
    parser.add_argument('--output_dir', type=str, default='examples/kolmogorov_flow/comparison',
                        help='輸出目錄')
    parser.add_argument('--times', type=float, nargs='+', default=[0.5, 1.0, 1.5, 1.8],
                        help='快照時間點列表')
    
    args = parser.parse_args()
    
    generate_snapshots(args.config_pirate, args.config_soap,
                      args.checkpoint_path_pirate, args.checkpoint_path_soap,
                      args.output_dir, args.times)


if __name__ == '__main__':
    main()
