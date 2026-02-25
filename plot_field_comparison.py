#!/usr/bin/env python3
"""
視覺化 DNS vs PINN Prediction vs Error Field
展示特定時間窗口的速度場和渦度場對比
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# 添加 jaxpi 到路徑
sys.path.insert(0, os.path.expanduser('~/jaxpi'))

# JAX/Flax 只在需要載入 checkpoint 時才導入
# 對於 demo 模式，不需要這些依賴


def load_dns_data(data_path):
    """載入 DNS 參考數據"""
    print(f"載入 DNS 數據: {data_path}")
    data = np.load(data_path, allow_pickle=True).item()
    
    print(f"  - 時間步數: {data['t'].shape}")
    print(f"  - 空間網格: {data['coords'].shape}")
    print(f"  - 速度場: {data['velocity'].shape}")
    print(f"  - 渦度場: {data['vorticity'].shape}")
    
    return data


def load_checkpoint_and_predict(ckpt_dir, config_name, dns_data, time_window):
    """
    載入 checkpoint 並生成預測
    
    注意: 這個函數需要在有完整 JAX/Flax 環境的機器上運行
    """
    print(f"載入 checkpoint: {ckpt_dir}/time_window_{time_window}")
    
    # 這裡需要完整的模型載入邏輯
    # 由於我們在 headnode 上可能遇到問題，這部分需要在計算節點上運行
    
    # TODO: 實現完整的模型載入和預測邏輯
    # 1. 載入配置
    # 2. 初始化模型
    # 3. 載入 checkpoint
    # 4. 在 DNS 網格上預測
    
    raise NotImplementedError("此功能需要在有 GPU 的計算節點上實現")


def plot_field_comparison_from_arrays(dns_u, dns_v, dns_w, 
                                      pred_u, pred_v, pred_w,
                                      coords, time_idx, 
                                      model_name="PINN",
                                      output_file="field_comparison.png"):
    """
    給定 DNS 和預測的數組，繪製對比圖
    
    參數:
        dns_u, dns_v, dns_w: DNS 參考數據 (N, ) 或 (H, W)
        pred_u, pred_v, pred_w: PINN 預測數據 (N, ) 或 (H, W)
        coords: 空間坐標 (N, 2) 或 None
        time_idx: 時間索引
        model_name: 模型名稱 (PIRATE/SOAP)
        output_file: 輸出文件名
    """
    
    # 確保數據是 2D 網格格式
    if dns_u.ndim == 1:
        # 假設是 256x256 網格
        grid_size = int(np.sqrt(len(dns_u)))
        dns_u = dns_u.reshape(grid_size, grid_size)
        dns_v = dns_v.reshape(grid_size, grid_size)
        dns_w = dns_w.reshape(grid_size, grid_size)
        pred_u = pred_u.reshape(grid_size, grid_size)
        pred_v = pred_v.reshape(grid_size, grid_size)
        pred_w = pred_w.reshape(grid_size, grid_size)
    
    # 計算誤差場
    error_u = np.abs(pred_u - dns_u)
    error_v = np.abs(pred_v - dns_v)
    error_w = np.abs(pred_w - dns_w)
    
    # 計算相對誤差
    rel_error_u = np.linalg.norm(error_u) / np.linalg.norm(dns_u)
    rel_error_v = np.linalg.norm(error_v) / np.linalg.norm(dns_v)
    rel_error_w = np.linalg.norm(error_w) / np.linalg.norm(dns_w)
    
    # 創建圖表
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 設定標題
    fig.suptitle(f'{model_name} vs DNS - Field Comparison (Time Index {time_idx})\n'
                 f'L2 Relative Errors: u={rel_error_u:.4f}, v={rel_error_v:.4f}, w={rel_error_w:.4f}',
                 fontsize=16, fontweight='bold')
    
    # === 第一行: u 速度分量 ===
    # DNS
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dns_u, cmap='RdBu_r', origin='lower', aspect='equal')
    ax1.set_title('DNS: u velocity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_u, cmap='RdBu_r', origin='lower', aspect='equal',
                     vmin=dns_u.min(), vmax=dns_u.max())
    ax2.set_title(f'{model_name}: u velocity', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Error
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(error_u, cmap='hot', origin='lower', aspect='equal')
    ax3.set_title(f'Absolute Error: u\n(L2 rel: {rel_error_u:.4f})', 
                  fontsize=13, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # === 第二行: v 速度分量 ===
    # DNS
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(dns_v, cmap='RdBu_r', origin='lower', aspect='equal')
    ax4.set_title('DNS: v velocity', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Prediction
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(pred_v, cmap='RdBu_r', origin='lower', aspect='equal',
                     vmin=dns_v.min(), vmax=dns_v.max())
    ax5.set_title(f'{model_name}: v velocity', fontsize=13, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Error
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(error_v, cmap='hot', origin='lower', aspect='equal')
    ax6.set_title(f'Absolute Error: v\n(L2 rel: {rel_error_v:.4f})', 
                  fontsize=13, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # === 第三行: 渦度 ===
    # DNS
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(dns_w, cmap='RdBu_r', origin='lower', aspect='equal')
    ax7.set_title('DNS: Vorticity', fontsize=13, fontweight='bold')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # Prediction
    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(pred_w, cmap='RdBu_r', origin='lower', aspect='equal',
                     vmin=dns_w.min(), vmax=dns_w.max())
    ax8.set_title(f'{model_name}: Vorticity', fontsize=13, fontweight='bold')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # Error
    ax9 = fig.add_subplot(gs[2, 2])
    im9 = ax9.imshow(error_w, cmap='hot', origin='lower', aspect='equal')
    ax9.set_title(f'Absolute Error: Vorticity\n(L2 rel: {rel_error_w:.4f})', 
                  fontsize=13, fontweight='bold')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 場對比圖已保存: {output_file}")
    
    return fig


def generate_field_comparison_demo():
    """
    生成示例對比圖（使用模擬數據）
    用於演示圖表格式
    """
    print("=" * 70)
    print("生成示例場對比圖（使用模擬數據）")
    print("=" * 70)
    print()
    
    # 創建模擬數據
    grid_size = 128
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 模擬 DNS 數據（Kolmogorov flow 的近似模式）
    dns_u = np.sin(Y) + 0.3 * np.cos(2*X) * np.sin(Y)
    dns_v = 0.2 * np.sin(2*X) * np.cos(Y)
    dns_w = np.cos(X) * np.sin(Y) - np.sin(X) * np.cos(Y)
    
    # 模擬 PIRATE 預測（添加較大誤差）
    pred_u_pirate = dns_u + 0.3 * np.random.randn(*dns_u.shape) * np.std(dns_u)
    pred_v_pirate = dns_v + 0.3 * np.random.randn(*dns_v.shape) * np.std(dns_v)
    pred_w_pirate = dns_w + 0.8 * np.random.randn(*dns_w.shape) * np.std(dns_w)
    
    # 模擬 SOAP 預測（添加較小誤差）
    pred_u_soap = dns_u + 0.05 * np.random.randn(*dns_u.shape) * np.std(dns_u)
    pred_v_soap = dns_v + 0.05 * np.random.randn(*dns_v.shape) * np.std(dns_v)
    pred_w_soap = dns_w + 0.15 * np.random.randn(*dns_w.shape) * np.std(dns_w)
    
    # 繪製 PIRATE 對比圖
    plot_field_comparison_from_arrays(
        dns_u, dns_v, dns_w,
        pred_u_pirate, pred_v_pirate, pred_w_pirate,
        None, time_idx=10,
        model_name="PIRATE (Adam)",
        output_file="field_comparison_pirate_demo.png"
    )
    
    # 繪製 SOAP 對比圖
    plot_field_comparison_from_arrays(
        dns_u, dns_v, dns_w,
        pred_u_soap, pred_v_soap, pred_w_soap,
        None, time_idx=10,
        model_name="SOAP",
        output_file="field_comparison_soap_demo.png"
    )
    
    print()
    print("=" * 70)
    print("示例圖表生成完成！")
    print("=" * 70)
    print()
    print("注意: 這些是使用模擬數據生成的示例圖表。")
    print("要生成真實的對比圖，需要:")
    print("  1. 載入 DNS 數據")
    print("  2. 載入訓練好的 PINN checkpoint")
    print("  3. 在 DNS 網格上進行預測")
    print("  4. 使用 plot_field_comparison_from_arrays() 繪製")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='視覺化 DNS vs PINN 場對比')
    parser.add_argument('--mode', type=str, default='demo', 
                        choices=['demo', 'real'],
                        help='運行模式: demo (模擬數據) 或 real (真實數據)')
    parser.add_argument('--dns-data', type=str, 
                        default='~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy',
                        help='DNS 數據路徑')
    parser.add_argument('--ckpt-dir', type=str,
                        help='Checkpoint 目錄')
    parser.add_argument('--time-window', type=int, default=10,
                        help='時間窗口編號')
    parser.add_argument('--model-name', type=str, default='PINN',
                        help='模型名稱 (PIRATE/SOAP)')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        generate_field_comparison_demo()
    else:
        print("真實數據模式需要完整實現...")
        print("請參考 plot_field_comparison_from_arrays() 函數")
        # TODO: 實現真實數據的完整流程
