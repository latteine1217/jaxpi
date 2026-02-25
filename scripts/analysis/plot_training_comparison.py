#!/usr/bin/env python3
"""
視覺化 PIRATE vs SOAP 訓練結果對比（基於物理時間）
修正版：使用時間對齊方法進行公平比較
"""
import matplotlib.pyplot as plt
import numpy as np

# DNS 時間參數
DNS_START_TIME = 0.002
DNS_END_TIME = 1.962
DNS_TOTAL_STEPS = 50
DNS_DT = (DNS_END_TIME - DNS_START_TIME) / (DNS_TOTAL_STEPS - 1)  # ≈ 0.040

# PIRATE: 10 窗口，每窗口 5 個 DNS 步
# 窗口終點時間 = 0.002 + (window_idx * 5) * 0.04
pirate_times = [DNS_START_TIME + i * 5 * DNS_DT for i in range(1, 11)]

# SOAP: 25 窗口，每窗口 2 個 DNS 步  
# 窗口終點時間 = 0.002 + (window_idx * 2) * 0.04
soap_times = [DNS_START_TIME + i * 2 * DNS_DT for i in range(1, 18)]

# PIRATE 數據（10 個窗口的終點時間）
pirate_data = {
    'windows': list(range(1, 11)),
    'times': pirate_times,
    'u_error': [0.005202, 0.007983, 0.011940, 0.016530, 0.025750, 
                0.052830, 0.082130, 0.155700, 0.314500, 0.351500],
    'v_error': [0.005086, 0.014150, 0.022410, 0.028460, 0.027440, 
                0.055250, 0.140400, 0.225200, 0.312200, 0.334900],
    'w_error': [0.013720, 0.048280, 0.091390, 0.122700, 0.142000, 
                0.181400, 0.365600, 0.574300, 0.855100, 0.935500],
}

# SOAP 數據（17 個窗口已完成的終點時間）
soap_data = {
    'windows': list(range(1, 18)),
    'times': soap_times,
    'u_error': [0.002386, 0.004212, 0.005751, 0.007416, 0.008897, 
                0.011060, 0.012820, 0.013960, 0.015420, 0.017400,
                0.019880, 0.025670, 0.035500, 0.046590, 0.055810, 
                0.067140, 0.079600],
    'v_error': [0.002383, 0.004460, 0.007307, 0.012330, 0.016960, 
                0.020910, 0.025360, 0.027110, 0.026800, 0.026330,
                0.025320, 0.025670, 0.031420, 0.043330, 0.067350, 
                0.103700, 0.144300],
    'w_error': [0.004596, 0.012070, 0.027660, 0.045080, 0.060820, 
                0.079340, 0.102400, 0.111900, 0.117100, 0.128900,
                0.138400, 0.144100, 0.144900, 0.160300, 0.205500, 
                0.289400, 0.377500],
}

def plot_comparison():
    """繪製 PIRATE vs SOAP 對比圖（基於物理時間）"""
    
    # 設定圖表風格
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PIRATE vs SOAP - Time-Aligned Comparison\n' +
                 'Kolmogorov Flow at Re=10,000 (Physical Time)', 
                 fontsize=16, fontweight='bold')
    
    # 顏色設定
    color_pirate = '#E74C3C'  # 紅色
    color_soap = '#3498DB'    # 藍色
    
    # === 圖 1: u_error 對比 ===
    ax1 = axes[0, 0]
    ax1.plot(pirate_data['times'], pirate_data['u_error'], 
             'o-', color=color_pirate, linewidth=2, markersize=8, 
             label='PIRATE (10 windows)', alpha=0.8)
    ax1.plot(soap_data['times'], soap_data['u_error'], 
             's-', color=color_soap, linewidth=2, markersize=8, 
             label='SOAP (25 windows)', alpha=0.8)
    ax1.set_xlabel('Physical Time (t)', fontsize=12)
    ax1.set_ylabel('u velocity - L2 Relative Error', fontsize=12)
    ax1.set_title('u-velocity Error vs Physical Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 標註關鍵時間點
    ax1.axvline(x=0.762, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.text(0.77, 0.002, 't≈0.76', fontsize=9, color='gray')
    
    # === 圖 2: v_error 對比 ===
    ax2 = axes[0, 1]
    ax2.plot(pirate_data['times'], pirate_data['v_error'], 
             'o-', color=color_pirate, linewidth=2, markersize=8, 
             label='PIRATE (10 windows)', alpha=0.8)
    ax2.plot(soap_data['times'], soap_data['v_error'], 
             's-', color=color_soap, linewidth=2, markersize=8, 
             label='SOAP (25 windows)', alpha=0.8)
    ax2.set_xlabel('Physical Time (t)', fontsize=12)
    ax2.set_ylabel('v velocity - L2 Relative Error', fontsize=12)
    ax2.set_title('v-velocity Error vs Physical Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax2.axvline(x=0.762, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.text(0.77, 0.002, 't≈0.76', fontsize=9, color='gray')
    
    # === 圖 3: w_error 對比 ===
    ax3 = axes[1, 0]
    ax3.plot(pirate_data['times'], pirate_data['w_error'], 
             'o-', color=color_pirate, linewidth=2, markersize=8, 
             label='PIRATE (10 windows)', alpha=0.8)
    ax3.plot(soap_data['times'], soap_data['w_error'], 
             's-', color=color_soap, linewidth=2, markersize=8, 
             label='SOAP (25 windows)', alpha=0.8)
    ax3.set_xlabel('Physical Time (t)', fontsize=12)
    ax3.set_ylabel('Vorticity - L2 Relative Error', fontsize=12)
    ax3.set_title('Vorticity Error vs Physical Time', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax3.axvline(x=0.762, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.text(0.77, 0.005, 't≈0.76', fontsize=9, color='gray')
    
    # === 圖 4: 時間窗口策略對比 ===
    ax4 = axes[1, 1]
    
    # 計算窗口時間範圍並繪製
    # PIRATE: 10 窗口，每窗口 5 步
    pirate_window_width = 5 * DNS_DT
    for i in range(10):
        t_start = DNS_START_TIME + i * pirate_window_width
        ax4.barh(1, pirate_window_width, left=t_start, height=0.3,
                color=color_pirate, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # SOAP: 25 窗口，每窗口 2 步
    soap_window_width = 2 * DNS_DT
    for i in range(17):  # 只顯示已完成的 17 個窗口
        t_start = DNS_START_TIME + i * soap_window_width
        ax4.barh(0, soap_window_width, left=t_start, height=0.3,
                color=color_soap, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['SOAP (25 windows, 2 steps/window)', 
                         'PIRATE (10 windows, 5 steps/window)'])
    ax4.set_xlabel('Physical Time (t)', fontsize=12)
    ax4.set_title('Time Window Discretization Strategy', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 1.4)
    
    # 標註關鍵時間點
    ax4.axvline(x=0.762, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax4.text(0.762, 1.5, 't≈0.76\n(P-W4 vs S-W10)', fontsize=9, ha='center')
    
    ax4.axvline(x=1.322, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax4.text(1.322, 1.5, 't≈1.32\n(P-W7 vs S-W17)', fontsize=9, ha='center')
    
    # 添加說明文字
    textstr = ('Key: At same time (t≈0.76, t≈1.32), errors are similar (<8% diff).\n'
               'SOAP uses 2.5× more windows → less error per window.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax4.text(0.02, 0.05, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 訓練對比圖已保存: training_comparison.png")
    
    return fig

def plot_error_growth():
    """繪製誤差增長趨勢圖（線性尺度，基於物理時間）"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Error Growth vs Physical Time: PIRATE vs SOAP', 
                 fontsize=16, fontweight='bold')
    
    color_pirate = '#E74C3C'
    color_soap = '#3498DB'
    
    # u_error
    ax1 = axes[0]
    ax1.plot(pirate_data['times'], pirate_data['u_error'], 
             'o-', color=color_pirate, linewidth=2.5, markersize=10, 
             label='PIRATE (10 windows)', alpha=0.8)
    ax1.plot(soap_data['times'], soap_data['u_error'], 
             's-', color=color_soap, linewidth=2.5, markersize=10, 
             label='SOAP (25 windows)', alpha=0.8)
    ax1.set_xlabel('Physical Time (t)', fontsize=13)
    ax1.set_ylabel('L2 Relative Error', fontsize=13)
    ax1.set_title('u-velocity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.762, color='gray', linestyle='--', alpha=0.4)
    ax1.axvline(x=1.322, color='gray', linestyle='--', alpha=0.4)
    
    # v_error
    ax2 = axes[1]
    ax2.plot(pirate_data['times'], pirate_data['v_error'], 
             'o-', color=color_pirate, linewidth=2.5, markersize=10, 
             label='PIRATE (10 windows)', alpha=0.8)
    ax2.plot(soap_data['times'], soap_data['v_error'], 
             's-', color=color_soap, linewidth=2.5, markersize=10, 
             label='SOAP (25 windows)', alpha=0.8)
    ax2.set_xlabel('Physical Time (t)', fontsize=13)
    ax2.set_ylabel('L2 Relative Error', fontsize=13)
    ax2.set_title('v-velocity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.762, color='gray', linestyle='--', alpha=0.4)
    ax2.axvline(x=1.322, color='gray', linestyle='--', alpha=0.4)
    
    # w_error
    ax3 = axes[2]
    ax3.plot(pirate_data['times'], pirate_data['w_error'], 
             'o-', color=color_pirate, linewidth=2.5, markersize=10, 
             label='PIRATE (10 windows)', alpha=0.8)
    ax3.plot(soap_data['times'], soap_data['w_error'], 
             's-', color=color_soap, linewidth=2.5, markersize=10, 
             label='SOAP (25 windows)', alpha=0.8)
    ax3.set_xlabel('Physical Time (t)', fontsize=13)
    ax3.set_ylabel('L2 Relative Error', fontsize=13)
    ax3.set_title('Vorticity', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0.762, color='gray', linestyle='--', alpha=0.4)
    ax3.axvline(x=1.322, color='gray', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('error_growth_linear.png', dpi=300, bbox_inches='tight')
    print("✅ 誤差增長圖已保存: error_growth_linear.png")
    
    return fig

if __name__ == "__main__":
    print("=" * 70)
    print("生成基於物理時間的訓練對比視覺化圖表")
    print("=" * 70)
    print()
    print(f"DNS 時間範圍: t ∈ [{DNS_START_TIME:.3f}, {DNS_END_TIME:.3f}]")
    print(f"DNS 時間步長: Δt ≈ {DNS_DT:.4f}")
    print()
    print("PIRATE: 10 窗口 × 5 步/窗口")
    print("SOAP: 25 窗口 × 2 步/窗口")
    print()
    print("關鍵時間點（公平比較）:")
    print("  • t ≈ 0.76: PIRATE Window 4 vs SOAP Window 10")
    print("  • t ≈ 1.32: PIRATE Window 7 vs SOAP Window 17")
    print("=" * 70)
    print()
    
    # 生成對比圖
    plot_comparison()
    
    # 生成誤差增長圖
    plot_error_growth()
    
    print()
    print("=" * 70)
    print("所有圖表已生成完成！")
    print("  • training_comparison.png - 基於物理時間的完整對比")
    print("  • error_growth_linear.png - 基於物理時間的誤差增長")
    print("=" * 70)
