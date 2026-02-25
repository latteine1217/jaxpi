"""
分析能量譜並計算 Kolmogorov -5/3 律擬合
評估 DNS 和 PINN 是否正確捕捉到湍流的慣性次範圍
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# 添加專案路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def power_law(k, A, alpha):
    """冪律函數 E(k) = A * k^alpha"""
    return A * k**alpha


def analyze_kolmogorov_spectrum(npz_file_path, config_name, output_dir):
    """
    分析能量譜並擬合 Kolmogorov -5/3 律
    
    Args:
        npz_file_path: 時間平均能量譜數據文件路徑
        config_name: 配置名稱 (pirate/soap)
        output_dir: 輸出目錄
    """
    
    print("=" * 80)
    print(f"分析 {config_name.upper()} 的 Kolmogorov 能量譜")
    print("=" * 80)
    print()
    
    # 載入數據
    print(f"載入數據: {npz_file_path}")
    data = np.load(npz_file_path)
    
    k = data['k']
    E_ref_mean = data['E_ref_mean']
    E_ref_std = data['E_ref_std']
    E_pred_mean = data['E_pred_mean']
    E_pred_std = data['E_pred_std']
    
    print(f"  - 波數範圍: [{k[0]:.2f}, {k[-1]:.2f}]")
    print(f"  - 數據點數: {len(k)}")
    print()
    
    # 過濾有效數據
    mask = (E_ref_mean > 1e-12) & (E_pred_mean > 1e-12) & (k > 0)
    k_valid = k[mask]
    E_ref_valid = E_ref_mean[mask]
    E_pred_valid = E_pred_mean[mask]
    
    print(f"有效數據點數: {len(k_valid)}")
    print()
    
    # === 擬合策略：選擇慣性次範圍 ===
    # 1. 跳過大尺度（能量注入範圍）：k < k_min
    # 2. 跳過小尺度（耗散範圍）：k > k_max
    # 3. 慣性次範圍通常在中間的對數範圍
    
    # 啟發式選擇：使用中間 50% 的對數範圍
    log_k = np.log10(k_valid)
    log_k_min = log_k[0]
    log_k_max = log_k[-1]
    log_k_range = log_k_max - log_k_min
    
    # 慣性次範圍：從 25% 到 75% 的對數範圍
    log_k_inertial_min = log_k_min + 0.25 * log_k_range
    log_k_inertial_max = log_k_min + 0.75 * log_k_range
    
    mask_inertial = (log_k >= log_k_inertial_min) & (log_k <= log_k_inertial_max)
    
    k_inertial = k_valid[mask_inertial]
    E_ref_inertial = E_ref_valid[mask_inertial]
    E_pred_inertial = E_pred_valid[mask_inertial]
    
    print(f"慣性次範圍估計:")
    print(f"  - k 範圍: [{k_inertial[0]:.2f}, {k_inertial[-1]:.2f}]")
    print(f"  - 數據點數: {len(k_inertial)}")
    print()
    
    # === 擬合 DNS 能量譜 ===
    print("擬合 DNS 能量譜...")
    
    # 使用對數線性回歸（更穩定）
    log_k_inertial = np.log10(k_inertial)
    log_E_ref_inertial = np.log10(E_ref_inertial)
    
    slope_ref, intercept_ref, r_value_ref, p_value_ref, std_err_ref = linregress(
        log_k_inertial, log_E_ref_inertial
    )
    
    # 將對數斜率轉換回冪律指數
    alpha_ref = slope_ref
    A_ref = 10**intercept_ref
    r2_ref = r_value_ref**2
    
    print(f"  DNS: E(k) = {A_ref:.4e} * k^({alpha_ref:.4f})")
    print(f"  R² = {r2_ref:.6f}")
    print(f"  理論 Kolmogorov: k^(-5/3) = k^({-5/3:.4f})")
    print(f"  偏差: {abs(alpha_ref - (-5/3)):.4f}")
    print()
    
    # === 擬合 PINN 能量譜 ===
    print("擬合 PINN 能量譜...")
    
    log_E_pred_inertial = np.log10(E_pred_inertial)
    
    slope_pred, intercept_pred, r_value_pred, p_value_pred, std_err_pred = linregress(
        log_k_inertial, log_E_pred_inertial
    )
    
    alpha_pred = slope_pred
    A_pred = 10**intercept_pred
    r2_pred = r_value_pred**2
    
    print(f"  PINN: E(k) = {A_pred:.4e} * k^({alpha_pred:.4f})")
    print(f"  R² = {r2_pred:.6f}")
    print(f"  理論 Kolmogorov: k^(-5/3) = k^({-5/3:.4f})")
    print(f"  偏差: {abs(alpha_pred - (-5/3)):.4f}")
    print()
    
    # === 生成擬合曲線 ===
    k_fit = np.logspace(np.log10(k_inertial[0]), np.log10(k_inertial[-1]), 100)
    E_ref_fit = A_ref * k_fit**alpha_ref
    E_pred_fit = A_pred * k_fit**alpha_pred
    E_kolmogorov = A_ref * k_fit**(-5/3)  # 使用 DNS 的振幅
    
    # === 繪製能量譜和擬合結果 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 左圖：完整能量譜 + 擬合 ---
    ax1.loglog(k_valid, E_ref_valid, 'b-', linewidth=2, label='DNS', 
               marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    ax1.loglog(k_valid, E_pred_valid, 'r--', linewidth=2, label='PINN', 
               marker='s', markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    
    # 標記慣性次範圍
    ax1.axvline(k_inertial[0], color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Inertial subrange')
    ax1.axvline(k_inertial[-1], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvspan(k_inertial[0], k_inertial[-1], alpha=0.1, color='green')
    
    # 繪製擬合曲線
    ax1.loglog(k_fit, E_ref_fit, 'b:', linewidth=2.5, alpha=0.8,
               label=f'DNS fit: $k^{{{alpha_ref:.3f}}}$ (R²={r2_ref:.4f})')
    ax1.loglog(k_fit, E_pred_fit, 'r:', linewidth=2.5, alpha=0.8,
               label=f'PINN fit: $k^{{{alpha_pred:.3f}}}$ (R²={r2_pred:.4f})')
    ax1.loglog(k_fit, E_kolmogorov, 'k--', linewidth=2, alpha=0.6,
               label=r'Kolmogorov: $k^{-5/3}$')
    
    ax1.set_xlabel('Wavenumber (k)', fontsize=14)
    ax1.set_ylabel('Energy Spectrum E(k)', fontsize=14)
    ax1.legend(fontsize=10, loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax1.set_title(f'Energy Spectrum with Kolmogorov -5/3 Fit ({config_name.upper()})', 
                  fontsize=14, fontweight='bold', pad=10)
    ax1.tick_params(labelsize=12)
    
    # --- 右圖：慣性次範圍放大 + 補償譜 ---
    # 補償譜：E(k) * k^(5/3) 應該是常數（如果符合 k^(-5/3)）
    E_ref_compensated = E_ref_inertial * k_inertial**(5/3)
    E_pred_compensated = E_pred_inertial * k_inertial**(5/3)
    
    ax2.semilogx(k_inertial, E_ref_compensated, 'b-', linewidth=2.5, 
                 label='DNS', marker='o', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    ax2.semilogx(k_inertial, E_pred_compensated, 'r--', linewidth=2.5, 
                 label='PINN', marker='s', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    
    # 如果完美符合 k^(-5/3)，補償譜應該是水平線
    ax2.axhline(np.mean(E_ref_compensated), color='b', linestyle=':', linewidth=2, alpha=0.5,
                label=f'DNS mean: {np.mean(E_ref_compensated):.4e}')
    ax2.axhline(np.mean(E_pred_compensated), color='r', linestyle=':', linewidth=2, alpha=0.5,
                label=f'PINN mean: {np.mean(E_pred_compensated):.4e}')
    
    ax2.set_xlabel('Wavenumber (k)', fontsize=14)
    ax2.set_ylabel(r'$E(k) \cdot k^{5/3}$ (Compensated Spectrum)', fontsize=14)
    ax2.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_title(f'Compensated Spectrum (Inertial Subrange)', fontsize=14, fontweight='bold', pad=10)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # 保存圖形
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'kolmogorov_fit_{config_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Kolmogorov 擬合圖已保存: {output_path}")
    
    plt.close()
    
    # === 保存擬合結果 ===
    fit_results = {
        'k_inertial': k_inertial,
        'E_ref_inertial': E_ref_inertial,
        'E_pred_inertial': E_pred_inertial,
        'alpha_ref': alpha_ref,
        'A_ref': A_ref,
        'r2_ref': r2_ref,
        'alpha_pred': alpha_pred,
        'A_pred': A_pred,
        'r2_pred': r2_pred,
        'kolmogorov_exponent': -5/3,
        'deviation_ref': abs(alpha_ref - (-5/3)),
        'deviation_pred': abs(alpha_pred - (-5/3))
    }
    
    fit_results_path = os.path.join(output_dir, f'kolmogorov_fit_results_{config_name}.npz')
    np.savez(fit_results_path, **fit_results)
    print(f"✓ 擬合結果已保存: {fit_results_path}")
    
    # === 生成文字報告 ===
    report_path = os.path.join(output_dir, f'KOLMOGOROV_ANALYSIS_{config_name.upper()}.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# Kolmogorov -5/3 律分析報告 ({config_name.upper()})\n\n")
        f.write("## 概述\n\n")
        f.write("本報告分析 DNS 參考解和 PINN 預測的能量譜是否符合 Kolmogorov 湍流理論中的 **-5/3 冪律**。\n\n")
        f.write("根據 Kolmogorov 1941 理論，在充分發展的湍流中，能量譜在慣性次範圍（inertial subrange）應該遵循：\n\n")
        f.write("$$E(k) \\sim k^{-5/3}$$\n\n")
        f.write("---\n\n")
        
        f.write("## 數據信息\n\n")
        f.write(f"- **波數範圍**: [{k[0]:.2f}, {k[-1]:.2f}]\n")
        f.write(f"- **總數據點數**: {len(k)}\n")
        f.write(f"- **有效數據點數**: {len(k_valid)}\n")
        f.write(f"- **慣性次範圍**: [{k_inertial[0]:.2f}, {k_inertial[-1]:.2f}] ({len(k_inertial)} 點)\n\n")
        f.write("---\n\n")
        
        f.write("## 擬合結果\n\n")
        f.write("### DNS 參考解\n\n")
        f.write(f"**擬合公式**: $E(k) = {A_ref:.4e} \\cdot k^{{{alpha_ref:.4f}}}$\n\n")
        f.write(f"- **冪律指數 (α)**: {alpha_ref:.4f}\n")
        f.write(f"- **理論值 (Kolmogorov)**: {-5/3:.4f}\n")
        f.write(f"- **偏差**: {abs(alpha_ref - (-5/3)):.4f}\n")
        f.write(f"- **擬合優度 (R²)**: {r2_ref:.6f}\n\n")
        
        if r2_ref > 0.95:
            f.write("✅ **結論**: DNS 能量譜在慣性次範圍展現出**優秀的擬合**。\n\n")
        elif r2_ref > 0.90:
            f.write("✅ **結論**: DNS 能量譜在慣性次範圍展現出**良好的擬合**。\n\n")
        else:
            f.write("⚠️ **結論**: DNS 能量譜在慣性次範圍的擬合**有待改善**。\n\n")
        
        f.write("### PINN 預測\n\n")
        f.write(f"**擬合公式**: $E(k) = {A_pred:.4e} \\cdot k^{{{alpha_pred:.4f}}}$\n\n")
        f.write(f"- **冪律指數 (α)**: {alpha_pred:.4f}\n")
        f.write(f"- **理論值 (Kolmogorov)**: {-5/3:.4f}\n")
        f.write(f"- **偏差**: {abs(alpha_pred - (-5/3)):.4f}\n")
        f.write(f"- **擬合優度 (R²)**: {r2_pred:.6f}\n\n")
        
        if r2_pred > 0.95:
            f.write("✅ **結論**: PINN 能量譜在慣性次範圍展現出**優秀的擬合**。\n\n")
        elif r2_pred > 0.90:
            f.write("✅ **結論**: PINN 能量譜在慣性次範圍展現出**良好的擬合**。\n\n")
        else:
            f.write("⚠️ **結論**: PINN 能量譜在慣性次範圍的擬合**有待改善**。\n\n")
        
        f.write("---\n\n")
        
        f.write("## 比較分析\n\n")
        f.write("| 指標 | DNS | PINN | 差異 |\n")
        f.write("|------|-----|------|------|\n")
        f.write(f"| 冪律指數 (α) | {alpha_ref:.4f} | {alpha_pred:.4f} | {abs(alpha_ref - alpha_pred):.4f} |\n")
        f.write(f"| 與理論偏差 | {abs(alpha_ref - (-5/3)):.4f} | {abs(alpha_pred - (-5/3)):.4f} | - |\n")
        f.write(f"| R² | {r2_ref:.6f} | {r2_pred:.6f} | {abs(r2_ref - r2_pred):.6f} |\n\n")
        
        # 判斷 PINN 是否捕捉到 Kolmogorov 規律
        deviation_diff = abs(alpha_pred - (-5/3)) - abs(alpha_ref - (-5/3))
        
        f.write("### 物理意義評估\n\n")
        
        if abs(deviation_diff) < 0.1:
            f.write("✅ **PINN 成功捕捉了 Kolmogorov -5/3 律**，與 DNS 參考解的準確度相當。\n\n")
        elif deviation_diff < 0:
            f.write("✅ **PINN 甚至比 DNS 更接近理論 -5/3 律**（可能是數值偶然或 DNS 本身的有限 Re 效應）。\n\n")
        else:
            f.write(f"⚠️ **PINN 偏離 Kolmogorov 律的程度比 DNS 大 {deviation_diff:.4f}**，表明在慣性次範圍的物理準確性有所下降。\n\n")
        
        f.write("---\n\n")
        
        f.write("## 技術細節\n\n")
        f.write("### 慣性次範圍選擇方法\n\n")
        f.write("本分析使用啟發式方法選擇慣性次範圍：\n\n")
        f.write("1. 計算波數的對數範圍 $\\log_{10}(k)$\n")
        f.write("2. 選擇中間 50% 的對數範圍（跳過前 25% 和後 25%）\n")
        f.write("3. 這樣可以避開能量注入範圍（大尺度）和耗散範圍（小尺度）\n\n")
        f.write("### 擬合方法\n\n")
        f.write("使用**對數線性回歸**（log-log plot 上的線性擬合）：\n\n")
        f.write("$$\\log_{10} E(k) = \\alpha \\cdot \\log_{10} k + \\log_{10} A$$\n\n")
        f.write("這比直接非線性擬合更穩定，且對數變換後的 R² 更能反映冪律擬合的質量。\n\n")
        f.write("### 補償譜\n\n")
        f.write("補償譜定義為：\n\n")
        f.write("$$E_{comp}(k) = E(k) \\cdot k^{5/3}$$\n\n")
        f.write("如果能量譜精確遵循 $k^{-5/3}$，則補償譜應該是常數（水平線）。\n")
        f.write("補償譜的波動程度反映了實際譜與 Kolmogorov 理論的偏離程度。\n\n")
        f.write("---\n\n")
        
        f.write("## 參考文獻\n\n")
        f.write("1. Kolmogorov, A. N. (1941). *The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers.* Doklady Akademii Nauk SSSR, 30, 301-305.\n")
        f.write("2. Pope, S. B. (2000). *Turbulent Flows.* Cambridge University Press.\n")
        f.write("3. Sagaut, P., & Cambon, C. (2008). *Homogeneous Turbulence Dynamics.* Cambridge University Press.\n\n")
    
    print(f"✓ 分析報告已保存: {report_path}")
    print()
    print("=" * 80)
    print("完成!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='分析 Kolmogorov -5/3 律')
    parser.add_argument('--data_file', type=str, required=True,
                        help='時間平均能量譜數據文件路徑 (.npz)')
    parser.add_argument('--config', type=str, required=True, choices=['soap', 'pirate'],
                        help='配置名稱')
    parser.add_argument('--output_dir', type=str, default='examples/kolmogorov_flow/comparison',
                        help='輸出目錄')
    
    args = parser.parse_args()
    
    analyze_kolmogorov_spectrum(args.data_file, args.config, args.output_dir)


if __name__ == '__main__':
    main()
