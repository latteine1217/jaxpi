"""
生成 SOAP (25窗口) vs PIRATE (10窗口) 的完整對比圖表
使用物理時間作為 X 軸
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 設置中文字體
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# PIRATE 數據 (10 windows)
pirate_windows = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
pirate_u_errors = np.array([0.005202, 0.007983, 0.011940, 0.016530, 0.025750, 
                             0.052830, 0.082130, 0.155700, 0.314500, 0.351500])
pirate_v_errors = np.array([0.005086, 0.014150, 0.022410, 0.028460, 0.027440,
                             0.055250, 0.140400, 0.225200, 0.312200, 0.334900])
pirate_w_errors = np.array([0.013720, 0.048280, 0.091390, 0.122700, 0.142000,
                             0.181400, 0.365600, 0.574300, 0.855100, 0.935500])

# SOAP 數據 (25 windows)
soap_windows = np.array(range(1, 26))
soap_u_errors = np.array([0.002386, 0.004212, 0.005751, 0.007416, 0.008897,
                          0.011060, 0.012820, 0.013960, 0.015420, 0.017400,
                          0.019880, 0.025670, 0.035500, 0.046590, 0.055810,
                          0.067140, 0.079600, 0.094330, 0.123900, 0.175400,
                          0.245100, 0.298200, 0.329400, 0.352300, 0.354600])
soap_v_errors = np.array([0.002383, 0.004460, 0.007307, 0.012330, 0.016960,
                          0.020910, 0.025360, 0.027110, 0.026800, 0.026330,
                          0.025320, 0.025670, 0.031420, 0.043330, 0.067350,
                          0.103700, 0.144300, 0.170600, 0.186300, 0.220000,
                          0.253700, 0.286600, 0.327100, 0.326500, 0.318000])
soap_w_errors = np.array([0.004596, 0.012070, 0.027660, 0.045080, 0.060820,
                          0.079340, 0.102400, 0.111900, 0.117100, 0.128900,
                          0.138400, 0.144100, 0.144900, 0.160300, 0.205500,
                          0.289400, 0.377500, 0.443100, 0.511600, 0.622100,
                          0.756300, 0.810100, 0.850600, 0.881700, 0.943400])

# 計算物理時間
# DNS: 50 time steps, t ∈ [0.002, 1.962], Δt ≈ 0.040
t_start = 0.002
t_end = 1.962
num_dns_steps = 50
dt_dns = (t_end - t_start) / (num_dns_steps - 1)

# PIRATE: 10 windows, 5 DNS steps per window
steps_per_window_pirate = 5
pirate_times = t_start + (pirate_windows - 0.5) * steps_per_window_pirate * dt_dns

# SOAP: 25 windows, 2 DNS steps per window  
steps_per_window_soap = 2
soap_times = t_start + (soap_windows - 0.5) * steps_per_window_soap * dt_dns

# 創建圖表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SOAP (25 Windows) vs PIRATE (10 Windows) - Complete Training Comparison\nRe=10,000 Kolmogorov Flow', 
             fontsize=14, fontweight='bold')

# 子圖 1: u velocity error (線性尺度)
ax = axes[0, 0]
ax.plot(pirate_times, pirate_u_errors * 100, 'o-', label='PIRATE (Adam, 10 windows)', 
        color='#e74c3c', linewidth=2, markersize=8)
ax.plot(soap_times, soap_u_errors * 100, 's-', label='SOAP (25 windows)', 
        color='#3498db', linewidth=2, markersize=6)
ax.axvline(x=0.76, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1.32, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.76, ax.get_ylim()[1]*0.95, 't≈0.76', ha='center', fontsize=9, color='gray')
ax.text(1.32, ax.get_ylim()[1]*0.95, 't≈1.32', ha='center', fontsize=9, color='gray')
ax.set_xlabel('Physical Time', fontsize=11)
ax.set_ylabel('u velocity L2 Error (%)', fontsize=11)
ax.set_title('(a) u velocity Error - Linear Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 子圖 2: v velocity error (線性尺度)
ax = axes[0, 1]
ax.plot(pirate_times, pirate_v_errors * 100, 'o-', label='PIRATE (Adam, 10 windows)', 
        color='#e74c3c', linewidth=2, markersize=8)
ax.plot(soap_times, soap_v_errors * 100, 's-', label='SOAP (25 windows)', 
        color='#3498db', linewidth=2, markersize=6)
ax.axvline(x=0.76, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1.32, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Physical Time', fontsize=11)
ax.set_ylabel('v velocity L2 Error (%)', fontsize=11)
ax.set_title('(b) v velocity Error - Linear Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 子圖 3: 渦度誤差 (線性尺度)
ax = axes[1, 0]
ax.plot(pirate_times, pirate_w_errors * 100, 'o-', label='PIRATE (Adam, 10 windows)', 
        color='#e74c3c', linewidth=2, markersize=8)
ax.plot(soap_times, soap_w_errors * 100, 's-', label='SOAP (25 windows)', 
        color='#3498db', linewidth=2, markersize=6)
ax.axvline(x=0.76, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=1.32, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Physical Time', fontsize=11)
ax.set_ylabel('Vorticity L2 Error (%)', fontsize=11)
ax.set_title('(c) Vorticity Error - Linear Scale', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 子圖 4: 時間對齊比較
ax = axes[1, 1]

# 在關鍵時間點對比
comparison_times = [0.76, 1.32, 1.90]
pirate_indices = [3, 6, 9]  # 對應 window 4, 7, 10
soap_indices = [9, 16, 24]  # 對應 window 10, 17, 25

x = np.arange(len(comparison_times))
width = 0.35

# u velocity 對比
pirate_u_at_times = [pirate_u_errors[i] * 100 for i in pirate_indices]
soap_u_at_times = [soap_u_errors[i] * 100 for i in soap_indices]

bars1 = ax.bar(x - width/2, pirate_u_at_times, width, label='PIRATE', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, soap_u_at_times, width, label='SOAP', color='#3498db', alpha=0.8)

# 添加數值標籤
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('u velocity L2 Error (%)', fontsize=11)
ax.set_title('(d) Time-Aligned Comparison at Key Points', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f't≈{t:.2f}' for t in comparison_times])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('training_comparison_complete.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: training_comparison_complete.png")

# 創建第二個圖表：對數尺度
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Error Growth Comparison (Log Scale) - SOAP (25 Windows) vs PIRATE (10 Windows)', 
             fontsize=14, fontweight='bold')

# u velocity (對數)
ax = axes[0]
ax.semilogy(pirate_times, pirate_u_errors * 100, 'o-', label='PIRATE', 
            color='#e74c3c', linewidth=2, markersize=8)
ax.semilogy(soap_times, soap_u_errors * 100, 's-', label='SOAP', 
            color='#3498db', linewidth=2, markersize=6)
ax.axvline(x=0.76, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1.32, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Physical Time', fontsize=11)
ax.set_ylabel('u velocity L2 Error (%) [log scale]', fontsize=11)
ax.set_title('(a) u velocity', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# v velocity (對數)
ax = axes[1]
ax.semilogy(pirate_times, pirate_v_errors * 100, 'o-', label='PIRATE', 
            color='#e74c3c', linewidth=2, markersize=8)
ax.semilogy(soap_times, soap_v_errors * 100, 's-', label='SOAP', 
            color='#3498db', linewidth=2, markersize=6)
ax.axvline(x=0.76, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1.32, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Physical Time', fontsize=11)
ax.set_ylabel('v velocity L2 Error (%) [log scale]', fontsize=11)
ax.set_title('(b) v velocity', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# 渦度 (對數)
ax = axes[2]
ax.semilogy(pirate_times, pirate_w_errors * 100, 'o-', label='PIRATE', 
            color='#e74c3c', linewidth=2, markersize=8)
ax.semilogy(soap_times, soap_w_errors * 100, 's-', label='SOAP', 
            color='#3498db', linewidth=2, markersize=6)
ax.axvline(x=0.76, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1.32, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Physical Time', fontsize=11)
ax.set_ylabel('Vorticity L2 Error (%) [log scale]', fontsize=11)
ax.set_title('(c) Vorticity', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('error_growth_logscale_complete.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: error_growth_logscale_complete.png")

# 打印關鍵發現
print("\n" + "="*80)
print("關鍵發現 - 時間對齊比較")
print("="*80)
print(f"\n在 t≈0.76:")
print(f"  PIRATE (Window 4):  u={pirate_u_errors[3]*100:.2f}%, v={pirate_v_errors[3]*100:.2f}%, w={pirate_w_errors[3]*100:.2f}%")
print(f"  SOAP   (Window 10): u={soap_u_errors[9]*100:.2f}%, v={soap_v_errors[9]*100:.2f}%, w={soap_w_errors[9]*100:.2f}%")
print(f"  差異: u={abs(pirate_u_errors[3]-soap_u_errors[9])/pirate_u_errors[3]*100:.1f}%")

print(f"\n在 t≈1.32:")
print(f"  PIRATE (Window 7):  u={pirate_u_errors[6]*100:.2f}%, v={pirate_v_errors[6]*100:.2f}%, w={pirate_w_errors[6]*100:.2f}%")
print(f"  SOAP   (Window 17): u={soap_u_errors[16]*100:.2f}%, v={soap_v_errors[16]*100:.2f}%, w={soap_w_errors[16]*100:.2f}%")
print(f"  差異: u={abs(pirate_u_errors[6]-soap_u_errors[16])/pirate_u_errors[6]*100:.1f}%")

print(f"\n在 t≈1.90:")
print(f"  PIRATE (Window 10): u={pirate_u_errors[9]*100:.2f}%, v={pirate_v_errors[9]*100:.2f}%, w={pirate_w_errors[9]*100:.2f}%")
print(f"  SOAP   (Window 25): u={soap_u_errors[24]*100:.2f}%, v={soap_v_errors[24]*100:.2f}%, w={soap_w_errors[24]*100:.2f}%")
print(f"  差異: u={abs(pirate_u_errors[9]-soap_u_errors[24])/pirate_u_errors[9]*100:.1f}%")

print("\n" + "="*80)
print("結論:")
print("="*80)
print("✓ 在相同物理時間點，SOAP 和 Adam 優化器的性能差異 <10%")
print("✓ SOAP 配置的主要優勢來自 2.5× 更多的時間窗口（25 vs 10）")
print("✓ 時間窗口離散化策略的影響遠大於優化器選擇")
print("="*80)
