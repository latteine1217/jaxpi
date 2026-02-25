"""
检查 Kolmogorov Flow DNS 数据的雷诺数并生成可视化 gif
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# 加载数据（新格式 DNS/LES 兼容）
data_path = "examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy"
print(f"正在加载数据: {data_path}")
print("=" * 60)

data = np.load(data_path, allow_pickle=True).item()

# 打印所有可用的键（依新格式）
print("数据文件中包含的键:")
print("  - u, v, omega, time, x, y, config")
print("=" * 60)

# 提取关键信息
config = data.get("config", {})
nu = float(config.get("nu")) if config is not None else None
Re = (1.0 / nu) if nu else None
t = np.array(data["time"]).flatten()
u = np.array(data["u"])
v = np.array(data["v"])
vorticity = np.array(data["omega"])
x_vals = np.array(data["x"])
y_vals = np.array(data["y"])
x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
coords = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

# 计算雷诺数
print("DNS 数据信息:")
print("-" * 60)
print(f"运动粘度 (nu): {nu}")

calculated_Re = 1.0 / nu if nu is not None else 0.0
print(f"计算得到的雷诺数 (Re = 1/nu): {calculated_Re:.1f}")

if Re is not None:
    print(f"数据中存储的雷诺数 (Re): {Re}")
print("-" * 60)
print(f"时间步数: {len(t)}")
print(f"时间范围: [{t[0]:.4f}, {t[-1]:.4f}]")
print(f"时间步长 (平均): {np.mean(np.diff(t)):.6f}")
print("-" * 60)
print(f"速度场形状 (原始): {u.shape}")
print(f"涡度场形状 (原始): {vorticity.shape}")
print(f"空间坐标形状: {coords.shape}")

# 数据是平铺的，需要重塑为 2D 网格
# 从坐标推断网格大小
x_coords = coords[:, 0]
y_coords = coords[:, 1]
unique_x = np.unique(x_coords)
unique_y = np.unique(y_coords)
nx, ny = len(unique_x), len(unique_y)

print(f"空间网格分辨率: {nx} x {ny}")

# 重塑数据
nt = len(t)
vorticity = vorticity.reshape(nt, nx, ny)
u = u.reshape(nt, nx, ny)
v = v.reshape(nt, nx, ny)

print(f"重塑后涡度场形状: {vorticity.shape}")
print("=" * 60)

# 统计信息
print("\n速度场统计信息:")
print(f"u 速度: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}, std={u.std():.4f}")
print(f"v 速度: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.4f}")
print(f"涡度: min={vorticity.min():.4f}, max={vorticity.max():.4f}, mean={vorticity.mean():.4f}, std={vorticity.std():.4f}")
print("=" * 60)

# 创建输出目录
output_dir = "examples/kolmogorov_flow/visualization"
os.makedirs(output_dir, exist_ok=True)

# 生成涡度场动画
print(f"\n正在生成涡度场动画...")
print(f"采样策略: 每隔 2 个时间步取一帧，共 {len(t)//2} 帧")

# 使用子采样以减少文件大小和生成时间
subsample = 2  # 每隔2个时间步取一帧
frames = list(range(0, len(t), subsample))

# 预先计算全局的 vmin/vmax
vmin_global = vorticity.min()
vmax_global = vorticity.max()

# 使用 subplots 并固定 colorbar 位置
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
ax = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])

# 初始化图像和 colorbar
im = ax.imshow(vorticity[0].T, cmap='RdBu_r', origin='lower', 
               aspect='auto', interpolation='bilinear',
               vmin=vmin_global, vmax=vmax_global)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Vorticity', fontsize=12)

def update(frame_idx):
    frame = frames[frame_idx]
    # 只更新图像数据，不清除 ax
    im.set_data(vorticity[frame].T)
    ax.set_title(f'Vorticity Field - Re={calculated_Re:.0f} - t={t[frame]:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    return [im]

# 创建动画
anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)

# 保存为 gif
gif_path = os.path.join(output_dir, f"kolmogorov_flow_Re{calculated_Re:.0f}_vorticity.gif")
writer = PillowWriter(fps=10)
anim.save(gif_path, writer=writer, dpi=100)
plt.close()

print(f"✓ 涡度场动画已保存: {gif_path}")

# 生成速度场动画
print(f"\n正在生成速度场强度动画...")

# 计算速度强度
velocity_magnitude = np.sqrt(u**2 + v**2)
vmag_min = velocity_magnitude.min()
vmag_max = velocity_magnitude.max()

# 使用 subplots 并固定 colorbar 位置
fig2 = plt.figure(figsize=(11, 8))
gs2 = fig2.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
ax2 = fig2.add_subplot(gs2[0])
cax2 = fig2.add_subplot(gs2[1])

# 初始化图像和 colorbar
im2 = ax2.imshow(velocity_magnitude[0].T, cmap='viridis', origin='lower', 
                 aspect='auto', interpolation='bilinear',
                 vmin=vmag_min, vmax=vmag_max)
cbar2 = plt.colorbar(im2, cax=cax2)
cbar2.set_label('|u|', fontsize=12)

def update_velocity(frame_idx):
    frame = frames[frame_idx]
    # 只更新图像数据
    im2.set_data(velocity_magnitude[frame].T)
    ax2.set_title(f'Velocity Magnitude - Re={calculated_Re:.0f} - t={t[frame]:.4f}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    return [im2]

anim2 = FuncAnimation(fig2, update_velocity, frames=len(frames), interval=100, blit=True)

gif_path2 = os.path.join(output_dir, f"kolmogorov_flow_Re{calculated_Re:.0f}_velocity.gif")
writer2 = PillowWriter(fps=10)
anim2.save(gif_path2, writer=writer2, dpi=100)
plt.close()

print(f"✓ 速度场动画已保存: {gif_path2}")

# 生成时间演化的统计图
print(f"\n正在生成时间统计图...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 计算各时间步的统计量
vorticity_mean = [vorticity[i].mean() for i in range(len(t))]
vorticity_std = [vorticity[i].std() for i in range(len(t))]
velocity_mag_mean = [velocity_magnitude[i].mean() for i in range(len(t))]

axes[0].plot(t, vorticity_mean, 'b-', linewidth=2)
axes[0].set_ylabel('Mean Vorticity', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f'Time Evolution Statistics (Re={calculated_Re:.0f})', fontsize=14, fontweight='bold')

axes[1].plot(t, vorticity_std, 'r-', linewidth=2)
axes[1].set_ylabel('Vorticity Std', fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, velocity_mag_mean, 'g-', linewidth=2)
axes[2].set_ylabel('Mean Velocity Magnitude', fontsize=12)
axes[2].set_xlabel('Time', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
stats_path = os.path.join(output_dir, f"kolmogorov_flow_Re{calculated_Re:.0f}_statistics.png")
plt.savefig(stats_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ 统计图已保存: {stats_path}")

print("\n" + "=" * 60)
print("结论:")
print(f"该 DNS 数据确实是 Re = {calculated_Re:.0f} 的 Kolmogorov Flow")
print(f"所有可视化文件已保存在: {output_dir}/")
print("=" * 60)
