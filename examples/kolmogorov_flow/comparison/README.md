# DNS vs PINN 对比分析结果

## 📊 概览

本目录包含 Kolmogorov Flow (Re=10000) 的 DNS 数据与 PINN 预测结果的对比分析。

## 📁 文件说明

### 1. `dns_vs_pinn_comparison_pirate.png` (474 KB)
主对比图，包含四个子图（2×2布局）：

```
┌─────────────────────┬─────────────────────┐
│  Relative L2 Error  │  Kinetic Energy     │
│  (u & v velocities) │  (DNS vs PINN)      │
├─────────────────────┼─────────────────────┤
│  Enstrophy          │  Energy Spectrum    │
│  (vorticity energy) │  (wavenumber k)     │
└─────────────────────┴─────────────────────┘
```

**图像规格**:
- 分辨率: 3451 × 2840 像素
- 格式: PNG (高清)
- 布局: 2×2 网格，避免挤压

### 2. `comparison_data_pirate.npz` (4.0 KB)
NumPy 压缩数据文件，包含：

| 数组名 | 形状 | 说明 |
|--------|------|------|
| `times` | (10,) | 采样时间点 [0.0, 1.8] 秒 |
| `errors_u` | (10,) | u 速度的相对 L2 误差 |
| `errors_v` | (10,) | v 速度的相对 L2 误差 |
| `errors_w` | (10,) | w 涡度的相对 L2 误差 |
| `ke_ref` | (10,) | DNS 动能 |
| `ke_pred` | (10,) | PINN 预测动能 |
| `ens_ref` | (10,) | DNS 涡度拟能 |
| `ens_pred` | (10,) | PINN 预测涡度拟能 |
| `spectrum_k` | (27,) | 能量谱波数 |
| `spectrum_ref` | (27,) | DNS 能量谱 |
| `spectrum_pred` | (27,) | PINN 预测能量谱 |

**加载示例**:
```python
import numpy as np
data = np.load('comparison_data_pirate.npz')
times = data['times']
errors_u = data['errors_u']
# ...
```

### 3. `ANALYSIS_REPORT_pirate.md` (6.3 KB)
详细分析报告，包含：
- 四张图的详细解读
- 误差统计分析
- 物理意义解释
- 改进建议

## 📈 关键结果摘要

### 误差统计

| 变量 | 平均误差 | 最小误差 | 最大误差 |
|------|---------|---------|---------|
| **u 速度** | 51.1% | 0.4% | 80.1% |
| **v 速度** | 60.9% | 0.4% | 91.2% |
| **w 涡度** | 88.5% | 0.5% | 113.1% |

### 全局量对比

| 物理量 | DNS 趋势 | PINN 趋势 | 平均相对误差 |
|--------|---------|-----------|-------------|
| **动能** | 0.161 → 1.016 | 0.161 → 0.974 | 10.5% |
| **涡度拟能** | 144 → 124 (峰值 186) | 145 → 103 (峰值 183) | 6.6% |

### 能量谱

- **大尺度** (k < 10): 相对误差 ~16.7%  ✅
- **小尺度** (k ≥ 10): 相对误差 >200%  ⚠️
- **谱斜率**: DNS (-8.82) vs PINN (-5.76) vs 理论 (-3 或 -5/3)

## 🎯 主要结论

### ✅ 成功之处
1. **初始条件**: 误差 < 0.5%
2. **全局趋势**: 动能增长、涡度演化趋势正确
3. **大尺度结构**: 能量谱低波数区域匹配良好

### ⚠️ 挑战
1. **时间外推**: 窗口边界误差跳跃
2. **小尺度结构**: 涡度和高波数能量谱误差大
3. **耗散建模**: 后期涡度拟能衰减过快

## 🔧 重现步骤

### 生成对比图

在伺服器上运行：

```bash
cd ~/jaxpi
python3 examples/kolmogorov_flow/generate_comparison_plots.py \
    --config pirate \
    --checkpoint_path ~/jaxpi/pirate/ckpt \
    --output_dir ~/jaxpi/examples/kolmogorov_flow/comparison \
    --time_interval 0.5
```

### 参数说明

- `--config`: 配置名称 (`pirate` 或 `soap`)
- `--checkpoint_path`: checkpoint 根目录
- `--output_dir`: 输出目录
- `--time_interval`: 采样时间间隔（秒，默认 0.5）

## 📚 相关文档

- 主项目 README: `../../README.md`
- DNS 数据验证: `../DNS_DATA_VERIFICATION.md`
- 配置文件: `../configs/pirate.py`

## 📝 引用

如使用本分析结果，请引用：

```
Kolmogorov Flow PINN Analysis (Re=10000)
Configuration: PIRATE with 10 time windows
Date: 2026-01-07
```

---

*生成时间: 2026-01-07*  
*工具版本: generate_comparison_plots.py v1.1*
