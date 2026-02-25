#!/usr/bin/env python3
"""
比較 PIRATE 和 SOAP 訓練結果
"""
import sys

def main():
    print("=" * 80)
    print("PIRATE vs SOAP - Training Results Comparison")
    print("Kolmogorov Flow at Re=10,000")
    print("=" * 80)
    print()
    
    print("Training Configuration:")
    print("-" * 80)
    print(f"{'Metric':<30} {'PIRATE':<25} {'SOAP':<25}")
    print("-" * 80)
    print(f"{'Optimizer':<30} {'Adam':<25} {'SOAP (schedule-free)':<25}")
    print(f"{'Total Windows':<30} {'10':<25} {'25 (17 completed)':<25}")
    print(f"{'Iterations per Window':<30} {'20,000':<25} {'20,000':<25}")
    print(f"{'Total Iterations':<30} {'200,000':<25} {'500,000 (target)':<25}")
    print(f"{'Hidden Dimension':<30} {'256':<25} {'256':<25}")
    print(f"{'Batch Size (total)':<30} {'8,192':<25} {'8,192':<25}")
    print(f"{'Training Time':<30} {'8h 12m':<25} {'~20h (in progress)':<25}")
    print()
    
    print("Final Errors by Window:")
    print("-" * 80)
    
    # PIRATE 數據（10 個窗口）
    pirate_data = [
        (1,  0.005202, 0.005086, 0.013720),
        (2,  0.007983, 0.014150, 0.048280),
        (3,  0.011940, 0.022410, 0.091390),
        (4,  0.016530, 0.028460, 0.122700),
        (5,  0.025750, 0.027440, 0.142000),
        (6,  0.052830, 0.055250, 0.181400),
        (7,  0.082130, 0.140400, 0.365600),
        (8,  0.155700, 0.225200, 0.574300),
        (9,  0.314500, 0.312200, 0.855100),
        (10, 0.351500, 0.334900, 0.935500),
    ]
    
    # SOAP 數據（17 個窗口已完成）
    soap_data = [
        (1,  0.002386, 0.002383, 0.004596),
        (2,  0.004212, 0.004460, 0.012070),
        (3,  0.005751, 0.007307, 0.027660),
        (4,  0.007416, 0.012330, 0.045080),
        (5,  0.008897, 0.016960, 0.060820),
        (6,  0.011060, 0.020910, 0.079340),
        (7,  0.012820, 0.025360, 0.102400),
        (8,  0.013960, 0.027110, 0.111900),
        (9,  0.015420, 0.026800, 0.117100),
        (10, 0.017400, 0.026330, 0.128900),
        (11, 0.019880, 0.025320, 0.138400),
        (12, 0.025670, 0.025670, 0.144100),
        (13, 0.035500, 0.031420, 0.144900),
        (14, 0.046590, 0.043330, 0.160300),
        (15, 0.055810, 0.067350, 0.205500),
        (16, 0.067140, 0.103700, 0.289400),
        (17, 0.079600, 0.144300, 0.377500),
    ]
    
    print(f"{'Window':<8} {'PIRATE u_error':<18} {'SOAP u_error':<18} {'Improvement':<15}")
    print("-" * 80)
    
    for i in range(10):
        p_w, p_u, p_v, p_w_err = pirate_data[i]
        s_w, s_u, s_v, s_w_err = soap_data[i]
        improvement = (p_u - s_u) / p_u * 100
        print(f"{p_w:<8} {p_u:<18.6f} {s_u:<18.6f} {improvement:>6.1f}%")
    
    print()
    print(f"{'Window':<8} {'PIRATE v_error':<18} {'SOAP v_error':<18} {'Improvement':<15}")
    print("-" * 80)
    
    for i in range(10):
        p_w, p_u, p_v, p_w_err = pirate_data[i]
        s_w, s_u, s_v, s_w_err = soap_data[i]
        improvement = (p_v - s_v) / p_v * 100
        print(f"{p_w:<8} {p_v:<18.6f} {s_v:<18.6f} {improvement:>6.1f}%")
    
    print()
    print(f"{'Window':<8} {'PIRATE w_error':<18} {'SOAP w_error':<18} {'Improvement':<15}")
    print("-" * 80)
    
    for i in range(10):
        p_w, p_u, p_v, p_w_err = pirate_data[i]
        s_w, s_u, s_v, s_w_err = soap_data[i]
        improvement = (p_w_err - s_w_err) / p_w_err * 100
        print(f"{p_w:<8} {p_w_err:<18.6f} {s_w_err:<18.6f} {improvement:>6.1f}%")
    
    print()
    print("=" * 80)
    print("Summary Statistics (First 10 Windows)")
    print("=" * 80)
    
    # 計算前 10 個窗口的統計
    pirate_u = [d[1] for d in pirate_data]
    pirate_v = [d[2] for d in pirate_data]
    pirate_w = [d[3] for d in pirate_data]
    
    soap_u = [d[1] for d in soap_data[:10]]
    soap_v = [d[2] for d in soap_data[:10]]
    soap_w = [d[3] for d in soap_data[:10]]
    
    print(f"\n{'Metric':<20} {'PIRATE':<20} {'SOAP':<20} {'Improvement':<15}")
    print("-" * 80)
    
    # u_error
    p_u_mean = sum(pirate_u) / len(pirate_u)
    s_u_mean = sum(soap_u) / len(soap_u)
    u_improvement = (p_u_mean - s_u_mean) / p_u_mean * 100
    print(f"{'u_error (mean)':<20} {p_u_mean:<20.6f} {s_u_mean:<20.6f} {u_improvement:>6.1f}%")
    print(f"{'u_error (final)':<20} {pirate_u[-1]:<20.6f} {soap_u[-1]:<20.6f} {((pirate_u[-1]-soap_u[-1])/pirate_u[-1]*100):>6.1f}%")
    
    # v_error
    p_v_mean = sum(pirate_v) / len(pirate_v)
    s_v_mean = sum(soap_v) / len(soap_v)
    v_improvement = (p_v_mean - s_v_mean) / p_v_mean * 100
    print(f"{'v_error (mean)':<20} {p_v_mean:<20.6f} {s_v_mean:<20.6f} {v_improvement:>6.1f}%")
    print(f"{'v_error (final)':<20} {pirate_v[-1]:<20.6f} {soap_v[-1]:<20.6f} {((pirate_v[-1]-soap_v[-1])/pirate_v[-1]*100):>6.1f}%")
    
    # w_error
    p_w_mean = sum(pirate_w) / len(pirate_w)
    s_w_mean = sum(soap_w) / len(soap_w)
    w_improvement = (p_w_mean - s_w_mean) / p_w_mean * 100
    print(f"{'w_error (mean)':<20} {p_w_mean:<20.6f} {s_w_mean:<20.6f} {w_improvement:>6.1f}%")
    print(f"{'w_error (final)':<20} {pirate_w[-1]:<20.6f} {soap_w[-1]:<20.6f} {((pirate_w[-1]-soap_w[-1])/pirate_w[-1]*100):>6.1f}%")
    
    print()
    print("=" * 80)
    print("SOAP Extended Windows (11-17)")
    print("=" * 80)
    print(f"\n{'Window':<8} {'u_error':<15} {'v_error':<15} {'w_error':<15}")
    print("-" * 80)
    
    for i in range(10, 17):
        s_w, s_u, s_v, s_w_err = soap_data[i]
        print(f"{s_w:<8} {s_u:<15.6f} {s_v:<15.6f} {s_w_err:<15.6f}")
    
    print()
    print("=" * 80)
    print("Key Findings:")
    print("=" * 80)
    print()
    print("1. Accuracy Improvement:")
    print(f"   - SOAP achieves {u_improvement:.1f}% better u_error on average")
    print(f"   - SOAP achieves {v_improvement:.1f}% better v_error on average")
    print(f"   - SOAP achieves {w_improvement:.1f}% better w_error on average")
    print()
    print("2. Error Stability:")
    print(f"   - PIRATE: errors grow rapidly after window 6 (u: {pirate_u[5]:.3f}→{pirate_u[-1]:.3f})")
    print(f"   - SOAP: shows more gradual error growth (u: {soap_u[5]:.3f}→{soap_u[-1]:.3f})")
    print()
    print("3. Final Window Performance (Window 10):")
    print(f"   - PIRATE: u={pirate_u[-1]:.3f}, v={pirate_v[-1]:.3f}, w={pirate_w[-1]:.3f}")
    print(f"   - SOAP: u={soap_u[-1]:.3f}, v={soap_v[-1]:.3f}, w={soap_w[-1]:.3f}")
    print(f"   - SOAP achieves {((pirate_u[-1]-soap_u[-1])/pirate_u[-1]*100):.1f}% reduction in u_error")
    print()
    print("4. Training Efficiency:")
    print("   - PIRATE: 8h 12m for 10 windows (~49 min/window)")
    print("   - SOAP: ~20h for 17 windows (~70 min/window, 43% slower per window)")
    print("   - Trade-off: SOAP is slower but achieves significantly better accuracy")
    print()
    print("5. Status:")
    print("   - PIRATE: ✅ Completed (10/10 windows)")
    print("   - SOAP: 🟡 In Progress (17/25 windows, 68% complete)")
    print("   - Expected SOAP completion: ~6-8 hours from now")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
