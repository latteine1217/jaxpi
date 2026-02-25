#!/usr/bin/env python3
"""
時間對齊的公平比較：在相同物理時間評估 PIRATE 和 SOAP
"""

import os
import sys
import numpy as np
import argparse

# 添加專案路徑
sys.path.insert(0, os.path.expanduser('~/jaxpi'))
sys.path.insert(0, os.path.expanduser('~/jaxpi/examples/kolmogorov_flow'))

import jax.numpy as jnp
from jaxpi.utils import restore_checkpoint
from examples.kolmogorov_flow.utils import get_dataset
from examples.kolmogorov_flow import models


def find_window_for_time(t_target, t_star, num_windows):
    """
    找出包含目標時間的時間窗口
    
    Args:
        t_target: 目標時間
        t_star: DNS 時間數組
        num_windows: 總窗口數
        
    Returns:
        window_idx: 窗口索引（1-based）
        None if target time is out of range
    """
    num_steps_per_window = len(t_star) // num_windows
    
    for w in range(1, num_windows + 1):
        start_idx = (w - 1) * num_steps_per_window
        end_idx = w * num_steps_per_window - 1
        
        if t_star[start_idx] <= t_target <= t_star[end_idx]:
            return w
    
    return None


def evaluate_at_time(config_name, checkpoint_base, window_idx, t_target, dns_data):
    """
    在指定時間評估模型
    
    Args:
        config_name: 'pirate' 或 'soap'
        checkpoint_base: checkpoint 根目錄
        window_idx: 時間窗口索引
        t_target: 目標評估時間
        dns_data: DNS 參考數據字典
        
    Returns:
        dict: {'u_error', 'v_error', 'w_error', 't_actual'}
    """
    
    # 載入配置
    if config_name == 'pirate':
        from configs import pirate as config_module
    elif config_name == 'soap':
        from configs import soap as config_module
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = config_module.get_config()
    
    # 解析 DNS 數據
    t_star = dns_data['t']
    coords = dns_data['coords']
    u_ref_all = dns_data['u_ref']
    v_ref_all = dns_data['v_ref']
    w_ref_all = dns_data['w_ref']
    nu = dns_data['nu']
    
    # 計算時間窗口範圍
    num_steps_per_window = len(t_star) // config.training.num_time_windows
    start_idx = (window_idx - 1) * num_steps_per_window
    end_idx = window_idx * num_steps_per_window
    
    # 獲取窗口的時間和數據
    t_window = t_star[start_idx:end_idx]
    u0 = u_ref_all[start_idx, :]
    v0 = v_ref_all[start_idx, :]
    w0 = w_ref_all[start_idx, :]
    
    # 驗證目標時間在窗口範圍內
    if not (t_window[0] <= t_target <= t_window[-1]):
        raise ValueError(f"Target time {t_target:.6f} is outside window {window_idx} range [{t_window[0]:.6f}, {t_window[-1]:.6f}]")
    
    # 初始化模型
    model = models.NavierStokes(config, t_window, coords, u0, v0, w0, nu)
    
    # 載入 checkpoint
    ckpt_dir = os.path.join(checkpoint_base, f'time_window_{window_idx}')
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    model.state = restore_checkpoint(model.state, ckpt_dir, step=20000)
    
    # 找到最接近的 DNS 時間點
    dns_time_idx = np.argmin(np.abs(t_star - t_target))
    t_actual = float(t_star[dns_time_idx])
    
    # 獲取 DNS 參考數據
    u_dns = u_ref_all[dns_time_idx, :]
    v_dns = v_ref_all[dns_time_idx, :]
    w_dns = w_ref_all[dns_time_idx, :]
    
    # 在目標時間評估模型
    t_array = jnp.full((coords.shape[0],), t_actual)
    
    # 計算誤差
    u_error, v_error, w_error = model.compute_l2_error(
        model.state.params,
        t_array,
        coords,
        u_dns,
        v_dns,
        w_dns
    )
    
    return {
        'u_error': float(u_error),
        'v_error': float(v_error),
        'w_error': float(w_error),
        't_actual': t_actual,
        'window': window_idx
    }


def generate_time_aligned_comparison(soap_max_window=None):
    """
    生成時間對齊的完整比較表
    
    Args:
        soap_max_window: SOAP 的最大可用窗口（None 表示自動檢測）
    """
    
    print("=" * 100)
    print("時間對齊的 PIRATE vs SOAP 公平比較")
    print("=" * 100)
    print()
    
    # 載入 DNS 數據
    print("載入 DNS 參考數據...")
    data_path = os.path.expanduser(
        '~/jaxpi/examples/kolmogorov_flow/data/kolmogorov_dns/kolmogorov_dns_10000.npy'
    )
    data = np.load(data_path, allow_pickle=True).item()
    t_star = np.array(data['time']).flatten()
    u_ref = np.array(data['u'])
    v_ref = np.array(data['v'])
    w_ref = np.array(data['omega'])
    config = data.get('config', {})
    nu = float(config.get('nu', 0.0)) if config is not None else 0.0

    x_vals = np.array(data['x'])
    y_vals = np.array(data['y'])
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')
    coords = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    
    dns_data = {
        't': t_star,
        'coords': coords,
        'u_ref': u_ref,
        'v_ref': v_ref,
        'w_ref': w_ref,
        'nu': nu
    }
    
    print(f"DNS 數據: {len(t_star)} 時間步, t ∈ [{t_star[0]:.6f}, {t_star[-1]:.6f}]")
    print()
    
    # 檢測可用的 SOAP 窗口
    if soap_max_window is None:
        soap_ckpt_dir = os.path.expanduser('~/jaxpi/soap_Re10000/ckpt')
        soap_windows = [int(d.split('_')[-1]) for d in os.listdir(soap_ckpt_dir) 
                       if d.startswith('time_window_')]
        soap_max_window = max(soap_windows) if soap_windows else 0
    
    print(f"可用的 SOAP 窗口: 1-{soap_max_window}")
    print()
    
    # PIRATE 配置
    pirate_checkpoint_base = os.path.expanduser('~/jaxpi/pirate/ckpt')
    pirate_num_windows = 10
    
    # SOAP 配置
    soap_checkpoint_base = os.path.expanduser('~/jaxpi/soap_Re10000/ckpt')
    soap_num_windows = 25
    
    # 生成比較表
    results = []
    
    print("=" * 100)
    print("開始評估...")
    print("=" * 100)
    print()
    
    # 對每個 SOAP 窗口，找到對應時間並評估 PIRATE
    for soap_w in range(1, min(soap_max_window, soap_num_windows) + 1):
        # SOAP 窗口的結束時間（最後一個時間步）
        soap_end_idx = soap_w * 2 - 1  # SOAP 每窗口 2 步
        t_soap_end = t_star[soap_end_idx]
        
        # 找出包含此時間的 PIRATE 窗口
        pirate_w = find_window_for_time(t_soap_end, t_star, pirate_num_windows)
        
        if pirate_w is None:
            print(f"⚠️  SOAP Window {soap_w}: 時間 {t_soap_end:.6f} 超出 PIRATE 範圍")
            continue
        
        print(f"評估 SOAP Window {soap_w} (t≈{t_soap_end:.3f}) vs PIRATE Window {pirate_w}...")
        
        try:
            # 評估 SOAP
            soap_result = evaluate_at_time('soap', soap_checkpoint_base, soap_w, 
                                          t_soap_end, dns_data)
            
            # 評估 PIRATE 在相同時間
            pirate_result = evaluate_at_time('pirate', pirate_checkpoint_base, pirate_w, 
                                            t_soap_end, dns_data)
            
            results.append({
                'soap_window': soap_w,
                'pirate_window': pirate_w,
                't': soap_result['t_actual'],
                'soap_u': soap_result['u_error'],
                'soap_v': soap_result['v_error'],
                'soap_w': soap_result['w_error'],
                'pirate_u': pirate_result['u_error'],
                'pirate_v': pirate_result['v_error'],
                'pirate_w': pirate_result['w_error'],
            })
            
            print(f"  ✓ t={soap_result['t_actual']:.3f}: SOAP u={soap_result['u_error']:.4f} vs PIRATE u={pirate_result['u_error']:.4f}")
            
        except Exception as e:
            print(f"  ✗ 評估失敗: {e}")
            continue
    
    print()
    print("=" * 100)
    print("比較結果")
    print("=" * 100)
    print()
    
    # 打印表格
    print(f"{'Time':<8} {'SOAP W':<8} {'PIRATE W':<10} {'SOAP u%':<10} {'PIRATE u%':<12} {'Δu%':<10} "
          f"{'SOAP v%':<10} {'PIRATE v%':<12} {'Δv%':<10} {'SOAP w%':<10} {'PIRATE w%':<12} {'Δw%':<10}")
    print("-" * 130)
    
    for r in results:
        delta_u = ((r['pirate_u'] - r['soap_u']) / r['pirate_u'] * 100) if r['pirate_u'] > 0 else 0
        delta_v = ((r['pirate_v'] - r['soap_v']) / r['pirate_v'] * 100) if r['pirate_v'] > 0 else 0
        delta_w = ((r['pirate_w'] - r['soap_w']) / r['pirate_w'] * 100) if r['pirate_w'] > 0 else 0
        
        print(f"{r['t']:<8.3f} {r['soap_window']:<8} {r['pirate_window']:<10} "
              f"{r['soap_u']*100:<10.2f} {r['pirate_u']*100:<12.2f} {delta_u:<10.1f} "
              f"{r['soap_v']*100:<10.2f} {r['pirate_v']*100:<12.2f} {delta_v:<10.1f} "
              f"{r['soap_w']*100:<10.2f} {r['pirate_w']*100:<12.2f} {delta_w:<10.1f}")
    
    # 統計
    if results:
        avg_delta_u = np.mean([((r['pirate_u'] - r['soap_u']) / r['pirate_u'] * 100) for r in results])
        avg_delta_v = np.mean([((r['pirate_v'] - r['soap_v']) / r['pirate_v'] * 100) for r in results])
        avg_delta_w = np.mean([((r['pirate_w'] - r['soap_w']) / r['pirate_w'] * 100) for r in results])
        
        print()
        print("=" * 100)
        print("統計摘要")
        print("=" * 100)
        print(f"比較的時間點數: {len(results)}")
        print(f"平均改善:")
        print(f"  u_error: {avg_delta_u:>6.1f}%")
        print(f"  v_error: {avg_delta_v:>6.1f}%")
        print(f"  w_error: {avg_delta_w:>6.1f}%")
        
        # 保存結果
        output_file = os.path.expanduser('~/jaxpi/time_aligned_comparison.npz')
        np.savez(output_file, results=results)
        print()
        print(f"結果已保存至: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='時間對齊的 PIRATE vs SOAP 公平比較')
    parser.add_argument('--soap-max-window', type=int, default=None,
                       help='SOAP 的最大可用窗口（默認自動檢測）')
    
    args = parser.parse_args()
    
    results = generate_time_aligned_comparison(args.soap_max_window)
    
    print()
    print("=" * 100)
    print("完成！")
    print("=" * 100)


if __name__ == '__main__':
    main()
