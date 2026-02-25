"""
從訓練日誌提取評估誤差（避免重新計算）
訓練過程中已經計算過誤差，我們直接從日誌中提取
"""

import re
import sys
import argparse


def extract_errors_from_log(log_file):
    """
    從訓練日誌中提取每個時間窗口的最終誤差
    """
    
    results = []
    current_window = None
    current_errors = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            # 檢測時間窗口
            if 'Starting time window' in line or 'Time window' in line:
                # 提取窗口編號
                match = re.search(r'[Tt]ime [Ww]indow[:\s]+(\d+)', line)
                if match:
                    # 保存前一個窗口的結果
                    if current_window and current_errors:
                        results.append({
                            'window': current_window,
                            **current_errors
                        })
                    
                    current_window = int(match.group(1))
                    current_errors = {}
            
            # 提取誤差（在訓練結束時輸出）
            if 'u_error' in line and current_window:
                match = re.search(r'u_error\s+([\d.e+-]+)', line)
                if match:
                    current_errors['u_error'] = float(match.group(1))
            
            if 'v_error' in line and current_window:
                match = re.search(r'v_error\s+([\d.e+-]+)', line)
                if match:
                    current_errors['v_error'] = float(match.group(1))
            
            if 'w_error' in line and current_window:
                match = re.search(r'w_error\s+([\d.e+-]+)', line)
                if match:
                    current_errors['w_error'] = float(match.group(1))
    
    # 保存最後一個窗口
    if current_window and current_errors:
        results.append({
            'window': current_window,
            **current_errors
        })
    
    return results


def extract_final_errors(log_file):
    """
    提取每個窗口訓練結束時（Iter 19900）的誤差
    """
    
    results = {}
    current_window = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # 檢測時間窗口開始
        if 'Starting time window' in line or 'time_window' in line:
            match = re.search(r'[Tt]ime[_\s][Ww]indow[_\s:]+(\d+)', line)
            if match:
                current_window = int(match.group(1))
        
        # 檢測 Iter 19900（最後一個迭代，或接近20000的）
        if current_window and ('Iter: 19900' in line or 'Iter: 19800' in line or 'Iter: 19700' in line):
            # 往後找10行，提取誤差
            u_error = None
            v_error = None
            w_error = None
            
            for j in range(i, min(i+15, len(lines))):
                if 'u_error' in lines[j]:
                    match = re.search(r'u_error\s+([\d.e+-]+)', lines[j])
                    if match:
                        u_error = float(match.group(1))
                
                if 'v_error' in lines[j]:
                    match = re.search(r'v_error\s+([\d.e+-]+)', lines[j])
                    if match:
                        v_error = float(match.group(1))
                
                if 'w_error' in lines[j]:
                    match = re.search(r'w_error\s+([\d.e+-]+)', lines[j])
                    if match:
                        w_error = float(match.group(1))
            
            if u_error is not None and v_error is not None and w_error is not None:
                results[current_window] = {
                    'window': current_window,
                    'u_error': u_error,
                    'v_error': v_error,
                    'w_error': w_error
                }
    
    # 轉換為列表並排序
    result_list = sorted(results.values(), key=lambda x: x['window'])
    
    return result_list


def main():
    parser = argparse.ArgumentParser(description='從訓練日誌提取誤差')
    parser.add_argument('log_file', type=str, help='訓練日誌文件路徑（.err 文件）')
    parser.add_argument('--output', type=str, default=None, help='輸出文件路徑')
    
    args = parser.parse_args()
    
    # 提取誤差
    results = extract_final_errors(args.log_file)
    
    if not results:
        print("⚠️ 未找到任何誤差數據")
        print("請檢查日誌文件格式")
        sys.exit(1)
    
    # 打印結果
    print("=" * 80)
    print(f"從日誌提取的誤差數據: {args.log_file}")
    print("=" * 80)
    print(f"{'Window':<10} {'u_error':<15} {'v_error':<15} {'w_error':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['window']:<10} {r['u_error']:<15.6f} {r['v_error']:<15.6f} {r['w_error']:<15.6f}")
    
    print("=" * 80)
    
    # 統計
    import numpy as np
    u_errors = np.array([r['u_error'] for r in results])
    v_errors = np.array([r['v_error'] for r in results])
    w_errors = np.array([r['w_error'] for r in results])
    
    print()
    print("統計摘要:")
    print(f"  u_error: mean={u_errors.mean():.6f}, min={u_errors.min():.6f}, max={u_errors.max():.6f}")
    print(f"  v_error: mean={v_errors.mean():.6f}, min={v_errors.min():.6f}, max={v_errors.max():.6f}")
    print(f"  w_error: mean={w_errors.mean():.6f}, min={w_errors.min():.6f}, max={w_errors.max():.6f}")
    print()
    print(f"總窗口數: {len(results)}")
    
    # 保存結果
    if args.output:
        np.savez(args.output, results=results)
        print(f"\n✓ 結果已保存到: {args.output}")


if __name__ == '__main__':
    main()
