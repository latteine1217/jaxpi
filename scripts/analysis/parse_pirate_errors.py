#!/usr/bin/env python3
"""
提取 PIRATE 訓練日誌中每個時間窗口的最終錯誤
"""
import re
import sys

def parse_pirate_log(log_file):
    """解析 PIRATE 訓練日誌，提取每個窗口的最終錯誤"""
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    results = []
    current_window = None
    last_iter_data = {}
    
    for i, line in enumerate(lines):
        # 檢測時間窗口
        window_match = re.search(r'Training time window (\d+)', line)
        if window_match:
            # 保存上一個窗口的數據
            if current_window is not None and last_iter_data:
                results.append({
                    'window': current_window,
                    'u_error': last_iter_data.get('u_error', 'N/A'),
                    'v_error': last_iter_data.get('v_error', 'N/A'),
                    'w_error': last_iter_data.get('w_error', 'N/A')
                })
            
            # 開始新窗口
            current_window = int(window_match.group(1))
            last_iter_data = {}
            continue
        
        # 檢測迭代次數
        iter_match = re.search(r'Iter:\s+(\d+)', line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            
            # 只關心 19900 迭代
            if current_iter == 19900:
                # 提取接下來幾行的錯誤數據
                for j in range(i+1, min(i+12, len(lines))):
                    error_line = lines[j]
                    
                    u_match = re.search(r'u_error\s+([\d.e+-]+)', error_line)
                    if u_match:
                        last_iter_data['u_error'] = float(u_match.group(1))
                    
                    v_match = re.search(r'v_error\s+([\d.e+-]+)', error_line)
                    if v_match:
                        last_iter_data['v_error'] = float(v_match.group(1))
                    
                    w_match = re.search(r'w_error\s+([\d.e+-]+)', error_line)
                    if w_match:
                        last_iter_data['w_error'] = float(w_match.group(1))
    
    # 保存最後一個窗口的數據
    if current_window is not None and last_iter_data:
        results.append({
            'window': current_window,
            'u_error': last_iter_data.get('u_error', 'N/A'),
            'v_error': last_iter_data.get('v_error', 'N/A'),
            'w_error': last_iter_data.get('w_error', 'N/A')
        })
    
    return results

def print_results(results):
    """打印結果表格"""
    print("=" * 70)
    print("PIRATE Training - Final Errors per Window (Iter 19900)")
    print("=" * 70)
    print()
    print(f"{'Window':<10} {'u_error':<15} {'v_error':<15} {'w_error':<15}")
    print("-" * 70)
    
    u_errors = []
    v_errors = []
    w_errors = []
    
    for res in results:
        window = res['window']
        u_err = res['u_error']
        v_err = res['v_error']
        w_err = res['w_error']
        
        # 格式化輸出
        u_str = f"{u_err:.6f}" if isinstance(u_err, float) else str(u_err)
        v_str = f"{v_err:.6f}" if isinstance(v_err, float) else str(v_err)
        w_str = f"{w_err:.6f}" if isinstance(w_err, float) else str(w_err)
        
        print(f"{window:<10} {u_str:<15} {v_str:<15} {w_str:<15}")
        
        # 收集數值用於統計
        if isinstance(u_err, float):
            u_errors.append(u_err)
        if isinstance(v_err, float):
            v_errors.append(v_err)
        if isinstance(w_err, float):
            w_errors.append(w_err)
    
    print()
    print("=" * 70)
    print("Statistics")
    print("=" * 70)
    
    if u_errors:
        print(f"u_error: mean={sum(u_errors)/len(u_errors):.6f}, min={min(u_errors):.6f}, max={max(u_errors):.6f}")
    if v_errors:
        print(f"v_error: mean={sum(v_errors)/len(v_errors):.6f}, min={min(v_errors):.6f}, max={max(v_errors):.6f}")
    if w_errors:
        print(f"w_error: mean={sum(w_errors)/len(w_errors):.6f}, min={min(w_errors):.6f}, max={max(w_errors):.6f}")
    
    print()
    print("=" * 70)
    print(f"Total windows: {len(results)}")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 parse_pirate_errors.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    results = parse_pirate_log(log_file)
    print_results(results)
