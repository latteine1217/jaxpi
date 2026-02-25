#!/bin/bash
# 提取 PIRATE 訓練的最終錯誤數據
# 每個時間窗口的最後一次迭代 (Iter 19900) 的 u_error, v_error, w_error

LOG_FILE="$HOME/jaxpi/logs/kf_pirate_2gpu_2578.err"

echo "=========================================="
echo "PIRATE Training - Final Errors per Window"
echo "=========================================="
echo ""
printf "%-8s  %-12s  %-12s  %-12s\n" "Window" "u_error" "v_error" "w_error"
echo "----------------------------------------------------------"

for window in {1..10}; do
    next_window=$((window + 1))
    
    # 提取當前窗口的最後一次迭代數據
    if [ $window -eq 10 ]; then
        # 最後一個窗口，取到文件結尾
        errors=$(awk "/Training time window $window/,EOF" "$LOG_FILE" | \
                 grep -A 8 "Iter: 19900" | tail -9 | \
                 grep -E "u_error|v_error|w_error" | \
                 awk '{print $2}')
    else
        # 其他窗口，取到下一個窗口開始
        errors=$(awk "/Training time window $window/,/Training time window $next_window/" "$LOG_FILE" | \
                 grep -A 8 "Iter: 19900" | tail -9 | \
                 grep -E "u_error|v_error|w_error" | \
                 awk '{print $2}')
    fi
    
    # 將錯誤值轉換為陣列
    error_array=($errors)
    
    # 輸出結果
    if [ ${#error_array[@]} -ge 3 ]; then
        u_err=${error_array[0]}
        v_err=${error_array[1]}
        w_err=${error_array[2]}
        printf "%-8s  %-12s  %-12s  %-12s\n" "$window" "$u_err" "$v_err" "$w_err"
    else
        printf "%-8s  %-12s  %-12s  %-12s\n" "$window" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "=========================================="
echo "Statistics"
echo "=========================================="

# 計算平均值 (簡單版)
echo "Note: Calculate mean/std using the extracted data above"
