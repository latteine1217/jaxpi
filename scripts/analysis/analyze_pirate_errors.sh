#!/bin/bash
# 從 PIRATE 訓練日誌中提取各個時間窗口的誤差統計

LOG_FILE=~/jaxpi/logs/kf_pirate_2gpu_2578.err

echo "========================================="
echo "PIRATE 訓練誤差分析"
echo "========================================="
echo ""
echo "從訓練日誌提取: $LOG_FILE"
echo ""

# 提取各個 Window 的最後幾個誤差記錄
for window in {1..10}; do
    echo "========================================="
    echo "Time Window $window"
    echo "========================================="
    
    # 找到這個 window 的訓練區間
    if [ $window -lt 10 ]; then
        next_window=$((window + 1))
        grep -A 200 "Training time window $window" $LOG_FILE | \
        grep -B 200 "Training time window $next_window" | \
        grep -E "Iter.*19[0-9]00|u_error|v_error|w_error" | \
        tail -24 | head -21 | \
        awk '
        /Iter/ {iter=$3}
        /u_error/ {u=$3}
        /v_error/ {v=$3}
        /w_error/ {
            w=$3
            if (iter != "") {
                printf "Iter: %s  u_error: %s  v_error: %s  w_error: %s\n", iter, u, v, w
                iter=""
            }
        }
        '
    else
        # Window 10 是最後一個，直接取最後的誤差
        grep -A 1000 "Training time window $window" $LOG_FILE | \
        grep -E "Iter.*19[0-9]00|u_error|v_error|w_error" | \
        tail -24 | head -21 | \
        awk '
        /Iter/ {iter=$3}
        /u_error/ {u=$3}
        /v_error/ {v=$3}
        /w_error/ {
            w=$3
            if (iter != "") {
                printf "Iter: %s  u_error: %s  v_error: %s  w_error: %s\n", iter, u, v, w
                iter=""
            }
        }
        '
    fi
    
    # 計算平均誤差
    echo ""
    echo "最後 1000 iterations 平均誤差:"
    if [ $window -lt 10 ]; then
        next_window=$((window + 1))
        grep -A 200 "Training time window $window" $LOG_FILE | \
        grep -B 200 "Training time window $next_window" | \
        grep -E "Iter.*19[0-9]00|u_error|v_error|w_error" | \
        tail -30 | \
        awk '
        /u_error/ {u_sum+=$3; u_count++}
        /v_error/ {v_sum+=$3; v_count++}
        /w_error/ {w_sum+=$3; w_count++}
        END {
            if (u_count > 0) printf "u_error: %.6f\n", u_sum/u_count
            if (v_count > 0) printf "v_error: %.6f\n", v_sum/v_count
            if (w_count > 0) printf "w_error: %.6f\n", w_sum/w_count
        }
        '
    else
        grep -A 1000 "Training time window $window" $LOG_FILE | \
        grep -E "u_error|v_error|w_error" | \
        tail -30 | \
        awk '
        /u_error/ {u_sum+=$3; u_count++}
        /v_error/ {v_sum+=$3; v_count++}
        /w_error/ {w_sum+=$3; w_count++}
        END {
            if (u_count > 0) printf "u_error: %.6f\n", u_sum/u_count
            if (v_count > 0) printf "v_error: %.6f\n", v_sum/v_count
            if (w_count > 0) printf "w_error: %.6f\n", w_sum/w_count
        }
        '
    fi
    
    echo ""
done

echo ""
echo "========================================="
echo "總結：所有時間窗口的最終誤差（Iter 20000）"
echo "========================================="
printf "%-10s %-15s %-15s %-15s\n" "Window" "u_error" "v_error" "w_error"
echo "---------------------------------------------------------------"

for window in {1..10}; do
    if [ $window -lt 10 ]; then
        next_window=$((window + 1))
        result=$(grep -A 200 "Training time window $window" $LOG_FILE | \
        grep -B 200 "Training time window $next_window" | \
        grep -E "Iter.*20000|u_error|v_error|w_error" | \
        tail -4 | \
        awk '
        /u_error/ {u=$3}
        /v_error/ {v=$3}
        /w_error/ {w=$3; printf "%s %s %s", u, v, w}
        ')
    else
        result=$(grep -A 1000 "Training time window $window" $LOG_FILE | \
        grep -E "u_error|v_error|w_error" | \
        tail -4 | \
        awk '
        /u_error/ {u=$3}
        /v_error/ {v=$3}
        /w_error/ {w=$3; printf "%s %s %s", u, v, w}
        ')
    fi
    
    if [ -n "$result" ]; then
        printf "%-10d %s\n" $window "$result" | awk '{printf "%-10s %-15s %-15s %-15s\n", $1, $2, $3, $4}'
    fi
done

echo "========================================="
