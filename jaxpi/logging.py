import logging
from tabulate import tabulate


def get_log_keys(log_dict):
    key_list = []
    for key in log_dict.keys():
        if key.endswith("_loss"):
            key_list.append(key)
        elif key.endswith("_error"):
            key_list.append(key)
    return key_list


class Logger:
    def __init__(self, name: str = "main"):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()
        formatter = logging.Formatter(
            "[%(asctime)s - %(name)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        streamhandler.setLevel(logging.INFO)
        self.logger.addHandler(streamhandler)
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(message)

    def log_iter(self, step, start_time, end_time, log_dict,
                 max_steps=None, time_window=None, num_time_windows=None):
        """
        記錄訓練迭代資訊（增強版）

        Args:
            step: 當前步數
            start_time: 步驟開始時間
            end_time: 步驟結束時間
            log_dict: 包含 losses, errors, weights 等的字典
            max_steps: 每個 time window 的最大步數（用於計算進度）
            time_window: 當前時間窗口編號（從 log_dict 中提取，或直接傳入）
            num_time_windows: 總時間窗口數
        """
        log_keys = get_log_keys(log_dict)
        step_time = end_time - start_time

        # 從 log_dict 中提取 time_window（如果沒有直接傳入）
        if time_window is None:
            time_window = log_dict.get("time_window", None)

        # 構建表頭資訊
        header_parts = []

        # Time Window 資訊
        if time_window is not None:
            if num_time_windows is not None:
                header_parts.append(f"Time Window: {time_window}/{num_time_windows}")
            else:
                header_parts.append(f"Time Window: {time_window}")

        # Step 資訊與進度
        if max_steps is not None:
            progress_pct = (step / max_steps) * 100
            header_parts.append(f"Step: {step}/{max_steps} ({progress_pct:.1f}%)")
        else:
            header_parts.append(f"Step: {step}")

        # Step Time
        step_time_ms = step_time * 1000.0
        header_parts.append(f"Step Time: {step_time_ms:.2f} ms")

        # 估算剩餘時間（ETA）
        if max_steps is not None and step > 0:
            steps_remaining = max_steps - step
            eta_seconds = steps_remaining * step_time
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)

            if eta_hours > 0:
                eta_str = f"{eta_hours}h {eta_minutes}m {eta_secs}s"
            elif eta_minutes > 0:
                eta_str = f"{eta_minutes}m {eta_secs}s"
            else:
                eta_str = f"{eta_secs}s"

            header_parts.append(f"ETA: {eta_str}")

        # 組合表頭
        header_line = " | ".join(header_parts)
        separator_length = max(80, len(header_line) + 4)
        separator = "=" * separator_length

        # 構建 loss/error 表格
        # 分離 losses, errors, weights 等不同類型的 metrics
        loss_items = []
        error_items = []
        weight_items = []
        other_items = []

        for key in log_keys:
            value_str = "{:.3e}".format(log_dict[key])

            if key.endswith("_loss"):
                # 檢查是否有對應的 weight
                weight_key = key.replace("_loss", "_weight")
                if weight_key in log_dict:
                    weight_str = "{:.3e}".format(log_dict[weight_key])
                else:
                    weight_str = "-"
                loss_items.append([key, value_str, weight_str])
            elif key.endswith("_error"):
                error_items.append([key, value_str, "-"])
            elif key.endswith("_weight") and not any(
                item[0] == key.replace("_weight", "_loss") for item in loss_items
            ):
                # 只顯示沒有對應 loss 的 weight
                weight_items.append([key, value_str, "-"])
            elif key not in ["time_window"]:  # 排除已在 header 中顯示的資訊
                other_items.append([key, value_str, "-"])

        # 合併所有項目（losses 優先，然後是 errors）
        table_data = loss_items + error_items + weight_items + other_items

        if table_data:
            table = tabulate(
                table_data,
                headers=["Loss/Error Term", "Value", "Weight"],
                tablefmt="simple",
                numalign="right",
                disable_numparse=True,
            )
        else:
            table = "(No metrics available)"

        # 組合完整訊息
        message_lines = [
            separator,
            header_line,
            separator,
            table,
            separator,
            ""  # 空行分隔
        ]

        # 輸出每一行
        for line in message_lines:
            self.logger.info(line)
