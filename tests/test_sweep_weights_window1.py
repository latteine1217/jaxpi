"""
What:
    驗證 window-1 weight sweep 的搜尋空間與 CLI 預設值。
Why:
    使用者要求先把 `data_weight` 範圍收斂到 `1~100`，
    並把目標門檻改成 `5e-5`；這兩者都屬於實驗邏輯，不應靠人工目測確認。
"""

import unittest
import sys
import types


def _install_stub_modules():
    """
    What:
        安裝最小 stub，避免測試因大型依賴缺失而無法匯入目標模組。
    Why:
        這支測試只驗證 CLI 與搜尋空間，不需要真的載入 Optuna/JAX/W&B。
    """

    if "optuna" not in sys.modules:
        optuna = types.ModuleType("optuna")
        optuna.Trial = object
        pruners = types.ModuleType("optuna.pruners")
        pruners.MedianPruner = object
        optuna.pruners = pruners
        sys.modules["optuna"] = optuna
        sys.modules["optuna.pruners"] = pruners

    if "jax" not in sys.modules:
        sys.modules["jax"] = types.ModuleType("jax")

    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")


_install_stub_modules()

from scripts.sweep import sweep_weights_window1


class DummyTrial:
    """
    What:
        最小 trial stub，只記錄 `suggest_float` 呼叫參數。
    Why:
        測試只需要驗證搜尋區間，不需要真的啟動 Optuna。
    """

    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, log=False):
        self.calls.append((name, low, high, log))
        return 10.0


class SweepWeightsWindow1Test(unittest.TestCase):
    def test_data_weight_search_range_is_1_to_100_log_scale(self):
        trial = DummyTrial()

        value = sweep_weights_window1.suggest_data_weight(trial)

        self.assertEqual(value, 10.0)
        self.assertEqual(trial.calls, [("data_weight", 1.0, 100.0, True)])

    def test_default_threshold_is_5e_minus_5(self):
        parser = sweep_weights_window1.build_arg_parser()

        args = parser.parse_args([])

        self.assertEqual(args.threshold, 5e-5)


if __name__ == "__main__":
    unittest.main()
