"""
What:
    驗證 window-1 固定權重驗證 config 的關鍵行為。
Why:
    這次 job 的目的不是再做 sweep，而是固定 `data_weight=23.1429`
    重跑單獨驗證並保留 checkpoint；這些條件不能靠人工目測確認。
"""

import sys
import types
import unittest


def _install_stub_modules():
    """
    What:
        安裝最小 stub，讓 config module 可以在本地缺少 JAX/ConfigDict 依賴時匯入。
    Why:
        測試只驗證 config 輸出，不需要真實數值運算。
    """

    if "jax" not in sys.modules:
        jax_mod = types.ModuleType("jax")
        jnp_mod = types.ModuleType("jax.numpy")
        jnp_mod.pi = 3.141592653589793
        jax_mod.numpy = jnp_mod
        sys.modules["jax"] = jax_mod
        sys.modules["jax.numpy"] = jnp_mod

    if "ml_collections" not in sys.modules:
        mlc = types.ModuleType("ml_collections")

        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        mlc.ConfigDict = ConfigDict
        sys.modules["ml_collections"] = mlc


_install_stub_modules()


class Window1FixedWeightEvalConfigTest(unittest.TestCase):
    def test_config_fixes_data_weight_and_saves_checkpoints(self):
        from examples.kolmogorov_flow.configs import (
            paper_repro_soap_sensor100_n512_w50_window1_dw231429_eval as cfg_mod,
        )

        config = cfg_mod.get_config()

        self.assertEqual(config.training.max_windows_to_run, 1)
        self.assertEqual(config.training.max_steps, 50000)
        self.assertEqual(config.weighting.init_weights.u_data, 23.1429)
        self.assertEqual(config.weighting.init_weights.v_data, 23.1429)
        self.assertEqual(config.saving.save_every_steps, 1000)
        self.assertIsNone(config.saving.ckpt_dir)


if __name__ == "__main__":
    unittest.main()
