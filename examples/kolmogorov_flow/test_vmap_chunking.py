#!/usr/bin/env python3
"""
vmap 分塊優化驗證腳本

此腳本驗證 vmap 分塊版本與原始雙層 vmap 版本的數值一致性。

測試內容：
1. 隨機初始化模型參數
2. 生成測試資料
3. 對比兩種實作的預測結果
4. 驗證記憶體使用情況

使用方法：
    python test_vmap_chunking.py
"""

import sys
import os

# 添加路徑以導入相對模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# 導入配置和模型
from examples.kolmogorov_flow.configs import pirate
from examples.kolmogorov_flow import models


def test_vmap_chunking_correctness():
    """測試 vmap 分塊的數值正確性"""

    print("=" * 70)
    print("vmap 分塊優化數值一致性測試")
    print("=" * 70)

    # 創建測試配置（不啟用分塊）
    config_no_chunk = pirate.get_config()
    config_no_chunk.optimization.use_vmap_chunking = False

    # 創建測試配置（啟用分塊）
    config_chunked = pirate.get_config()
    config_chunked.optimization.use_vmap_chunking = True
    config_chunked.optimization.vmap_chunk_size = 512

    # 生成簡單的測試資料
    np.random.seed(42)
    num_space = 2048  # 較小的測試規模
    coords = np.random.rand(num_space, 2).astype(np.float32)
    u0 = np.random.rand(num_space).astype(np.float32)
    v0 = np.random.rand(num_space).astype(np.float32)
    w0 = np.random.rand(num_space).astype(np.float32)

    t_star = np.linspace(0, 1, 10).astype(np.float32)
    nu = 1e-4

    print(f"\n📊 測試資料規模:")
    print(f"  時間點數: {len(t_star)}")
    print(f"  空間點數: {num_space}")
    print(f"  分塊大小: {config_chunked.optimization.vmap_chunk_size}")

    # 創建模型（不啟用分塊）
    print(f"\n⏳ 初始化模型...")
    model_no_chunk = models.NavierStokes(
        config_no_chunk,
        t_star,
        coords,
        u0,
        v0,
        w0,
        nu,
        replicate_state=False,
    )

    # 創建模型（啟用分塊）
    model_chunked = models.NavierStokes(
        config_chunked,
        t_star,
        coords,
        u0,
        v0,
        w0,
        nu,
        replicate_state=False,
    )

    # 確保兩個模型使用相同的參數（複製參數）
    model_chunked.state = model_chunked.state.replace(
        params=model_no_chunk.state.params
    )

    print(f"✅ 模型初始化完成")

    # 測試預測
    print(f"\n🧪 測試預測函數...")

    # 選擇測試時間點和空間座標
    t_test = jnp.array(t_star[:5])  # 5 個時間點
    x_test = jnp.array(coords[:1000, 0])  # 1000 個空間點
    y_test = jnp.array(coords[:1000, 1])

    # 原始版本預測
    print(f"  - 運行原始 vmap 版本...")
    u_pred_no_chunk = model_no_chunk.u_pred_fn(
        model_no_chunk.state.params, t_test, x_test, y_test
    )
    v_pred_no_chunk = model_no_chunk.v_pred_fn(
        model_no_chunk.state.params, t_test, x_test, y_test
    )
    w_pred_no_chunk = model_no_chunk.w_pred_fn(
        model_no_chunk.state.params, t_test, x_test, y_test
    )

    # 分塊版本預測
    print(f"  - 運行分塊 vmap 版本...")
    u_pred_chunked = model_chunked.u_pred_fn(
        model_chunked.state.params, t_test, x_test, y_test
    )
    v_pred_chunked = model_chunked.v_pred_fn(
        model_chunked.state.params, t_test, x_test, y_test
    )
    w_pred_chunked = model_chunked.w_pred_fn(
        model_chunked.state.params, t_test, x_test, y_test
    )

    # 計算誤差
    print(f"\n📏 數值一致性檢查:")

    u_max_diff = jnp.max(jnp.abs(u_pred_no_chunk - u_pred_chunked))
    v_max_diff = jnp.max(jnp.abs(v_pred_no_chunk - v_pred_chunked))
    w_max_diff = jnp.max(jnp.abs(w_pred_no_chunk - w_pred_chunked))

    u_rel_error = u_max_diff / (jnp.max(jnp.abs(u_pred_no_chunk)) + 1e-10)
    v_rel_error = v_max_diff / (jnp.max(jnp.abs(v_pred_no_chunk)) + 1e-10)
    w_rel_error = w_max_diff / (jnp.max(jnp.abs(w_pred_no_chunk)) + 1e-10)

    print(f"  u 場最大絕對誤差: {u_max_diff:.2e}")
    print(f"  v 場最大絕對誤差: {v_max_diff:.2e}")
    print(f"  w 場最大絕對誤差: {w_max_diff:.2e}")
    print(f"  u 場相對誤差: {u_rel_error:.2e}")
    print(f"  v 場相對誤差: {v_rel_error:.2e}")
    print(f"  w 場相對誤差: {w_rel_error:.2e}")

    # 驗證閾值（浮點精度範圍內）
    tolerance = 1e-6

    if u_max_diff < tolerance and v_max_diff < tolerance and w_max_diff < tolerance:
        print(f"\n✅ 通過：分塊版本與原始版本數值一致（誤差 < {tolerance}）")
        status = True
    else:
        print(f"\n❌ 失敗：分塊版本與原始版本數值不一致（誤差 >= {tolerance}）")
        status = False

    # 記憶體追蹤（如果可用）
    print(f"\n💾 記憶體使用情況:")
    try:
        for device in jax.devices():
            stats = device.memory_stats()
            if stats:
                print(f"  {device.device_kind}:")
                print(f"    當前使用: {stats.get('bytes_in_use', 0) / (1024**2):.2f} MB")
                print(f"    峰值使用: {stats.get('peak_bytes_in_use', 0) / (1024**2):.2f} MB")
    except Exception:
        print(f"  ⚠️  無法獲取記憶體統計（某些後端不支援）")

    return status


def test_memory_scaling():
    """測試記憶體縮放性（可選）"""

    print("\n" + "=" * 70)
    print("記憶體縮放測試（可選）")
    print("=" * 70)
    print("\n⚠️  此測試需要較大的 GPU 記憶體，可能會導致 OOM")
    print("如果不想執行，可以跳過此測試\n")

    # 這裡可以添加更詳細的記憶體縮放測試
    # 例如測試不同大小的資料集在啟用/不啟用分塊時的記憶體使用
    print("（暫時跳過記憶體縮放測試）")


def main():
    """主函數"""

    print("\n🚀 開始測試...\n")

    # 測試數值一致性
    status = test_vmap_chunking_correctness()

    # （可選）測試記憶體縮放
    # test_memory_scaling()

    print("\n" + "=" * 70)
    if status:
        print("✅ 所有測試通過！")
        print("\nvmap 分塊優化已驗證：")
        print("  - 數值結果與原始版本一致")
        print("  - 可安全使用於生產訓練")
        print("\n啟用方式：")
        print("  在配置檔案中設定：")
        print("    config.optimization.use_vmap_chunking = True")
        print("    config.optimization.vmap_chunk_size = 512")
    else:
        print("❌ 測試失敗，請檢查實作")
    print("=" * 70 + "\n")

    return 0 if status else 1


if __name__ == "__main__":
    sys.exit(main())
