from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowLayout:
    num_time_steps: int
    time_remainder: int
    total_time_steps: int
    num_time_windows: int


def to_window_local_time(t_window: np.ndarray) -> np.ndarray:
    """
    What:
        將單一 window 的絕對時間軸轉成以 0 起算的 local time。
    Why:
        time-window checkpoint 的訓練定義依賴 local time；評估若直接餵入
        absolute time，會把不屬於 checkpoint 的時間偏移混進誤差。
    """
    t_window = np.asarray(t_window, dtype=float)
    if t_window.ndim != 1 or t_window.size == 0:
        raise ValueError(f"t_window 必須是一維且非空，實際 shape={t_window.shape}")
    return t_window - float(t_window[0])


def resolve_window_layout(
    t_values: np.ndarray,
    num_time_windows: int,
    *,
    expected_time_remainder: int | None = None,
) -> WindowLayout:
    """
    What:
        根據時間軸與 window 數量解析每個 window 的步數。
    Why:
        `len(t) // num_time_windows` 會在有餘數時靜默丟掉最後幾個時間點；
        對評估腳本來說，這種行為必須顯式宣告，不能默默接受。
    """
    t_values = np.asarray(t_values)
    if t_values.ndim != 1:
        raise ValueError(f"t_values 必須是一維，實際 shape={t_values.shape}")
    if num_time_windows < 1:
        raise ValueError(f"num_time_windows 必須 >= 1，實際為 {num_time_windows}")

    total_time_steps = int(t_values.shape[0])
    num_time_steps, time_remainder = divmod(total_time_steps, int(num_time_windows))

    if num_time_steps < 1:
        raise ValueError(
            "每個 time window 至少需要 1 個時間點："
            f"total_time_steps={total_time_steps}, num_time_windows={num_time_windows}"
        )

    if expected_time_remainder is None:
        if time_remainder != 0:
            raise ValueError(
                "Detected trailing time steps that would be silently dropped: "
                f"total_time_steps={total_time_steps}, num_time_windows={num_time_windows}, "
                f"num_time_steps={num_time_steps}, trailing time steps={time_remainder}. "
                "若這是刻意沿用訓練定義，請在 config.eval.expected_time_remainder 明確宣告。"
            )
    else:
        expected_time_remainder = int(expected_time_remainder)
        if time_remainder != expected_time_remainder:
            raise ValueError(
                "time-window remainder 與 config 宣告不一致："
                f"expected={expected_time_remainder}, actual={time_remainder}, "
                f"total_time_steps={total_time_steps}, num_time_windows={num_time_windows}"
            )

    return WindowLayout(
        num_time_steps=num_time_steps,
        time_remainder=time_remainder,
        total_time_steps=total_time_steps,
        num_time_windows=int(num_time_windows),
    )


def resolve_window_layout_from_config(t_values: np.ndarray, config) -> WindowLayout:
    """
    What:
        從 config 解析 window layout 的顯式契約。
    Why:
        評估應與該 config 對 dataset slicing 的明示假設對齊，而不是依賴隱含慣例。
    """
    eval_cfg = getattr(config, "eval", None)
    expected_time_remainder = (
        getattr(eval_cfg, "expected_time_remainder", None) if eval_cfg is not None else None
    )
    return resolve_window_layout(
        t_values,
        int(config.training.num_time_windows),
        expected_time_remainder=expected_time_remainder,
    )


def infer_domain_lengths(coords: np.ndarray) -> tuple[float, float]:
    """
    What:
        從扁平化 `(x, y)` 座標推斷週期域長度 `[0, Lx) × [0, Ly)`。
    Why:
        FFT 波數軸必須使用資料的實際網格間距；把 unit-domain 資料硬當成
        `[0, 2π)` 會直接造成 spectral axis misalignment。
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords 必須是 shape=(N, 2) 的二維陣列，實際 shape={coords.shape}")

    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    if x_unique.size == 0 or y_unique.size == 0:
        raise ValueError("coords 不能為空")

    if x_unique.size > 1:
        dx = float(np.median(np.diff(x_unique)))
        lx = float(x_unique[-1] - x_unique[0] + dx)
    else:
        lx = 1.0

    if y_unique.size > 1:
        dy = float(np.median(np.diff(y_unique)))
        ly = float(y_unique[-1] - y_unique[0] + dy)
    else:
        ly = 1.0

    if lx <= 0 or ly <= 0:
        raise ValueError(f"推斷出的 domain length 非法：Lx={lx}, Ly={ly}")

    return lx, ly


def compute_energy_spectrum(
    u2d: np.ndarray,
    v2d: np.ndarray,
    *,
    lx: float,
    ly: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    What:
        計算 2D 速度場的等向性積分能量譜。
    Why:
        所有 eval driver 應使用同一套波數定義，避免不同腳本各自推 FFT 軸。
    """
    u2d = np.asarray(u2d, dtype=float)
    v2d = np.asarray(v2d, dtype=float)
    if u2d.shape != v2d.shape or u2d.ndim != 2:
        raise ValueError(
            f"u2d/v2d 必須是相同 shape 的 2D 場，實際為 {u2d.shape} vs {v2d.shape}"
        )

    nx, ny = u2d.shape
    if nx < 2 or ny < 2:
        raise ValueError(f"頻譜計算至少需要 2x2 網格，實際為 {u2d.shape}")

    u_fft = np.fft.fftshift(np.fft.fft2(u2d))
    v_fft = np.fft.fftshift(np.fft.fft2(v2d))
    energy_density = 0.5 * (np.abs(u_fft) ** 2 + np.abs(v_fft) ** 2)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=lx / nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=ly / ny))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_radius = np.sqrt(kx_grid**2 + ky_grid**2)

    k_max = int(np.max(k_radius))
    k_bins = np.arange(1, min(k_max, min(nx, ny) // 2))
    energy_shell = []
    k_out = []
    for k_i in k_bins:
        mask = (k_radius >= k_i - 0.5) & (k_radius < k_i + 0.5)
        n_points = int(np.sum(mask))
        if n_points > 0:
            energy_shell.append(float(np.sum(energy_density[mask]) / n_points))
            k_out.append(int(k_i))

    return np.asarray(k_out, dtype=float), np.asarray(energy_shell, dtype=float)
