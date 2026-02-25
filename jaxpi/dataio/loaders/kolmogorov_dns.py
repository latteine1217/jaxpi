"""Kolmogorov DNS/LES loader compatible with pinns-sparse-flow data."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from jaxpi.dataio.dns_format import load_dns_npy, select_time_window


def load_kolmogorov_dns(
    path: str,
    time_range: Optional[Tuple[float, float]] = None,
    time_stride: int = 1,
    normalize_time: bool = True,
):
    dns = load_dns_npy(path)
    time_selected, fields = select_time_window(dns, time_range=time_range, time_stride=time_stride)

    u = fields.get("u")
    v = fields.get("v")
    w = fields.get("w")
    omega = fields.get("omega")

    if u is None or v is None:
        raise ValueError("DNS 檔案缺少 u/v 欄位")

    if normalize_time:
        time_selected = time_selected - time_selected[0]

    coords = np.asarray(dns.coords, dtype=float)

    def to_np(array: Optional[np.ndarray]):
        return None if array is None else np.array(array)

    return (
        to_np(u),
        to_np(v),
        to_np(w) if w is not None else np.zeros_like(u),
        to_np(omega) if omega is not None else np.zeros_like(u),
        np.array(time_selected),
        np.array(coords),
        dns.config,
    )
