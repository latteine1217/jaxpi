"""DNS dataset format utilities (pinns-sparse-flow compatible)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DNSData:
    coords: np.ndarray
    time: np.ndarray
    fields: Dict[str, np.ndarray]
    config: Dict


@dataclass(frozen=True)
class DNSMetadata:
    ndim: int
    n_time: int
    n_points: int
    grid_shape: Tuple[int, ...]
    domain_size: Tuple[float, ...]


def _infer_coords_from_config(data: Dict) -> np.ndarray:
    config = data.get("config", {})
    n = int(config.get("N", data["u"].shape[-1]))
    domain_length = float(config.get("L", 2 * np.pi))
    x = np.linspace(0.0, domain_length, n, endpoint=False)
    y = np.linspace(0.0, domain_length, n, endpoint=False)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    coords = np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=1)
    return coords


def load_dns_npy(path: str) -> DNSData:
    payload = np.load(path, allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    if not isinstance(data, dict):
        raise ValueError(f"DNS 檔案格式錯誤: {path}")

    time = np.asarray(data.get("time", data.get("t")), dtype=float)
    if time is None:
        raise ValueError("DNS 檔案缺少 time/t 欄位")

    coords = data.get("coords")
    if coords is None:
        coords = _infer_coords_from_config(data)
    coords = np.asarray(coords, dtype=float)

    fields = {}
    for key in ("u", "v", "w", "p", "omega"):
        if key in data:
            fields[key] = np.asarray(data[key], dtype=float)

    config = data.get("config", {})
    return DNSData(coords=coords, time=time, fields=fields, config=config)


def infer_metadata(dns: DNSData) -> DNSMetadata:
    n_time = dns.time.shape[0]
    n_points = dns.coords.shape[0]
    ndim = dns.coords.shape[1]

    config = dns.config or {}
    n = int(config.get("N", 0))
    if n > 0 and ndim == 2:
        grid_shape = (n, n)
    elif n > 0 and ndim == 3:
        grid_shape = (n, n, n)
    else:
        grid_shape = (n_points,)

    domain_length = float(config.get("L", 0.0))
    if domain_length > 0:
        domain_size = tuple(domain_length for _ in range(ndim))
    else:
        domain_size = tuple(np.ptp(dns.coords[:, i]) for i in range(ndim))

    return DNSMetadata(
        ndim=ndim,
        n_time=n_time,
        n_points=n_points,
        grid_shape=grid_shape,
        domain_size=domain_size,
    )


def select_time_window(
    dns: DNSData,
    time_range: Optional[Tuple[float, float]] = None,
    time_stride: int = 1,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    time_all = dns.time
    if time_range is None:
        time_mask = np.ones_like(time_all, dtype=bool)
    else:
        t_start, t_end = time_range
        time_mask = (time_all >= t_start) & (time_all <= t_end)

    indices = np.where(time_mask)[0][::max(time_stride, 1)]
    time_selected = time_all[indices]

    fields = {
        key: value[indices]
        for key, value in dns.fields.items()
        if value.shape[0] == time_all.shape[0]
    }

    return time_selected, fields
