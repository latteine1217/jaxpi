"""Sensor data format utilities (pinns-sparse-flow compatible)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import json
import numpy as np


@dataclass(frozen=True)
class SensorData:
    indices: np.ndarray
    coords: np.ndarray
    metadata: Dict
    dns_values: Optional[Dict[str, np.ndarray]] = None


def load_sensor_json(path: str) -> SensorData:
    with open(path, "r") as handle:
        payload = json.load(handle)

    indices = np.asarray(payload.get("indices", []), dtype=int)
    coords = np.asarray(payload.get("selected_coordinates", []), dtype=float)
    metadata = {k: v for k, v in payload.items() if k not in {"indices", "selected_coordinates"}}

    return SensorData(indices=indices, coords=coords, metadata=metadata)


def load_dns_values_npz(path: str) -> Dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: np.asarray(payload[key]) for key in payload.files}


def attach_dns_values(sensor: SensorData, npz_path: Optional[str]) -> SensorData:
    if not npz_path:
        return sensor
    dns_values = load_dns_values_npz(npz_path)
    return SensorData(
        indices=sensor.indices,
        coords=sensor.coords,
        metadata=sensor.metadata,
        dns_values=dns_values,
    )
