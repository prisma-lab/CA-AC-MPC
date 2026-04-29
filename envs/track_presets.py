from __future__ import annotations

import math
from typing import List

import numpy as np

from utils.track_generator import Gate


def make_track(track: str) -> List[Gate]:
    """Build a named built-in gate layout."""
    gates: List[Gate] = []
    track = track.lower()

    if track == "horizontal":
        for i in range(6):
            pose = np.array([5.0 + 4.0 * i, 0.0, 1.0, 0.0], dtype=np.float32)
            gates.append(Gate(width=1.0, height=1.0, pose=pose))
    elif track == "vertical":
        for i in range(6):
            pose = np.array([0.0, 0.0, 1.0 + 2.0 * i, 0.0], dtype=np.float32)
            gates.append(Gate(width=1.0, height=1.0, pose=pose))
    elif track == "circle":
        radius = 6.0
        n_gates = 8
        for k in range(n_gates):
            ang = 2 * math.pi * k / n_gates
            x = radius * math.cos(ang)
            y = radius * math.sin(ang)
            yaw = math.atan2(math.cos(ang), -math.sin(ang))
            pose = np.array([x, y, 1.5, yaw], dtype=np.float32)
            gates.append(Gate(width=1.0, height=1.0, pose=pose))
    elif track == "splits":
        centers = [
            [3.0, 2.0, 1.0],
            [4.0, -6.0, 1.0],
            [10.0, -5.0, 1.0],
            [14.0, 1.0, 3.5],
            [14.0, -9.0, 1.5],
            [0.5, -10.5, 1.0],
        ]
        yaws_deg = [-45.0, 70.0, 160.0, 0.0, -135.0, 180.0]
        for center, yaw_deg in zip(centers, yaws_deg):
            pose = np.array(
                [center[0], center[1], center[2], math.radians(yaw_deg)],
                dtype=np.float32,
            )
            gates.append(Gate(width=1.0, height=1.0, pose=pose))
    else:
        raise ValueError(f"Unknown track: {track}")
    return gates


def make_straight_track(
    n_gates: int = 4,
    start_x: float = 3.0,
    spacing: float = 2.0,
    y: float = 0.0,
    z: float = 1.0,
    width: float = 1.0,
    height: float = 1.0,
) -> List[Gate]:
    gates: List[Gate] = []
    for i in range(int(n_gates)):
        pose = np.array([start_x + spacing * i, y, z, 0.0], dtype=np.float32)
        gates.append(Gate(width=width, height=height, pose=pose))
    return gates
