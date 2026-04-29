import yaml
import numpy as np
import math
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


def rot_from_yaw(yaw: float) -> np.ndarray:
    """
    Build a 3D yaw rotation matrix.

    Columns encode local gate axes in world coordinates:
    x = forward/normal, y = right, z = up.
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

@dataclass
class Gate:
    width: float
    height: float
    pose: np.ndarray  # [x, y, z, yaw_radians]
    gate_id: str = ""

    def __post_init__(self):
        self.center = self.pose[0:3].astype(np.float32)
        yaw = float(self.pose[3])

        rot_mat = rot_from_yaw(yaw)

        self.normal = rot_mat[:, 0]
        right = rot_mat[:, 1]
        self.up = rot_mat[:, 2]

        self.R = np.stack([right, self.up, self.normal], axis=1)

    def corners(self) -> np.ndarray:
        """Return the four 3D gate corners."""
        right = self.R[:, 0]
        up = self.R[:, 1]
        
        hw = 0.5 * self.width
        hh = 0.5 * self.height
        
        return np.stack(
            [
                self.center + right * hw + up * hh,  # Top-Right
                self.center - right * hw + up * hh,  # Top-Left
                self.center - right * hw - up * hh,  # Bottom-Left
                self.center + right * hw - up * hh,  # Bottom-Right
            ],
            axis=0,
        )

def _natural_sort_key(s):
    """Sort gate_1, gate_2, gate_10 in numeric order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def load_track_from_yaml(yaml_path: str) -> Tuple[List[Gate], Dict[str, Any]]:
    """Load ordered gates and track metadata from a YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    gates = []
    metadata = {}

    # Estrai le chiavi che iniziano con "gate_"
    gate_keys = [k for k in config.keys() if k.startswith('gate_')]
    # Ordina le chiavi (gate_1 -> gate_2 -> gate_10)
    gate_keys.sort(key=_natural_sort_key)

    for key in gate_keys:
        values = config[key]
        x, y, z = values[0], values[1], values[2]
        yaw_deg = values[3]
        w = values[4] if len(values) > 4 else 1.5
        h = values[5] if len(values) > 5 else 1.5

        # Conversione in radianti per la classe Gate
        yaw_rad = math.radians(yaw_deg)
        
        # Creazione array posa [x, y, z, yaw]
        pose = np.array([x, y, z, yaw_rad], dtype=np.float32)
        gates.append(Gate(width=w, height=h, pose=pose, gate_id=key))

    if "start_pos" in config:
        metadata["start_pos"] = config["start_pos"] # [x, y, z, yaw]
    
    if "track_id" in config:
        metadata["track_id"] = config["track_id"]

    return gates, metadata
