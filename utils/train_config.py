from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update nested dicts.
    """
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict.")
    return data


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply CLI overrides like:
      ppo.learning_rate=1e-4
      env.kwargs.gate_pass_radius=0.6

    Values are parsed using yaml.safe_load for basic typing.
    """
    out = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override (missing '='): {ov}")
        key, raw = ov.split("=", 1)
        value = yaml.safe_load(raw)
        parts = [p for p in key.split(".") if p]
        if not parts:
            raise ValueError(f"Invalid override key: {ov}")
        cur: Any = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
    return out

