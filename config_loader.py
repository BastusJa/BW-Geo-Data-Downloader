"""Shared configuration loading for the downloader scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ADVANCED_CONFIG_FILENAME = "advanced_config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the user config and merge it over the technical defaults."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}

    if not isinstance(user_config, dict):
        raise ValueError("YAML configuration must contain a mapping/object.")

    advanced_path = config_path.with_name(ADVANCED_CONFIG_FILENAME)
    if not advanced_path.exists():
        return user_config

    with advanced_path.open("r", encoding="utf-8") as file:
        advanced_config = yaml.safe_load(file) or {}

    if not isinstance(advanced_config, dict):
        raise ValueError(f"{ADVANCED_CONFIG_FILENAME} must contain a YAML mapping.")

    return _deep_merge(advanced_config, user_config)
