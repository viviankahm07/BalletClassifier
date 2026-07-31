"""
config.py
---------
Load YAML configuration files.
"""

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data_config(path: str = "data_config.yaml") -> dict:
    return load_config(path)
