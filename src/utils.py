# utils.py
import os
import yaml # type: ignore

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
