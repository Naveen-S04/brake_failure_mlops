from pathlib import Path
import yaml

def load_params(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)