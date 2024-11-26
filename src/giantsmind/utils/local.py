import json
from pathlib import Path
from typing import Dict, List

from platformdirs import user_data_dir

Collections = Dict[str, List[str]]


def get_local_data_path() -> Path:
    local_path = Path(user_data_dir("giantsmind"))
    local_path.mkdir(exist_ok=True)
    return local_path


def load_collections(path: str | Path) -> Collections:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        collections = json.load(f)
    return collections


def save_collections(collections: Collections, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(collections, f)
