from pathlib import Path
from typing import Dict, List
import pandas as pd
import json

Collections = Dict[str, List[str]]


def get_local_data_path() -> str:
    local_path = Path.home() / ".local" / "share" / "giantsmind"
    local_path.mkdir(exist_ok=True)
    return str(local_path)


def load_collections(path: str) -> Collections:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        collections = json.load(f)
    return collections


def save_collections(collections: Collections, path: str) -> None:
    with open(path, "w") as f:
        json.dump(collections, f)


def load_metadata_df() -> pd.DataFrame:
    metadata_path = Path(get_local_data_path()) / "metadata.pkl"
    if metadata_path.exists():
        metadata_df = pd.read_pickle(metadata_path)
    else:
        metadata_df = pd.DataFrame()
    return metadata_df


def delete_metadata_df() -> None:
    metadata_path = Path(get_local_data_path()) / "metadata.pkl"
    if metadata_path.exists():
        metadata_path.unlink()


def save_metadata_df(metadata_df: pd.DataFrame) -> None:
    metadata_path = Path(get_local_data_path()) / "metadata.pkl"
    metadata_df.to_pickle(metadata_path)
