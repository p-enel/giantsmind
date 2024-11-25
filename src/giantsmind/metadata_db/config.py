from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from giantsmind.utils import local

local_data_path = Path(local.get_local_data_path())
local_data_path.mkdir(exist_ok=True)
DEFAULT_DATABASE_PATH = Path(local_data_path / "papers.db")
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DATABASE_PATH}"
