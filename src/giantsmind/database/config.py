from pathlib import Path
from giantsmind.utils import local

local_data_path = Path(local.get_local_data_path())
local_data_path.mkdir(exist_ok=True)
database_path = local_data_path / "papers.db"

DATABASE_URL = f"sqlite:///{database_path}"
