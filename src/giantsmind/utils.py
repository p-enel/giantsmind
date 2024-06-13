from pathlib import Path
import os
import giantsmind


def set_env_vars():
    env_file = Path(giantsmind.__file__).parent.parent.parent / ".env"

    if not env_file.exists():
        raise Exception(".env file not found.")

    for line in env_file.read_text().split("\n"):
        if not line:
            continue
        key, value = line.split("=")
        os.environ[key] = value


def get_local_data_path() -> str:
    local_path = Path.home() / ".local" / "share" / "giantsmind"
    local_path.mkdir(exist_ok=True)
    return str(local_path)
