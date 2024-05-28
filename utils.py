from pathlib import Path
import os


def set_env_vars():
    env_file = Path(".env")

    if not env_file.exists():
        raise Exception(".env file not found.")

    for line in env_file.read_text().split("\n"):
        if not line:
            continue
        key, value = line.split("=")
        os.environ[key] = value
