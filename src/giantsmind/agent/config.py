from importlib import resources
from pathlib import Path

# SQL Agent Constants
SQL_SYSTEM_MESSAGE_PATH = resources.files("giantsmind.agent.resources.messages").joinpath(
    "sql_system_message.txt"
)
DEFAULT_MODEL = "claude-3-5-sonnet-latest"
NO_QUERY = "NO_QUERY"
SQL_PREFIX = "SQL: "
