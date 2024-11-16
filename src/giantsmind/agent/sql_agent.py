import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string

import giantsmind.database as db_module
import giantsmind.database.config as db_config
from giantsmind.agent.config import (
    DEFAULT_MODEL,
    NO_QUERY,
    SQL_PREFIX,
    SQL_SYSTEM_MESSAGE_PATH,
)
from giantsmind.database import database_functions as db_functions
from giantsmind.database.models import DatabaseConfig, SQLiteConnection
from giantsmind.database.query_executor import _process_results, execute_metadata_query
from giantsmind.utils import utils
from giantsmind.utils.logging import logger


def _get_sql_schema() -> str:
    path = str(Path(db_module.__file__).parent / "schema.sql")
    with open(path, "r") as f:
        schema = f.read()
    return schema


def _sql_sys_msg(schema: str, collection_id: int) -> SystemMessage:
    with open(SQL_SYSTEM_MESSAGE_PATH, "r") as file:
        system_sql_template = file.read()
    return SystemMessage(system_sql_template.format(schema=schema, collection_id=collection_id))


def _get_llm_model() -> BaseChatModel:
    utils.set_env_vars()
    return ChatAnthropic(model=DEFAULT_MODEL)


def _preprocess_query(query: str) -> str | None:
    if query == NO_QUERY:
        return None
    if query.startswith(SQL_PREFIX):
        return query[len(SQL_PREFIX) :]
    logger.error(f"Cannot preprocess query: Invalid query {query}")
    raise ValueError(f"Wrong query {query}")


def get_sql_query(user_message: str, collection_id: int = 1) -> str:
    schema = _get_sql_schema()
    system_sql = _sql_sys_msg(schema, collection_id)
    model = _get_llm_model()
    messages = [system_sql, HumanMessage(user_message)]
    answer = model.invoke(messages)
    return answer.content


def metadata_query(query: str) -> List[Dict[str, Any]]:
    dbconf = DatabaseConfig(
        path=db_config.DATABASE_PATH,
        db_functions=[db_functions.levenshtein_func, db_functions.author_name_distance_func],
    )
    db_connector = SQLiteConnection()

    try:
        return execute_metadata_query(
            query=query,
            preprocess_query=_preprocess_query,
            process_results=_process_results,
            db_connector=db_connector,
            config=dbconf,
        )
    except (ValueError, sqlite3.Error) as e:
        raise


if __name__ == "__main__":
    # Testing the SQL agent
    query = get_sql_query("Get all the papers from Emily Johnson", 1)
    query = get_sql_query("I want all the papers from Patel published in Nature", 1)
    query = get_sql_query("Has sarah williams published in 2022?", 1)
    execute_metadata_query(query)
    answer = get_sql_query(
        "Get all the papers that authors Robert Kennedy and Jennifer Lawrence published together in Plos comp biology published after 2010",
        1,
    )

    print(get_buffer_string([answer]))

    # Which papers are machine learning papers?
