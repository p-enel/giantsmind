from pathlib import Path
from typing import List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)

import giantsmind.database as db_module
import giantsmind.database.query as db_query
from giantsmind.utils import utils
from giantsmind.utils.logging import logger

# Constants
SQL_SYSTEM_MESSAGE_PATH = Path(__file__).parent / "messages" / "sql_system_message.txt"
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
NO_QUERY = "NO_QUERY"
SQL_PREFIX = "SQL: "


def _get_sql_schema() -> str:
    path = str(Path(db_module.__file__).parent / "schema.sql")
    with open(path, "r") as f:
        schema = f.read()
    return schema


def _sql_sys_msg(schema: str, collection_id: int) -> SystemMessage:
    with open(SQL_SYSTEM_MESSAGE_PATH, "r") as file:
        system_sql_template = file.read()
    system_sql = SystemMessage(system_sql_template.format(schema=schema, collection_id=collection_id))
    return system_sql


def _get_chat_model() -> BaseChatModel:
    utils.set_env_vars()
    model = ChatAnthropic(model=DEFAULT_MODEL)
    return model


def get_sql_query(user_message: str, collection_id: int = 1) -> str:
    schema = _get_sql_schema()
    system_sql = _sql_sys_msg(schema, collection_id)
    model = _get_chat_model()
    messages = [system_sql, HumanMessage(user_message)]
    answer = model.invoke(messages)
    return answer.content


def _preprocess_query(query: str) -> str:
    if query == NO_QUERY:
        return None
    if query.startswith(SQL_PREFIX):
        return query[len(SQL_PREFIX) :]
    logger.error(f"Cannot preprocess query: Invalid query {query}")
    raise ValueError(f"Wrong query {query}")


def _process_results(results: List[Tuple[str]]) -> List[str]:
    return [tuple_[0] for tuple_ in results]


def execute_query(query: str) -> List[str]:
    query = _preprocess_query(query)
    results = db_query.execute_query(query)
    return _process_results(results)


if __name__ == "__main__":
    query = get_sql_query("Get all the papers from Emily Johnson", 1)
    query = get_sql_query("I want all the papers from Patel published in Nature", 1)
    query = get_sql_query("Has sarah williams published in 2022?", 1)
    execute_query(query)
    answer = get_sql_query(
        "Get all the papers that authors Robert Kennedy and Jennifer Lawrence published together in Plos comp biology published after 2010",
        1,
    )

    print(get_buffer_string([answer]))
