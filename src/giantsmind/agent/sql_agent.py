import sqlite3
from importlib import resources
from typing import Any, Callable, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string

from giantsmind.agent import config as agent_cfg
from giantsmind.database import config as db_config
from giantsmind.database import database_functions as db_functions
from giantsmind.database.models import DatabaseConfig, SQLiteConnection
from giantsmind.database.query_executor import execute_metadata_query
from giantsmind.utils.logging import logger

load_dotenv()


def _get_sql_schema() -> str:
    with resources.files("giantsmind.database.resources").joinpath("schema.sql").open("r") as f:
        schema = f.read()
    return schema


def _sql_sys_msg(schema: str, collection_id: int) -> SystemMessage:
    with open(agent_cfg.SQL_SYSTEM_MESSAGE_PATH, "r") as file:
        system_sql_template = file.read()
    return SystemMessage(system_sql_template.format(schema=schema, collection_id=collection_id))


def _get_llm_model() -> BaseChatModel:
    return ChatAnthropic(model=agent_cfg.DEFAULT_MODEL)


def _preprocess_query(query: str) -> str | None:
    if query == agent_cfg.NO_QUERY:
        return None
    if query.startswith(agent_cfg.SQL_PREFIX):
        return query[len(agent_cfg.SQL_PREFIX) :]
    logger.error(f"Cannot preprocess query: Invalid query {query}")
    raise ValueError(f"Wrong query {query}")


def get_sql_query(user_message: str, collection_id: int = 1) -> str:
    schema = _get_sql_schema()
    system_sql = _sql_sys_msg(schema, collection_id)
    model = _get_llm_model()
    messages = [system_sql, HumanMessage(user_message)]
    answer = model.invoke(messages)
    return answer.content


def _process_results(results: List[Tuple[str]]) -> List[Dict[str, Any]]:
    res_list_dict = []
    for result in results:
        res_list_dict.append(
            {
                "title": result[0],
                "journal": result[1],
                "publication_date": result[2],
                "authors": result[3],
                "paper_id": result[4],
                "url": result[5],
            }
        )
    return res_list_dict


def metadata_query(
    query: str,
    preprocess_query: Callable[[str], str] = _preprocess_query,
    process_results: Callable[[List[Tuple]], List[str]] = _process_results,
) -> List[Dict[str, Any]]:
    dbconf = DatabaseConfig(
        path=db_config.DATABASE_PATH,
        db_functions=[db_functions.levenshtein_func, db_functions.author_name_distance_func],
    )
    db_connector = SQLiteConnection()

    processed_query = preprocess_query(query)

    try:
        query_results = execute_metadata_query(
            query=processed_query,
            db_connector=db_connector,
            config=dbconf,
        )
    except (ValueError, sqlite3.Error) as e:
        raise

    return process_results(query_results)


if __name__ == "__main__":
    # Testing the SQL agent
    answer = get_sql_query(
        "Get all the papers that authors Robert Kennedy and Jennifer Lawrence published together in Plos comp biology published after 2010",
        1,
    )

    print(get_buffer_string([answer]))

    # Which papers are machine learning papers?
