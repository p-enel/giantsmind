from pathlib import Path
from typing import List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (  # AIMessage,; FunctionMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)

import giantsmind.database as db_module
import giantsmind.database.query as db_query
from giantsmind.utils import utils
from giantsmind.utils.logging import logger

# from langchain_core.tools import tool


def _get_sql_schema() -> str:
    path = str(Path(db_module.__file__).parent / "schema.sql")
    with open(path, "r") as f:
        schema = f.read()
    return schema


# @tool
# def multiply(ab: int, ba: int) -> int:
#     """Multiply two numbers."""
#     return ab * ba


def _sql_sys_msg(schema: str, collection_id: int) -> SystemMessage:
    system_sql = SystemMessage(
        f"""<system> You are an LLM agent that assists users in
finding scientific articles in a database accessed with SQL. The
database contains the metadata of scientific papers. Here is a
description of the schema of the table containing articles metadata:

<database_schema>
{schema}
</database_schema>

To find authors use the AUTHOR_NAME_DISTANCE function that takes
two strings and returns the Levenshtein distance between them. To
find a journal use the LEVENSHTEIN function that takes two strings
and returns the Levenshtein distance between them.

Your role is to interpret what papers the user is looking for and
transform it into a SQL query that only returns the paper IDs
corresponding to the user's request by looking for papers in the
collection ID {collection_id}. One important constraint: this is a SQL
query for a SQLite 3 database, so e.g. keywords such as `Extract` are
not available. E.g. 'Human: I want all the papers that authors robert
kennedy and jennifer Lawrence published together in Plos comp biology
published after 2010' would require an SQL query to be generated:

SQL: SELECT DISTINCT p.paper_id
FROM papers p
JOIN author_paper ap1 ON p.paper_id = ap1.paper_id
JOIN authors a1 ON ap1.author_id = a1.author_id
JOIN author_paper ap2 ON p.paper_id = ap2.paper_id
JOIN authors a2 ON ap2.author_id = a2.author_id
JOIN journals j ON p.journal_id = j.journal_id
JOIN paper_collection pc ON p.paper_id = pc.paper_id
WHERE author_name_distance(a1.name, 'Robert Kennedy') <= 3
  AND author_name_distance(a2.name, 'Jennifer Lawrence') <= 3
  AND a1.author_id < a2.author_id
  AND LEVENSHTEIN(j.name, 'Plos comp biology') <= 5
  AND p.publication_date > '2010-12-31'
  AND pc.collection_id = 3
ORDER BY p.publication_date DESC;

Always include the collection_id {collection_id} in the query. Do not
add text before or after the query but always start the query with
'SQL: '.

Your answer should only contain the SQL query. If the user asks a
question that does not require any query, answer:

NO_QUERY
</system>
    """
    )
    return system_sql


def _get_chat_model() -> BaseChatModel:
    utils.set_env_vars()
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    return model


def get_sql_query(user_message: str, collection_id: int) -> str:
    schema = _get_sql_schema()
    system_sql = _sql_sys_msg(schema, collection_id)
    model = _get_chat_model()
    messages = [system_sql, HumanMessage(user_message)]
    answer = model.invoke(messages)
    return answer.content


def _preprocess_query(query: str) -> str:
    if query == "NO_QUERY":
        return None
    if query.startswith("SQL: "):
        return query[5:]
    logger.error(f"Cannot preprocess query: Invalid query {query}")
    raise ValueError(f"Wrong query {query}")


def _process_results(results: List[Tuple[str]]) -> List[str]:
    return [tuple_[0] for tuple_ in results]


def execute_query(query: str):
    query = _preprocess_query(query)
    results = db_query.execute_query(query)
    return _process_results(results)


if __name__ == "__main__":
    query = get_sql_query("Get all the papers from Emily Johnson")
    query = get_sql_query("I want all the papers from Patel published in Nature")
    query = get_sql_query("Has sarah williams published in 2022?")
    execute_query(query)
    answer = get_sql_query(
        "Get all the papers that authors Robert Kennedy and Jennifer Lawrence published together in Plos comp biology published after 2010"
    )

    print(get_buffer_string([answer]))
