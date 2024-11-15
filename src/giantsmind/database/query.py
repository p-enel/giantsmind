from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any, Callable, List

import Levenshtein
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text as sql_txt

from giantsmind.database.config import DATABASE_URL

engine = create_engine(DATABASE_URL)


def author_name_distance(db_name, query_name):
    db_name = db_name.lower()
    query_name = query_name.lower()

    db_parts = db_name.split()
    query_parts = query_name.split()

    if len(query_parts) == 1:
        if len(db_parts) > 1:
            return levenshtein(db_parts[1], query_name)
        return levenshtein(db_name, query_name)

    elif len(query_parts) == 2:
        if len(db_parts) >= 2:
            normal_order = levenshtein(db_parts[0], query_parts[0]) + levenshtein(
                db_parts[-1], query_parts[1]
            )
            swapped_order = levenshtein(db_parts[0], query_parts[1]) + levenshtein(
                db_parts[-1], query_parts[0]
            )
            return min(normal_order, swapped_order)
        return min(
            levenshtein(db_name, query_name),
            levenshtein(db_name, f"{query_parts[1]} {query_parts[0]}"),
        )

    return levenshtein(db_name, query_name)


@event.listens_for(engine, "connect")
def string_match(conn, rec):
    conn.create_function("levenshtein", 2, levenshtein)


@event.listens_for(engine, "connect")
def author_match(conn, rec):
    conn.create_function("author_name_distance", 2, author_name_distance)


@dataclass
class DatabaseFunction:
    """
    Represents a custom database function configuration.

    Attributes:
        name: Name of the function as it will be used in SQL
        num_params: Number of parameters the function accepts
        func: The Python function to be registered
    """

    name: str
    num_params: int
    func: Callable[..., Any]


# Custom functions
levenshtein_func = DatabaseFunction(
    "levenshtein",
    2,
    lambda s1, s2: Levenshtein.distance(s1.lower(), s2.lower()),
)


def connect_function_sqlite(connection: Connection, custom_functions: List[DatabaseFunction]) -> None:
    """
    Setup database connection with multiple custom functions.

    Args:
        connection: SQLite database connection
        custom_functions: List of DatabaseFunction objects to register

    Example:
        custom_functions = [
            DatabaseFunction(
                name="levenshtein",
                num_params=2,
                func=levenshtein_function
            ),
            DatabaseFunction(
                name="custom_concat",
                num_params=3,
                func=concat_function
            )
        ]
        setup_database_connection(connection, custom_functions)
    """
    for func in custom_functions:
        connection.create_function(func.name, func.num_params, func.func)


def execute_query(query, engine: Engine = engine):
    with engine.connect() as connection:
        connection.execute(sql_txt("PRAGMA group_concat_max_len = 10;"))
        results = connection.execute(sql_txt(query)).fetchall()

    return results


def retrive_papers_metadata(paper_ids: List[str], engine: Engine = engine):
    query = f"""SELECT 
    papers.title,
    papers.journal_id,
    papers.publication_date,
    GROUP_CONCAT(authors.name, ', ') AS authors
FROM papers
LEFT JOIN author_paper ON papers.paper_id = author_paper.paper_id
LEFT JOIN authors ON author_paper.author_id = authors.author_id
LEFT JOIN journals ON papers.journal_id = journals.journal_id
WHERE papers.paper_id IN {tuple(paper_ids)}
GROUP BY papers.paper_id, journals.name;
    """
    return execute_query(query, engine)


if __name__ == "__main__":
    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Your SQL query as a string with fixed values

    sql_query = text(
        """
        SELECT DISTINCT p.paper_id, p.title, p.publication_date, j.name AS journal_name,
               a1.name AS author1, a2.name AS author2
        FROM papers p
        JOIN author_paper ap1 ON p.paper_id = ap1.paper_id
        JOIN authors a1 ON ap1.author_id = a1.author_id
        JOIN author_paper ap2 ON p.paper_id = ap2.paper_id
        JOIN authors a2 ON ap2.author_id = a2.author_id
        JOIN journals j ON p.journal_id = j.journal_id
        WHERE AUTHOR_NAME_DISTANCE(a1.name, 'smith') <= 3
          AND AUTHOR_NAME_DISTANCE(a2.name, 'lisa anderson') <= 3
          AND a1.author_id < a2.author_id
          AND LEVENSHTEIN(j.name, 'Nature') <= 5
          AND p.publication_date > '2010-12-31'
        ORDER BY p.publication_date DESC
    """
    )

    # Execute the query
    result = session.execute(sql_query)

    # Fetch and print the results
    for row in result:
        print(f"Paper ID: {row.paper_id}")
        print(f"Title: {row.title}")
        print(f"Publication Date: {row.publication_date}")
        print(f"Journal: {row.journal_name}")
        print(f"Authors: {row.author1}, {row.author2}")
        print("---")

    # Don't forget to close the session
    session.close()
