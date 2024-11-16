from sqlite3 import Connection
from typing import Callable, List

from giantsmind.database.database_functions import DatabaseFunction


def setup_database_connection(connection: Connection, db_functions: List[DatabaseFunction]) -> None:
    """Setup database connection with required functions."""
    for func in db_functions:
        connection.create_function(func.name, func.num_params, func.func)


def create_paper_ids_clause(paper_ids: List[str]) -> str:
    """Create SQL clause for paper IDs filtering."""
    if len(paper_ids) == 1:
        return f"= '{paper_ids[0]}'"
    return f"IN {tuple(paper_ids)}"


def get_papers_query(paper_ids_txt: str) -> str:
    """Generate SQL query for fetching paper metadata."""
    return """SELECT
        papers.title,
        papers.journal_id,
        papers.publication_date,
        GROUP_CONCAT(authors.name, ', ') AS authors,
        papers.paper_id,
        papers.url
    FROM papers
    LEFT JOIN author_paper ON papers.paper_id = author_paper.paper_id
    LEFT JOIN authors ON author_paper.author_id = authors.author_id
    LEFT JOIN journals ON papers.journal_id = journals.journal_id
    WHERE papers.paper_id {}
    GROUP BY papers.paper_id, journals.name;
    """.format(
        paper_ids_txt
    )
