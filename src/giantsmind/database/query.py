from typing import List

from sqlalchemy import create_engine
from sqlalchemy import text as sql_txt
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from giantsmind.database.config import DATABASE_URL
from giantsmind.database.database_functions import setup_db_functions

engine = create_engine(DATABASE_URL)
setup_db_functions(engine)


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

    sql_query = sql_txt(
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
