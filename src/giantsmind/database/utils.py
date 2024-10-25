from datetime import date
from typing import List, Tuple

import textdistance
from sqlalchemy import and_
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


from giantsmind.database.schema import Author, Paper, Journal, engine
from giantsmind.database import collection_operations as col_ops
from giantsmind.database import paper_operations as paper_ops
from giantsmind.utils.logging import logger


def _get_unique_values(rows, column):
    return list(set([getattr(row, column) for row in rows]))


def _get_distance(values: List[str], search_term: str) -> Tuple[List[str], List[float]]:
    distances = [textdistance.damerau_levenshtein.distance(search_term, val) for val in values]
    return list(distances)


def _sort(values: List[str], distances: List[float]) -> Tuple[List[str], List[float]]:
    values, distances = zip(*sorted(zip(values, distances), key=lambda x: x[1]))
    return list(values), list(distances)


# def _column_fuzzy_match(
#     col_name: str, search_term: str, engine: Engine = engine
# ) -> Tuple[List[str], List[float]]:
#     with Session(engine) as session:
#         unique_vals = _list_unique(session, col_name)
#     distances = _get_distance(unique_vals, search_term)
#     values, distances = _sort(unique_vals, distances)
#     return values, distances


# def _list_unique(session, attr: str):
#     values = session.query(getattr(Paper, attr)).distinct().all()
#     return [value[0] for value in values]


# def list_unique_vals(name: str, engine: Engine = engine):
#     return _list_unique(engine, name)


def search_string_in_column(
    table, column_name: str, search_term: str, engine: Engine = engine
) -> Tuple[List[str], List[float]]:
    """Search for a string in a column of a table.

    Returns the values of the column that match the search term in the
    column of the table.

    """
    field = getattr(Paper, column_name)
    with Session(engine) as session:
        # Try to find an exact match
        exact_match = session.query(table).filter(field == search_term).one_or_none()
        if exact_match:
            return [exact_match.__dict__[column_name]], [0.0]

        # Try to find a match using the LIKE operator
        like_matches = session.query(table).filter(field.like(f"%{search_term}%")).all()
        if like_matches:
            unique_vals = _get_unique_values(like_matches, column_name)
            distances = _get_distance(unique_vals, search_term)
            unique_vals, distances = _sort(unique_vals, distances)
            return unique_vals, distances

        # Try to find a match using fuzzy matching
        results, distances = _column_fuzzy_match(engine, column_name, search_term)
        return results, distances


def _find_papers(
    session: Session,
    journals: List[str] = None,
    authors: List[str] = None,
    titles: List[str] = None,
    year_ranges: List[str] = None,
) -> List[Paper]:

    filters = []

    if journals:
        filters.append(Journal.name.in_(journals))
    if authors:
        filters.append(Author.name.in_(authors))
    if titles:
        filters.append(Paper.title.in_(titles))
    if year_ranges:
        for year_range in year_ranges:
            start_year, end_year = [int(year) for year in year_range.split("-")]
            filters.append(
                and_(
                    Paper.publication_date >= date(start_year, 1, 1),
                    Paper.publication_date <= date(end_year, 12, 31),
                )
            )
    papers = session.query(Paper).filter(and_(*filters)).all()
    return papers


def find_papers(
    journal: str = None,
    author: str = None,
    title: str = None,
    year_range: str = None,
    engine: Engine = engine,
) -> List[dict]:
    papers = _find_papers(engine, journal, author, title, year_range)
    col_ops.create_collection(
        engine, name="search results", paper_ids=[paper.paper_id for paper in papers], overwrite=True
    )
    papers_dict = [_paper_to_dict(paper) for paper in papers]
    return papers_dict


def _paper_to_dict(paper: Paper) -> dict:
    paper_dict = {
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "journal": paper.journal.name,
        "publication_date": paper.publication_date,
        "paper_id": paper.paper_id,
        "url": paper.url,
        "file_path": paper.file_path,
    }
    return paper_dict


def _authors2str(authors: List[Author]):
    return ", ".join([author.name for author in authors])


def print_papers(papers):
    for paper in papers:
        print("".join(["-"] * 80))
        print(
            f"Title: {paper.title}, Author: {_authors2str(paper.authors)}, Journal: {paper.journal.name}, Publication Date: {paper.publication_date}"
        )
    print("".join(["-"] * 80))


def print_papers_from_collection(collection_id: int, engine: Engine = engine) -> None:
    with Session(engine) as session:
        collection = col_ops._get_collection(session, collection_id)
        print_papers(collection.papers)
