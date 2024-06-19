from datetime import date
from typing import List

from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker

from giantsmind.article_metadata.schema import Collection, Paper


def _add_paper(session, metadata: dict) -> Paper:
    new_paper = Paper(
        journal=metadata["journal"],
        file_path=metadata["file_path"],
        publication_date=metadata["publication_date"],
        title=metadata["title"],
        author=metadata["author"],
        url=metadata["url"],
    )
    session.add(new_paper)
    session.commit()
    return Paper


def add_papers(engine, metadatas: list[dict]) -> list[Paper]:
    session = sessionmaker(bind=engine)()
    papers = []
    for metadata in metadatas:
        paper = _add_paper(session, metadata)
        papers.append(paper)
    session.close()
    return papers


def _remove_paper(session, paper_id):
    paper = session.query(Paper).filter_by(paper_id=paper_id).one_or_none()
    if paper:
        session.delete(paper)
        session.commit()
    else:
        print(f"Paper ID '{paper_id}' not found.")


def remove_papers(engine, paper_ids: List[str]) -> None:
    session = sessionmaker(bind=engine)()
    for paper_id in paper_ids:
        _remove_paper(session, paper_id)
    session.close()


def create_collection(engine, name, paper_ids):
    session = sessionmaker(bind=engine)()
    new_collection = Collection(name=name)
    papers = session.query(Paper).filter(Paper.paper_id.in_(paper_ids)).all()
    new_collection.papers = papers
    session.add(new_collection)
    session.commit()
    session.close()
    print(f"Collection '{name}' created successfully with papers: {paper_ids}")


def add_paper_to_collection(engine, paper_id, collection_id):
    session = sessionmaker(bind=engine)()
    paper = session.query(Paper).filter_by(paper_id=paper_id).one_or_none()
    collection = session.query(Collection).filter_by(collection_id=collection_id).one_or_none()
    if paper and collection:
        collection.papers.append(paper)
        session.commit()
        print(f"Paper ID '{paper_id}' added to Collection ID '{collection_id}' successfully.")
    else:
        print(f"Paper ID '{paper_id}' or Collection ID '{collection_id}' not found.")
    session.close()


def remove_paper_from_collection(engine, paper_id, collection_id):
    session = sessionmaker(bind=engine)()
    collection = session.query(Collection).filter_by(collection_id=collection_id).one_or_none()
    if collection:
        paper = session.query(Paper).filter_by(paper_id=paper_id).one_or_none()
        if paper in collection.papers:
            collection.papers.remove(paper)
            session.commit()
            print(f"Paper ID '{paper_id}' removed from Collection ID '{collection_id}' successfully.")
        else:
            print(f"Paper ID '{paper_id}' not found in Collection ID '{collection_id}'.")
    else:
        print(f"Collection ID '{collection_id}' not found.")
    session.close()


def delete_collection(engine, collection_id):
    session = sessionmaker(bind=engine)()
    collection = session.query(Collection).filter_by(collection_id=collection_id).one_or_none()
    if collection:
        session.delete(collection)
        session.commit()
        print(f"Collection ID '{collection_id}' deleted successfully.")
    else:
        print(f"Collection ID '{collection_id}' not found.")
    session.close()


def find_papers(engine, journal=None, author=None, title=None, year_range=None):
    session = sessionmaker(bind=engine)()
    filters = []
    if journal:
        filters.append(Paper.journal == journal)
    if author:
        filters.append(Paper.author == author)
    if title:
        filters.append(Paper.title == title)
    if year_range:
        start_year, end_year = year_range
        filters.append(
            and_(
                Paper.publication_date >= date(start_year, 1, 1),
                Paper.publication_date <= date(end_year, 12, 31),
            )
        )

    papers = session.query(Paper).filter(and_(*filters)).all()
    session.close()
    return papers


def print_papers(papers):
    for paper in papers:
        print(
            f"Paper ID: {paper.paper_id}, Title: {paper.title}, Author: {paper.author}, Journal: {paper.journal}, Publication Date: {paper.publication_date}"
        )
