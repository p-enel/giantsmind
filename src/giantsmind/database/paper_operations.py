from typing import Callable, List
from datetime import datetime

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from giantsmind.database import author_operations as author_ops
from giantsmind.database import journal_operations as journal_ops
from giantsmind.database.schema import Author, ChunkIDs, Journal, Paper, engine
from giantsmind.utils.logging import logger


class PaperNotFoundError(BaseException):
    def __init__(self, paper: str, message="No papers found"):
        self.message = f"{message} with ID '{paper}'."
        super().__init__(message)

    def __str__(self):
        return self.message


class PaperExistsError(BaseException):
    def __init__(self, paper_id: str, message="Paper already exists"):
        self.message = f"{message} with ID '{paper_id}'."
        super().__init__(message)

    def __str__(self):
        return self.message


def _get_paper(session: Session, paper_id: str) -> Paper:
    paper = session.query(Paper).filter_by(paper_id=paper_id).one_or_none()
    return paper


def get_all_papers(session: Session):
    return session.query(Paper).all()


def _add_paper(
    session: Session, metadata: dict, get_paper_fn: Callable[[Session, str], Paper] = _get_paper
) -> Paper:
    paper = get_paper_fn(session, metadata["paper_id"])
    if paper:
        logger.error(f"Paper with ID '{metadata['paper_id']}' already exists.")
        return None

    authors = []
    for author_str in metadata["authors"]:
        author_db = author_ops._get_author(session, author_str)
        if not author_db:
            author_db = Author(name=author_str)
            session.add(author_db)
            session.commit()
        authors.append(author_db)

    journal_id = journal_ops._get_journal_id_from_paper_id(metadata["paper_id"])
    if not journal_id:
        logger.error(f"Could not get journal ID from paper ID '{metadata['paper_id']}'.")
        raise ValueError(f"Could not get journal ID from paper ID '{metadata['paper_id']}'.")

    journal = journal_ops._get_journal_from_id(session, journal_id)

    if not journal:
        journal = Journal(name=metadata["journal"], journal_id=journal_id)
        session.add(journal)
        session.commit()

    publication_date = datetime.strptime(metadata["publication_date"], "%Y-%m-%d").date()

    new_paper = Paper(
        paper_id=metadata["paper_id"],
        journal=journal,
        file_path=metadata["file_path"],
        publication_date=publication_date,
        title=metadata["title"],
        authors=authors,
        url=metadata["url"],
    )
    session.add(new_paper)
    session.commit()
    return Paper


def _add_papers(session: Session, metadatas: list[dict]) -> list[Paper]:
    papers = []
    for metadata in metadatas:
        paper = _add_paper(session, metadata)
        papers.append(paper)
    return papers


def get_papers(paper_ids: List[str], engine: Engine = engine) -> List[Paper]:
    results = []
    with Session(engine) as session:
        for paper_id in paper_ids:
            paper = _get_paper(session, paper_id)
            if not paper:
                logger.error(f"Could not find paper with ID '{paper_id}'.")
                raise PaperNotFoundError(paper_id)
            results.append(paper)
    return results


def add_papers(
    metadatas: list[dict],
    engine: Engine = engine,
) -> list[str]:
    with Session(engine) as session:
        papers = _add_papers(session, metadatas)
    return papers


def _remove_paper(session: Session, paper: Paper) -> None:
    session.delete(paper)
    session.commit()


def remove_papers(
    paper_ids: List[str],
    engine: Engine = engine,
) -> None:
    session = Session(engine)
    for paper_id in paper_ids:
        paper = _get_paper(session, paper_id)
        if not paper:
            logger.error(f"Paper {paper_id} not found. Could not remove paper.")
            raise PaperNotFoundError(paper_id)
        _remove_paper(session, paper)
    session.close()


def _add_chunks(session: Session, chunk_ids: List[str], paper: Paper) -> None:
    for chunk_id in chunk_ids:
        new_chunk = ChunkIDs(chunk_id=chunk_id, paper=paper)
        session.add(new_chunk)
    session.commit()


def add_chunks(chunk_ids: List[str], paper_id: str, engine: Engine = engine) -> None:
    with Session(engine) as session:
        paper = _get_paper(session, paper_id)
        if not paper:
            logger.error(f"Could not add chunks to paper: {paper_id}")
            raise PaperNotFoundError(paper_id)
        _add_chunks(session, chunk_ids, paper)
