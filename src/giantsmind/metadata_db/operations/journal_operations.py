from sqlalchemy.orm import Session

from giantsmind.metadata_db.schema import Journal


def _get_journal(session: Session, journal: str) -> Journal:
    journal = session.query(Journal).filter_by(name=journal).one_or_none()
    return journal


def _get_journal_from_id(session: Session, journal_id: str) -> Journal:
    journal = session.query(Journal).filter_by(journal_id=journal_id).one_or_none()
    return journal


def _get_journal_id_from_paper_id(paper_id: str) -> str:
    id_type, id_ = paper_id.split(":")
    journal_id = None
    if id_type == "doi":
        journal_id = id_.split("/")[0]
    elif id_type == "arXiv":
        journal_id = "arXiv"
    return journal_id
