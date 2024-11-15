from sqlalchemy.orm import Session

from giantsmind.database.schema import Author


def _get_author(session: Session, author: str) -> Author:
    author = session.query(Author).filter_by(name=author).one_or_none()
    return author
