from datetime import date
from typing import Callable, List, Tuple

import textdistance
from sqlalchemy import and_
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


from giantsmind.database.schema import Author, ChunkIDs, Collection, Paper, Journal, engine
from giantsmind.utils.logging import logger
from giantsmind.database import paper_operations as paper_ops


def _get_author(session: Session, author: str) -> Author:
    author = session.query(Author).filter_by(name=author).one_or_none()
    return author
