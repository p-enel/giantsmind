from sqlalchemy import Column, Date, ForeignKey, Integer, String, Table, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from giantsmind.metadata_db.config import DEFAULT_DATABASE_URL
from giantsmind.utils.local import get_local_data_path
from giantsmind.utils.logging import logger

Base = declarative_base()
engine: Engine = create_engine(DEFAULT_DATABASE_URL)

paper_collection_association = Table(
    "paper_collection",
    Base.metadata,
    Column("paper_id", Integer, ForeignKey("papers.paper_id")),
    Column("collection_id", Integer, ForeignKey("collections.collection_id")),
)

author_paper_association = Table(
    "author_paper",
    Base.metadata,
    Column("author_id", Integer, ForeignKey("authors.author_id")),
    Column("paper_id", Integer, ForeignKey("papers.paper_id")),
)


class Collection(Base):
    __tablename__ = "collections"
    collection_id = Column(Integer, primary_key=True)
    name = Column(String)
    papers = relationship("Paper", secondary=paper_collection_association, back_populates="collections")


class Author(Base):
    __tablename__ = "authors"
    author_id = Column(Integer, primary_key=True)
    name = Column(String)
    papers = relationship("Paper", secondary=author_paper_association, back_populates="authors")


class Journal(Base):
    __tablename__ = "journals"
    journal_id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    papers = relationship("Paper", back_populates="journal")


class Paper(Base):
    __tablename__ = "papers"
    paper_id = Column(String, primary_key=True)
    journal_id = Column(Integer, ForeignKey("journals.journal_id"))
    journal = relationship("Journal", back_populates="papers")
    file_path = Column(String)
    publication_date = Column(Date)
    title = Column(String, nullable=False)
    url = Column(String)
    collections = relationship("Collection", secondary=paper_collection_association, back_populates="papers")
    chunks_ids = relationship("ChunkIDs", back_populates="paper")
    authors = relationship("Author", secondary=author_paper_association, back_populates="papers")


class ChunkIDs(Base):
    __tablename__ = "chunk_ids"
    chunk_id = Column(String, primary_key=True)
    paper_id = Column(String, ForeignKey("papers.paper_id"))
    paper = relationship("Paper", back_populates="chunks_ids")


def init_db():
    Base.metadata.create_all(engine)
    logger.info("Database and tables created successfully and saved to disk as 'papers.db'.")


# def get_session():
#     Session = sessionmaker(bind=engine)
#     return Session()

if (get_local_data_path() / "papers.db").exists():
    logger.info("Database already exists. Skipping database initialization.")
else:
    init_db()
    logger.info("Database initialized successfully.")
