from sqlalchemy import create_engine, Column, Integer, String, Date, Table, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from giantsmind.article_metadata.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

Base = declarative_base()

paper_collection_association = Table(
    "paper_collection",
    Base.metadata,
    Column("paper_id", Integer, ForeignKey("papers.paper_id")),
    Column("collection_id", Integer, ForeignKey("collections.collection_id")),
)


class Paper(Base):
    __tablename__ = "papers"
    paper_id = Column(Integer, primary_key=True)
    journal = Column(String)
    file_path = Column(String)
    publication_date = Column(Date)
    title = Column(String)
    author = Column(String)
    url = Column(String)
    collections = relationship("Collection", secondary=paper_collection_association, back_populates="papers")


class Collection(Base):
    __tablename__ = "collections"
    collection_id = Column(Integer, primary_key=True)
    name = Column(String)
    papers = relationship("Paper", secondary=paper_collection_association, back_populates="collections")


def init_db():
    Base.metadata.create_all(engine)
    print("Database and tables created successfully and saved to disk as 'papers.db'.")


def get_session():
    Session = sessionmaker(bind=engine)
    return Session()
