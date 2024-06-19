import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from giantsmind.database import schema, operations

engine = create_engine("sqlite:///:memory:")
schema.Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)


@pytest.fixture(scope="module")
def session():
    """Create a new database session for the tests."""
    session = Session()
    yield session
    session.close()


def test_add_paper(session):
    session.connection()
    operations.add_paper(
        session=session,
        journal="Journal of AI Research",
        file_path="/path/to/paper1.pdf",
        publication_date=date(2023, 5, 24),
        title="Advancements in AI",
        author="John Doe",
        url="http://example.com/paper1",
    )
    paper = session.query(schema.Paper).filter_by(title="Advancements in AI").one()
    assert paper.author == "John Doe"


def test_remove_paper(session):
    operations.add_paper(
        session=session,
        journal="Journal of AI Research",
        file_path="/path/to/paper_to_remove.pdf",
        publication_date=date(2023, 5, 24),
        title="Paper to Remove",
        author="Jane Doe",
        url="http://example.com/paper_to_remove",
    )
    paper = session.query(schema.Paper).filter_by(title="Paper to Remove").one()
    operations.remove_paper(session, paper.paper_id)
    paper = session.query(schema.Paper).filter_by(title="Paper to Remove").one_or_none()
    assert paper is None


def test_create_collection(session):
    operations.add_paper(
        session=session,
        journal="Journal of AI Research",
        file_path="/path/to/paper2.pdf",
        publication_date=date(2023, 5, 24),
        title="Another AI Paper",
        author="Alice Smith",
        url="http://example.com/paper2",
    )
    paper = session.query(schema.Paper).filter_by(title="Another AI Paper").one()
    operations.create_collection(session, "Test Collection", [paper.paper_id])
    collection = session.query(schema.Collection).filter_by(name="Test Collection").one()
    assert len(collection.papers) == 1


def test_add_paper_to_collection(session):
    paper = session.query(schema.Paper).filter_by(title="Advancements in AI").one()
    collection = session.query(schema.Collection).filter_by(name="Test Collection").one()
    operations.add_paper_to_collection(session, paper.paper_id, collection.collection_id)
    collection = session.query(schema.Collection).filter_by(name="Test Collection").one()
    assert len(collection.papers) == 2


def test_remove_paper_from_collection(session):
    paper = session.query(schema.Paper).filter_by(title="Advancements in AI").one()
    collection = session.query(schema.Collection).filter_by(name="Test Collection").one()
    operations.remove_paper_from_collection(session, paper.paper_id, collection.collection_id)
    collection = session.query(schema.Collection).filter_by(name="Test Collection").one()
    assert len(collection.papers) == 1


def test_delete_collection(session):
    collection = session.query(schema.Collection).filter_by(name="Test Collection").one()
    operations.delete_collection(session, collection.collection_id)
    collection = session.query(schema.Collection).filter_by(name="Test Collection").one_or_none()
    assert collection is None


def test_find_papers(session):
    papers = operations.find_papers(
        session, journal="Journal of AI Research", author="John Doe", year_range=(2023, 2023)
    )
    assert len(papers) == 1
    assert papers[0].title == "Advancements in AI"
