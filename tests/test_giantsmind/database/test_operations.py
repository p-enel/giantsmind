import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from giantsmind.database import operations as db_ops
from giantsmind.database.schema import Base, Paper, Author, Journal, Collection, ChunkIDs


@pytest.fixture(scope="module")
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def sample_data(test_db):
    journal1 = Journal(journal_id="10.1000", name="Test Journal")
    journal2 = Journal(journal_id="arXiv", name="Another Journal")
    test_db.add_all([journal1, journal2])

    author1 = Author(name="John Doe")
    author2 = Author(name="Jane Smith")
    test_db.add_all([author1, author2])

    paper1 = Paper(
        paper_id="TEST123",
        journal=journal1,
        title="Test Paper",
        publication_date=date(2023, 1, 1),
        authors=[author1, author2],
        url="http://example.com/test",
        file_path="/path/to/test.pdf",
    )
    paper2 = Paper(
        paper_id="TEST456",
        journal=journal2,
        title="Another Test Paper",
        publication_date=date(2023, 2, 1),
        authors=[author1],
        url="http://example.com/another",
        file_path="/path/to/another.pdf",
    )
    test_db.add_all([paper1, paper2])

    collection1 = Collection(name="Test Collection", papers=[paper1])
    collection2 = Collection(name="Another Collection", papers=[paper2])
    test_db.add_all([collection1, collection2])

    chunk1 = ChunkIDs(chunk_id="CHUNK1", paper=paper1)
    chunk2 = ChunkIDs(chunk_id="CHUNK2", paper=paper1)
    test_db.add_all([chunk1, chunk2])

    test_db.commit()

    yield test_db

    test_db.query(ChunkIDs).delete()
    test_db.query(Collection).delete()
    test_db.query(Paper).delete()
    test_db.query(Author).delete()
    test_db.query(Journal).delete()
    test_db.commit()


def test_get_paper(sample_data):
    paper = db_ops._get_paper(sample_data, "TEST123")
    assert paper is not None
    assert paper.title == "Test Paper"


def test_get_collection(sample_data):
    collection = db_ops._get_collection(sample_data, 1)
    assert collection is not None
    assert collection.name == "Test Collection"


def test_get_collection_from_name(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    assert collection is not None
    assert collection.name == "Test Collection"


def test_delete_collection(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    db_ops._delete_collection(sample_data, collection)
    assert db_ops._get_collection_from_name(sample_data, "Test Collection") is None


def test_create_collection_core(sample_data):
    papers = sample_data.query(Paper).all()
    new_collection = db_ops._create_collection_core(sample_data, "New Collection", papers)
    assert new_collection is not None
    assert new_collection.name == "New Collection"
    assert len(new_collection.papers) == 2


def test_create_collection(sample_data):
    papers = sample_data.query(Paper).all()
    new_collection = db_ops._create_collection(sample_data, "New Collection", papers)
    assert new_collection is not None
    assert new_collection.name == "New Collection"
    assert len(new_collection.papers) == 2

    # Test overwrite
    overwritten_collection = db_ops._create_collection(
        sample_data, "New Collection", [papers[0]], overwrite=True
    )
    assert len(overwritten_collection.papers) == 1


def test_remove_paper_from_collection(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    paper = db_ops._get_paper(sample_data, "TEST123")
    db_ops._remove_paper_from_collection(sample_data, paper, collection)
    assert len(collection.papers) == 0


def test_get_all_papers(sample_data):
    papers = db_ops._get_all_papers(sample_data)
    assert len(papers) == 2


def test_get_all_papers_collection(sample_data):
    all_papers_collection = db_ops._get_all_papers_collection(sample_data)
    assert all_papers_collection is not None
    assert all_papers_collection.name == "all papers"
    assert len(all_papers_collection.papers) == 2


def test_get_author(sample_data):
    author = db_ops._get_author(sample_data, "John Doe")
    assert author is not None
    assert author.name == "John Doe"


def test_get_journal(sample_data):
    journal = db_ops._get_journal(sample_data, "Test Journal")
    assert journal is not None
    assert journal.name == "Test Journal"


def test_add_paper(sample_data):
    metadata = {
        "paper_id": "TEST789",
        "journal": "New Journal",
        "file_path": "/path/to/new.pdf",
        "publication_date": date(2023, 3, 1),
        "title": "New Test Paper",
        "authors": ["Alice Johnson"],
        "url": "http://example.com/new",
    }
    new_paper = db_ops._add_paper(sample_data, metadata)
    assert new_paper is not None
    assert new_paper.paper_id == "TEST789"
    assert new_paper.title == "New Test Paper"


def test_add_chunks(sample_data):
    paper = db_ops._get_paper(sample_data, "TEST123")
    chunk_ids = ["CHUNK3", "CHUNK4"]
    db_ops._add_chunks(sample_data, chunk_ids, paper)
    chunks = sample_data.query(ChunkIDs).filter_by(paper_id="TEST123").all()
    assert len(chunks) == 4  # 2 existing + 2 new


def test_get_papers(sample_data):
    papers = db_ops.get_papers(["TEST123", "TEST456"])
    assert len(papers) == 2
    assert papers[0].paper_id == "TEST123"
    assert papers[1].paper_id == "TEST456"


def test_get_all_papers_collectionid(sample_data):
    collection_id = db_ops.get_all_papers_collectionid()
    assert collection_id is not None


def test_add_papers(sample_data):
    metadata = [
        {
            "paper_id": "TEST999",
            "journal": "New Journal",
            "file_path": "/path/to/new.pdf",
            "publication_date": date(2023, 4, 1),
            "title": "Another New Test Paper",
            "authors": ["Bob Wilson"],
            "url": "http://example.com/another_new",
        }
    ]
    new_papers = db_ops.add_papers(metadata)
    assert len(new_papers) == 1
    assert new_papers[0].paper_id == "TEST999"


def test_add_chunks_public(sample_data):
    db_ops.add_chunks(["CHUNK5", "CHUNK6"], "TEST123")
    chunks = sample_data.query(ChunkIDs).filter_by(paper_id="TEST123").all()
    assert len(chunks) == 4  # 2 existing + 2 new


def test_remove_papers(sample_data):
    db_ops.remove_papers(["TEST123"])
    paper = db_ops._get_paper(sample_data, "TEST123")
    assert paper is None


def test_create_collection_public(sample_data):
    collection_id = db_ops.create_collection("New Public Collection", ["TEST456"])
    assert collection_id is not None
    collection = db_ops._get_collection(sample_data, collection_id)
    assert collection.name == "New Public Collection"
    assert len(collection.papers) == 1


def test_add_paper_to_collection_public(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Another Collection")
    db_ops.add_paper_to_collection("TEST123", collection.collection_id)
    updated_collection = db_ops._get_collection_from_name(sample_data, "Another Collection")
    assert len(updated_collection.papers) == 2


def test_remove_papers_from_collection_public(sample_data):
    db_ops.remove_papers_from_collection(["TEST456"], "Another Collection")
    collection = db_ops._get_collection_from_name(sample_data, "Another Collection")
    assert len(collection.papers) == 1


def test_delete_collection_public(sample_data):
    db_ops.delete_collection("Another Collection")
    collection = db_ops._get_collection_from_name(sample_data, "Another Collection")
    assert collection is None


def test_delete_collection_by_id(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    db_ops.delete_collection_by_id(collection.collection_id)
    assert db_ops._get_collection_from_name(sample_data, "Test Collection") is None


def test_list_unique_vals(sample_data):
    unique_journals = db_ops.list_unique_vals("journal")
    assert "Test Journal" in unique_journals
    assert "Another Journal" in unique_journals


def test_rename_collection(sample_data):
    db_ops.rename_collection("Test Collection", "Renamed Collection")
    assert db_ops._get_collection_from_name(sample_data, "Test Collection") is None
    assert db_ops._get_collection_from_name(sample_data, "Renamed Collection") is not None


def test_duplicate_collection(sample_data):
    original_collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    new_collection_id = db_ops.duplicate_collection(
        original_collection.collection_id, "Duplicated Collection"
    )
    new_collection = db_ops._get_collection(sample_data, new_collection_id)
    assert new_collection is not None
    assert new_collection.name == "Duplicated Collection"
    assert len(new_collection.papers) == len(original_collection.papers)


def test_merge_collections(sample_data):
    collection1 = db_ops._get_collection_from_name(sample_data, "Test Collection")
    collection2 = db_ops._get_collection_from_name(sample_data, "Another Collection")
    merged_collection_id = db_ops.merge_collections(
        [collection1.collection_id, collection2.collection_id], "Merged Collection"
    )
    merged_collection = db_ops._get_collection(sample_data, merged_collection_id)
    assert merged_collection is not None
    assert merged_collection.name == "Merged Collection"
    assert len(merged_collection.papers) == len(collection1.papers) + len(collection2.papers)


def test_get_collection_id(sample_data):
    collection_id = db_ops.get_collection_id("Test Collection")
    assert collection_id is not None
    collection = db_ops._get_collection(sample_data, collection_id)
    assert collection.name == "Test Collection"


def test_get_collection_name(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    collection_name = db_ops.get_collection_name(collection.collection_id)
    assert collection_name == "Test Collection"


def test_get_all_collections(sample_data):
    collection_ids, collection_names = db_ops.get_all_collections()
    assert len(collection_ids) == len(collection_names)
    assert "Test Collection" in collection_names
    assert "Another Collection" in collection_names


def test_get_paper_paths_from_collection_id(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    paper_paths = db_ops.get_paper_paths_from_collection_id(collection.collection_id)
    assert len(paper_paths) == 1
    assert paper_paths[0] == "/path/to/test.pdf"


def test_get_metadata_from_collection_id(sample_data):
    collection = db_ops._get_collection_from_name(sample_data, "Test Collection")
    metadata = db_ops.get_metadata_from_collection_id(collection.collection_id)
    assert len(metadata) == 1
    assert metadata[0]["title"] == "Test Paper"
    assert metadata[0]["paper_id"] == "TEST123"


# Add more tests for edge cases and error conditions
def test_paper_not_found_error(sample_data):
    with pytest.raises(db_ops.PaperNotFoundError):
        db_ops.get_papers(["NONEXISTENT"])


def test_collection_not_found_error(sample_data):
    with pytest.raises(db_ops.CollectionNotFoundError):
        db_ops.delete_collection("Nonexistent Collection")


def test_collection_exists_error(sample_data):
    with pytest.raises(db_ops.CollectionExistsError):
        db_ops.create_collection("Test Collection", ["TEST123"])


# Add more edge case and error condition tests as needed
