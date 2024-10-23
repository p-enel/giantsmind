import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from giantsmind.database.collection_operations import (
    create_collection,
    add_paper_to_collection,
    remove_papers_from_collection,
    delete_collection,
    rename_collection,
    duplicate_collection,
    merge_collections,
    get_collection_id,
    get_collection_name,
    get_all_collections,
    get_paper_paths_from_collection_id,
    get_metadata_from_collection_id,
    CollectionNotFoundError,
    CollectionExistsError,
)
from giantsmind.database.schema import Base, Paper, Collection
from giantsmind.database import paper_operations


@pytest.fixture(scope="module")
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


@pytest.fixture(scope="module")
def sample_papers(test_db):
    papers = [
        Paper(paper_id="1", title="Test Paper 1", file_path="/path/to/paper1.pdf"),
        Paper(paper_id="2", title="Test Paper 2", file_path="/path/to/paper2.pdf"),
        Paper(paper_id="3", title="Test Paper 3", file_path="/path/to/paper3.pdf"),
    ]
    test_db.add_all(papers)
    test_db.commit()
    return papers


# Test create_collection
def test_create_collection(test_db, sample_papers):
    collection_id = create_collection("Test Collection", ["1", "2"], engine=test_db.bind)
    assert collection_id is not None
    collection = test_db.query(Collection).filter_by(name="Test Collection").one()
    assert len(collection.papers) == 2
    assert collection.papers[0].paper_id == "1"
    assert collection.papers[1].paper_id == "2"


# Test add_paper_to_collection
def test_add_paper_to_collection(test_db, sample_papers):
    collection = Collection(name="Add Paper Test")
    test_db.add(collection)
    test_db.commit()

    add_paper_to_collection("3", collection.collection_id, engine=test_db.bind)

    updated_collection = test_db.query(Collection).filter_by(name="Add Paper Test").one()
    assert len(updated_collection.papers) == 1
    assert updated_collection.papers[0].paper_id == "3"


# # Test remove_papers_from_collection
# def test_remove_papers_from_collection(test_db, sample_papers):
#     collection = Collection(name="Remove Paper Test")
#     collection.papers = sample_papers
#     test_db.add(collection)
#     test_db.commit()

#     remove_papers_from_collection(["1", "2"], "Remove Paper Test", engine=test_db.bind)

#     # # Refresh the session to get the latest data
#     # test_db.refresh(collection)

#     assert len(collection.papers) == 1
#     assert collection.papers[0].paper_id == "3"


def test_remove_papers_from_nonexistent_collection(test_db):
    with pytest.raises(CollectionNotFoundError):
        remove_papers_from_collection(["1", "2"], "Nonexistent Collection", engine=test_db.bind)


# # Test to ensure proper handling when trying to remove non-existent papers
# def test_remove_nonexistent_papers_from_collection(test_db, sample_papers):
#     collection = Collection(name="Remove Nonexistent Papers Test")
#     collection.papers = sample_papers
#     test_db.add(collection)
#     test_db.commit()

#     # This should not raise an error, but should log a warning
#     remove_papers_from_collection(["4", "5"], "Remove Nonexistent Papers Test", engine=test_db.bind)

#     # Refresh the session to get the latest data
#     test_db.refresh(collection)

#     # The collection should still have all its original papers
#     assert len(collection.papers) == 3


# Test delete_collection
def test_delete_collection(test_db):
    collection = Collection(name="Delete Test")
    test_db.add(collection)
    test_db.commit()

    delete_collection("Delete Test", engine=test_db.bind)

    assert test_db.query(Collection).filter_by(name="Delete Test").one_or_none() is None


# Test rename_collection
def test_rename_collection(test_db):
    collection = Collection(name="Old Name")
    test_db.add(collection)
    test_db.commit()

    rename_collection("Old Name", "New Name", engine=test_db.bind)

    assert test_db.query(Collection).filter_by(name="Old Name").one_or_none() is None
    assert test_db.query(Collection).filter_by(name="New Name").one() is not None


# Test duplicate_collection
def test_duplicate_collection(test_db, sample_papers):
    original = Collection(name="Original")
    original.papers = sample_papers
    test_db.add(original)
    test_db.commit()

    new_id = duplicate_collection(original.collection_id, "Duplicate", engine=test_db.bind)

    duplicate = test_db.query(Collection).filter_by(collection_id=new_id).one()
    assert duplicate.name == "Duplicate"
    assert len(duplicate.papers) == len(sample_papers)


# Test merge_collections
def test_merge_collections(test_db, sample_papers):
    col1 = Collection(name="Merge1")
    col1.papers = sample_papers[:2]
    col2 = Collection(name="Merge2")
    col2.papers = [sample_papers[2]]
    test_db.add_all([col1, col2])
    test_db.commit()

    merged_id = merge_collections([col1.collection_id, col2.collection_id], "Merged", engine=test_db.bind)

    merged = test_db.query(Collection).filter_by(collection_id=merged_id).one()
    assert merged.name == "Merged"
    assert len(merged.papers) == 3


# Test get_collection_id and get_collection_name
def test_get_collection_id_and_name(test_db):
    collection = Collection(name="Get Test")
    test_db.add(collection)
    test_db.commit()

    col_id = get_collection_id("Get Test", engine=test_db.bind)
    assert col_id == collection.collection_id

    col_name = get_collection_name(col_id, engine=test_db.bind)
    assert col_name == "Get Test"


# Test get_all_collections
def test_get_all_collections(test_db):
    collections = [
        Collection(name="All Test 1"),
        Collection(name="All Test 2"),
        Collection(name="All Test 3"),
    ]
    test_db.add_all(collections)
    test_db.commit()

    ids, names = get_all_collections(engine=test_db.bind)
    assert len(ids) == len(names)
    assert "All Test 1" in names
    assert "All Test 2" in names
    assert "All Test 3" in names


# Test get_paper_paths_from_collection_id
def test_get_paper_paths_from_collection_id(test_db, sample_papers):
    collection = Collection(name="Paths Test")
    collection.papers = sample_papers
    test_db.add(collection)
    test_db.commit()

    paths = get_paper_paths_from_collection_id(collection.collection_id, engine=test_db.bind)
    assert len(paths) == 3
    assert "/path/to/paper1.pdf" in paths
    assert "/path/to/paper2.pdf" in paths
    assert "/path/to/paper3.pdf" in paths


# Test get_metadata_from_collection_id
def test_get_metadata_from_collection_id(test_db, sample_papers):
    collection = Collection(name="Metadata Test")
    collection.papers = sample_papers
    test_db.add(collection)
    test_db.commit()

    metadata = get_metadata_from_collection_id(collection.collection_id, engine=test_db.bind)
    assert len(metadata) == 3
    assert metadata[0]["title"] == "Test Paper 1"
    assert metadata[1]["title"] == "Test Paper 2"
    assert metadata[2]["title"] == "Test Paper 3"


# Test error cases
def test_collection_not_found_error(test_db):
    with pytest.raises(CollectionNotFoundError):
        delete_collection("Non-existent Collection", engine=test_db.bind)


def test_collection_exists_error(test_db):
    collection = Collection(name="Existing Collection")
    test_db.add(collection)
    test_db.commit()

    with pytest.raises(CollectionExistsError):
        create_collection("Existing Collection", [], engine=test_db.bind)
