from typing import Callable, List, Tuple

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from giantsmind.metadata_db import paper_operations as paper_ops
from giantsmind.metadata_db import utils as db_utils
from giantsmind.metadata_db.schema import Collection, Paper, engine
from giantsmind.utils.logging import logger


class CollectionNotFoundError(BaseException):
    def __init__(self, collection_id: int, type="ID", message="Collection not found"):
        self.message = f"{message} with {type} '{collection_id}'."
        super().__init__(message)

    def __str__(self):
        return self.message


class CollectionExistsError(BaseException):
    def __init__(self, collection_name: str, message="Collection already exists"):
        self.message = f"{message} with name '{collection_name}'."
        super().__init__(message)

    def __str__(self):
        return self.message


def _get_collection(session: Session, collection_id: int) -> Collection:
    collection = session.query(Collection).filter_by(collection_id=collection_id).one_or_none()
    return collection


def _get_collection_from_name(session: Session, name: str) -> Collection:
    collection = session.query(Collection).filter_by(name=name).one_or_none()
    return collection


def _delete_collection(session: Session, collection: Collection) -> None:
    session.delete(collection)
    session.commit()


def _create_collection_core(session: Session, name: str, papers: List[Paper]) -> Collection:
    new_collection = Collection(name=name)
    new_collection.papers = papers
    session.add(new_collection)
    session.commit()
    return new_collection


def _create_collection(
    session: Session,
    name: str,
    papers: List[Paper],
    overwrite: bool = False,
    core_func: Callable[[Session, str, List[Paper]], Collection] = _create_collection_core,
) -> Collection:
    # Fetch collection with same name if exists
    collection = core_func(session, name)
    name_tmp = None
    if collection:  # If collection with same name exists
        if not overwrite:
            # Raise error if overwrite is False
            logger.error(f"Collection '{name}' already exists.")
            raise CollectionExistsError(name)
        logger.debug(f"Collection '{name}' already exists, will overwrite.")
        # Keep existing collection until new collection successfully added
        # but rename it to avoid collision with new collection
        name_tmp = f"{name}_tmp"
        _rename_collection(session, collection, name_tmp)

    try:
        # Create new collection
        new_collection = _create_collection_core(session, name, papers)
    except Exception as e:
        logger.error(f"Could not create collection {name}.")
        if name_tmp:
            # If creation fails and collection with same name existed,
            # existing collection is renamed back to its original name.
            _rename_collection(session, collection, name)
        raise e

    if name_tmp:
        # Delete collection with same name if it existed
        _delete_collection(session, collection)

    return new_collection


def _remove_paper_from_collection(session: Session, paper: Paper, collection: Collection):
    try:
        collection.papers.remove(paper)
    except ValueError as e:
        logger.error(f"Paper ID '{paper.paper_id}' not found in collection '{collection.name}'.")
        raise e
    finally:
        session.commit()


def _get_all_papers_collection(
    session: Session,
    get_col_from_name_fn: Callable[[Session, str], Collection] = _get_collection_from_name,
    create_collection_fn: Callable[[Session, Callable], Collection] = _create_collection,
) -> Collection:
    all_papers_col = get_col_from_name_fn(session, "all papers")
    if not all_papers_col:
        logger.debug("'all papers' collection not found. Creating it.")
        all_papers = paper_ops.get_all_papers(session)
        all_papers_col = create_collection_fn(session, "all papers", all_papers)

    return all_papers_col


def _handle_missing_papers(paper_results: List[Paper | None], paper_ids: List[str], collection_name: str):
    missing_indices = [index for index, paper in enumerate(paper_results) if paper is None]
    missing_paper_ids = tuple([paper_ids[index] for index in missing_indices])
    logger.error(
        f"Could not create collection '{collection_name}'. Paper(s) {missing_paper_ids} have not been found."
    )
    raise paper_ops.PaperNotFoundError(missing_paper_ids)


def create_collection(
    name: str, paper_ids: List[str], overwrite: bool = False, engine: Engine = engine
) -> int:
    logger.debug(f"Creating collection '{name}' with papers: {paper_ids}. Overwrite: {overwrite}.")
    with Session(engine) as session:
        papers = [paper_ops._get_paper(session, paper_id) for paper_id in paper_ids]
        if None in papers:
            _handle_missing_papers(papers, paper_ids, name)

        new_collection_id = _create_collection(session, name, papers, overwrite).collection_id

    return new_collection_id


def add_paper_to_collection(paper_id: str, collection_id: int, engine: Engine = engine) -> None:
    with Session(engine) as session:
        paper = paper_ops._get_paper(session, paper_id)
        if not paper:
            logger.error(f"Paper ID '{paper_id}' not found.")
            raise paper_ops.PaperNotFoundError(paper_id)
        collection = _get_collection(session, collection_id)
        if not collection:
            logger.error(f"Collection ID '{collection_id}' not found.")
            raise CollectionNotFoundError(collection_id)
        collection.papers.append(paper)
        session.commit()
        logger.info(f"Paper ID '{paper_id}' added to Collection ID '{collection_id}' successfully.")


def remove_papers_from_collection(
    paper_ids: List[str], collection_name: str, engine: Engine = engine
) -> None:
    with Session(engine) as session:
        collection = _get_collection_from_name(session, collection_name)
        if collection is None:
            logger.error(f"Collection '{collection_name}' not found.")
            raise CollectionNotFoundError(collection_name)

        papers = paper_ops.get_papers(paper_ids, engine)
        for paper in papers:
            if paper not in collection.papers:
                logger.warning(f"Paper ID '{paper.paper_id}' not found in collection '{collection_name}'.")
                continue
            _remove_paper_from_collection(session, paper, collection)
        logger.info(f"Papers {paper_ids} removed from collection '{collection_name}' successfully.")


def delete_collection(name: str, engine: Engine = engine) -> None:
    with Session(engine) as session:
        collection = _get_collection_from_name(session, name)
        if not collection:
            logger.error(f"Collection '{name}' not found.")
            raise CollectionNotFoundError(name)
        _delete_collection(session, collection)


def delete_collection_by_id(collection_id: int, engine: Engine = engine) -> None:
    with Session(engine) as session:
        collection = _get_collection(session, collection_id)
        if not collection:
            logger.error(f"Collection ID '{collection_id}' not found.")
            raise CollectionNotFoundError(collection_id)
        _delete_collection(session, collection)
        logger.info(f"Collection ID '{collection_id}' deleted successfully.")


def _rename_collection(session: Session, collection: Collection, new_name: str) -> None:
    collection.name = new_name
    session.commit()


def rename_collection(old_name: str, new_name: str, engine: Engine = engine) -> None:
    with Session(engine) as session:
        collection_old = _get_collection_from_name(session, old_name)
        if not collection_old:
            logger.error(f"Collection '{old_name}' not found.")
            raise CollectionNotFoundError(old_name)
        collection_new = _get_collection_from_name(session, new_name)
        if collection_new:
            logger.error(f"Collection '{new_name}' already exists.")
            raise CollectionExistsError(new_name)
        _rename_collection(session, collection_old, new_name)
        logger.info(f"Successfully rename collection '{old_name}' to '{new_name}'.")


def duplicate_collection(collection_id: int, new_name: str, engine: Engine = engine) -> int:
    with Session(engine) as session:
        collection = _get_collection(session, collection_id)
        if not collection:
            logger.error(f"Collection ID '{collection_id}' not found.")
            raise CollectionNotFoundError(collection_id)
        try:
            new_collection = _create_collection(session, new_name, collection.papers, overwrite=False)
            new_collection_id = new_collection.collection_id
        except CollectionExistsError as e:
            logger.error(f"Cannot duplicate collection, name '{new_name}' already exists.")
            raise e

    return new_collection_id


def _merge_collections(
    session: Session, collections: List[Collection], new_name: str, overwrite: bool = False
) -> Collection:
    papers = []
    for collection in collections:
        papers.extend(collection.papers)
    new_collection = _create_collection(session, new_name, papers, overwrite=overwrite)
    return new_collection


def merge_collections(
    collection_ids: List[int],
    new_name: str,
    overwrite: bool = False,
    engine: Engine = engine,
) -> int:
    if len(collection_ids) < 2:
        raise ValueError("At least two collections are required to merge.")
    with Session(engine) as session:
        collections = session.query(Collection).filter(Collection.collection_id.in_(collection_ids)).all()
        if len(collections) < 2:
            raise ValueError("Less than two collections found.")
        if collections:
            merged_collection = _merge_collections(session, collections, new_name, overwrite=overwrite)
            new_collection_id = merged_collection.collection_id
        else:
            new_collection_id = None
    return new_collection_id


def get_collection_id(collection_name: str, engine: Engine = engine) -> int:
    with Session(engine) as session:
        collection = session.query(Collection).filter_by(name=collection_name).one_or_none()
        if not collection:
            return None
        collection_id = collection.collection_id
    return collection_id


def get_collection_name(collection_id: int, engine: Engine = engine) -> str:
    with Session(engine) as session:
        collection = session.query(Collection).filter_by(collection_id=collection_id).one_or_none()
        if not collection:
            return None
        collection_name = collection.name
    return collection_name


def get_all_collections(engine: Engine = engine) -> Tuple[List[int], List[str]]:
    with Session(engine) as session:
        collections = session.query(Collection).all()
        collection_names = [collection.name for collection in collections]
        collection_ids = [collection.collection_id for collection in collections]
    return collection_ids, collection_names


def get_paper_paths_from_collection_id(collection_id: int, engine: Engine = engine) -> List[str]:
    with Session(engine) as session:
        collection = _get_collection(session, collection_id)
        if not collection:
            logger.error(f"Collection ID '{collection_id}' not found.")
            raise CollectionNotFoundError(collection_id)
        file_paths = [paper.file_path for paper in collection.papers]
    return file_paths


def get_metadata_from_collection_id(collection_id: int, engine: Engine = engine) -> List[dict]:
    with Session(engine) as session:
        collection = _get_collection(session, collection_id)
        if not collection:
            logger.error(f"Collection ID '{collection_id}' not found.")
            raise CollectionNotFoundError(collection_id)
        metadata = [db_utils._paper_to_dict(paper) for paper in collection.papers]
    return metadata


def get_all_papers_collectionid(engine: Engine = engine) -> int:
    with Session(engine) as session:
        all_papers_col = _get_collection_from_name(session, "all papers")
        if not all_papers_col:
            logger.debug("'all papers' collection not found. Creating it.")
            all_papers_col = _get_all_papers_collection(session)
            logger.debug("Created 'all papers' collection.")
        all_papers_id = all_papers_col.collection_id
    logger.debug(f"all_papers_id: {all_papers_id}")
    return all_papers_id


def get_paper_ids_from_collectionid(collection_id: int, engine: Engine = engine) -> List[str]:
    with Session(engine) as session:
        collection = _get_collection(session, collection_id)
        if not collection:
            logger.error(f"Collection ID '{collection_id}' not found.")
            raise CollectionNotFoundError(collection_id)
        paper_ids = [paper.paper_id for paper in collection.papers]
    return paper_ids
