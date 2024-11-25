from typing import List, Tuple

from giantsmind.metadata_db import collection_operations as col_ops
from giantsmind.utils.logging import logger

from . import helpers as cli_helpers
from .context import CollectionId, Context
from .state import states


def _select_papers(collection_id: CollectionId, keep: bool) -> CollectionId:
    logger.debug(f"Selecting papers from collection {collection_id}...")
    from random import randint

    return CollectionId(randint(0, 10**9))


def _delete_collection_by_id(collection_id: CollectionId) -> None:
    col_ops.delete_collection_by_id(collection_id.value)


def _delete_collection_by_name(name: str) -> None:
    col_ops.delete_collection(name)


def _get_collection_id(collection_name: str) -> int:
    collection_id_int = col_ops.get_collection_id(collection_name)
    return CollectionId(collection_id_int)


def save_collection(collection_id: CollectionId):
    while True:
        name = input("Enter a name for the new collection (c for cancel):\n")
        if name == "c":
            return
        try:
            col_ops.duplicate_collection(collection_id.value, name)
        except col_ops.CollectionNotFoundError:
            logger.error(f"Cannot save collection {collection_id.value}.")
            return
        except col_ops.CollectionExistsError:
            logger.debug(f"Cannot collection with name {name} because it already exists.")
            print(f"A collection with name {name} already exists, choose another name.")
            continue
        except Exception as e:
            raise e
        break


def get_collections() -> Tuple[List[int], List[str]]:
    collection_ints, collections_names = col_ops.get_all_collections()
    collection_ids = [CollectionId(collection_int) for collection_int in collection_ints]
    return collection_ids, collections_names


def select_collection(collections: List[CollectionId]) -> CollectionId:
    collection_names = [col_ops.get_collection_name(collection_id.value) for collection_id in collections]
    options = cli_helpers.list2optargs(
        [
            (collection_name, collection_id)
            for collection_name, collection_id in zip(collection_names, collections)
        ]
    ) + [("c", "Cancel", None)]
    choice = cli_helpers.get_choice(options)
    if choice == "c":
        return None
    return options[choice]


def get_all_papers_collection() -> CollectionId:
    return CollectionId(col_ops.get_all_papers_collectionid())


def act_set_results_as_working_papers(context: Context) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    new_context = context.copy()
    _delete_collection_by_id(new_context.papers)
    new_context.papers = new_context.results
    logger.debug(f"\nnew_context:\n{new_context}")
    return new_context


def act_select_papers(context: Context, keep: bool) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    new_context: Context = context.copy()
    table_selected: CollectionId = context.papers
    if context.results:
        table_selected = context.results
    new_context.selected_papers = _select_papers(table_selected, keep=keep)
    logger.debug(f"\nnew_context:\n{new_context}")
    return new_context


def act_apply_selection(context: Context) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    new_context = context.copy()
    if new_context.previous_state == states.AFTER_SEARCH:
        _delete_collection_by_id(new_context.results)
        new_context.results = new_context.selected_papers
    elif new_context.previous_state == states.INTERACT_WITH_PAPERS:
        _delete_collection_by_id(new_context.papers)
        new_context.papers = new_context.selected_papers
    new_context.current_state = new_context.previous_state
    new_context.selected_papers = None
    logger.debug(f"\nnew_context:\n{new_context}")
    return new_context


def act_load_collection(context: Context, collection_name: str) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    new_context: Context = context.copy()
    new_context.current_state = states.AFTER_SEARCH
    new_context.results = _get_collection_id(collection_name)
    logger.debug(f"\nnew_context:\n{new_context}")
    return new_context


def _merge_collections(collection1: CollectionId, collection2: CollectionId, name: str) -> CollectionId:
    new_collection_id = col_ops.merge_collections([collection1, collection2], name)
    return CollectionId(new_collection_id)


def act_add_results_to_paper_context(context: Context) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    new_context = context.copy()
    new_context.papers = _merge_collections(new_context.papers, context.results)
    logger.debug(f"\nnew_context:\n{new_context}")
    return context


def _add_to_collection(collection_id: CollectionId, papers: CollectionId, name: str) -> None:
    col_ops.merge_collections([collection_id, papers], name, overwrite=True)


def act_add_to_collection(context: Context) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    collection_ids, collection_name = get_collections()
    collection_selected = select_collection(collection_ids)
    if collection_selected is None:
        logger.debug("No collection selected, aborting.")
        return context
    context.results = _add_to_collection(collection_selected, context.results, "search results")
    logger.debug("Adding results to a collection...")
    return context


def act_clean_up(context: Context) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    logger.debug("Cleaning up...")
    _delete_collection_by_name("working_papers")
    _delete_collection_by_name("papers_selected")
    _delete_collection_by_name("results")
    return context


def act_create_new_collection(context: Context) -> Context:
    logger.debug(f"\ninput context:\n{context}")
    if context.current_state == states.AFTER_SEARCH:
        to_save = context.results
    else:
        to_save = context.papers
    if not to_save:
        logger.debug("No papers to save.")
        return context
    save_collection(to_save)
    logger.debug(f"Creating a new collection with collection_id: {to_save}")
    return context
