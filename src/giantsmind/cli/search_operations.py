from context import Context, CollectionId
from state import states
from giantsmind.agent.sql_agent import get_sql_query, execute_query
from typing import List
from giantsmind.database import collection_operations as col_ops
from giantsmind.database import utils as db_utils
from giantsmind.utils.logging import logger


def _check_sql_results(results: List[str]):
    logger.debug(f"Checking SQL results: {results}")
    if not isinstance(results, list):
        logger.error(f"The results of the SQL query is not a list. Results: {results}")
        raise TypeError(f"The results of the SQL query must be a list. Results: {results}")
    if not all(isinstance(result, str) for result in results):
        logger.error("Each result of the SQL query is not a string.")
        raise TypeError(f"Each result of the SQL query must be a string. Results: {results}")
    logger.debug("SQL results are valid.")


def metadata_search(search_scope: CollectionId) -> CollectionId:
    while True:
        answer = input("What papers are you looking for? (c for cancel)\n")
        if answer == "c":
            logger.debug("Search cancelled by user.")
            return
        query = get_sql_query(answer, search_scope.value)
        if not query:
            logger.debug("No query was generated.")
            print("No query was generated. Did you ask for papers?")
            continue
        results = execute_query(query)
        _check_sql_results(results)
        collection_id = col_ops.create_collection("results", results, overwrite=True)
        break
    logger.debug(f"Metadata search results: {collection_id}")
    return CollectionId(collection_id)


def print_papers_from_collection(collection: CollectionId):
    try:
        db_utils.print_papers_from_collection(collection)
    except col_ops.CollectionNotFoundError:
        logger.error(f"Collection with ID '{collection}' not found.")


def act_search_with_metadata(context: Context) -> Context:
    logger.debug(str(context))
    new_context = context.copy()
    collection_results = metadata_search(new_context.search_scope)
    if not collection_results:
        logger.info("Search cancelled.")
        return new_context

    new_context.results = collection_results
    new_context.current_state = states.AFTER_SEARCH
    print("\nSearch results:")
    print_papers_from_collection(collection_results.value)
    print()
    logger.debug(str(new_context))
    return new_context


def semantic_search(search_scope: CollectionId) -> int:
    # Dummy function to simulate search
    from random import randint

    return CollectionId(randint(0, 10**9))


def act_search_with_content(context: Context) -> Context:
    print(context)
    new_context = context.copy()
    new_context.results = semantic_search(new_context.search_scope)
    new_context.current_state = states.AFTER_SEARCH
    print(new_context)
    return new_context


def act_refine_search(context: Context) -> Context:
    print(context)
    new_context = context.copy()
    new_context.search_scope = new_context.results
    new_context.current_state = states.SEARCH_PAPERS
    print("new_context", new_context)
    return new_context
