from typing import List, Tuple

from langchain_core.documents.base import Document

from giantsmind.agents import answering, question_parsing, sql
from giantsmind.core import process_results as proc_res
from giantsmind.core.models import MetadataResult, ParsedElements, SearchResults
from giantsmind.utils.logging import logger
from giantsmind.vector_db import search


def get_user_question() -> str:
    """Get the user's question."""
    return input("Please enter your question: ")


def parse_user_question(question: str) -> ParsedElements:
    """Parse the user's question using parsing_agent and handle potential errors."""
    print("Parsing question...  ", end="", flush=True)
    parser = question_parsing.create_default_parser()
    try:
        parsed_elements = parser.parse_question(question)
        if parsed_elements == {}:
            raise ValueError("Parsing did not return any elements.")
        if "error" in parsed_elements:
            raise ValueError(parsed_elements["error"])
    except Exception as e:
        logger.error(f"Error parsing question: {e}")
        print(f"Error parsing question: {e}")
        return ParsedElements()

    logger.info(f"Parsed elements: {parsed_elements}")
    print("done.")
    if "error" in parsed_elements:
        logger.error(parsed_elements["error"])
        print(parsed_elements["error"])
        return ParsedElements()

    return ParsedElements(**parsed_elements)


def display_parsed_elements(parsed_elements: ParsedElements) -> None:
    """Display parsed elements to the user."""
    txt_to_print = """Parsed elements:
Metadata search: {metadata_search}
Content search: {content_search}
General knowledge: {general_knowledge}
"""
    print(txt_to_print.format(**parsed_elements))


def modify_parsed_elements(parsed_elements: ParsedElements) -> ParsedElements:
    """Allow the user to modify each parsed element individually."""
    for key in parsed_elements:
        user_input = input(
            f"Do you want to modify '{key}'? (current value: '{parsed_elements[key]}') (y/[n]): "
        )
        if user_input.lower() == "y":
            new_value = input(f"Enter new value for '{key}': ")
            parsed_elements[key] = new_value
    return ParsedElements(**parsed_elements)


def prompt_question() -> Tuple[str, ParsedElements]:
    """Main loop to handle user input and parsed elements."""

    while True:
        user_question = get_user_question()
        parsed_elements = parse_user_question(user_question)
        if parsed_elements:
            break

    if parsed_elements is None:
        print("An error occurred while parsing the question. Please try again.")
        return user_question, {}

    display_parsed_elements(parsed_elements)
    parsed_elements = handle_user_modifications(parsed_elements)
    return user_question, parsed_elements


def handle_user_modifications(parsed_elements: ParsedElements) -> ParsedElements:
    """Handle user modifications to the parsed elements."""
    while True:
        user_input = input("Do you want to make any changes to the parsed elements (y/[n])")
        if user_input.lower() == "y":
            parsed_elements = modify_parsed_elements(parsed_elements)
            display_parsed_elements(parsed_elements)
        elif user_input.lower() in ["n", ""]:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    return parsed_elements


def get_metadata(metadata_query: str, collection_id: int) -> List[MetadataResult]:
    print("Generate SQL query...  ", end="", flush=True)
    sql_query = sql.get_sql_query(metadata_query, collection_id=collection_id)
    print("done.")
    logger.info(f"SQL query: {sql_query}")

    raw_results = sql.metadata_query(sql_query)
    metadata_results = [MetadataResult(**result) for result in raw_results]
    logger.info(f"Metadata results: {metadata_results}")
    return metadata_results


def content_search(
    content_search: str, metadata_results: List[MetadataResult]
) -> Tuple[List[Document], List[float]]:
    paper_ids = proc_res.extract_paper_ids(metadata_results)
    logger.info(f"Paper IDs: {paper_ids}")
    content_results, content_scores = search.execute_content_search(content_search, paper_ids=paper_ids)
    logger.info(f"Content results: {content_results}")
    logger.info(f"Content scores: {content_scores}")
    return content_results, content_scores


def answer_question(user_question: str, aggregated_context: str) -> str:
    print("Answering question...  ", end="", flush=True)
    final_answer = answering.invoke(user_question, aggregated_context)
    print("done.")
    logger.info(f"Final answer: {final_answer}")
    return final_answer


def print_results(final_answer: str) -> None:
    print(f'\n{"-"*70}')
    print(f"Answer: {final_answer}")


def one_question_chain(collection_id: int) -> None:
    user_question, parsed_elements = prompt_question()

    results: SearchResults = {}
    if parsed_elements.get("metadata_search"):
        results["metadata"] = get_metadata(parsed_elements["metadata_search"], collection_id)

    if parsed_elements.get("content_search"):
        content_results, content_scores = content_search(
            parsed_elements["content_search"], results.get("metadata", [])
        )
        results["content"] = content_results

    if parsed_elements.get("general_knowledge"):
        results["general"] = parsed_elements["general_knowledge"]

    aggregated_context = proc_res.aggregate_results(parsed_elements, results)
    logger.info(f"Aggregated context: {aggregated_context}")

    final_answer = answer_question(user_question, aggregated_context)
    print_results(final_answer)


if __name__ == "__main__":
    one_question_chain(1)
