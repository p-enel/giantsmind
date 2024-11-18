from typing import Dict, List, Tuple

from langchain_core.documents.base import Document

from giantsmind.agent import answering_agent, parsing_agent, sql_agent
from giantsmind.core import process_results as proc_res
from giantsmind.core import search
from giantsmind.utils.logging import logger

# def main():
logger.info("Starting main function")


def get_user_question() -> str:
    """Get the user's question."""
    return input("Please enter your question: ")


def parse_user_question(question: str) -> Dict[str, str]:
    """Parse the user's question using parsing_agent and handle potential errors."""
    print("Parsing question...  ", end="", flush=True)
    parsed_elements = parsing_agent.parse_question(question)
    logger.info(f"Parsed elements: {parsed_elements}")
    print("done.")
    if "error" in parsed_elements:
        logger.error(parsed_elements["error"])
        print(parsed_elements["error"])
        return None
    return parsed_elements


def display_parsed_elements(parsed_elements: Dict[str, str]) -> None:
    """Display parsed elements to the user."""
    txt_to_print = """Parsed elements:
Metadata search: {metadata_search}
Content search: {content_search}
General knowledge: {general_knowledge}
"""
    print(txt_to_print.format(**parsed_elements))


def modify_parsed_elements(parsed_elements: Dict[str, str]) -> Dict[str, str]:
    """Allow the user to modify each parsed element individually."""
    for key in parsed_elements:
        user_input = input(
            f"Do you want to modify '{key}'? (current value: '{parsed_elements[key]}') (y/[n]): "
        )
        if user_input.lower() == "y":
            new_value = input(f"Enter new value for '{key}': ")
            parsed_elements[key] = new_value
    return parsed_elements


def prompt_question() -> Tuple[str, Dict[str, str]]:
    """Main loop to handle user input and parsed elements."""
    user_question = get_user_question()
    parsed_elements = parse_user_question(user_question)

    if parsed_elements is None:
        print("An error occurred while parsing the question. Please try again.")

    display_parsed_elements(parsed_elements)

    while True:
        user_input = input("Do you want to make any changes to the parsed elements (y/[n])")
        if user_input.lower() == "y":
            parsed_elements = modify_parsed_elements(parsed_elements)
            display_parsed_elements(parsed_elements)
        elif user_input.lower() in ["n", ""]:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    return user_question, parsed_elements


def get_metadata(metadata_query: str, collection_id: int) -> List[Dict[str, str]]:
    print("Generate SQL query...  ", end="", flush=True)
    sql_query = sql_agent.get_sql_query(metadata_query, collection_id=collection_id)
    print("done.")
    logger.info(f"SQL query: {sql_query}")

    metadata_results = sql_agent.metadata_query(sql_query)
    logger.info(f"Metadata results: {metadata_results}")
    return metadata_results


def content_search(
    content_search: str, metadata_results: List[Dict[str, str]]
) -> Tuple[List[Document], List[float]]:
    paper_ids = proc_res.extract_paper_ids(metadata_results)
    logger.info(f"Paper IDs: {paper_ids}")
    content_results, content_scores = search.execute_content_search(content_search, paper_ids)
    logger.info(f"Content results: {content_results}")
    logger.info(f"Content scores: {content_scores}")
    return content_results, content_scores


def answer_question(user_question: str, aggregated_context: str) -> str:
    print("Answering question...  ", end="", flush=True)
    final_answer = answering_agent.invoke(user_question, aggregated_context)
    print("done.")
    logger.info(f"Final answer: {final_answer}")
    return final_answer


def print_results(final_answer: str) -> None:
    print(f'\n{"".join(["-"]*70)}')
    print(f"\nAnswer: {final_answer}")


def one_question_chain(collection_id: int) -> None:
    user_question, parsed_elements = prompt_question()

    metadata_results = None
    if parsed_elements.get("metadata_search"):
        metadata_results = get_metadata(parsed_elements["metadata_search"], collection_id)

    content_results = None
    if parsed_elements.get("content_search"):
        content_results, content_scores = content_search(parsed_elements["content_search"], metadata_results)

    aggregated_context = proc_res.aggregate_results(
        user_question,
        parsed_elements,
        metadata_results=metadata_results,
        content_results=content_results,
        general_knowledge=parsed_elements.get("general_knowledge"),
    )
    logger.info(f"Aggregated context: {aggregated_context}")

    final_answer = answer_question(user_question, aggregated_context)
    print_results(final_answer)


if __name__ == "__main__":
    one_question_chain(1)
    # What is the subject of articles published after 2020?
    # Give me a list of the 10 first papers in the database.
    # How is a microstate defined in the papers in the database?
    # From all the papers in the database find how is defined microstate (in the context of neuroscience).

    # 1. "To reverse engineer an entire nervous system"
    # 2. "Data-Efficient Multimodal Fusion on a Single GPU"
    # 3. "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"
    # 4. "Personality Moderates Intra-Individual Variability in EEG Microstates and Spontaneous Thoughts"
    # 5. "Spontaneous thought and microstate activity modulation by social imitation"
    # 6. "The importance of mixed selectivity in complex cognitive tasks"
    # 7. "Mastering the game of Go without human knowledge"
    # 8. "Biosynthesis of the nosiheptide indole side ring centers on a cryptic carrier protein NosJ"
    # 9. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
    # 10. "VILA: Improving Structured Content Extraction from Scientific PDFs Using Visual Layout Groups"

    # ['doi:10.1038/nature12160', 'doi:10.1038/nature24270', 'doi:10.1038/s41467-017-00439-1', 'doi:10.1126/science.aar6404']

    # How is mixed selectivity defined in the paper from Rigoti?
