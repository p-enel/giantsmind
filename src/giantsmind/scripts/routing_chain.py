from giantsmind.agent import answering_agent, parsing_agent, sql_agent
from giantsmind.core import process_results as proc_res
from giantsmind.core import search
from giantsmind.utils.logging import logger


def main():
    logger.info("Starting main function")
    
    user_question = input("Please enter your question: ")
    logger.info(f"User question: {user_question}")

    parsed_elements = parsing_agent.parse_question(user_question)
    logger.info(f"Parsed elements: {parsed_elements}")

    if "error" in parsed_elements:
        logger.error(parsed_elements["error"])
        print(parsed_elements["error"])
        return

    metadata_results = None
    if parsed_elements.get("metadata_search"):
        sql_query = sql_agent.get_sql_query(parsed_elements["metadata_search"])
        logger.info(f"SQL query: {sql_query}")
        metadata_results = sql_agent.execute_metadata_query(sql_query)
        logger.info(f"Metadata results: {metadata_results}")

    content_results = None
    if parsed_elements.get("content_search"):
        paper_ids = proc_res.extract_paper_ids(metadata_results)
        logger.info(f"Paper IDs: {paper_ids}")
        content_results, content_scores = search.execute_content_search(
            parsed_elements["content_search"], paper_ids
        )
        logger.info(f"Content results: {content_results}")
        logger.info(f"Content scores: {content_scores}")

    aggregated_context = proc_res.aggregate_results(
        user_question,
        parsed_elements,
        metadata_results=metadata_results,
        content_results=content_results,
        general_knowledge=parsed_elements.get("general_knowledge"),
    )
    logger.info(f"Aggregated context: {aggregated_context}")

    final_answer = answering_agent.invoke(user_question, aggregated_context)
    logger.info(f"Final answer: {final_answer}")

    print(f"\nAnswer: {final_answer}")
    logger.info("End of main function")


if __name__ == "__main__":
    main()
    # What is the subject of articles published after 2020?
    # Give me a list of the 10 first papers in the database.
