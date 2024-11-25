from typing import Callable, Dict, List, Optional, Sequence

from langchain_core.documents.base import Document

from giantsmind.core.models import MetadataResult, ParsedElements, SearchResults

REQUIRED_METADATA_KEYS = {"title", "authors", "journal", "publication_date", "paper_id"}
DEFAULT_SEPARATOR_LENGTH = 80


def extract_paper_ids(metadata_results: Optional[List[Dict[str, str]]]) -> List[str]:
    """Extract paper IDs from metadata results."""
    if not metadata_results:
        return []
    return [result["paper_id"] for result in metadata_results]


def combine_docs(documents: Sequence[Document], separator_length: int = DEFAULT_SEPARATOR_LENGTH) -> str:
    """Combine multiple documents into a single formatted string for display.

    Args:
        documents: Sequence of Document objects containing paper content and metadata.
        separator_length: Length of the separator line between documents (default: 80).

    Returns:
        A formatted string containing all documents with their metadata.

    Raises:
        ValueError: If documents is empty or None, or if required metadata is missing.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty or None")

    output = ["-" * separator_length]

    for doc in documents:
        # Validate metadata
        missing_keys = REQUIRED_METADATA_KEYS - doc.metadata.keys()
        if missing_keys:
            raise ValueError(f"Document missing required metadata: {missing_keys}")

        metadata_str = (
            f"The following text is an excerpt of the article entitled '{doc.metadata['title']}' "
            f"from author(s) {doc.metadata['authors']}. It was published in {doc.metadata['journal']} "
            f"in (year-month-day) {doc.metadata['publication_date']}. "
            f"Paper ID: {doc.metadata['paper_id']}.\n\n<excerpt>"
        )
        output.append(metadata_str)
        output.append(doc.page_content)
        output.append(f"</excerpt>\n{'-' * separator_length}")

    output.append("End of excerpts.")
    return "\n".join(output)


def format_metadata_results(metadata_results: Optional[List[MetadataResult]]) -> str:
    """Format metadata results into a human-readable string.

    Args:
        metadata_results: List of dictionaries containing paper metadata.
            Each dictionary must contain: title, authors, publication_date,
            journal, and paper_id.

    Returns:
        A formatted string containing all metadata results.
        Returns "No metadata results found." if input is None or empty.

    Raises:
        ValueError: If any metadata result is missing required keys.
    """
    if not metadata_results:
        return "No metadata results found."

    formatted_results = []
    for result in metadata_results:
        missing_keys = REQUIRED_METADATA_KEYS - result.keys()
        if missing_keys:
            raise ValueError(f"Metadata result missing required keys: {missing_keys}")

        formatted_results.append(
            "Title: {title}\n"
            "Authors: {authors}\n"
            "Publication Date: {publication_date}\n"
            "Journal: {journal}\n"
            "Paper ID: {paper_id}\n".format(**result)
        )

    return "".join(formatted_results)


def aggregate_results(
    parsed_elements: ParsedElements,
    results: SearchResults,
    format_metadata: Callable[[Optional[List[MetadataResult]]], str] = format_metadata_results,
    combine_documents: Callable[[Sequence[Document], int], str] = combine_docs,
) -> str:
    """Aggregate search results and general knowledge into a formatted context string.

    Args:
        user_question: The original question asked by the user.
        parsed_elements: Dictionary containing search parameters.
        results: Dictionary containing search results:
            - metadata: List of metadata results
            - content: List of document results
            - general: General knowledge text
        format_metadata: Function to format metadata results (injectable).
        combine_documents: Function to combine documents (injectable).

    Returns:
        A formatted string containing all aggregated results.

    Raises:
        ValueError: If parsed_elements is empty or required keys are missing for active searches.
    """
    if not parsed_elements:
        raise ValueError("parsed_elements cannot be empty")

    sections = []

    if results.get("metadata"):
        if "metadata_search" not in parsed_elements:
            raise ValueError("metadata_search required in parsed_elements when metadata results present")
        sections.append(
            f"# This question required a metadata search: \"{parsed_elements['metadata_search']}\"\n"
            f"Here are the results:\n{format_metadata(results['metadata'])}"
        )

    if results.get("content"):
        if "content_search" not in parsed_elements:
            raise ValueError("content_search required in parsed_elements when content results present")
        sections.append(
            f"# This question required content search for: {parsed_elements['content_search']}\n"
            f"Here are the results:\n{combine_documents(results['content'])}\n"
        )

    if results.get("general"):
        if "general_knowledge" not in parsed_elements:
            raise ValueError("general_knowledge required in parsed_elements when general knowledge present")
        sections.append(f"# General knowledge required: {parsed_elements['general_knowledge']}")

    return "\n".join(sections)
