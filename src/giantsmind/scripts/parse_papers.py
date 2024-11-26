from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents.base import Document

from giantsmind.core import get_metadata, parse_documents
from giantsmind.metadata_db.operations import paper_operations as paper_ops
from giantsmind.utils import local, pdf_tools, utils
from giantsmind.utils.logging import logger
from giantsmind.vector_db import base, chroma_client, prep_docs

MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}
PARSE_INSTRUCTIONS = """This is a scientific article. Please extract the text from the document and return it in markdown format."""
DEFAULT_COLLECTION = "main_collection"
EMBEDDINGS_MODEL = "bge-small"

load_dotenv()


def add_paper_to_dbs(vc_client: base.VectorDBClient, paper_chunks: List[Document], metadata: Dict[str, str]):
    """Add a paper to the databases."""
    try:
        n_chunks = len(paper_chunks)
        ids = vc_client.add_documents(paper_chunks)
        if len(ids) != n_chunks:
            raise ValueError(f"Expected {n_chunks} IDs, got {len(ids)}")
        metadata["chunks"] = ids
        paper_ops.add_papers([metadata])
    except Exception as e:
        logger.error(f"Failed to add paper '{metadata.get('title', 'Unknown')}' to databases: {str(e)}")
        raise


def setup_pdf_processing(pdf_folder: Path) -> List[Path]:
    """Setup and validate PDF processing environment."""
    if not pdf_folder.is_dir():
        logger.error(f"Invalid directory: {pdf_folder}")
        raise NotADirectoryError(f"{pdf_folder} is not a valid directory.")

    pdf_paths = pdf_tools.get_pdf_paths(pdf_folder)
    if not pdf_paths:
        logger.warning("No PDF files found in the specified directory")
        return []

    logger.info(f"Found {len(pdf_paths)} PDF files to process")
    return pdf_paths


def process_documents(pdf_paths: List[Path]) -> tuple[List[Document], List[Dict]]:
    """Process PDF documents and extract metadata."""
    try:
        logger.info("Processing metadata and parsing documents")
        metadatas = get_metadata.process_metadata(pdf_paths)
        parsed_docs = parse_documents.parse_pdfs(pdf_paths)

        for doc, metadata in zip(parsed_docs, metadatas):
            metadata["authors"] = "; ".join(metadata["authors"])
            doc.metadata = metadata

        return parsed_docs, metadatas
    except Exception as e:
        logger.error(f"Failed to process documents: {str(e)}")
        raise


def process_database_operations(parsed_docs: List[Document], metadatas: List[Dict], persist_directory: Path):
    """Handle database operations for document processing."""
    try:
        ids = [metadata["paper_id"] for metadata in metadatas]

        logger.info("Checking for existing papers in database")
        embeddings = FastEmbedEmbeddings(
            model_name=MODELS[EMBEDDINGS_MODEL]["model"], cache_dir=str(persist_directory)
        )
        client = chroma_client.ChromadbClient(
            DEFAULT_COLLECTION, embeddings, persist_directory=str(persist_directory)
        )
        index_to_process = utils.get_exist_absent(ids, lambda ids: client.check_ids_exist(ids))[-1]

        if not index_to_process:
            logger.info("All papers already exist in database. Nothing to process.")
            return

        logger.info(f"Processing {len(index_to_process)} new papers")
        parsed_docs_to_db, metadatas_to_db = zip(*[(parsed_docs[i], metadatas[i]) for i in index_to_process])

        logger.info("Chunking documents")
        chunked_docs = prep_docs.chunk_documents(parsed_docs_to_db)

        process_papers(client, chunked_docs, metadatas_to_db)
    except Exception as e:
        logger.error(f"Database operation error: {str(e)}")
        raise


def process_papers(
    client: base.VectorDBClient, chunked_docs: List[List[Document]], metadatas_to_db: List[Dict]
):
    """Process individual papers and add them to the database."""
    failed_papers = []
    for i, (paper_chunks, metadata) in enumerate(zip(chunked_docs, metadatas_to_db), 1):
        try:
            logger.info(f"Processing paper {i}/{len(chunked_docs)}: {metadata.get('title', 'Unknown title')}")
            add_paper_to_dbs(client, paper_chunks, metadata)
        except Exception as e:
            logger.error(f"Failed to process paper {metadata.get('title', 'Unknown')}: {str(e)}")
            failed_papers.append(metadata.get("title", "Unknown"))
            continue

    if failed_papers:
        logger.warning(f"Failed to process {len(failed_papers)} papers: {', '.join(failed_papers)}")


def parse_papers(pdf_path: str) -> int:
    """Handle PDF parsing operation."""
    try:
        logger.info("Starting PDF parsing process")
        pdf_paths = setup_pdf_processing(Path(pdf_path))
        if not pdf_paths:
            return 1

        parsed_docs, metadatas = process_documents(pdf_paths)
        process_database_operations(parsed_docs, metadatas, local.get_local_data_path())

        logger.info("PDF parsing and database update completed")
        return 0
    except Exception as e:
        logger.error(f"Critical error in parsing process: {str(e)}")
        return 1


if __name__ == "__main__":
    import os

    parse_papers(os.getenv("DEFAULT_PDF_PATH"))
