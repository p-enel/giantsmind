import asyncio
import os
from itertools import chain
from pathlib import Path
from typing import List, Sequence
import traceback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents.base import Document
from llama_parse import LlamaParse

from giantsmind.utils import utils

MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}
PARSE_INSTRUCTIONS = """This is a scientific article. Please extract the text from the document and return it in markdown format."""


def load_markdown(document_path: str) -> List[Document]:
    loader = UnstructuredMarkdownLoader(document_path)
    return loader.load()


def get_pdf_paths(folder_path: str) -> list:
    """Get a list of PDF files in a folder."""
    folder = Path(folder_path)
    pdf_files = [str(file) for file in folder.glob("*.pdf")]
    return pdf_files


def parse_document(file_path: str | Path, instruction: str) -> Document:
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=20000,
    )
    return parser.load_data(file_path)


def parse_files(files: Sequence[str], instruction: str) -> list[Document]:
    if len(files) == 0:
        print("No files to parse.")
        return []

    output_folder = Path(utils.get_local_data_path()) / "parsed_docs"
    output_folder.mkdir(exist_ok=True)

    parsed_documents = []
    for file_path in files:
        print(f"Parsing {file_path}...")
        parsed_doc = parse_document(file_path, instruction)
        if len(parsed_doc) > 1:
            raise Exception("Unexpected behavior: multiple documents returned.")
        output_path = output_folder / Path(file_path).with_suffix(".md").name
        parsed_documents.append(str(output_path))
        with output_path.open("w") as f:
            f.write(parsed_doc[0].text)
    return parsed_documents


def _initialize_parser(instruction: str) -> LlamaParse:
    return LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        result_type="markdown",
        parsing_instruction=instruction,
        check_interval=0.5,
        max_timeout=20000,
    )


async def _attempt_parse(parser: LlamaParse, file_path: str) -> Document:
    docs = await parser.aload_data(file_path)
    return docs[0]


async def aparse_document(file_path: str, instruction: str, retries: int = 2) -> Document:
    """Asynchronously parse a single document."""
    parser = _initialize_parser(instruction)
    for attempt in range(1, retries + 2):  # retries + 1 attempts
        try:
            return await _attempt_parse(parser, file_path)
        except Exception as error:
            print(f"Error during attempt {attempt}/{retries + 1}:\n{type(error)}:{error}")
            if attempt == retries + 1:
                print(f"Failed to parse {file_path} after {retries + 1} attempts.")
                raise
            await asyncio.sleep(2)  # Optional: Wait a bit before retrying


async def aparse_files(file_paths: List[str], instruction: str) -> List[Document | None]:
    if len(file_paths) == 0:
        print("No files to parse.")
        return []

    parsing_crs = [aparse_document(file_path, instruction) for file_path in file_paths]
    parsing_results = await asyncio.gather(*parsing_crs, return_exceptions=True)

    processed_results: List[Document | None] = []
    for i, result in enumerate(parsing_results):
        if isinstance(result, Exception):
            print(f"Error parsing file {file_paths[i]}:\n{type(result)} {result}")
            traceback.print_exception(type(result), result, result.__traceback__)
            processed_results.append(None)
            continue
        processed_results.append(result)

    return processed_results


def create_output_folder() -> str:
    output_folder = Path(utils.get_local_data_path()) / "parsed_docs"
    output_folder.mkdir(exist_ok=True)
    return str(output_folder)


def write_single_parsed_doc(parsing_result: Document, output_folder: str, file_path: str) -> str:
    output_path = Path(output_folder) / Path(file_path).with_suffix(".md").name
    with output_path.open("w") as f:
        f.write(parsing_result.text)
    return str(output_path)


def write_parsed_docs(file_paths: List[str], parsing_results: List[Document | None]) -> List[str]:
    output_folder = create_output_folder()
    parsed_file_paths: List[str | None] = []

    for file_path, parsing_result in zip(file_paths, parsing_results):
        if parsing_result is None:
            parsed_file_paths.append(None)
            continue

        parsed_file_paths.append(write_single_parsed_doc(parsing_result, output_folder, file_path))

    return parsed_file_paths


def load_parsed_documents(parsed_files: List[str]) -> List[Document]:
    return [load_markdown(doc)[0] for doc in parsed_files]


def chunk_documents(
    documents: Sequence[Document], chunk_size: int = 4096, chunk_overlap: int = 256
) -> List[List[Document]]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = [text_splitter.split_documents([doc]) for doc in documents]
    return chunked_docs


def chunk_pdfs(
    pdf_files: Sequence[str],
    chunk_size: int = 4096,
    chunk_overlap: int = 256,
) -> List[List[Document]]:
    parsed_docs = asyncio.run(aparse_files(pdf_files, PARSE_INSTRUCTIONS))
    parsed_files = write_parsed_docs(pdf_files, parsed_docs)
    parsed_docs = load_parsed_documents(parsed_files)
    chunked_docs = chunk_documents(parsed_docs)
    return chunked_docs


if __name__ == "__main__":
    utils.set_env_vars()

    from langchain.vectorstores import Qdrant
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from giantsmind.core import qdrant as gm_qdrant
    from giantsmind.core import get_metadata

    collection = "test"
    embeddings_model = "bge-small"
    pdf_folder = "/home/pierre/Data/giants"

    embeddings = FastEmbedEmbeddings(model_name=MODELS[embeddings_model]["model"])
    client = gm_qdrant.setup_database_and_collection(
        collection, MODELS[embeddings_model]["vector_size"], embeddings_model
    )

    unproc_hashes, unproc_files = gm_qdrant.get_unprocessed_pdf_files(
        client, collection, get_pdf_paths(pdf_folder)
    )

    if len(unproc_files) == 0:
        print("All files have already been processed.")
        exit()

    metadatas = get_metadata.process_metadata(unproc_files, unproc_hashes)
    payload_files = gm_qdrant.process_and_save_payloads(metadatas, unproc_files, unproc_hashes)

    chunked_docs = chunk_pdfs(unproc_files)
    chunked_docs_with_payload = gm_qdrant.load_payloads_and_update_chunked_documents(
        chunked_docs, payload_files
    )
    chunked_docs_with_payload = list(chain(*chunked_docs_with_payload))

    qdrant = Qdrant(client, collection, embeddings)
    qdrant.add_documents(chunked_docs_with_payload)
