import asyncio
import os
import traceback
from pathlib import Path
from typing import List, Sequence

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents.base import Document as LangchainDocument
from llama_parse import LlamaParse
from llama_parse.base import Document as LlamaDocument

from giantsmind.utils import local, utils

MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}
PARSE_INSTRUCTIONS = """Extract the text from this scientific article and return it in markdown format without delimiters. Do not add any text to the document."""


def load_markdown(document_path: str) -> List[LangchainDocument]:
    loader = UnstructuredMarkdownLoader(document_path)
    return loader.load()


def parse_document(file_path: str | Path, instruction: str) -> LlamaDocument:
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=20000,
    )
    return parser.load_data(file_path)


def parse_files(files: Sequence[str], instruction: str) -> list[LlamaDocument]:
    if len(files) == 0:
        print("No files to parse.")
        return []

    output_folder = Path(local.get_local_data_path()) / "parsed_docs"
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


def _initialize_parser(instruction: str) -> LlamaDocument:
    return LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        result_type="markdown",
        parsing_instruction=instruction,
        check_interval=1,
        max_timeout=20000,
    )


async def _attempt_parse(parser: LlamaParse, file_path: str) -> LlamaDocument:
    docs = await parser.aload_data(file_path)
    return docs


async def aparse_document(pdf_path: str, instruction: str, retries: int = 2) -> LlamaDocument:
    """Asynchronously parse a single document."""
    parser = _initialize_parser(instruction)
    for attempt in range(1, retries + 2):
        try:
            return await _attempt_parse(parser, pdf_path)
        except Exception as error:
            print(f"Error during attempt {attempt}/{retries + 1}:\n{type(error)}:{error}")
            if attempt == retries + 1:
                print(f"Failed to parse {pdf_path} after {retries + 1} attempts.")
                raise
            await asyncio.sleep(2)


def _check_exist_load_parsed_doc(pdf_path: str, verbose: bool = False) -> LangchainDocument | None:
    """Check if the document has already been parsed"""
    fname = Path(pdf_path).name
    doc_fname = Path(fname).with_suffix(".md")
    doc_path = Path(local.get_local_data_path()) / "parsed_docs" / doc_fname
    if doc_path.exists():
        if verbose:
            print(f"Document '{fname}' has already been parsed.")
        return load_markdown(doc_path)

    return None


async def aparse_files(file_paths: List[str], instruction: str) -> List[LlamaDocument | None]:
    if len(file_paths) == 0:
        print("No files to parse.")
        return []

    parsing_crs = [aparse_document(file_path, instruction) for file_path in file_paths]
    parsing_results = await asyncio.gather(*parsing_crs, return_exceptions=True)

    processed_results: List[LlamaDocument | None] = []
    for i, result in enumerate(parsing_results):
        if isinstance(result, Exception):
            print(f"Error parsing file {file_paths[i]}:\n{type(result)} {result}")
            traceback.print_exception(type(result), result, result.__traceback__)
            processed_results.append(None)
            continue
        processed_results.append(result)

    return processed_results


def create_output_folder() -> str:
    output_folder = Path(local.get_local_data_path()) / "parsed_docs"
    output_folder.mkdir(exist_ok=True)
    return str(output_folder)


def write_single_parsed_file(parsing_result: LlamaDocument, output_folder: str, file_path: str) -> str:
    output_path = Path(output_folder) / Path(file_path).with_suffix(".md").name
    for page_doc in parsing_result:
        with output_path.open("a") as f:
            f.write(f"{page_doc.text}\n")
    return str(output_path)


def write_parsed_docs(file_paths: List[str], parsing_results: List[LlamaDocument | None]) -> List[str]:
    output_folder = create_output_folder()
    parsed_file_paths: List[str | None] = []

    for file_path, parsing_result in zip(file_paths, parsing_results):
        if parsing_result is None:
            parsed_file_paths.append(None)
            continue

        parsed_file_paths.append(write_single_parsed_file(parsing_result, output_folder, file_path))

    return parsed_file_paths


def _pdfs_path_to_md_path(pdf_paths: List[str]) -> List[str]:
    return [
        Path(local.get_local_data_path()) / "parsed_docs" / Path(pdf_path).with_suffix(".md").name
        for pdf_path in pdf_paths
    ]


def load_parsed_documents(parsed_files: List[str]) -> List[LangchainDocument]:
    return [load_markdown(doc)[0] for doc in parsed_files]


def load_parsed_documents_with_pdf_path(pdf_paths: List[str]) -> List[LangchainDocument]:
    parsed_files = _pdfs_path_to_md_path(pdf_paths)
    return load_parsed_documents(parsed_files)


def _check_markdown_exist(pdf_path: str) -> bool:
    fname = Path(pdf_path).name
    doc_fname = Path(fname).with_suffix(".md")
    doc_path = Path(local.get_local_data_path()) / "parsed_docs" / doc_fname
    return doc_path.exists()


def check_markdowns_exist(pdf_paths: List[str]) -> List[bool]:
    return [_check_markdown_exist(pdf_path) for pdf_path in pdf_paths]


def parse_pdfs(
    pdf_paths: Sequence[str],
    chunk_size: int = 4096,
    chunk_overlap: int = 256,
) -> List[List[LangchainDocument]]:
    pdf_paths_exist, index_exist, pdf_paths_to_process, index_to_process = utils.get_exist_absent(
        pdf_paths, check_markdowns_exist
    )
    parsed_docs = asyncio.run(aparse_files(pdf_paths_to_process, PARSE_INSTRUCTIONS))
    # parse_files(pdf_paths_to_process, PARSE_INSTRUCTIONS)
    write_parsed_docs(pdf_paths_to_process, parsed_docs)
    langchain_docs = load_parsed_documents_with_pdf_path(pdf_paths)
    # langchain_docs = utils.reorder_merge_lists(parsed_docs, parsed_docs_existing, index_to_process, index_exist)
    return langchain_docs
