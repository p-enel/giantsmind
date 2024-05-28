import asyncio
import hashlib
import json
import os
import textwrap
from itertools import chain
from pathlib import Path
from typing import List, Sequence, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from get_metadata import get_metadata


MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}


def print_response(response) -> None:
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            print()
            continue
        print("\n".join(textwrap.wrap(chunk, 80, break_long_words=False)))


def perform_similarity_search(qdrant, query):
    similar_docs = qdrant.similarity_search_with_score(query)
    for doc, score in similar_docs:
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {score}")
        print("-" * 80)
        print()


def retrieve_documents(qdrant, query, k=5):
    retriever = qdrant.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def compress_retrieve_documents(retriever, query):
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever.invoke(query)


def create_qa_system(llm_model_name, retriever, temperature=0):
    llm = ChatGroq(temperature=temperature, model_name=llm_model_name)
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Answer the question and provide additional helpful information,
    based on the pieces of information, if applicable. Be succinct.

    Responses should be properly formatted to be easily read.
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
    )


async def main():

    documents = load_markdown(save_path)
    docs = split_documents(documents)
    print("len(docs) =", len(docs))
    print(docs[0].page_content)

    qdrant = initialize_qdrant(docs, "BAAI/bge-base-en-v1.5")

    query = "Is noise used in this work? If so, how is it used?"
    perform_similarity_search(qdrant, query)

    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retrieve_documents(retriever, query)
    for doc in retrieved_docs:
        print(f"id: {doc.metadata['_id']}\n")
        print(f"text: {doc.page_content[:256]}\n")
        print("-" * 80)
        print()

    reranked_docs = compress_retrieve_documents(retriever, query)
    print("len(reranked_docs) =", len(reranked_docs))
    for doc in reranked_docs:
        print(f"id: {doc.metadata['_id']}\n")
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {doc.metadata['relevance_score']}")
        print("-" * 80)
        print()

    qa_system = create_qa_system("llama3-70b-8192", retriever)
    response = qa_system.invoke("Is noise used in this work? If so, how is it used?")
    print_response(response)

    response = qa_system.invoke(
        "Explain to me what is the goal of the trained context neuron."
    )
    print_response(response)

    response = qa_system.invoke("What is mixed selectivity?")
    print_response(response)


###############################################################################


instruction = """This is a scientific article. Please extract the text from the document and return it in markdown format."""


def set_env_vars():
    env_file = Path(".env")

    if not env_file.exists():
        raise Exception(".env file not found.")

    for line in env_file.read_text().split("\n"):
        if not line:
            continue
        key, value = line.split("=")
        os.environ[key] = value


def create_client() -> QdrantClient:
    """Create a Qdrant client from Qdrant cloud."""
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    return client


def create_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    """Create a collection in Qdrant."""
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created.")


def load_markdown(document_path):
    loader = UnstructuredMarkdownLoader(document_path)
    return loader.load()


def get_pdf_files(folder_path: str | Path) -> list:
    """Get a list of PDF files in a folder."""
    folder = Path(folder_path)
    pdf_files = [file for file in folder.glob("*.pdf")]
    return pdf_files


def get_file_hashes(files: list) -> list:
    """Get the hash of each file in a list of files."""
    hashes = []
    for file in files:
        with open(file, "rb") as f:
            data = f.read()
            hash = hashlib.sha256(data).hexdigest()
            hashes.append(hash)
    return hashes


def search_for_hashes(
    client: QdrantClient, collection_name: str, hashes: list
) -> list[str]:
    """Search for hashes in a Qdrant collection."""
    records = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="hash",
                    match=models.MatchAny(any=hashes),
                ),
            ]
        ),
    )
    hashes_found = [record.payload["hash"] for record in records[0]]
    return hashes_found


def save_payloads_to_json(
    payloads: Sequence[dict],
    pdf_files: Sequence[str | Path],
    folder: str | Path = "./Parsed_docs",
) -> None:
    """Save metadata to a JSON file."""
    folder_path = Path(folder)
    if not folder_path.exists():
        folder_path.mkdir()
    for payload, file_name in zip(payloads, pdf_files):
        output_path = folder_path / Path(file_name).with_suffix(".json").name
        with output_path.open("w") as f:
            json.dump(payload, f, indent=4)


async def parse_document(file_path: str | Path, instruction: str):
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=5000,
    )
    return await parser.aload_data(file_path)


def parse_files(
    files: Sequence[str | Path], instruction: str, output_folder: str = "./Parsed_docs"
) -> list[str]:
    """Parse a list of files using Llama."""
    folder_path = Path(output_folder)
    if len(files) != 0 and not folder_path.exists():
        folder_path.mkdir()
    parsed_documents = []
    for file_path in files:
        print(f"Parsing {file_path}...")
        parsed_doc = asyncio.run(parse_document(file_path, instruction))
        if len(parsed_doc) > 1:
            raise Exception("Unexpected behavior: multiple documents returned.")
        output_path = folder_path / Path(file_path).with_suffix(".md").name
        parsed_documents.append(str(output_path))
        with output_path.open("w") as f:
            f.write(parsed_doc[0].text)
    return parsed_documents


def chunk_documents(
    documents: Sequence[str | Path], chunk_size: int = 4096, chunk_overlap: int = 256
) -> List[List[str]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = [text_splitter.split_documents([doc]) for doc in documents]
    chunked_docs = [[chunk.page_content for chunk in doc] for doc in chunked_docs]
    return chunked_docs


def embed_documents(
    embeddings_model: str, documents: List[List[str]]
) -> List[List[List[float]]]:
    embeddings = FastEmbedEmbeddings(model_name=embeddings_model)
    embedded_docs = []
    for doc in documents:
        embedded_docs.append(embeddings.embed_documents(doc))
    return embedded_docs


def load_payloads_from_json(json_files: Sequence[str | Path]) -> List[dict]:
    payloads = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            payload = json.load(f)
            payloads.append(payload)
    return payloads


def prepare_documents(
    documents: List[str] | List[Path], embeddings_model: str = "BAAI/bge-base-en-v1.5"
) -> Tuple[List[float], List[dict]]:
    parsed_docs = [load_markdown(doc)[0] for doc in parsed_files]
    chunked_docs = chunk_documents(parsed_docs)
    embedded_docs = embed_documents(embeddings_model, chunked_docs)
    payload_files = [Path(doc).with_suffix(".json") for doc in parsed_files]
    payloads = load_payloads_from_json(payload_files)
    # Multiply payloads by the number of chunks for each document
    payloads = [
        [payload for _ in range(len(chunked_doc))]
        for payload, chunked_doc in zip(payloads, chunked_docs)
    ]
    return list(chain(*embedded_docs)), list(chain(*payloads))


def setup_database_and_collection(
    collection_name: str, embeddings_model: str
) -> QdrantClient:
    client = create_client()
    vector_size = MODELS[embeddings_model]["vector_size"]
    create_collection(client, collection_name, vector_size)
    return client


def get_unprocessed_pdf_files(
    client: QdrantClient, collection_name: str, pdf_folder: str | Path
) -> Tuple[Tuple[str], Tuple[str | Path]]:
    pdf_files = get_pdf_files(pdf_folder)
    hashes = dict(zip(get_file_hashes(pdf_files), pdf_files))
    hashes_redundant = search_for_hashes(client, collection_name, list(hashes.keys()))
    unproc_hashes, unproc_files = zip(
        *[(hash_, hashes[hash_]) for hash_ in hashes if hash_ not in hashes_redundant]
    )
    return unproc_hashes, unproc_files


def metadata_to_payload(metadata: dict, hash: str, file_path: str | Path) -> dict:
    payload = {
        "hash": hash,
        "file_path": str(file_path),
        "paper_metadata": {
            "title": metadata["title"],
            "author": metadata["author"],
            "journal": metadata["journal"],
            "publication_date": metadata["publication_date"],
            "keywords": metadata["keywords"],
            "doi": metadata["doi"],
            "url": metadata["url"],
        },
    }
    return payload


def process_metadata(
    files: Sequence[str | Path], hashes: Sequence[str], verbose: bool = True
) -> None:
    """Get and save metadata for a list of files."""
    metadatas = [get_metadata(file, verbose=True) for file in files]
    payloads = [
        metadata_to_payload(metadata, hash, file)
        for metadata, hash, file in zip(metadatas, hashes, files)
    ]
    save_payloads_to_json(payloads, files)


# =============================================================================
if __name__ == "__main__":
    set_env_vars()

    collection = "test"
    embeddings_model = "bge-small"
    pdf_folder = "/home/pierre/Data/giants"

    client = setup_database_and_collection(collection, embeddings_model)

    unproc_hashes, unproc_files = get_unprocessed_pdf_files(
        client, collection, pdf_folder
    )
    process_metadata(unproc_files, unproc_hashes)
    parsed_files = parse_files(unproc_files, instruction)
    vectors, payloads = prepare_documents(
        parsed_files, MODELS[embeddings_model]["model"]
    )

    client.upload_collection(
        collection_name=collection,
        vectors=embedded_docs,
        payload=payloads,
        ids=None,
    )
