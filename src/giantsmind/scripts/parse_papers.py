from itertools import chain
from typing import Dict, List

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents.base import Document

from giantsmind.core import get_metadata, parse_documents
from giantsmind.database import paper_operations as paper_ops
from giantsmind.utils import local, pdf_tools, utils
from giantsmind.vector_db import base, prep_docs
from giantsmind.vector_db import chroma_client

PDF_FOLDER = "/home/pierre/Data/giants"
MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}
PARSE_INSTRUCTIONS = """This is a scientific article. Please extract the text from the document and return it in markdown format."""


def add_paper_to_dbs(vc_client: base.VectorDBClient, paper_chunks: List[Document], metadata: Dict[str, str]):
    """Add a paper to the databases.

    Adds chunks of a paper to the vector database and metadata to the
    metadata database.
    """
    n_chunks = len(paper_chunks)
    ids = vc_client.add_documents(paper_chunks)
    if len(ids) != n_chunks:
        raise ValueError(f"Expected {n_chunks} IDs, got {len(ids)}")
    metadata["chunks"] = ids
    paper_ops.add_papers([metadata])


if __name__ == "__main__":

    collection = "main_collection"
    embeddings_model = "bge-small"
    persist_directory = local.get_local_data_path()

    # # With Qdrant
    # embeddings = FastEmbedEmbeddings(model_name=MODELS[embeddings_model]["model"])
    # client = gm_qdrant.setup_database_and_collection(
    #     collection, MODELS[embeddings_model]["vector_size"], embeddings_model
    # )

    pdf_paths = pdf_tools.get_pdf_paths(PDF_FOLDER)

    metadatas = get_metadata.process_metadata(pdf_paths)
    parsed_docs = parse_documents.parse_pdfs(pdf_paths)

    for doc, metadata in zip(parsed_docs, metadatas):
        metadata["authors"] = "; ".join(metadata["authors"])
        doc.metadata = metadata

    IDs = [metadata["paper_id"] for metadata in metadatas]

    # Qdrant
    # index_to_process = utils.get_exist_absent(
    #     IDs, lambda IDs: gm_qdrant.check_ids_exist(client, collection, IDs)
    # )[-1]

    # Check if the papers already exist in the chroma database
    embeddings = FastEmbedEmbeddings(model_name=MODELS[embeddings_model]["model"])
    client = chroma_client.ChromadbClient(collection, embeddings, persist_directory=persist_directory)
    index_to_process = utils.get_exist_absent(IDs, lambda IDs: client.check_ids_exist(IDs))[-1]

    parsed_docs_to_db, metadatas_to_db = zip(*[(parsed_docs[i], metadatas[i]) for i in index_to_process])

    chunked_docs = prep_docs.chunk_documents(parsed_docs_to_db)

    # Qdrant
    # payloads = gm_qdrant.generate_payloads(metadatas, pdf_paths)

    # chunked_docs_with_payload = gm_qdrant.update_chunked_documents_with_payloads(chunked_docs, payloads)

    # qdrant_lc = Qdrant(client, collection, embeddings)
    # qdrant_lc.add_documents(chunked_docs_with_payload)

    for paper_chunks, metadata in zip(chunked_docs, metadatas_to_db):
        add_paper_to_dbs(client, paper_chunks, metadata)
