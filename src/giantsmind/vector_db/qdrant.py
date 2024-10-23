import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from langchain_core.documents.base import Document
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from giantsmind.utils import local, utils


def get_year_from_metadata(year_metadata: List[str]) -> List[models.FieldCondition]:
    must = []

    # Process year ranges and validate
    year_ranges = {}
    non_range_years = []

    for year in year_metadata:
        if ":" in year:
            parts = year.split(":")
            if len(parts) != 2:
                raise ValueError(
                    "Invalid value: year range should be in the format 'range:year'. E.g. 'lt:2021'"
                )
            range_key, range_value = parts
            if range_key not in ["lt", "lte", "gt", "gte"]:
                raise ValueError(
                    "Invalid value: year range should start with 'lt', 'lte', 'gt', or 'gte'. E.g. 'lt:2021'"
                )
            year_ranges[range_key] = int(range_value)
        else:
            non_range_years.append(int(year))

    # Add the non-range years to the must list
    if non_range_years:
        if len(non_range_years) == 1:
            must.append(
                models.FieldCondition(
                    key="paper_metadata.year", match=models.MatchValue(value=non_range_years[0])
                )
            )
        else:
            must.append(
                models.FieldCondition(key="paper_metadata.year", match=models.MatchAny(any=non_range_years))
            )

    # Add the range years to the must list
    if year_ranges:
        must.append(models.FieldCondition(key="paper_metadata.year", range=models.Range(**year_ranges)))

    return must


def metadata_dict_to_filter(metadata_search: Dict[str, List[str]]) -> models.Filter:
    """Convert metadata search to a Qdrant filter."""
    if not metadata_search:
        raise ValueError("Invalid input: metadata_search cannot be empty.")

    for key, values in metadata_search.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid key: {key}. All keys should be strings.")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Invalid value: {values}. All values should be lists of strings.")
        if any(":" in value for value in values) and key != "year":
            raise ValueError(
                f"Invalid value in {key}: values should not contain ':' unless it's a year range."
            )

    if not all(key in ["author", "title", "journal", "year"] for key in metadata_search.keys()):
        raise ValueError("Invalid key: keys should be 'author', 'title', 'journal', or 'year'.")

    metadata_search = deepcopy(metadata_search)

    must = []
    if "year" in metadata_search:
        must.extend(get_year_from_metadata(metadata_search["year"]))
        del metadata_search["year"]

    must.extend(
        models.FieldCondition(
            key=f"paper_metadata.{key}",
            match=models.MatchAny(any=values) if len(values) > 1 else models.MatchValue(value=values[0]),
        )
        for key, values in metadata_search.items()
        if values
    )

    if not must:
        raise ValueError("Invalid value: at least one metadata field should be provided.")

    return models.Filter(must=must)


def paper_id_list_to_filter(paper_id_list: List[str]):
    if (
        not paper_id_list
        or not all(isinstance(paper_id, str) for paper_id in paper_id_list)
        or not all(":" in paper_id for paper_id in paper_id_list)
    ):
        raise ValueError("Argument should be a list of strings: 'id_type:id_value'.")

    must = [models.FieldCondition(key="paper_metadata.id", match=models.MatchAny(any=paper_id_list))]

    return models.Filter(must=must)


def perform_similarity_search(qdrant, query, **query_args):
    similar_docs = qdrant.similarity_search_with_score(query, **query_args)
    for doc, score in similar_docs:
        print(f"text: {doc.page_content[:256]}\n")
        print(f"score: {score}")
        print("-" * 80)
        print()


def create_client() -> QdrantClient:
    """Create a Qdrant client from Qdrant cloud."""
    utils.set_env_vars()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    return client


def create_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create a collection in Qdrant."""
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created.")


def setup_database_and_collection(
    collection_name: str, vector_size: int, embeddings_model: str
) -> QdrantClient:
    client = create_client()
    create_collection(client, collection_name, vector_size)
    return client


def metadata_to_payload(metadata: dict, file_path: str | Path) -> dict:
    payload = {
        "file_path": str(file_path),
        "id": metadata["id"],
        "paper_metadata": {
            "title": metadata["title"],
            "author": metadata["author"],
            "journal": metadata["journal"],
            "publication_date": metadata["publication_date"],
            "url": metadata["url"],
        },
    }
    return payload


def generate_payloads(metadatas: List[dict], files: Sequence[str]) -> List[dict]:
    return [metadata_to_payload(metadata, file) for metadata, file in zip(metadatas, files)]


def save_payloads_to_json(
    payloads: Sequence[dict],
    pdf_files: Sequence[str | Path],
) -> List[str]:
    """Save metadata to a JSON file."""
    folder_path = Path(local.get_local_data_path()) / "parsed_docs"
    folder_path.mkdir(exist_ok=True)
    output_paths = []
    for payload, file_name in zip(payloads, pdf_files):
        output_path = folder_path / Path(file_name).with_suffix(".json").name
        with output_path.open("w") as f:
            json.dump(payload, f, indent=4)
        output_paths.append(str(output_path))
    return output_paths


def process_and_save_payloads(metadatas: List[dict], files: Sequence[str], verbose: bool = True) -> List[str]:
    payloads = generate_payloads(metadatas, files)
    payload_files = save_payloads_to_json(payloads, files)
    return payload_files


def load_payloads(payload_files: Sequence[str | Path]) -> List[dict]:
    payloads = []
    for json_file in payload_files:
        with open(json_file, "r") as f:
            payload = json.load(f)
            payloads.append(payload)
    return payloads


def update_chunked_documents_with_payloads(
    chunked_docs: List[List[Document]], payloads: List[dict]
) -> List[List[Document]]:
    """Update chunked documents with payloads.

    Warning: This function modifies the input chunked_docs in place and returns it.
    """
    for payload, chunked_doc in zip(payloads, chunked_docs):
        for i_chunk, chunk in enumerate(chunked_doc):
            payload_copy = deepcopy(payload)
            payload_copy["chunk_index"] = i_chunk
            chunk.metadata.update(payload_copy)
    return chunked_docs


def load_payloads_and_update_chunked_documents(
    chunked_docs: List[List[Document]], payload_files: List[str]
) -> List[List[Document]]:
    payloads = load_payloads(payload_files)
    chunked_docs_new = update_chunked_documents_with_payloads(chunked_docs, payloads)
    return chunked_docs_new


def _order_documents(documents: List[Document]) -> List[Document]:
    """Order documents by chunk index."""
    return sorted(documents, key=lambda doc: doc.metadata["chunk_index"])


def record_to_document(record: models.Record) -> Document:
    return Document(
        page_content=record.payload["page_content"],
        metadata=record.payload["metadata"],
    )


def _get_article_chunks(client: QdrantClient, collection: str, paper_id: str) -> List[models.Record]:
    records, position = client.scroll(
        collection,
        models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.paper_metadata.ID", match=models.MatchValue(value=paper_id)
                )
            ]
        ),
        limit=200,
    )
    if len(records) == 200:
        warnings.warn("More than 200 chunks found. Consider increasing the limit.")
    return records, position


def get_article_chunks(client: QdrantClient, collection: str, paper_id: str) -> List[Document]:
    records, position = _get_article_chunks(client, collection, paper_id)
    docs = [record_to_document(record) for record in records]
    docs = _order_documents(docs)

    return docs


def get_text_from_article(client: QdrantClient, collection: str, paper_id: str) -> str:
    docs = get_article_chunks(client, collection, paper_id)
    text = "\n".join([doc.page_content for doc in docs])
    return text


def check_ids_exist_batch(client: QdrantClient, collection: str, IDs: List[str]) -> bool:
    records = client.scroll(
        collection,
        models.Filter(
            should=[models.FieldCondition(key="metadata.paper_metadata.ID", match=models.MatchAny(any=IDs))]
        ),
    )
    return len(records[0]) > 0


def check_id_exists(client: QdrantClient, collection: str, ID: str) -> bool:
    records = client.scroll(
        collection,
        models.Filter(
            should=[
                models.FieldCondition(key="metadata.paper_metadata.ID", match=models.MatchValue(value=ID))
            ]
        ),
    )

    return len(records[0]) > 0


def check_ids_exist(client: QdrantClient, collection: str, IDs: List[str]) -> Tuple[bool, List[str]]:
    return [check_id_exists(client, collection, ID) for ID in IDs]


# DEPRECATED
# def search_for_hashes(client: QdrantClient, collection_name: str, hashes: list) -> list[str]:
#     """Search for hashes in a Qdrant collection."""
#     records = client.scroll(
#         collection_name=collection_name,
#         scroll_filter=models.Filter(
#             should=[
#                 models.FieldCondition(
#                     key="hash",
#                     match=models.MatchAny(any=hashes),
#                 ),
#             ]
#         ),
#     )
#     hashes_found = list(set([record.payload["hash"] for record in records[0]]))
#     return hashes_found


# def get_unprocessed_pdf_files(
#     client: QdrantClient, collection_name: str, pdf_paths: List[str]
# ) -> Tuple[Tuple[str], Tuple[str]]:
#     hashes = dict(zip(pdf_tools.get_pdf_hashes(pdf_paths), pdf_paths))
#     hashes_redundant = search_for_hashes(client, collection_name, list(hashes.keys()))
#     unproc_hashes, unproc_files = [], []
#     for hash_ in hashes:
#         if hash_ not in hashes_redundant:
#             unproc_hashes.append(hash_)
#             unproc_files.append(hashes[hash_])
#     return unproc_hashes, unproc_files
