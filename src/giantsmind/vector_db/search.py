from pathlib import Path
from typing import Dict, List, Optional, Tuple

import langchain.vectorstores
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents.base import Document

from giantsmind.utils.local import get_local_data_path
from giantsmind.vector_db.chroma_client import ChromadbClient

MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}


def get_id_from_documents(documents: List[Document]) -> List[str]:
    """Get the IDs from a list of payloads."""
    return [document.metadata["hash"] for document in documents]


def get_metadata_from_payload(payload: Dict[str | dict, str]) -> Dict[str, str]:
    """Flatten the metadata payload."""
    metadata = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            metadata.update(get_metadata_from_payload(value))
        else:
            metadata[key] = value
    return metadata


def search_articles_with_similarity(
    vectorstore: langchain.vectorstores, query: str, **search_kwargs
) -> List[Document]:
    documents = retrieve_documents(vectorstore, query, **search_kwargs)
    return documents


def retrieve_documents(vectorstore: langchain.vectorstores, query: str, **search_kwargs) -> List[Document]:
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def create_embeddings(model_name: str) -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=MODELS[model_name]["model"])


def create_vectorstore_client(
    collection_name: str, embeddings: FastEmbedEmbeddings, persist_directory: Path
) -> ChromadbClient:
    return ChromadbClient(
        collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory
    )


def perform_similarity_search(
    client: ChromadbClient, query: str, paper_ids: Optional[List[str]] = None
) -> Tuple[List[Document], List[float]]:
    if paper_ids:
        if not client.check_ids_exist(paper_ids):
            raise ValueError("Some paper IDs do not exist in the database.")
        results = client.similarity_search(query, filter={"paper_id": {"$in": paper_ids}})
    else:
        results = client.similarity_search(query)
    return zip(*results)


def execute_content_search(
    content_query: str,
    embeddings_model: str = "bge-small",
    collection_name: str = "main_collection",
    paper_ids: Optional[List[str]] = None,
    persist_directory: Optional[Path] = None,
) -> Tuple[List[Document], List[float]]:
    if persist_directory is None:
        persist_directory = get_local_data_path()

    embeddings = create_embeddings(embeddings_model)
    client = create_vectorstore_client(collection_name, embeddings, persist_directory)
    docs, scores = perform_similarity_search(client, content_query, paper_ids)

    return list(docs), list(scores)


if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    # metadata_df = get_metadata.load_metadata_df()
    # metadata_search = {"year": [">=:2021"], "journal": ["arxiv"], "author": ["kording"]}
    # metadata_search = {"author": ["kording"]}
    # search_for_articles_metadata(metadata_df, metadata_search)

    ############################################

    from langchain.vectorstores import Qdrant
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    # from giantsmind import get_metadata
    from giantsmind import utils
    from giantsmind.vector_db import qdrant as gm_qdrant

    load_dotenv()

    collection = "test"
    embeddings_model = "bge-small"
    pdf_folder = "/home/pierre/Data/giants"
    MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}

    embeddings = FastEmbedEmbeddings(model_name=MODELS[embeddings_model]["model"])
    client = gm_qdrant.setup_database_and_collection(
        collection, MODELS[embeddings_model]["vector_size"], embeddings_model
    )

    qdrant = Qdrant(client, collection, embeddings)

    query = "mixed selectitivity in the prefrontal cortex"
    query = "What are properties of inter-day variations in brain functioning?"
    search_kwargs = {"k": 5}
    result_docs = search_articles_with_similarity(qdrant, query, **search_kwargs)
    len(result_docs)

    import qdrant
    import tiktoken

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    paper_id = "doi:10.1038/nature12160"
    text = gm_qdrant.get_text_from_article(client, collection, paper_id)


# def get_matching_publication_dates(metadata_df: pd.DataFrame, publication_dates: List[str]) -> pd.DataFrame:
#     """Get articles with matching publication dates."""
#     operator_dict = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}

#     year_ranges = {}
#     non_range_years = []

#     for date in publication_dates:
#         if ":" in date:
#             range_key, range_value = date.split(":")
#             if range_key not in operator_dict:
#                 raise ValueError(f"Invalid operator in {date}. Use one of {list(operator_dict.keys())}.")
#             year_ranges[range_key] = int(range_value)
#         else:
#             non_range_years.append(int(date))

#     cond = np.zeros(len(metadata_df), dtype=bool)
#     cond_range = np.zeros(
#         len(metadata_df), dtype=bool
#     )  # Default to False mask for OR operation if no range years

#     if non_range_years:
#         cond = metadata_df["publication_date"].apply(lambda x: int(x.split("-")[0]) in non_range_years)

#     if year_ranges:
#         # Switch to True mask for AND operation
#         cond_range = np.ones(len(metadata_df), dtype=bool)
#         for op, year in year_ranges.items():
#             cond_range &= metadata_df["publication_date"].apply(
#                 lambda x: operator_dict[op](int(x.split("-")[0]), year)
#             )

#     return metadata_df[cond | cond_range]


# def search_for_articles_metadata(metadata_df: pd.DataFrame, search: Dict[str, List[str]]) -> pd.DataFrame:
#     """Search for articles in a metadata dataframe."""
#     allowed_keys = ["author", "journal", "year"]
#     if not all(key in allowed_keys for key in search.keys()):
#         raise ValueError(f"Invalid key: keys should be {allowed_keys}.")
#     results = deepcopy(metadata_df)
#     if "author" in search:
#         results = results[
#             results["author"].apply(lambda x: any(author.lower() in x.lower() for author in search["author"]))
#         ]
#     if "journal" in search:
#         results = results[
#             results["journal"].apply(
#                 lambda x: any(journal.casefold() in x.casefold() for journal in search["journal"])
#             )
#         ]
#     if "year" in search:
#         results = get_matching_publication_dates(results, search["year"])

#     return results
