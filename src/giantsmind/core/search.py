import operator
from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd

import langchain.vectorstores
from langchain_core.documents.base import Document

# TODO: Use thefuzz package to search for partial matches in metadata


def get_matching_publication_dates(metadata_df: pd.DataFrame, publication_dates: List[str]) -> pd.DataFrame:
    """Get articles with matching publication dates."""
    operator_dict = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}

    year_ranges = {}
    non_range_years = []

    for date in publication_dates:
        if ":" in date:
            range_key, range_value = date.split(":")
            if range_key not in operator_dict:
                raise ValueError(f"Invalid operator in {date}. Use one of {list(operator_dict.keys())}.")
            year_ranges[range_key] = int(range_value)
        else:
            non_range_years.append(int(date))

    cond = np.zeros(len(metadata_df), dtype=bool)
    cond_range = np.zeros(
        len(metadata_df), dtype=bool
    )  # Default to False mask for OR operation if no range years

    if non_range_years:
        cond = metadata_df["publication_date"].apply(lambda x: int(x.split("-")[0]) in non_range_years)

    if year_ranges:
        # Switch to True mask for AND operation
        cond_range = np.ones(len(metadata_df), dtype=bool)
        for op, year in year_ranges.items():
            cond_range &= metadata_df["publication_date"].apply(
                lambda x: operator_dict[op](int(x.split("-")[0]), year)
            )

    return metadata_df[cond | cond_range]


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


def get_metadata_from_payload(payload: Dict[str | dict, str]) -> Dict[str, str]:
    """Flatten the metadata payload."""
    metadata = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            metadata.update(get_metadata_from_payload(value))
        else:
            metadata[key] = value
    return metadata


def get_id_from_documents(documents: List[Document]) -> List[str]:
    """Get the IDs from a list of payloads."""
    return [document.metadata["hash"] for document in documents]


def search_articles_with_similarity(
    vectorstore: langchain.vectorstores, query: str, **search_kwargs
) -> List[Document]:
    documents = retrieve_documents(vectorstore, query, **search_kwargs)
    return documents


def retrieve_documents(vectorstore: langchain.vectorstores, query: str, **search_kwargs) -> List[Document]:
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


if __name__ == "__main__":

    from giantsmind import get_metadata
    from giantsmind import utils

    utils.set_env_vars()

    # metadata_df = get_metadata.load_metadata_df()
    # metadata_search = {"year": [">=:2021"], "journal": ["arxiv"], "author": ["kording"]}
    # metadata_search = {"author": ["kording"]}
    # search_for_articles_metadata(metadata_df, metadata_search)

    ############################################

    from langchain.vectorstores import Qdrant
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from giantsmind.vector_db import qdrant as gm_qdrant

    # from giantsmind import get_metadata
    from giantsmind import utils

    utils.set_env_vars()

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

    import tiktoken
    import qdrant

    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    paper_id = "doi:10.1038/nature12160"
    text = gm_qdrant.get_text_from_article(client, collection, paper_id)