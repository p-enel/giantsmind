import operator
from copy import deepcopy
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from giantsmind.parse_documents import load_metadata_df
from giantsmind.utils import set_env_vars

# TODO: Use thefuzz package to search for partial matches in metadata


def get_articles_from_metadata_dict(client: QdrantClient, metadata_search: Mapping[str, str]) -> List[str]:
    """Get articles hash from metadata search."""
    pass


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


def search_for_articles_metadata(metadata_df: pd.DataFrame, search: Dict[str, List[str]]) -> pd.DataFrame:
    """Search for articles in a metadata dataframe."""
    allowed_keys = ["author", "journal", "year"]
    if not all(key in allowed_keys for key in search.keys()):
        raise ValueError(f"Invalid key: keys should be {allowed_keys}.")
    results = deepcopy(metadata_df)
    if "author" in search:
        results = results[
            results["author"].apply(lambda x: any(author.lower() in x.lower() for author in search["author"]))
        ]
    if "journal" in search:
        results = results[
            results["journal"].apply(
                lambda x: any(journal.casefold() in x.casefold() for journal in search["journal"])
            )
        ]
    if "year" in search:
        results = get_matching_publication_dates(results, search["year"])

    return results


def records_to_hashes(records: List[models.Record]) -> List[str]:
    pass


if __name__ == "__main__":

    set_env_vars()

    metadata_df = load_metadata_df()
    metadata_search = {"year": [">=:2021"], "journal": ["arxiv"], "author": ["kording"]}
    metadata_search = {"author": ["kording"]}
    search_for_articles_metadata(metadata_df, metadata_search)
