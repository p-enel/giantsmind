import pandas as pd
import pytest
from qdrant_client import models
from test_search_data import (
    get_year_from_metadata_test_data,
    metadata_to_filter_test_data,
    search_for_articles_metadata_df,
    test_search_for_articles_metadata_test_data,
)

from giantsmind import search


def test_get_articles_from_metadata_dict():
    pass


# @pytest.mark.parametrize("year_metadata, expected", get_year_from_metadata_test_data)
# def test_get_year_from_metadata(year_metadata, expected):
#     if not isinstance(expected, list):
#         with expected:
#             search.get_year_from_metadata(year_metadata)
#         return

#     actual = search.get_year_from_metadata(year_metadata)
#     assert actual == expected


# @pytest.mark.parametrize("metadata, expected", metadata_to_filter_test_data)
# def test_metadata_dict_to_filter(metadata, expected):
#     if not isinstance(expected, models.Filter):
#         with expected:
#             search.metadata_dict_to_filter(metadata)
#         return

#     actual = search.metadata_dict_to_filter(metadata)
#     assert actual == expected


@pytest.mark.parametrize("search_dict, expected", test_search_for_articles_metadata_test_data)
def test_search_for_articles_metadata(search_dict, expected):
    if not isinstance(expected, list):
        with expected:
            search.search_for_articles_metadata(search_for_articles_metadata_df, search_dict)
        return

    actual = search.search_for_articles_metadata(search_for_articles_metadata_df, search_dict)
    assert actual.to_dict(orient="records") == expected


def test_records_to_hashes():
    pass
