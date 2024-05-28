import pandas as pd
import pytest
from qdrant_client import models

get_year_from_metadata_test_data = [
    (
        ["lt:2021", "gt:2019"],
        [
            models.FieldCondition(
                key="metadata.year",
                range=models.Range(
                    gt=2019,
                    lt=2021,
                ),
            )
        ],
    ),
    (
        ["2021"],
        [
            models.FieldCondition(
                key="metadata.year",
                match=models.MatchValue(value=2021),
            )
        ],
    ),
    (
        ["2021", "2022"],
        [
            models.FieldCondition(
                key="metadata.year",
                match=models.MatchAny(any=[2021, 2022]),
            )
        ],
    ),
    (
        ["sdf:2021"],
        pytest.raises(ValueError),
    ),
    (
        ["lt:2021:2022"],
        pytest.raises(ValueError),
    ),
]


metadata_to_filter_test_data = [
    (
        {"author": ["John Doe"]},
        models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.author",
                    match=models.MatchValue(value="John Doe"),
                )
            ]
        ),
    ),
    (
        {"author": ["John Doe"], "year": ["2021"]},
        models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.year",
                    match=models.MatchValue(value=2021),
                ),
                models.FieldCondition(
                    key="metadata.author",
                    match=models.MatchValue(value="John Doe"),
                ),
            ]
        ),
    ),
    (
        {"author": ["John Doe"], "year": ["2021"], "journal": ["Nature"]},
        models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.year",
                    match=models.MatchValue(value=2021),
                ),
                models.FieldCondition(
                    key="metadata.author",
                    match=models.MatchValue(value="John Doe"),
                ),
                models.FieldCondition(
                    key="metadata.journal",
                    match=models.MatchValue(value="Nature"),
                ),
            ]
        ),
    ),
    (
        {
            "author": ["John Doe", "Isaac Newton"],
            "year": ["2021", "1789"],
            "journal": ["Nature", "Science"],
        },
        models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.year",
                    match=models.MatchAny(any=[2021, 1789]),
                ),
                models.FieldCondition(
                    key="metadata.author",
                    match=models.MatchAny(any=["John Doe", "Isaac Newton"]),
                ),
                models.FieldCondition(
                    key="metadata.journal",
                    match=models.MatchAny(any=["Nature", "Science"]),
                ),
            ]
        ),
    ),
    (
        {"author": ["lt:John Doe"]},
        pytest.raises(ValueError),
    ),
    (
        {"year": [2134]},
        pytest.raises(ValueError),
    ),
    (
        {"asdflkj": ["anything"]},
        pytest.raises(ValueError),
    ),
    (
        {234325: ["anything"]},
        pytest.raises(ValueError),
    ),
    (
        {},
        pytest.raises(ValueError),
    ),
]

search_for_articles_metadata_df = pd.DataFrame(
    [
        {
            "title": "Letter from anonymous authors.",
            "author": "John Doe; Jane Dane",
            "journal": "Nature",
            "publication_date": "2021-01-01",
        },
        {
            "title": "Cross-temporal collaboration.",
            "author": "Isaac Newton; Geoffrey Hinton",
            "journal": "Science",
            "publication_date": "3502-12-31",
        },
        {
            "title": "The future of AI.",
            "author": "Geoffrey Hinton",
            "journal": "Computer Science",
            "publication_date": "2021-01-01",
        },
        {
            "title": "The past of AI.",
            "author": "Isaac Newton",
            "journal": "Publica",
            "publication_date": "1789-05-10",
        },
        {
            "title": "Another anonymous letter.",
            "author": "John Doe; Jane Dane",
            "journal": "Science",
            "publication_date": "2021-01-01",
        },
    ]
)

test_search_for_articles_metadata_test_data = [
    (
        {"author": ["John Doe"]},
        search_for_articles_metadata_df.iloc[[0, 4]].to_dict(orient="records"),
    ),
    (
        {"author": ["doe"]},
        search_for_articles_metadata_df.iloc[[0, 4]].to_dict(orient="records"),
    ),
    (
        {"author": ["John Doe", "Isaac Newton"]},
        search_for_articles_metadata_df.iloc[[0, 1, 3, 4]].to_dict(orient="records"),
    ),
    (
        {"journal": ["Nature"]},
        search_for_articles_metadata_df.iloc[[0]].to_dict(orient="records"),
    ),
    (
        {"journal": ["nature"]},
        search_for_articles_metadata_df.iloc[[0]].to_dict(orient="records"),
    ),
    (
        {"journal": ["NATURE"]},
        search_for_articles_metadata_df.iloc[[0]].to_dict(orient="records"),
    ),
    (
        {"year": ["2021"]},
        search_for_articles_metadata_df.iloc[[0, 2, 4]].to_dict(orient="records"),
    ),
    (
        {"year": ["2021", "1789"]},
        search_for_articles_metadata_df.iloc[[0, 2, 3, 4]].to_dict(orient="records"),
    ),
    (
        {"author": ["Isaac Newton"], "year": ["1789"]},
        search_for_articles_metadata_df.iloc[[3]].to_dict(orient="records"),
    ),
    (
        {"author": ["John Doe"], "year": ["2021"], "journal": ["Science"]},
        search_for_articles_metadata_df.iloc[[4]].to_dict(orient="records"),
    ),
    (
        {"author": ["John Doe"], "year": ["2021"], "journal": ["Nature", "Science"]},
        search_for_articles_metadata_df.iloc[[0, 4]].to_dict(orient="records"),
    ),
    (
        {"author": ["John Doe"], "year": ["2021", "1789"], "journal": ["Nature", "Science"]},
        search_for_articles_metadata_df.iloc[[0, 4]].to_dict(orient="records"),
    ),
    ({"year": ["<:2021"]}, search_for_articles_metadata_df.iloc[[3]].to_dict(orient="records")),
    ({"year": [">:2021"]}, search_for_articles_metadata_df.iloc[[1]].to_dict(orient="records")),
    (
        {"year": [">=:2021"]},
        search_for_articles_metadata_df.iloc[[0, 1, 2, 4]].to_dict(orient="records"),
    ),
    (
        {"year": ["<:2021", ">:1700"]},
        search_for_articles_metadata_df.iloc[[3]].to_dict(orient="records"),
    ),
]
