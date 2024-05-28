import pytest

from giantsmind import get_metadata


fetch_metadata_from_doi_test_data = [
    (
        "10.1162/tacl_a_00466",
        {
            "title": "VILA: Improving Structured Content Extraction from Scientific PDFs Using Visual Layout Groups",
            "author": "Zejiang Shen; Kyle Lo; Lucy Lu Wang; Bailey Kuehl; Daniel S. Weld; Doug Downey",
            "url": "http://dx.doi.org/10.1162/tacl_a_00466",
            "journal": "Transactions of the Association for Computational Linguistics",
            "publication_date": "2022-04-06",
            "ID": "doi:10.1162/tacl_a_00466",
        },
    ),
    (
        "10.1038/nature24270",
        {
            "title": "Mastering the game of Go without human knowledge",
            "author": "David Silver; Julian Schrittwieser; Karen Simonyan; Ioannis Antonoglou; Aja Huang; Arthur Guez; Thomas Hubert; Lucas Baker; Matthew Lai; Adrian Bolton; Yutian Chen; Timothy Lillicrap; Fan Hui; Laurent Sifre; George van den Driessche; Thore Graepel; Demis Hassabis",
            "url": "http://dx.doi.org/10.1038/nature24270",
            "journal": "Nature",
            "publication_date": "2017-10-19",
            "ID": "doi:10.1038/nature24270",
        },
    ),
    ("1", {}),
]


@pytest.mark.parametrize("doi, expected", fetch_metadata_from_doi_test_data)
def test_fetch_metadata_from_doi(doi, expected):
    actual = get_metadata.fetch_metadata_from_doi(doi)
    assert actual == expected


fetch_metadata_from_arxiv_test_data = [
    (
        "2404.05961",
        {
            "title": "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders",
            "author": "Parishad BehnamGhader; Vaibhav Adlakha; Marius Mosbach; Dzmitry Bahdanau; Nicolas Chapados; Siva Reddy",
            "url": "http://arxiv.org/abs/2404.05961v1",
            "journal": "arXiv",
            "publication_date": "2024-04-09",
            "ID": "arXiv:2404.05961",
        },
    ),
    (
        "1712.01815v1",
        {
            "title": "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
            "author": "David Silver; Thomas Hubert; Julian Schrittwieser; Ioannis Antonoglou; Matthew Lai; Arthur Guez; Marc Lanctot; Laurent Sifre; Dharshan Kumaran; Thore Graepel; Timothy Lillicrap; Karen Simonyan; Demis Hassabis",
            "url": "http://arxiv.org/abs/1712.01815v1",
            "journal": "arXiv",
            "publication_date": "2017-12-05",
            "ID": "arXiv:1712.01815v1",
        },
    ),
    (
        "42",
        {},
    ),
]


@pytest.mark.parametrize("arxiv_id, expected", fetch_metadata_from_arxiv_test_data)
def test_fetch_metadata_from_arxiv(arxiv_id, expected):
    actual = get_metadata.fetch_metadata_from_arxiv(arxiv_id)
    assert actual == expected
