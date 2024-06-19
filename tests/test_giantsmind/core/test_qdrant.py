import pytest

from qdrant_client import models

from giantsmind import qdrant as qdrant_


def test_paper_id_list_to_filter_valid_input():
    paper_id_list = ["type1:id1", "type2:id2", "type3:id3"]
    filter_obj = qdrant_.paper_id_list_to_filter(paper_id_list)

    assert isinstance(filter_obj, models.Filter)
    assert filter_obj.must[0].key == "paper_metadata.ID"


def test_paper_id_list_to_filter_invalid_input_type():
    paper_id_list = [123, "type2:id2"]

    with pytest.raises(ValueError, match="Argument should be a list of strings: 'id_type:id_value'."):
        qdrant_.paper_id_list_to_filter(paper_id_list)


def test_paper_id_list_to_filter_invalid_input_format():
    paper_id_list = ["id1", "type2:id2"]

    with pytest.raises(ValueError, match="Argument should be a list of strings: 'id_type:id_value'."):
        qdrant_.paper_id_list_to_filter(paper_id_list)


def test_paper_id_list_to_filter_empty_list():
    paper_id_list = []

    with pytest.raises(ValueError, match="Argument should be a list of strings: 'id_type:id_value'."):
        qdrant_.paper_id_list_to_filter(paper_id_list)
