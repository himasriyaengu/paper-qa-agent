import pytest
from pydantic import ValidationError
from structured import Answer

def test_valid_answer():
    a = Answer.model_validate_json('{"answer":"1595","value":1595,"refused":false,"confidence":0.9}')
    assert a.value == 1595

def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Answer.model_validate_json('{"answer":"x","refused":false,"confidence":2}')

def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        Answer.model_validate_json('{"answer":"x","confidence":0.5}')

def test_citation_shorthand():
    from citations import parse_cites
    assert parse_cites("[reflexion p.4, 6]") == {"reflexion p.4", "reflexion p.6"}
