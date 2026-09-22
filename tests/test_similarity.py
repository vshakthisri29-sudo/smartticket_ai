import pytest
from src.similarity import check_repeat_complaint

def test_similarity_bounds():
    query = "My payment failed but amount was deducted from my bank."
    res = check_repeat_complaint(query, threshold=0.75)
    assert 0.0 <= res["similarity_score"] <= 1.0
    assert isinstance(res["is_repeat"], bool)
    assert "method" in res

def test_similarity_empty():
    res = check_repeat_complaint("")
    assert res["is_repeat"] is False
    assert res["similarity_score"] == 0.0
