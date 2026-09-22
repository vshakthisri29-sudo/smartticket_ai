import pytest
from src.preprocessing import clean_text

def test_clean_text_basic():
    raw = "  My Order Has NOT arrived!!! Please check ASAP.  "
    cleaned = clean_text(raw)
    assert "not" in cleaned
    assert "asap" in cleaned
    assert "!" not in cleaned
    assert cleaned == "my order has not arrived please check asap"

def test_clean_text_null_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   ") == ""

def test_clean_text_preserves_financial_and_urgency():
    raw = "The bank DEDUCTED $500 twice for UNAUTHORIZED transaction."
    cleaned = clean_text(raw)
    assert "deducted" in cleaned
    assert "unauthorized" in cleaned
    assert "twice" in cleaned

def test_clean_text_whitespace_normalization():
    raw = "line1\n\nline2\t\tline3   word"
    cleaned = clean_text(raw)
    assert cleaned == "line1 line2 line3 word"
