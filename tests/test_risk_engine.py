import pytest
from src.risk_engine import calculate_risk

def test_risk_score_bounds():
    dummy_pred = {
        "category": "Payment",
        "priority": "Critical",
        "priority_confidence": 0.95,
        "emotion": "Angry",
        "similarity_score": 0.90,
        "is_repeat": True
    }
    res = calculate_risk(dummy_pred, raw_text="Urgent emergency! Charged twice!")
    assert 0 <= res["risk_score"] <= 100
    assert res["risk_level"] in ["Low", "Medium", "High", "Critical"]
    assert isinstance(res["contributing_factors"], list)
    assert len(res["contributing_factors"]) > 0
    assert res["escalation_required"] is True

def test_risk_score_low_case():
    dummy_pred = {
        "category": "Product",
        "priority": "Low",
        "priority_confidence": 0.60,
        "emotion": "Satisfied",
        "similarity_score": 0.10,
        "is_repeat": False
    }
    res = calculate_risk(dummy_pred, raw_text="Thank you for the quick help.")
    assert 0 <= res["risk_score"] <= 50
    assert res["risk_level"] in ["Low", "Medium"]
    assert res["escalation_required"] is False
