import os
import pytest
from config import CATEGORY_MODEL_PATH, PRIORITY_MODEL_PATH
from src.predict import predict_ticket

def test_prediction_output_structure():
    if not os.path.exists(CATEGORY_MODEL_PATH) or not os.path.exists(PRIORITY_MODEL_PATH):
        pytest.skip("Models not trained yet on local environment")
        
    ticket = "I was charged twice and my order was cancelled immediately. Need money back ASAP."
    res = predict_ticket(ticket)
    
    # Required keys
    assert "category" in res
    assert "category_confidence" in res
    assert "priority" in res
    assert "priority_confidence" in res
    assert "emotion" in res
    assert "emotion_confidence" in res
    assert "risk_score" in res
    assert "risk_level" in res
    assert "recommended_action" in res
    assert "top_features" in res
    assert "is_repeat" in res
    
    # Confidence range
    assert 0.0 <= res["category_confidence"] <= 1.0
    assert 0.0 <= res["priority_confidence"] <= 1.0
    assert 0.0 <= res["emotion_confidence"] <= 1.0
    assert 0 <= res["risk_score"] <= 100
