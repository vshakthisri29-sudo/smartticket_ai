import os
import joblib
from config import (
    MODEL_PIPELINE_PATH,
    CATEGORY_MODEL_PATH,
    PRIORITY_MODEL_PATH,
    EMOTION_MODEL_PATH,
    CONFIDENCE_WARNING_THRESHOLD,
    CONFIDENCE_CRITICAL_THRESHOLD,
    SIMILARITY_THRESHOLD
)
from src.preprocessing import clean_text
from src.emotion import predict_emotion
from src.similarity import check_repeat_complaint
from src.explain import explain_prediction
from src.risk_engine import calculate_risk
from src.recommendations import recommend_action

def predict_ticket(ticket_text: str, similarity_threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """
    Unified ML prediction pipeline using the self-contained MultiOutputClassifier pipeline.
    Directly infers Category, Priority, and Emotion along with their respective posterior
    probabilities, explainable token attribution, similarity repeat detection, and risk analysis.
    """
    if not ticket_text or not ticket_text.strip():
        raise ValueError("Ticket text is empty or contains no valid text characters.")
        
    cleaned = clean_text(ticket_text)
    
    # 1. Prediction using unified MultiOutputClassifier pipeline
    if os.path.exists(MODEL_PIPELINE_PATH):
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        clf = pipeline.named_steps["classifier"]
        
        # Single predict call for all 3 targets
        multi_preds = pipeline.predict([ticket_text])[0]
        category = str(multi_preds[0])
        priority = str(multi_preds[1])
        emotion = str(multi_preds[2])
        
        # Single predict_proba call returning list of 3 probability distributions
        multi_probs = pipeline.predict_proba([ticket_text])
        
        # Target 0: Category
        cat_classes = list(clf.estimators_[0].classes_)
        cat_probs = multi_probs[0][0]
        cat_best_idx = cat_probs.argmax()
        category_conf = round(float(cat_probs[cat_best_idx]), 4)
        cat_prob_dict = {cls: round(float(p), 4) for cls, p in zip(cat_classes, cat_probs)}
        
        # Target 1: Priority
        prio_classes = list(clf.estimators_[1].classes_)
        prio_probs = multi_probs[1][0]
        prio_best_idx = prio_probs.argmax()
        priority_conf = round(float(prio_probs[prio_best_idx]), 4)
        prio_prob_dict = {cls: round(float(p), 4) for cls, p in zip(prio_classes, prio_probs)}
        
        # Target 2: Emotion
        emo_classes = list(clf.estimators_[2].classes_)
        emo_probs = multi_probs[2][0]
        emo_best_idx = emo_probs.argmax()
        emotion_conf = round(float(emo_probs[emo_best_idx]), 4)
        emo_prob_dict = {cls: round(float(p), 4) for cls, p in zip(emo_classes, emo_probs)}
        emotion_method = "MultiOutputClassifier — Emotion ML Estimator"
        
    elif os.path.exists(CATEGORY_MODEL_PATH) and os.path.exists(PRIORITY_MODEL_PATH):
        # Fallback to separate models if unified is not yet present
        cat_pipe = joblib.load(CATEGORY_MODEL_PATH)
        prio_pipe = joblib.load(PRIORITY_MODEL_PATH)
        
        cat_probs = cat_pipe.predict_proba([ticket_text])[0]
        cat_classes = list(cat_pipe.classes_)
        category = str(cat_pipe.predict([ticket_text])[0])
        category_conf = round(float(cat_probs[cat_probs.argmax()]), 4)
        cat_prob_dict = {cls: round(float(p), 4) for cls, p in zip(cat_classes, cat_probs)}
        
        prio_probs = prio_pipe.predict_proba([ticket_text])[0]
        prio_classes = list(prio_pipe.classes_)
        priority = str(prio_pipe.predict([ticket_text])[0])
        priority_conf = round(float(prio_probs[prio_probs.argmax()]), 4)
        prio_prob_dict = {cls: round(float(p), 4) for cls, p in zip(prio_classes, prio_probs)}
        
        emotion_res = predict_emotion(ticket_text)
        emotion = emotion_res["emotion"]
        emotion_conf = emotion_res["emotion_confidence"]
        emo_prob_dict = emotion_res["emotion_probabilities"]
        emotion_method = emotion_res.get("method")
    else:
        raise FileNotFoundError("ML models are not trained yet. Run model training first.")
        
    # 2. Repeat Complaint Similarity Detection
    sim_res = check_repeat_complaint(ticket_text, threshold=similarity_threshold)
    
    # 3. Explainable AI Feature Attribution (uses unified or category pipeline)
    explain_res = explain_prediction(ticket_text)
    
    output = {
        "ticket_text": ticket_text,
        "cleaned_text": cleaned,
        
        # Category Output
        "category": category,
        "category_confidence": category_conf,
        "category_probabilities": cat_prob_dict,
        
        # Priority Output
        "priority": priority,
        "priority_confidence": priority_conf,
        "priority_probabilities": prio_prob_dict,
        
        # Emotion Output
        "emotion": emotion,
        "emotion_confidence": emotion_conf,
        "emotion_probabilities": emo_prob_dict,
        "emotion_method": emotion_method,
        
        # Repeat Detection
        "is_repeat": sim_res["is_repeat"],
        "similarity_score": sim_res["similarity_score"],
        "matched_ticket_id": sim_res.get("matched_ticket_id"),
        "matched_ticket_text": sim_res.get("matched_ticket_text"),
        "matched_category": sim_res.get("matched_category"),
        
        # Explainability
        "top_features": explain_res.get("top_features", []),
        "important_terms": [f["term"] for f in explain_res.get("top_features", [])],
        "highlighted_html": explain_res.get("highlighted_html", ticket_text),
        "explanation_disclaimer": explain_res.get("disclaimer"),
        
        # Calibration / Confidence Disclaimers
        "confidence_disclaimer": "Confidence represents the model's estimated probability for the selected class and is not a guarantee of correctness.",
        "low_confidence_warning": bool(category_conf < CONFIDENCE_WARNING_THRESHOLD or priority_conf < CONFIDENCE_WARNING_THRESHOLD),
        "very_low_confidence_warning": bool(category_conf < CONFIDENCE_CRITICAL_THRESHOLD or priority_conf < CONFIDENCE_CRITICAL_THRESHOLD)
    }
    
    # 4. Transparent Rule-Based Risk Engine Layer
    risk_output = calculate_risk(output, raw_text=ticket_text)
    output.update(risk_output)
    
    # 5. Synthesize Recommended Support Action
    action_output = recommend_action(output, risk_output)
    output["recommended_action"] = action_output["action_text"]
    output["recommendation_details"] = action_output
    
    return output
