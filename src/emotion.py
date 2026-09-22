import os
import joblib
from config import MODEL_PIPELINE_PATH, EMOTION_MODEL_PATH
from src.preprocessing import clean_text

# Rule-based transparent baseline dictionary for fallback
BASELINE_EMOTION_KEYWORDS = {
    "Angry": ["furious", "unacceptable", "outrageous", "disrespectful", "demand", "lawsuit", "terrible", "worst"],
    "Frustrated": ["frustrating", "tired", "wasting", "multiple times", "no progress", "headache", "still waiting"],
    "Worried": ["concerned", "worried", "anxious", "uneasy", "stressed", "security breach", "hacked", "identity"],
    "Satisfied": ["thank", "thanks", "appreciate", "grateful", "satisfied", "great", "helpful", "prompt"],
    "Neutral": ["inquiry", "assist", "status", "advise", "question", "procedure", "check"]
}

def predict_emotion_baseline(text: str) -> dict:
    cleaned = clean_text(text)
    scores = {emotion: 0.1 for emotion in BASELINE_EMOTION_KEYWORDS}
    
    for emotion, keywords in BASELINE_EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in cleaned:
                scores[emotion] += 1.0
                
    total = sum(scores.values())
    probs = {k: round(v / total, 4) for k, v in scores.items()}
    best_emotion = max(probs, key=probs.get)
    
    return {
        "emotion": best_emotion,
        "emotion_confidence": probs[best_emotion],
        "emotion_probabilities": probs,
        "is_baseline": True,
        "method": "Emotion Analysis — Baseline Module"
    }

def predict_emotion(text: str) -> dict:
    # 1. Try unified MultiOutputClassifier pipeline
    if os.path.exists(MODEL_PIPELINE_PATH):
        try:
            pipeline = joblib.load(MODEL_PIPELINE_PATH)
            clf = pipeline.named_steps["classifier"]
            emo_clf = clf.estimators_[2]
            
            # Predict probabilities
            tfidf_vec = pipeline.named_steps["tfidf"].transform([text])
            probs = emo_clf.predict_proba(tfidf_vec)[0]
            classes = list(emo_clf.classes_)
            prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
            best_idx = probs.argmax()
            
            return {
                "emotion": str(classes[best_idx]),
                "emotion_confidence": round(float(probs[best_idx]), 4),
                "emotion_probabilities": prob_dict,
                "is_baseline": False,
                "method": "Emotion Analysis — MultiOutputClassifier ML Pipeline"
            }
        except Exception:
            pass
            
    # 2. Try standalone emotion pipeline
    if os.path.exists(EMOTION_MODEL_PATH):
        try:
            pipeline = joblib.load(EMOTION_MODEL_PATH)
            probs = pipeline.predict_proba([text])[0]
            classes = list(pipeline.classes_)
            prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
            best_idx = probs.argmax()
            
            return {
                "emotion": str(classes[best_idx]),
                "emotion_confidence": round(float(probs[best_idx]), 4),
                "emotion_probabilities": prob_dict,
                "is_baseline": False,
                "method": "Emotion Analysis — ML Pipeline"
            }
        except Exception:
            return predict_emotion_baseline(text)
    else:
        return predict_emotion_baseline(text)
