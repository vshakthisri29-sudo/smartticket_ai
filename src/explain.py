import os
import re
import joblib
import numpy as np
from config import MODEL_PIPELINE_PATH, CATEGORY_MODEL_PATH
from src.preprocessing import clean_text

def explain_prediction(ticket_text: str, model_path: str = None, top_n: int = 6) -> dict:
    """
    Generate model-intrinsic feature importance for the predicted category.
    Supports either unified MultiOutputClassifier pipeline or individual Category pipeline.
    """
    selected_path = model_path
    if selected_path is None:
        if os.path.exists(MODEL_PIPELINE_PATH):
            selected_path = MODEL_PIPELINE_PATH
        elif os.path.exists(CATEGORY_MODEL_PATH):
            selected_path = CATEGORY_MODEL_PATH
        else:
            return {
                "top_features": [],
                "highlighted_html": ticket_text,
                "disclaimer": "Models are not trained yet. Run model training to enable explainability.",
                "error": "Model file not found."
            }
            
    try:
        pipeline = joblib.load(selected_path)
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["classifier"]
        
        # If MultiOutputClassifier, estimator 0 is Category
        if hasattr(clf, "estimators_"):
            cat_clf = clf.estimators_[0]
        else:
            cat_clf = clf
            
        # Transform text (tfidf has embedded preprocessor=clean_text)
        tfidf_vec = tfidf.transform([ticket_text])
        pred_class = cat_clf.predict(tfidf_vec)[0]
        class_idx = list(cat_clf.classes_).index(pred_class)
        
        feature_names = tfidf.get_feature_names_out()
        
        if cat_clf.coef_.shape[0] == 1 and len(cat_clf.classes_) == 2:
            coefs = cat_clf.coef_[0] if class_idx == 1 else -cat_clf.coef_[0]
        else:
            coefs = cat_clf.coef_[class_idx]
            
        non_zero_indices = tfidf_vec.nonzero()[1]
        
        contributions = []
        for idx in non_zero_indices:
            term = feature_names[idx]
            x_val = float(tfidf_vec[0, idx])
            weight = float(coefs[idx])
            contribution = x_val * weight
            
            contributions.append({
                "term": term,
                "feature_value": round(x_val, 4),
                "model_weight": round(weight, 4),
                "contribution": round(contribution, 4),
                "direction": "Positive" if contribution > 0 else "Negative"
            })
            
        positive_contributions = [c for c in contributions if c["contribution"] > 0]
        positive_contributions.sort(key=lambda x: x["contribution"], reverse=True)
        top_features = positive_contributions[:top_n]
        
        highlighted_html = ticket_text
        highlighted_terms = [item["term"] for item in top_features]
        
        for term in sorted(highlighted_terms, key=len, reverse=True):
            pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
            highlighted_html = pattern.sub(
                r'<mark style="background: rgba(13, 148, 136, 0.35); color: #FFFFFF; font-weight: 700; padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid #14B8A6;">\1</mark>',
                highlighted_html
            )
            
        return {
            "predicted_class": str(pred_class),
            "top_features": top_features,
            "highlighted_terms": highlighted_terms,
            "highlighted_html": highlighted_html,
            "disclaimer": "These terms are features that contributed to the model's prediction. They should not be interpreted as causal explanations."
        }
    except Exception as exc:
        return {
            "top_features": [],
            "highlighted_html": ticket_text,
            "disclaimer": f"Explainability extraction error: {str(exc)}",
            "error": str(exc)
        }
