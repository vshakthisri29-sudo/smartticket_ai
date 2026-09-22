import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import TICKETS_DATA_PATH, TICKET_HISTORY_PATH, SIMILARITY_THRESHOLD
from src.preprocessing import clean_text

def check_repeat_complaint(ticket_text: str, threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """
    Detect potential repeat complaints using TF-IDF feature extraction
    and Cosine Similarity against historical ticket repository.
    """
    cleaned_input = clean_text(ticket_text)
    if not cleaned_input:
        return {
            "is_repeat": False,
            "similarity_score": 0.0,
            "matched_ticket_id": None,
            "matched_ticket_text": None,
            "matched_category": None,
            "method": "Similarity-based repeat complaint detection"
        }
    
    # Load candidate comparison tickets: prefer history, fallback to dataset
    history_df = None
    if os.path.exists(TICKET_HISTORY_PATH):
        try:
            hdf = pd.read_csv(TICKET_HISTORY_PATH)
            if len(hdf) > 0 and "ticket_text" in hdf.columns:
                history_df = hdf
        except Exception:
            pass
            
    if history_df is None and os.path.exists(TICKETS_DATA_PATH):
        try:
            df = pd.read_csv(TICKETS_DATA_PATH)
            history_df = df.sample(min(300, len(df)), random_state=42)
        except Exception:
            pass
            
    if history_df is None or len(history_df) == 0:
        return {
            "is_repeat": False,
            "similarity_score": 0.0,
            "matched_ticket_id": None,
            "matched_ticket_text": None,
            "matched_category": None,
            "method": "Similarity-based repeat complaint detection (No history available)"
        }
        
    corpus = [clean_text(t) for t in history_df["ticket_text"].fillna("")]
    all_texts = [cleaned_input] + corpus
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Query vector is index 0; corpus vectors are indices 1..N
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    best_idx = sims.argmax()
    best_score = float(sims[best_idx])
    
    is_repeat = bool(best_score >= threshold)
    matched_row = history_df.iloc[best_idx]
    
    ticket_id = matched_row.get("ticket_id", f"HIST-{best_idx+1}")
    cat = matched_row.get("category", "General")
    raw_text = matched_row.get("ticket_text", "")
    
    return {
        "is_repeat": is_repeat,
        "similarity_score": round(best_score, 4),
        "matched_ticket_id": str(ticket_id),
        "matched_ticket_text": str(raw_text),
        "matched_category": str(cat),
        "threshold": threshold,
        "method": "Similarity-based repeat complaint detection"
    }
