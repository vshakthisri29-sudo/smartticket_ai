import os
import uuid
from datetime import datetime
import pandas as pd
from config import TICKET_HISTORY_PATH, DATA_DIR

def get_next_history_id() -> str:
    return f"HIST-{uuid.uuid4().hex[:6].upper()}"

def save_ticket(ticket_text: str, result: dict, resolution_status: str = "Open", resolution_message: str = "") -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    
    ticket_id = result.get("ticket_id") or get_next_history_id()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    row = {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "category": result.get("category", "General"),
        "category_confidence": result.get("category_confidence", 0.0),
        "priority": result.get("priority", "Medium"),
        "priority_confidence": result.get("priority_confidence", 0.0),
        "emotion": result.get("emotion", "Neutral"),
        "emotion_confidence": result.get("emotion_confidence", 0.0),
        "is_repeat": result.get("is_repeat", False),
        "similarity_score": result.get("similarity_score", 0.0),
        "risk_score": result.get("risk_score", 0),
        "risk_level": result.get("risk_level", "Medium"),
        "recommended_action": result.get("recommended_action", ""),
        "resolution_status": resolution_status,
        "resolution_message": resolution_message,
        "created_at": now_str
    }
    
    df_new = pd.DataFrame([row])
    
    if os.path.exists(TICKET_HISTORY_PATH):
        try:
            df_existing = pd.read_csv(TICKET_HISTORY_PATH)
            # Prepend newest ticket at top
            df_combined = pd.concat([df_new, df_existing], ignore_index=True)
            df_combined.to_csv(TICKET_HISTORY_PATH, index=False)
        except Exception:
            df_new.to_csv(TICKET_HISTORY_PATH, index=False)
    else:
        df_new.to_csv(TICKET_HISTORY_PATH, index=False)
        
    return row

def load_history() -> pd.DataFrame:
    if not os.path.exists(TICKET_HISTORY_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TICKET_HISTORY_PATH)
        return df
    except Exception:
        return pd.DataFrame()

def update_resolution(ticket_id: str, new_status: str, resolution_message: str = "") -> bool:
    if not os.path.exists(TICKET_HISTORY_PATH):
        return False
    try:
        df = pd.read_csv(TICKET_HISTORY_PATH)
        idx = df[df["ticket_id"] == ticket_id].index
        if len(idx) > 0:
            df.loc[idx, "resolution_status"] = new_status
            if resolution_message:
                df.loc[idx, "resolution_message"] = resolution_message
            df.to_csv(TICKET_HISTORY_PATH, index=False)
            return True
        return False
    except Exception:
        return False
