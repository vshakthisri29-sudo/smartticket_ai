import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# File paths
TICKETS_DATA_PATH = os.path.join(DATA_DIR, "tickets.csv")
TICKET_HISTORY_PATH = os.path.join(DATA_DIR, "ticket_history.csv")
DATASET_METADATA_PATH = os.path.join(DATA_DIR, "dataset_metadata.json")

MODEL_PIPELINE_PATH = os.path.join(MODELS_DIR, "smartticket_pipeline.joblib")
CATEGORY_MODEL_PATH = os.path.join(MODELS_DIR, "category_pipeline.joblib")
PRIORITY_MODEL_PATH = os.path.join(MODELS_DIR, "priority_pipeline.joblib")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
EVALUATION_PATH = os.path.join(MODELS_DIR, "evaluation.json")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Classes
CATEGORIES = ["Payment", "Delivery", "Account", "Technical", "Refund", "Product"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
EMOTIONS = ["Neutral", "Frustrated", "Angry", "Worried", "Satisfied"]
RESOLUTION_STATUSES = ["Open", "Under Review", "In Progress", "Resolved", "Escalated"]

# Thresholds
SIMILARITY_THRESHOLD = 0.75
CONFIDENCE_WARNING_THRESHOLD = 0.70
CONFIDENCE_CRITICAL_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 70
CRITICAL_RISK_THRESHOLD = 85

# Styling Tokens
THEME_COLORS = {
    "primary": "#4F46E5",        # Vibrant Indigo
    "secondary": "#0F172A",      # Deep Slate Headings
    "accent": "#10B981",         # Emerald Green
    "warning": "#F59E0B",        # Amber
    "danger": "#EF4444",         # Rose Red
    "background": "#FFFFFF",     # Clean White
    "card_bg": "#FFFFFF",        # White Card
    "text": "#0F172A"            # Crisp Dark Text
}
