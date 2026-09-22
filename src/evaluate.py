import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    TICKETS_DATA_PATH,
    MODEL_PIPELINE_PATH,
    CATEGORY_MODEL_PATH,
    EVALUATION_PATH,
    MODELS_DIR
)
from src.train import TARGET_COLUMNS

def evaluate_models():
    if not os.path.exists(TICKETS_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {TICKETS_DATA_PATH}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(TICKETS_DATA_PATH)
    
    # Ensure required target columns exist in df
    for col, default_val in [("category", "General"), ("priority", "Medium"), ("emotion", "Neutral")]:
        if col not in df.columns:
            df[col] = default_val
            
    # Raw input text
    X = df["ticket_text"].fillna("").astype(str)
    Y = df[TARGET_COLUMNS].values
    
    # Safe stratification
    cat_counts = df["category"].value_counts()
    can_stratify = (len(cat_counts) > 1) and (cat_counts.min() >= 2) and (len(df) >= 10)
    stratify_target = df["category"] if can_stratify else None
    
    if len(df) < 4:
        X_test, Y_test = X, Y
    else:
        test_size = 0.20 if len(df) >= 10 else 0.25
        _, X_test, _, Y_test = train_test_split(
            X,
            Y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_target
        )
    
    # Load unified MultiOutputClassifier pipeline if available
    pipeline = None
    if os.path.exists(MODEL_PIPELINE_PATH):
        pipeline = joblib.load(MODEL_PIPELINE_PATH)
        print(f"Loaded unified pipeline from {MODEL_PIPELINE_PATH}")
        Y_pred = pipeline.predict(X_test)
    elif os.path.exists(CATEGORY_MODEL_PATH):
        # Fallback to separate models
        Y_pred = []
        models = [
            joblib.load(os.path.join(MODELS_DIR, f"{t}_pipeline.joblib"))
            for t in TARGET_COLUMNS
        ]
        for m in models:
            Y_pred.append(m.predict(X_test))
        import numpy as np
        Y_pred = np.column_stack(Y_pred)
    else:
        raise FileNotFoundError(f"No trained model pipeline found at {MODEL_PIPELINE_PATH}")
        
    evaluation_results = {}
    
    for idx, target_name in enumerate(TARGET_COLUMNS):
        y_true = Y_test[:, idx]
        y_pred = Y_pred[:, idx]
        
        if pipeline is not None:
            classes_in_model = list(pipeline.named_steps["classifier"].estimators_[idx].classes_)
        else:
            classes_in_model = sorted(list(set(y_true)))
            
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=classes_in_model).tolist()
        
        evaluation_results[target_name] = {
            "classes": classes_in_model,
            "classification_report": report,
            "confusion_matrix": cm,
            "accuracy": report["accuracy"]
        }
        print(f"    {target_name.capitalize()} Model Accuracy: {report['accuracy']:.2%}")
        
    with open(EVALUATION_PATH, "w") as f:
        json.dump(evaluation_results, f, indent=2)
        
    print(f"\nEvaluation successfully saved to {EVALUATION_PATH}")
    return evaluation_results

if __name__ == "__main__":
    evaluate_models()
