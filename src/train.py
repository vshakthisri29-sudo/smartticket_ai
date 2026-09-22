import os
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    MODELS_DIR,
    TICKETS_DATA_PATH,
    MODEL_PIPELINE_PATH,
    CATEGORY_MODEL_PATH,
    PRIORITY_MODEL_PATH,
    EMOTION_MODEL_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    CATEGORIES,
    PRIORITIES,
    EMOTIONS
)
from src.preprocessing import clean_text
from src.generate_dataset import generate_dataset

TARGET_COLUMNS = ["category", "priority", "emotion"]

def train_all_models():
    """
    Train a single unified MultiOutputClassifier ML pipeline with embedded
    clean_text preprocessing inside TfidfVectorizer.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(TICKETS_DATA_PATH):
        print(f"Dataset not found at {TICKETS_DATA_PATH}. Generating synthetic tickets...")
        generate_dataset()
        
    print(f"Loading dataset from {TICKETS_DATA_PATH}...")
    df = pd.read_csv(TICKETS_DATA_PATH)
    
    # Ensure required target columns exist in df
    for col, default_val in [("category", "General"), ("priority", "Medium"), ("emotion", "Neutral")]:
        if col not in df.columns:
            df[col] = default_val
            
    # Raw input text - clean_text is embedded directly in TfidfVectorizer preprocessor!
    X = df["ticket_text"].fillna("").astype(str)
    Y = df[TARGET_COLUMNS].values
    
    # Safe stratification: check if category has multiple classes and all classes have >= 2 members
    cat_counts = df["category"].value_counts()
    can_stratify = (len(cat_counts) > 1) and (cat_counts.min() >= 2) and (len(df) >= 10)
    stratify_target = df["category"] if can_stratify else None
    
    if len(df) < 4:
        X_train, X_test = X, X
        Y_train, Y_test = Y, Y
    else:
        test_size = 0.20 if len(df) >= 10 else 0.25
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_target
        )
    
    print("\n--> Building unified MultiOutputClassifier ML Pipeline...")
    print("    • Preprocessor: Embedded src.preprocessing.clean_text")
    print("    • Vectorizer: TF-IDF (1-2 ngrams, 10000 max features, sublinear tf)")
    print("    • Estimator: MultiOutputClassifier(LogisticRegression(max_iter=2000, random_state=42))")
    
    # Single self-contained pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )),
        ("classifier", MultiOutputClassifier(
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        ))
    ])
    
    print("\n--> Fitting MultiOutput Pipeline on training set...")
    pipeline.fit(X_train, Y_train)
    
    print("--> Predicting on held-out test set...")
    Y_pred = pipeline.predict(X_test)
    
    # Save the unified multi-output pipeline
    joblib.dump(pipeline, MODEL_PIPELINE_PATH)
    print(f"    Saved unified pipeline to: {MODEL_PIPELINE_PATH}")
    
    # Also save individual standalone pipeline views for backwards compatibility
    tfidf_step = pipeline.named_steps["tfidf"]
    clf_step = pipeline.named_steps["classifier"]
    
    individual_paths = [CATEGORY_MODEL_PATH, PRIORITY_MODEL_PATH, EMOTION_MODEL_PATH]
    for idx, (target_name, ind_path) in enumerate(zip(TARGET_COLUMNS, individual_paths)):
        sub_pipeline = Pipeline([
            ("tfidf", tfidf_step),
            ("classifier", clf_step.estimators_[idx])
        ])
        joblib.dump(sub_pipeline, ind_path)
    
    # Compute per-target metrics
    metrics = {}
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "architecture": "Single Unified MultiOutputClassifier Pipeline",
        "preprocessor": "src.preprocessing.clean_text (embedded in TfidfVectorizer)",
        "vectorizer_config": {
            "ngram_range": [1, 2],
            "max_features": 10000,
            "sublinear_tf": True
        },
        "classifier_config": {
            "type": "MultiOutputClassifier",
            "base_estimator": "LogisticRegression",
            "max_iter": 2000,
            "random_state": 42
        },
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(tfidf_step.vocabulary_),
        "target_columns": TARGET_COLUMNS,
        "models": {}
    }
    
    for idx, target_name in enumerate(TARGET_COLUMNS):
        y_true = Y_test[:, idx]
        y_pred = Y_pred[:, idx]
        
        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        
        metrics[target_name] = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }
        
        classes_list = list(clf_step.estimators_[idx].classes_)
        metadata["models"][target_name] = {
            "classes": classes_list,
            "num_classes": len(classes_list),
            "metrics": metrics[target_name]
        }
        print(f"    Target '{target_name}': Accuracy: {acc:.2%} | F1 Macro: {f1_macro:.2%}")
        
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\nUnified MultiOutputClassifier pipeline trained and saved successfully.")
    return metrics

if __name__ == "__main__":
    train_all_models()
