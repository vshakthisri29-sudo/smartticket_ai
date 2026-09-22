import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import config
from config import (
    CATEGORY_MODEL_PATH,
    PRIORITY_MODEL_PATH,
    EMOTION_MODEL_PATH,
    ASSETS_DIR,
    THEME_COLORS
)
MODEL_PIPELINE_PATH = getattr(
    config,
    "MODEL_PIPELINE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "smartticket_pipeline.joblib")
)

def load_custom_css():
    css_path = os.path.join(ASSETS_DIR, "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def check_models_loaded() -> dict:
    pipe_exists = os.path.exists(MODEL_PIPELINE_PATH)
    cat_exists = os.path.exists(CATEGORY_MODEL_PATH) or pipe_exists
    prio_exists = os.path.exists(PRIORITY_MODEL_PATH) or pipe_exists
    emo_exists = os.path.exists(EMOTION_MODEL_PATH) or pipe_exists
    all_ready = pipe_exists or (cat_exists and prio_exists and emo_exists)
    
    return {
        "all_ready": all_ready,
        "unified_pipeline_loaded": pipe_exists,
        "category_loaded": cat_exists,
        "priority_loaded": prio_exists,
        "emotion_loaded": emo_exists,
        "vectorizers_loaded": all_ready,
        "explainability_ready": all_ready
    }

def render_pipeline_flow(active_step: int = 1):
    steps = [
        ("TICKET", "Input"),
        ("CLEANING", "NLP Normalization"),
        ("TF-IDF", "Feature Extraction"),
        ("ML MODELS", "Logistic Regression"),
        ("EXPLAINABLE AI", "Word Attribution"),
        ("RISK ENGINE", "Rule Analysis"),
        ("ACTION", "Decision Support")
    ]
    
    html = '<div class="pipeline-container">'
    for idx, (badge, label) in enumerate(steps):
        is_active = (idx + 1) == active_step
        active_cls = "active" if is_active else ""
        html += f'''
        <div class="pipeline-node">
            <div class="pipeline-badge {active_cls}">{badge}</div>
            <div class="pipeline-label">{label}</div>
        </div>
        '''
        if idx < len(steps) - 1:
            html += '<div class="pipeline-arrow">→</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def create_feature_importance_chart(features: list, title: str = "Feature Importance / Word Contribution"):
    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No features available", showarrow=False, font=dict(color="#64748B"))
        fig.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        return fig
        
    terms = [f["term"] for f in reversed(features)]
    contribs = [f["contribution"] for f in reversed(features)]
    
    colors = ["#4F46E5" if c >= 0 else "#EF4444" for c in contribs]
    
    fig = go.Figure(go.Bar(
        x=contribs,
        y=terms,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#E2E8F0", width=1)),
        text=[f"{c:+.3f}" for c in contribs],
        textposition="outside",
        textfont=dict(color="#1E293B", size=11)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0F172A", size=14, family="Plus Jakarta Sans, sans-serif")),
        xaxis=dict(title="Linear Contribution (TF-IDF Weight × Coef)", color="#64748B", gridcolor="#F1F5F9"),
        yaxis=dict(color="#1E293B"),
        height=260,
        margin=dict(l=80, r=40, t=40, b=30),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="JetBrains Mono, monospace")
    )
    return fig

def create_confusion_matrix_chart(cm_data: list, class_names: list, title: str):
    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=class_names,
        y=class_names,
        colorscale=[[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#4F46E5"]],
        text=cm_data,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#0F172A"),
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0F172A", size=15)),
        xaxis=dict(title="Predicted Class", color="#475569"),
        yaxis=dict(title="True Class", color="#475569", autorange="reversed"),
        height=380,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, sans-serif")
    )
    return fig
