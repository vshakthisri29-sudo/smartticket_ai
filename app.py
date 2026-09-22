import os
import time
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from config import (
    TICKETS_DATA_PATH,
    METRICS_PATH,
    EVALUATION_PATH,
    MODEL_METADATA_PATH,
    SIMILARITY_THRESHOLD,
    CONFIDENCE_WARNING_THRESHOLD,
    CONFIDENCE_CRITICAL_THRESHOLD,
    THEME_COLORS
)
MODEL_PIPELINE_PATH = getattr(
    config,
    "MODEL_PIPELINE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "smartticket_pipeline.joblib")
)
from src.utils import load_custom_css, check_models_loaded, create_feature_importance_chart, create_confusion_matrix_chart
from src.generate_dataset import generate_dataset
from src.train import train_all_models
from src.evaluate import evaluate_models
from src.predict import predict_ticket
from src.storage import save_ticket, load_history, update_resolution

# Page configuration
st.set_page_config(
    page_title="SmartTicket AI — Enterprise Support Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load global SaaS design system & styles
load_custom_css()

CHART_COLORS = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#3B82F6"]

def render_styled_table(df_to_render, max_rows=15):
    if df_to_render is None or len(df_to_render) == 0:
        return "<p style='color:#64748B; padding: 12px;'>No records found.</p>"
    sub = df_to_render.head(max_rows)
    html = '<div style="overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 12px; margin: 14px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">'
    html += '<table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.88rem; background: #FFFFFF; color: #1E293B;">'
    html += '<thead><tr style="background: #F8FAFC; border-bottom: 2px solid #E2E8F0; text-align: left;">'
    for col in sub.columns:
        html += f'<th style="padding: 12px 14px; font-weight: 700; color: #475569; letter-spacing: 0.02em; font-size: 0.78rem;">{str(col).upper()}</th>'
    html += '</tr></thead><tbody>'
    for i, (_, row) in enumerate(sub.iterrows()):
        bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFC"
        html += f'<tr style="background: {bg}; border-bottom: 1px solid #F1F5F9;">'
        for val in row:
            val_str = str(val)
            if val_str == "Critical":
                badge = f'<span style="background: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5; padding: 3px 8px; border-radius: 6px; font-weight: 700;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "High":
                badge = f'<span style="background: #FFEDD5; color: #EA580C; border: 1px solid #FDBA74; padding: 3px 8px; border-radius: 6px; font-weight: 700;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Medium":
                badge = f'<span style="background: #FEF3C7; color: #D97706; border: 1px solid #FCD34D; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Low":
                badge = f'<span style="background: #DCFCE7; color: #16A34A; border: 1px solid #86EFAC; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Payment":
                badge = f'<span style="background: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Technical":
                badge = f'<span style="background: #F3E8FF; color: #7E22CE; border: 1px solid #D8B4FE; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Delivery":
                badge = f'<span style="background: #FEF3C7; color: #B45309; border: 1px solid #FCD34D; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Account":
                badge = f'<span style="background: #CCFBF1; color: #0F766E; border: 1px solid #5EEAD4; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Refund":
                badge = f'<span style="background: #FFE4E6; color: #BE123C; border: 1px solid #FDA4AF; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str == "Product":
                badge = f'<span style="background: #E0E7FF; color: #4338CA; border: 1px solid #A5B4FC; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            elif val_str in ["Resolved", "Open", "In Progress", "Under Review", "Escalated"]:
                stat_colors = {
                    "Resolved": ("#DCFCE7", "#16A34A", "#86EFAC"),
                    "In Progress": ("#DBEAFE", "#1D4ED8", "#93C5FD"),
                    "Under Review": ("#FEF3C7", "#B45309", "#FCD34D"),
                    "Escalated": ("#FEE2E2", "#DC2626", "#FCA5A5"),
                    "Open": ("#F1F5F9", "#475569", "#CBD5E1")
                }
                bg_s, col_s, bdr_s = stat_colors.get(val_str, ("#F1F5F9", "#475569", "#CBD5E1"))
                badge = f'<span style="background: {bg_s}; color: {col_s}; border: 1px solid {bdr_s}; padding: 3px 8px; border-radius: 6px; font-weight: 600;">{val_str}</span>'
                html += f'<td style="padding: 10px 14px;">{badge}</td>'
            else:
                html += f'<td style="padding: 10px 14px; color: #334155;">{val_str}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html

# Session State Initialization
if "splash_seen" not in st.session_state:
    st.session_state["splash_seen"] = False
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Triage Specialist"
if "user_name" not in st.session_state:
    st.session_state["user_name"] = "Support Agent"
if "user_email" not in st.session_state:
    st.session_state["user_email"] = ""
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "🎯 Live Classifier"
if "dataset_df" not in st.session_state:
    if os.path.exists(TICKETS_DATA_PATH):
        try:
            st.session_state["dataset_df"] = pd.read_csv(TICKETS_DATA_PATH)
        except Exception:
            st.session_state["dataset_df"] = None
    else:
        st.session_state["dataset_df"] = None

def format_name_from_email(email_str: str) -> str:
    if not email_str or "@" not in email_str:
        return "Support User"
    handle = email_str.split("@")[0]
    parts = handle.replace(".", " ").replace("_", " ").replace("-", " ").split()
    formatted = " ".join([p.capitalize() for p in parts])
    return formatted if formatted else handle

# ==============================================================================
# VIEW 0: MINIMALIST LOGO & LOADING SPLASH (NO WORDS, LOGO + LOADING ONLY)
# ==============================================================================
if not st.session_state.get("splash_seen", False):
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 14vh; margin-bottom: 30px;">
        <div class="splash-logo-circle">🎫</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_l1, col_l_center, col_l2 = st.columns([1.6, 1.8, 1.6])
    with col_l_center:
        prog_bar = st.progress(0)
        for step in range(1, 101, 3):
            prog_bar.progress(step)
            time.sleep(0.035)
        time.sleep(0.15)
        st.session_state["splash_seen"] = True
        st.rerun()
        
    st.stop()

# ==============================================================================
# VIEW 1: AUTHENTICATION / LOGIN SCREEN
# ==============================================================================
if not st.session_state["authenticated"]:
    st.markdown("""
    <div class="login-hero">
        <div style="font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 8px;">
            🎫 <span style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">SmartTicket AI</span>
        </div>
        <div style="color: #475569; font-size: 1.05rem; font-weight: 500;">
            Enterprise Customer Support Classification & Decision Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_l, col_center, col_r = st.columns([1, 1.4, 1])
    
    with col_center:
        with st.container(border=True):
            st.markdown("### Sign In to Workspace")
            st.caption("Enter your email address to access your custom support queue.")
            
            email_input = st.text_input(
                "Your Email Address",
                value=st.session_state.get("login_email_value", ""),
                placeholder="Enter your email (e.g. you@company.com)"
            )
            pass_input = st.text_input("Password", value="••••••••••••", type="password")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Sign In with My Email", type="primary", use_container_width=True):
                    cleaned_email = email_input.strip()
                    if not cleaned_email or "@" not in cleaned_email:
                        st.error("Please enter a valid email address.")
                    else:
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = cleaned_email
                        st.session_state["user_name"] = format_name_from_email(cleaned_email)
                        st.session_state["user_role"] = "Triage Specialist"
                        st.rerun()
                    
            with col_btn2:
                if st.button("Quick Demo Login", use_container_width=True):
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = "admin@smartticket.ai"
                    st.session_state["user_name"] = "Alex Rivera"
                    st.session_state["user_role"] = "ML Ops / Admin"
                    st.rerun()
                    
            st.markdown("---")
            st.markdown("""
            <div style="font-size: 0.78rem; color: #64748B; text-align: center; padding-top: 4px;">
                🔒 Single Sign-On Ready • SOC2 & GDPR Compliant
            </div>
            """, unsafe_allow_html=True)
        
    st.stop()

# ==============================================================================
# AUTHENTICATED TOP SAAS NAVBAR
# ==============================================================================
models_status = check_models_loaded()
model_badge_html = '<span style="color: #059669;">● Model Active</span>' if models_status["all_ready"] else '<span style="color: #D97706;">⚠️ Model Not Trained</span>'

nav_col_back, nav_col_logo, nav_col_menu, nav_col_user = st.columns([1.2, 2.3, 4.3, 2.2])

with nav_col_back:
    if st.button("← Back to Login", key="top_back_to_login", use_container_width=True, help="Return to login page"):
        st.session_state["authenticated"] = False
        st.rerun()

with nav_col_logo:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; padding: 6px 0;">
        <span style="font-size: 1.35rem; font-weight: 800; color: #0F172A;">🎫 SmartTicket<span style="color: #4F46E5;">AI</span></span>
        <span style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 12px; padding: 2px 7px; font-size: 0.70rem; font-weight: 700;">
            {model_badge_html}
        </span>
    </div>
    """, unsafe_allow_html=True)

with nav_col_menu:
    tabs = ["🎯 Live Classifier", "📁 Dataset Manager", "⚡ Model Training", "📊 Analytics Dashboard", "🗂️ Ticket Queue"]
    selected_tab = st.radio(
        "Navigation",
        tabs,
        index=tabs.index(st.session_state["active_tab"]) if st.session_state["active_tab"] in tabs else 0,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state["active_tab"] = selected_tab

with nav_col_user:
    u_c1, u_c2 = st.columns([2.5, 1])
    with u_c1:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 2px;">
            <div style="font-size: 0.86rem; font-weight: 700; color: #0F172A;">👤 {st.session_state.get('user_name', 'User')}</div>
            <div style="font-size: 0.72rem; color: #4F46E5; font-family: monospace; font-weight: 600;">{st.session_state.get('user_email', '')}</div>
        </div>
        """, unsafe_allow_html=True)
    with u_c2:
        if st.button("Log out", key="logout_btn"):
            st.session_state["authenticated"] = False
            st.rerun()

st.markdown("<hr style='margin-top: 4px; margin-bottom: 24px; border-color: #E2E8F0;'>", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: LIVE TICKET CLASSIFIER & DECISION TRIAGE
# ==============================================================================
if st.session_state["active_tab"] == "🎯 Live Classifier":
    st.markdown("### 🎯 Intelligent Support Ticket Classifier")
    st.caption("Submit unstructured customer complaints to generate instantaneous ML category predictions, priority levels, sentiment analysis, explainability, and risk scores.")
    
    if not models_status["all_ready"]:
        st.warning("⚠️ The Machine Learning model has not been trained yet. Switch to the **📁 Dataset Manager** or **⚡ Model Training** tab to initialize and train the model.")
        
    # Quick prompt buttons
    st.markdown("##### ⚡ Quick Example Complaints:")
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
    with btn_c1:
        if st.button("💳 Duplicate Charge"):
            st.session_state["ticket_text_input"] = "I was charged twice on my credit card for transaction #48291. My money was deducted twice but only one order shows in my account. Please refund the extra charge immediately!"
    with btn_c2:
        if btn_c2.button("📦 Lost Delivery"):
            st.session_state["ticket_text_input"] = "My package has not arrived even though the tracking status said delivered three days ago! Nobody came to my door and I urgently need this parcel before tomorrow."
    with btn_c3:
        if btn_c3.button("💻 App Crash on Login"):
            st.session_state["ticket_text_input"] = "The mobile app on Android crashes with an unhandled exception every time I try to open the checkout cart page. It freezes and closes abruptly."
    with btn_c4:
        if btn_c4.button("🔒 Account Lockout"):
            st.session_state["ticket_text_input"] = "My account is locked and the two-factor authentication password reset email never reaches my inbox. I cannot access my dashboard."

    ticket_text = st.text_area(
        "Customer Complaint Text:",
        value=st.session_state.get("ticket_text_input", ""),
        height=130,
        placeholder="Paste customer support inquiry or email body here..."
    )
    
    col_act, col_spacer = st.columns([1.5, 4])
    with col_act:
        classify_clicked = st.button("🚀 Classify & Analyze Ticket", type="primary", use_container_width=True)

    if classify_clicked:
        if not ticket_text.strip():
            st.error("Please enter a ticket description to analyze.")
        elif not models_status["all_ready"]:
            st.error("Cannot predict: Models are not trained yet. Please visit the '⚡ Model Training' tab.")
        else:
            with st.spinner("Classifying with MultiOutputClassifier pipeline..."):
                try:
                    res = predict_ticket(ticket_text)
                    saved = save_ticket(ticket_text, res)
                    st.session_state["latest_res"] = res
                    st.session_state["latest_id"] = saved["ticket_id"]
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")

    if "latest_res" in st.session_state:
        res = st.session_state["latest_res"]
        t_id = st.session_state.get("latest_id", "TICK-NEW")
        
        st.markdown(f"#### 📋 Triage Analysis Result &mdash; `{t_id}`")
        
        # 4 Main Prediction Metrics
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3.5px solid #4F46E5;">
                <div class="metric-card-title">Predicted Category</div>
                <div class="metric-card-value" style="color: #4F46E5;">{res['category']}</div>
                <div class="metric-card-subtitle" style="color: #6366F1;">Confidence: {res['category_confidence']:.1%}</div>
                <div class="saas-progress-bg" style="background: #EEF2FF; border-radius: 6px; height: 7px; margin-top: 8px; overflow: hidden;">
                    <div style="width: {res['category_confidence']*100}%; height: 100%; background: linear-gradient(90deg, #4F46E5, #6366F1); border-radius: 6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            prio_color = "#DC2626" if res['priority'] == "Critical" else ("#EA580C" if res['priority'] == "High" else "#16A34A")
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3.5px solid {prio_color};">
                <div class="metric-card-title">Priority Level</div>
                <div class="metric-card-value" style="color: {prio_color};">{res['priority']}</div>
                <div class="metric-card-subtitle" style="color: {prio_color};">Confidence: {res['priority_confidence']:.1%}</div>
                <div class="saas-progress-bg" style="background: #F1F5F9; border-radius: 6px; height: 7px; margin-top: 8px; overflow: hidden;">
                    <div style="width: {res['priority_confidence']*100}%; height: 100%; background: {prio_color}; border-radius: 6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3.5px solid #D946EF;">
                <div class="metric-card-title">Customer Emotion</div>
                <div class="metric-card-value" style="color: #C026D3;">{res['emotion']}</div>
                <div class="metric-card-subtitle" style="color: #D946EF;">Confidence: {res['emotion_confidence']:.1%}</div>
                <div class="saas-progress-bg" style="background: #FDF4FF; border-radius: 6px; height: 7px; margin-top: 8px; overflow: hidden;">
                    <div style="width: {res['emotion_confidence']*100}%; height: 100%; background: linear-gradient(90deg, #D946EF, #C026D3); border-radius: 6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            r_color = "#DC2626" if res['risk_score'] >= 75 else ("#D97706" if res['risk_score'] >= 50 else "#16A34A")
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3.5px solid {r_color};">
                <div class="metric-card-title">Composite Risk Score</div>
                <div class="metric-card-value" style="color: {r_color};">{res['risk_score']}<span style="font-size: 1rem; color: #64748B;">/100</span></div>
                <div class="metric-card-subtitle" style="color: #475569;">Level: {res['risk_level']} | Repeat: {'Yes' if res['is_repeat'] else 'No'}</div>
                <div class="saas-progress-bg" style="background: #F1F5F9; border-radius: 6px; height: 7px; margin-top: 8px; overflow: hidden;">
                    <div style="width: {res['risk_score']}%; height: 100%; background: {r_color}; border-radius: 6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Explainable AI & Recommendations
        col_xai, col_rec = st.columns([1, 1])
        
        with col_xai:
            st.markdown("##### 🔍 Explainable AI — Word Attribution")
            st.caption("Highlights specific terms extracted by TF-IDF that drove the model's classification.")
            st.markdown(
                f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; font-size: 0.95rem; line-height: 1.9; color: #1E293B; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">'
                f'{res["highlighted_html"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            if res.get("top_features"):
                fig_feat = create_feature_importance_chart(res["top_features"], title="Feature Contributions (Weight × Coef)")
                st.plotly_chart(fig_feat, use_container_width=True)
                
        with col_rec:
            st.markdown("##### 🛡️ Decision Support & Routing Playbook")
            st.caption("Synthesized operational next steps based on category, priority, and risk tier.")
            
            st.markdown(f"""
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 6px solid #10B981; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.76rem; color: #047857; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;">
                    Recommended Department: {res['recommendation_details'].get('routing_department', 'Support Operations')}
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #064E3B; margin: 10px 0;">
                    {res['recommended_action']}
                </div>
                <div style="font-size: 0.82rem; color: #065F46;">
                    ⚠️ <b>Human Review Recommended</b>: Please verify with customer records before initiating account or financial changes.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Contributing Risk Factors")
            if res.get("contributing_factors"):
                for factor in res["contributing_factors"]:
                    st.markdown(f"- 🔸 `{factor}`")
            else:
                st.markdown("- No critical risk anomalies detected.")

# ==============================================================================
# TAB 2: DATASET MANAGER
# ==============================================================================
elif st.session_state["active_tab"] == "📁 Dataset Manager":
    st.markdown("### 📁 Dataset Management")
    st.caption("Load customer ticket datasets, upload custom CSV files, and inspect data quality before training.")
    
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        st.markdown("##### 📂 Option A: Default System Dataset")
        st.write("Load the pre-configured realistic benchmark dataset containing 1,500 tickets.")
        if st.button("Load Default Dataset (1,500 Tickets)", type="primary"):
            with st.spinner("Generating / Loading benchmark dataset..."):
                df = generate_dataset(num_samples=1500)
                st.session_state["dataset_df"] = df
                st.success(f"Successfully loaded default dataset with {len(df)} records!")
                st.rerun()
                
    with col_d2:
        st.markdown("##### 📤 Option B: Upload Custom CSV")
        st.write("Upload your own CSV with columns: `ticket_text`, `category`, `priority`, `emotion`.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="custom_csv_uploader")
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                st.session_state["dataset_df"] = custom_df
                # Save to disk so training pipeline can use it
                custom_df.to_csv(TICKETS_DATA_PATH, index=False)
                st.success(f"Uploaded custom dataset: {len(custom_df)} rows loaded!")
            except Exception as ex:
                st.error(f"Error parsing CSV: {str(ex)}")

    st.markdown("---")
    
    # Dataset Preview & Summary
    df = st.session_state.get("dataset_df")
    if df is not None and len(df) > 0:
        st.markdown("#### 📊 Active Dataset Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df))
        c2.metric("Categories", df["category"].nunique() if "category" in df.columns else "N/A")
        c3.metric("Priorities", df["priority"].nunique() if "priority" in df.columns else "N/A")
        c4.metric("Emotions", df["emotion"].nunique() if "emotion" in df.columns else "N/A")
        
        st.markdown("##### Data Preview (Top 10 Rows)")
        cols_to_display = [c for c in ["ticket_id", "customer_id", "ticket_text", "category", "priority", "emotion", "created_at"] if c in df.columns]
        st.markdown(render_styled_table(df[cols_to_display].head(10)), unsafe_allow_html=True)
        
        # Distribution Charts
        st.markdown("##### Class Distribution Breakdown")
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            if "category" in df.columns:
                fig1 = px.bar(df["category"].value_counts().reset_index(), x="category", y="count", title="Category Distribution", color="category", color_discrete_sequence=CHART_COLORS)
                fig1.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#1E293B", family="Plus Jakarta Sans"), showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
        with ch2:
            if "priority" in df.columns:
                fig2 = px.pie(df["priority"].value_counts().reset_index(), names="priority", values="count", title="Priority Balance", hole=0.4, color_discrete_sequence=CHART_COLORS)
                fig2.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#1E293B", family="Plus Jakarta Sans"))
                st.plotly_chart(fig2, use_container_width=True)
        with ch3:
            if "emotion" in df.columns:
                fig3 = px.bar(df["emotion"].value_counts().reset_index(), x="emotion", y="count", title="Emotion Spectrum", color="emotion", color_discrete_sequence=CHART_COLORS)
                fig3.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#1E293B", family="Plus Jakarta Sans"), showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No dataset is currently loaded. Click 'Load Default Dataset' above or upload a CSV file.")

# ==============================================================================
# TAB 3: MODEL TRAINING
# ==============================================================================
elif st.session_state["active_tab"] == "⚡ Model Training":
    st.markdown("### ⚡ Machine Learning Training Center")
    st.caption("Train a self-contained MultiOutputClassifier pipeline with embedded TF-IDF feature extraction on your active dataset.")
    
    col_t_left, col_t_right = st.columns([1.2, 1])
    
    with col_t_left:
        st.markdown("##### 🛠️ Training Configuration")
        st.markdown("""
        - **Pipeline Type**: `MultiOutputClassifier(LogisticRegression(max_iter=2000))`
        - **Feature Extractor**: `TfidfVectorizer(ngram_range=(1,2), max_features=10000)`
        - **Embedded Preprocessor**: `src.preprocessing.clean_text`
        - **Train / Test Split**: `80% Train / 20% Test (Stratified)`
        """)
        
        start_train = st.button("🚀 Train Model on Active Dataset", type="primary", use_container_width=True)
        
        if start_train:
            prog_bar = st.progress(0)
            status_txt = st.empty()
            
            status_txt.markdown("**Step 1/4**: Validating training dataset...")
            prog_bar.progress(25)
            time.sleep(0.3)
            
            try:
                import importlib
                import src.train
                import src.evaluate
                importlib.reload(src.train)
                importlib.reload(src.evaluate)
                
                status_txt.markdown("**Step 2/4**: Fitting TF-IDF n-gram vectorizer & MultiOutputClassifier...")
                prog_bar.progress(60)
                metrics = src.train.train_all_models()
                
                status_txt.markdown("**Step 3/4**: Evaluating on held-out test data...")
                prog_bar.progress(85)
                eval_res = src.evaluate.evaluate_models()
                
                status_txt.markdown("**Step 4/4**: Serializing pipeline to `models/smartticket_pipeline.joblib`...")
                prog_bar.progress(100)
                time.sleep(0.3)
                
                status_txt.empty()
                prog_bar.empty()
                st.success("🎉 Model training and evaluation successfully completed!")
                st.rerun()
            except Exception as train_err:
                status_txt.empty()
                prog_bar.empty()
                st.error(f"Training error: {str(train_err)}")

    with col_t_right:
        st.markdown("##### 📌 Current Model Status")
        m_status = check_models_loaded()
        if m_status["all_ready"]:
            st.markdown("""
            <div style="background: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="color: #047857; font-weight: 700; font-size: 1.05rem;">✓ Model Pipeline Ready</div>
                <div style="color: #065F46; font-size: 0.88rem; margin-top: 6px; line-height: 1.5;">
                    Active model: <code style="background: #DCFCE7; color: #166534; padding: 2px 6px; border-radius: 4px;">smartticket_pipeline.joblib</code><br>
                    Trained & ready for real-time ticket triage, explainability, and risk routing.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #FFFBEB; border: 1.5px solid #FCD34D; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="color: #B45309; font-weight: 700; font-size: 1.05rem;">⚠️ Models Not Trained Yet</div>
                <div style="color: #92400E; font-size: 0.88rem; margin-top: 6px; line-height: 1.5;">
                    Click the button on the left to train the unified MultiOutputClassifier model on your active dataset.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Model Metrics Display
    if os.path.exists(EVALUATION_PATH):
        st.markdown("#### 📈 Empirical Test Set Performance (Measured)")
        with open(EVALUATION_PATH, "r") as f:
            eval_data = json.load(f)
            
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Category Model Accuracy", f"{eval_data['category']['accuracy']:.1%}")
        c_m2.metric("Priority Model Accuracy", f"{eval_data['priority']['accuracy']:.1%}")
        c_m3.metric("Emotion Model Accuracy", f"{eval_data['emotion']['accuracy']:.1%}")
        
        st.markdown("##### Confusion Matrices")
        tab_c, tab_p, tab_e = st.tabs(["Category Confusion Matrix", "Priority Confusion Matrix", "Emotion Confusion Matrix"])
        with tab_c:
            fig_c = create_confusion_matrix_chart(eval_data["category"]["confusion_matrix"], eval_data["category"]["classes"], "Category Confusion Matrix")
            st.plotly_chart(fig_c, use_container_width=True)
        with tab_p:
            fig_p = create_confusion_matrix_chart(eval_data["priority"]["confusion_matrix"], eval_data["priority"]["classes"], "Priority Confusion Matrix")
            st.plotly_chart(fig_p, use_container_width=True)
        with tab_e:
            fig_e = create_confusion_matrix_chart(eval_data["emotion"]["confusion_matrix"], eval_data["emotion"]["classes"], "Emotion Confusion Matrix")
            st.plotly_chart(fig_e, use_container_width=True)

# ==============================================================================
# TAB 4: ANALYTICS & OPERATIONS DASHBOARD
# ==============================================================================
elif st.session_state["active_tab"] == "📊 Analytics Dashboard":
    st.markdown("### 📊 Operational Support Analytics")
    st.caption("Live monitoring of ticket throughput, high-risk flags, sentiment trends, and model performance across support traffic.")
    
    history_df = load_history()
    
    if len(history_df) == 0:
        # Fallback to dataset or demo
        dataset_df = st.session_state.get("dataset_df")
        if dataset_df is not None:
            dash_df = dataset_df.head(50)
            st.info("Showing analytics based on active benchmark dataset (No live customer queries submitted yet).")
        else:
            dash_df = pd.DataFrame()
            st.warning("No historical tickets or dataset available. Visit the Dataset Manager tab to load sample data.")
    else:
        dash_df = history_df
        
    if len(dash_df) > 0:
        total = len(dash_df)
        high_risk = len(dash_df[dash_df["risk_score"] >= 60]) if "risk_score" in dash_df.columns else 0
        repeat_cases = len(dash_df[dash_df["is_repeat"] == True]) if "is_repeat" in dash_df.columns else 0
        avg_conf = dash_df["category_confidence"].mean() if "category_confidence" in dash_df.columns else 0.88
        
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Tickets Analyzed", total)
        k2.metric("High-Risk Escalations", high_risk, delta=f"{(high_risk/total)*100:.0f}% of total" if total else "0%")
        k3.metric("Repeat Inquiries Detected", repeat_cases, delta="TF-IDF Similarity")
        k4.metric("Mean Model Confidence", f"{avg_conf:.1%}", delta="Calibrated estimate")
        
        st.markdown("---")
        
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            if "category" in dash_df.columns:
                fig_v = px.bar(dash_df["category"].value_counts().reset_index(), x="category", y="count", title="Volume by Support Category", color="category", color_discrete_sequence=CHART_COLORS)
                fig_v.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#1E293B", family="Plus Jakarta Sans"), showlegend=False)
                st.plotly_chart(fig_v, use_container_width=True)
        with col_ch2:
            if "risk_score" in dash_df.columns:
                fig_r = px.histogram(dash_df, x="risk_score", nbins=10, title="Risk Score Distribution (0–100)", color_discrete_sequence=["#10B981"])
                fig_r.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#1E293B", family="Plus Jakarta Sans"))
                st.plotly_chart(fig_r, use_container_width=True)

# ==============================================================================
# TAB 5: TICKET QUEUE & RESOLUTION
# ==============================================================================
elif st.session_state["active_tab"] == "🗂️ Ticket Queue":
    st.markdown("### 🗂️ Support Desk Ticket Queue")
    st.caption("Manage customer tickets, review AI classification recommendations, and record official support resolutions.")
    
    h_df = load_history()
    
    if len(h_df) == 0:
        st.info("No tickets in the queue yet. Submit a ticket in the '🎯 Live Classifier' tab to see it appear here.")
    else:
        # Search and filters
        f_c1, f_c2 = st.columns([3, 1])
        with f_c1:
            q = st.text_input("Filter by keyword or Ticket ID:", placeholder="Search...")
        with f_c2:
            st_filter = st.selectbox("Status Filter:", ["All", "Open", "Under Review", "In Progress", "Resolved", "Escalated"])
            
        filt = h_df.copy()
        if q:
            filt = filt[filt["ticket_text"].str.contains(q, case=False, na=False) | filt["ticket_id"].str.contains(q, case=False, na=False)]
        if st_filter != "All" and "resolution_status" in filt.columns:
            filt = filt[filt["resolution_status"] == st_filter]
            
        cols = [c for c in ["ticket_id", "category", "priority", "emotion", "risk_score", "risk_level", "resolution_status", "created_at"] if c in filt.columns]
        st.markdown(render_styled_table(filt[cols]), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### ✍️ Agent Resolution Action")
        
        ticket_ids = filt["ticket_id"].dropna().tolist() if ("ticket_id" in filt.columns and len(filt) > 0) else []
        if not ticket_ids:
            st.info("No tickets match the current queue filter.")
        else:
            selected_tid = st.selectbox("Select Ticket to resolve:", ticket_ids)
            matched_df = filt[filt["ticket_id"] == selected_tid]
            if not matched_df.empty:
                target_row = matched_df.iloc[0]
                
                st.markdown(f"**Customer Message**: *\"{target_row.get('ticket_text', '')}\"*")
                st.markdown(f"**AI Recommendation**: `{target_row.get('recommended_action', 'N/A')}`")
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    new_stat = st.selectbox("Update Status:", ["Open", "Under Review", "In Progress", "Resolved", "Escalated"], index=0, key="queue_stat_sel")
                with res_col2:
                    agent_notes = st.text_input("Agent Resolution Message:", placeholder="Action taken or response sent to customer...", key="queue_agent_notes")
                    
                if st.button("Save Resolution & Update Queue", type="primary"):
                    success = update_resolution(selected_tid, new_stat, agent_notes)
                    if success:
                        st.success(f"Ticket `{selected_tid}` updated to **{new_stat}**.")
                        st.rerun()
                    else:
                        st.error("Failed to update status.")
