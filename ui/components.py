"""
ChurnPredictor AI - Premium Design System & UI Components.
High-aesthetic components with glassmorphism and modern fintech styling.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict, Tuple, Any


def create_premium_css() -> str:
    """Inject premium CSS tokens and glassmorphism styling."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --glass-bg: rgba(17, 25, 40, 0.75);
        --glass-border: rgba(255, 255, 255, 0.125);
        --accent-primary: #00D2FF;
        --accent-secondary: #3A7BD5;
        --text-primary: #FFFFFF;
        --text-secondary: #94A3B8;
        --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .hero-title { font-family: 'Outfit', sans-serif; }

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: var(--text-primary);
    }

    /* Hide Streamlit elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Glass Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 210, 255, 0.5);
        transform: translateY(-4px);
    }

    .metric-title {
        color: var(--text-secondary);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFFFFF, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Premium Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.1), rgba(58, 123, 213, 0.1));
        padding: 60px 40px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 40px;
        backdrop-filter: blur(20px);
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #00D2FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
    }

    .hero-sub {
        color: var(--text-secondary);
        font-size: 1.25rem;
        margin-top: 10px;
        font-weight: 400;
    }

    /* Section Headers */
    .section-header {
        border-left: 4px solid var(--accent-primary);
        padding-left: 15px;
        margin: 40px 0 20px 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }

    /* Custom Form Styling */
    .stForm {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    /* Modern Buttons */
    div[data-testid="stFormSubmitButton"] > button, .main-btn {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        font-size: 1rem !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.4) !important;
    }

    /* Badge */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* Sidebar Fixes */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 210, 255, 0.1) !important;
        color: #00D2FF !important;
    }

    /* AI Reasoning Box */
    .ai-reasoning-container {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 16px;
        padding: 25px;
        font-family: 'Inter', sans-serif;
        line-height: 1.8;
        color: #e2e8f0;
        position: relative;
        overflow: hidden;
    }
    .ai-reasoning-container::before {
        content: "🤖 AI INSIGHT";
        position: absolute;
        top: 0;
        right: 0;
        background: var(--accent-primary);
        color: #0f172a;
        padding: 4px 12px;
        font-size: 0.65rem;
        font-weight: 800;
        border-bottom-left-radius: 12px;
    }
    </style>
    """


def render_premium_header() -> None:
    """Render the high-aesthetic glassmorphism hero banner."""
    st.markdown("""
        <div class="hero-banner">
            <h1 class="hero-title">ChurnPredictor AI</h1>
            <p class="hero-sub">Enterprise-Grade Customer Retention Intelligence Powered by Agentic AI</p>
            <div style="margin-top: 20px;">
                <span style="color: #00D2FF; font-weight: 700;">● RAG</span> &nbsp; 
                <span style="color: #94A3B8;">|</span> &nbsp;
                <span style="color: #92FE9D; font-weight: 700;">● LLM</span> &nbsp; 
                <span style="color: #94A3B8;">|</span> &nbsp;
                <span style="color: #FF4B4B; font-weight: 700;">● ML</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_glass_metric(label: str, value: str, subtext: str = "", color: str = "#00D2FF") -> None:
    """Render a modern glass card metric."""
    st.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">{label}</div>
            <div class="metric-value" style="background: linear-gradient(90deg, #FFFFFF, {color}); -webkit-background-clip: text;">{value}</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 10px;">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)


def render_premium_risk_gauge(probability: float) -> None:
    """Sleek minimalist risk gauge."""
    colors = ["#10B981", "#F59E0B", "#EF4444"]
    risk_idx = 0 if probability < 0.3 else 1 if probability < 0.6 else 2
    accent = colors[risk_idx]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'color': 'white', 'size': 50, 'family': 'Outfit'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': accent, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': "rgba(255,255,255,0.02)"}
            ],
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", font={'color': "#94A3B8", 'family': "Inter"}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_modern_risk_factors(factors: List[Dict[str, str]]) -> None:
    """Clean list of risk factors with modern iconography."""
    if not factors:
        st.info("System found no critical risk triggers.")
        return

    for f in factors:
        impact_color = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}.get(f["impact"], "#94A3B8")
        st.markdown(f"""
            <div style="display: flex; align-items: start; gap: 15px; margin-bottom: 20px; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 12px; border-left: 3px solid {impact_color};">
                <div style="font-size: 1.2rem;">⚠️</div>
                <div>
                    <div style="font-weight: 700; font-size: 0.95rem; color: white;">{f['factor']}</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">{f['reason']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_feature_importance_card(features: list) -> None:
    """Modern feature importance bar chart."""
    if not features: return
    
    names = [f[0] for f in features]
    vals = [f[1] for f in features]
    
    fig = px.bar(
        x=vals, y=names, orientation='h',
        color=vals, color_continuous_scale='Blues',
    )
    fig.update_layout(
        height=300, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, coloraxis_showscale=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#94A3B8", size=10))
    )
    fig.update_traces(marker_line_width=0, opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)


def render_ai_reasoning_panel(content: str) -> None:
    """Premium AI reasoning component."""
    st.markdown(f"""
        <div class="ai-reasoning-container">
            {content}
        </div>
    """, unsafe_allow_html=True)


def render_customer_input_form_premium() -> Dict[str, Any]:
    """Glassmorphism form layout."""
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form("premium_input_form"):
            st.markdown("### 📋 Customer Profile Input")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("CORE DETAILS")
                tenure = st.number_input("Tenure (mos)", 0, 72, 12)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                monthly = st.number_input("Monthly Bill ($)", 0.0, 200.0, 65.0)
            
            with c2:
                st.caption("SERVICE CONFIG")
                internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming = st.selectbox("Streaming", ["No", "Yes", "No internet service"])

            with c3:
                st.caption("ACCOUNT SETUP")
                billing = st.selectbox("Paperless", ["Yes", "No"])
                payment = st.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
                senior = st.selectbox("Senior Citizen", ["No", "Yes"])

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 INITIATE ANALYSIS")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        return {
            "tenure": tenure, "Contract": contract, "MonthlyCharges": monthly,
            "InternetService": internet, "TechSupport": tech_support, 
            "StreamingTV": streaming, "PaperlessBilling": billing, 
            "PaymentMethod": payment, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "submitted": True, "TotalCharges": tenure * monthly,
            "gender": "Male", "Partner": "No", "Dependents": "No", 
            "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "StreamingMovies": "No"
        }
    return {"submitted": False}
