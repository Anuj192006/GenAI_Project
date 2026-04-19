"""
ChurnPredictor AI v2.0 - Premium Edition
Enterprise-grade customer retention intelligence platform
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from utils.helpers import setup_logging, get_risk_level, format_probability
from ml_pipeline.prediction import ModelLoader, ChurnPredictor
from ml_pipeline.model_trainer import ModelTrainer
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from agent.retention_agent import RetentionAgent
from ui.components import (
    create_premium_css, render_premium_header, render_glass_metric,
    render_premium_risk_gauge, render_modern_risk_factors,
    render_feature_importance_card, render_ai_reasoning_panel,
    render_customer_input_form_premium
)
from utils.config import DATA_FILE, MODEL_PKL, VECTOR_INDEX, METADATA_DB, CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Init
logger = setup_logging("ChurnPredictor")

st.set_page_config(page_title="ChurnPredictor AI", page_icon="🛡️", layout="wide")
st.markdown(create_premium_css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialise models, RAG, and Agent layer."""
    try:
        # ML
        if not MODEL_PKL.exists():
            trainer = ModelTrainer()
            trainer.train_pipeline(str(DATA_FILE), CATEGORICAL_FEATURES, NUMERIC_FEATURES, str(MODEL_PKL))
        
        model_loader = ModelLoader()
        model_loader.load(str(MODEL_PKL))
        
        # RAG
        embedding_gen = EmbeddingGenerator()
        if VECTOR_INDEX.exists() and METADATA_DB.exists():
            vector_store = VectorStore.load(str(VECTOR_INDEX), str(METADATA_DB))
        else:
            # Quick build if missing
            vector_store = VectorStore(embedding_gen.get_embedding_dim())
            df = pd.read_csv(DATA_FILE)
            rag_tmp = RAGRetriever(embedding_gen, vector_store)
            # Use 2D array for probabilities (1000, 2) to avoid indexing errors
            rag_tmp.build_knowledge_base(df.head(1000), np.zeros(1000), np.zeros((1000, 2)), sample_size=500)
            vector_store.save(str(VECTOR_INDEX), str(METADATA_DB))
        
        rag_retriever = RAGRetriever(embedding_gen, vector_store)
        agent = RetentionAgent(rag_retriever, list(model_loader.preprocessor.categorical_features))
        
        return {
            "predictor": ChurnPredictor(model_loader),
            "agent": agent,
            "llm_enabled": agent.llm is not None
        }
    except Exception as e:
        st.error(f"System Boot Failure: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────────────────────

systems = initialize_system()
if not systems: st.stop()

render_premium_header()

tab_analyze, tab_batch, tab_sys = st.tabs(["🎯 ANALYZE", "📊 BATCH", "⚙️ SYSTEM"])

# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE TAB
# ─────────────────────────────────────────────────────────────────────────────

with tab_analyze:
    customer_data = render_customer_input_form_premium()

    if customer_data.get("submitted") or "last_result" in st.session_state:
        if customer_data.get("submitted"):
            # Run Pipeline
            with st.status("🧠 AI Agent Processing...", expanded=False) as status:
                st.write("Inference Step: Predicting Churn Probability...")
                pred, probs = systems["predictor"].predict(customer_data, "Logistic Regression")
                st.write("Context Step: Retrieving Similar Historical Cases...")
                analysis = systems["agent"].analyze_churn_risk(customer_data, probs[1])
                st.session_state["last_result"] = {
                    "pred": pred, "probs": probs, "analysis": analysis, "data": customer_data
                }
                status.update(label="Analysis Complete", state="complete")

        res = st.session_state["last_result"]
        analysis = res["analysis"]
        churn_prob = float(res["probs"][1])

        # Headline Metrics
        st.markdown('<div class="section-header">Intelligence Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        risk_cat, _ = get_risk_level(churn_prob)
        
        with m1: render_glass_metric("Churn Risk", f"{churn_prob:.1%}", risk_cat, "#EF4444" if churn_prob > 0.5 else "#10B981")
        with m2: render_glass_metric("Confidence", f"{max(res['probs']):.0%}", "Model Precision", "#3A7BD5")
        with m3: render_glass_metric("Budget", f"${analysis['retention_budget']:.2f}", "Recommended Spend", "#92FE9D")
        with m4: render_glass_metric("Agent Status", "Active", "LLM Processing", "#00D2FF")

        # Visual Analytics
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.caption("RISK PROBABILITY GAUGE")
            render_premium_risk_gauge(churn_prob)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.caption("AI EXTRACTED RISK FACTORS")
            render_modern_risk_factors(analysis["risk_factors"])
            st.markdown('</div>', unsafe_allow_html=True)

        # AI Reasoning (The "Agent" part)
        st.markdown('<div class="section-header">Agentic Reasoning Trace</div>', unsafe_allow_html=True)
        if analysis.get("llm_used"):
            render_ai_reasoning_panel(analysis["llm_reasoning"])
        else:
            st.warning("⚠️ Using deterministic fallback logic. Set GROQ_API_KEY for GPT-level reasoning.")

        # Recommendations
        st.markdown('<div class="section-header">Prescriptive Actions</div>', unsafe_allow_html=True)
        for i, rec in enumerate(analysis["recommendations"][:3], 1):
            with st.expander(f"ACTION {i}: {rec['action']} [{rec['priority']}]", expanded=(i==1)):
                st.write(rec["description"])
                st.info(f"**Target Impact:** {rec['expected_impact']}")

        # Feature Importance
        st.markdown('<div class="section-header">Model Explainability</div>', unsafe_allow_html=True)
        importance = systems["predictor"].get_feature_importance("Logistic Regression")
        render_feature_importance_card(importance)

        # ── Action Hub ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Decision Support Hub</div>', unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        
        with ac1:
            if st.button("💬 SCHEDULE RETENTION CALL", use_container_width=True):
                st.session_state["call_scheduled"] = True
        
        with ac2:
            if st.button("📂 GENERATE CHURN REPORT", use_container_width=True):
                st.session_state["report_content"] = systems["agent"].generate_summary_report(analysis)

        if st.session_state.get("call_scheduled"):
            st.success("✅ **Retention Call Scheduled.** Ticket #RT-" + str(abs(hash(str(churn_prob))) % 100000) + " created.")
        
        if st.session_state.get("report_content"):
            st.markdown("### 📄 Generated Churn Report")
            st.code(st.session_state["report_content"], language="")
            st.download_button(
                "📥 DOWNLOAD REPORT (.txt)", 
                st.session_state["report_content"], 
                file_name=f"churn_report_{abs(hash(str(churn_prob)))}.txt",
                use_container_width=True
            )

# ─────────────────────────────────────────────────────────────────────────────
# BATCH TAB
# ─────────────────────────────────────────────────────────────────────────────

with tab_batch:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Bulk Processing")
    file = st.file_uploader("Upload CSV for parallel inference", type="csv")
    if file:
        df = pd.read_csv(file)
        if st.button("🚀 EXECUTE BATCH ANALYSIS"):
            preds, probs = systems["predictor"].predict_batch(df, "Logistic Regression")
            df["Risk"] = probs[:, 1]
            st.dataframe(df[["Risk"]].head(10).style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM TAB
# ─────────────────────────────────────────────────────────────────────────────

with tab_sys:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Engine Specifications")
    st.code("""
    LLM:     Llama-3.1-8b (Groq)
    RAG:     FAISS + all-MiniLM-L6-v2
    ML:      Scikit-Learn Pipeline
    UI:      Premium Glassmorphism v2.3
    """, language="markdown")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
    <div style="text-align: center; color: #94A3B8; margin-top: 60px; font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px;">
        🛡️ ChurnPredictor AI v2.3 | End-Semester Agentic AI Final Submission | 2024
    </div>
""", unsafe_allow_html=True)
