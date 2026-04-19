"""
Configuration management for ChurnPredictor AI.
Handles environment variables, model paths, and system parameters.
"""

import os
from pathlib import Path
from typing import Optional

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAG_DIR = PROJECT_ROOT / "rag"

# Data paths
DATA_FILE = DATA_DIR / "telco_churn.csv"
MODEL_PKL = MODELS_DIR / "churn_model.pkl"
VECTOR_INDEX = RAG_DIR / "vector_index.faiss"
METADATA_DB = RAG_DIR / "metadata.pkl"

# ─── Model Configuration ─────────────────────────────────────────────────────
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
    "decision_tree": {"random_state": 42, "max_depth": 15},
}

# ─── RAG Configuration ───────────────────────────────────────────────────────
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_threshold": 0.5,
    "top_k": 5,
    "batch_size": 32,
}

# ─── LLM Configuration ───────────────────────────────────────────────────────
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),  # ollama, openai, anthropic
    "model": os.getenv("LLM_MODEL", "mistral"),
    "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434"),
    "api_key": os.getenv("LLM_API_KEY", ""),
    "temperature": 0.7,
    "max_tokens": 1024,
}

# ─── Churn Risk Thresholds ──────────────────────────────────────────────────
CHURN_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0,
}

# ─── Feature Names (Must match training data) ────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# ─── Caching Configuration ──────────────────────────────────────────────────
CACHE_CONFIG = {
    "model_cache_ttl": 3600,  # 1 hour
    "embeddings_cache_ttl": 3600,
    "vector_index_cache_ttl": 3600,
}


def ensure_paths_exist():
    """Create necessary directories if they don't exist."""
    for path in [DATA_DIR, MODELS_DIR, RAG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


ensure_paths_exist()
