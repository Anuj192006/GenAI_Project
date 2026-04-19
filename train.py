#!/usr/bin/env python3
"""
Standalone training script for ChurnPredictor AI models.
Run this to train or retrain the ML models and build RAG knowledge base.

Usage:
    python train.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    DATA_FILE, MODEL_PKL, VECTOR_INDEX, METADATA_DB,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES
)
from utils.helpers import setup_logging
from ml_pipeline.model_trainer import ModelTrainer
from ml_pipeline.preprocessing import DataPreprocessor
from ml_pipeline.prediction import ModelLoader, ChurnPredictor
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

logger = setup_logging("ChurnPredictor-Trainer")


def train_models():
    """Train ML models."""
    logger.info("=" * 70)
    logger.info("TRAINING ML MODELS")
    logger.info("=" * 70)
    
    trainer = ModelTrainer()
    
    result = trainer.train_pipeline(
        str(DATA_FILE),
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        str(MODEL_PKL)
    )
    
    logger.info(f"\n✅ Models trained successfully!")
    logger.info(f"   Features: {result['n_features']}")
    logger.info(f"\n📊 Model Metrics:")
    for model_name, metrics in result['metrics'].items():
        logger.info(f"\n   {model_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"     {metric_name}: {value:.4f}")
    
    return trainer


def build_rag_knowledge_base(trainer):
    """Build RAG knowledge base from training data."""
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING RAG KNOWLEDGE BASE")
    logger.info("=" * 70)
    
    try:
        # Initialize RAG components
        logger.info("Initializing embedding generator...")
        embedding_gen = EmbeddingGenerator()
        logger.info(f"✅ Embedding dimension: {embedding_gen.get_embedding_dim()}")
        
        logger.info("Initializing vector store...")
        vector_store = VectorStore(embedding_gen.get_embedding_dim())
        
        logger.info("Initializing RAG retriever...")
        rag_retriever = RAGRetriever(embedding_gen, vector_store)
        
        # Load training data
        logger.info(f"Loading training data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        
        # Load models for predictions
        logger.info("Loading trained models...")
        model_loader = ModelLoader()
        model_loader.load(str(MODEL_PKL))
        predictor = ChurnPredictor(model_loader)
        
        # Get predictions for knowledge base
        logger.info("Generating predictions for knowledge base...")
        X_pred = df.drop(columns=["customerID", "Churn"] 
                         if "Churn" in df.columns else ["customerID"])
        predictions, probabilities = predictor.predict_batch(
            X_pred, "Logistic Regression"
        )
        
        # Build knowledge base (sample for performance)
        logger.info(f"Building knowledge base (sampling {len(df)} records)...")
        rag_retriever.build_knowledge_base(
            df,
            predictions,
            probabilities,
            sample_size=1000
        )
        
        # Save vector store
        logger.info("Saving vector store...")
        VECTOR_INDEX.parent.mkdir(parents=True, exist_ok=True)
        METADATA_DB.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save(str(VECTOR_INDEX), str(METADATA_DB))
        
        logger.info(f"✅ RAG knowledge base built!")
        logger.info(f"   Vector index: {VECTOR_INDEX}")
        logger.info(f"   Metadata DB: {METADATA_DB}")
        logger.info(f"   Total cases: {vector_store.size}")
        
    except Exception as e:
        logger.error(f"Error building RAG knowledge base: {e}", exc_info=True)
        raise


def main():
    """Main training pipeline."""
    
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "  🛡️  ChurnPredictor AI - Model Training Pipeline  ".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    logger.info("\n")
    
    try:
        # Check data file
        if not DATA_FILE.exists():
            logger.error(f"❌ Data file not found: {DATA_FILE}")
            logger.info(f"Please ensure {DATA_FILE} exists before training.")
            return False
        
        logger.info(f"✅ Data file found: {DATA_FILE}")
        
        # Train models
        trainer = train_models()
        
        # Build RAG KB
        build_rag_knowledge_base(trainer)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TRAINING COMPLETE!")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("1. Run Streamlit app: streamlit run ui/streamlit_app.py")
        logger.info("2. Open browser: http://localhost:8501")
        logger.info("\n")
        
        return True
    
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
