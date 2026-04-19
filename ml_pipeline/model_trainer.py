"""
Model Training Pipeline.
Trains and saves Logistic Regression and Decision Tree models.
"""

import pandas as pd
import pickle
import logging
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from .preprocessing import DataPreprocessor
from utils.config import MODEL_CONFIG, MODEL_PKL, DATA_FILE

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and manages ML models for churn prediction.
    
    Models:
    - Logistic Regression
    - Decision Tree
    """
    
    def __init__(self):
        """Initialize model trainer."""
        self.preprocessor = None
        self.models = {}
        self.metrics = {}
        self.splits = {}
    
    def load_and_preprocess(self, filepath: str, 
                           categorical_features: list,
                           numeric_features: list) -> Tuple[Any, Any, list]:
        """
        Load and preprocess data.
        
        Args:
            filepath: Path to CSV file
            categorical_features: List of categorical feature names
            numeric_features: List of numeric feature names
            
        Returns:
            Tuple of (X_scaled, y, feature_names)
        """
        logger.info("Loading and preprocessing data")
        
        self.preprocessor = DataPreprocessor(categorical_features, numeric_features)
        df = self.preprocessor.load_data(filepath)
        X_scaled, y, feature_names = self.preprocessor.fit_transform(df)
        
        return X_scaled, y, feature_names
    
    def split_data(self, X: Any, y: Any
                  ) -> Tuple[Any, Any, Any, Any]:
        """
        Split data into train/test sets.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=MODEL_CONFIG["test_size"],
            random_state=MODEL_CONFIG["random_state"]
        )
        
        self.splits = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: Any, y_train: Any) -> None:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training Logistic Regression")
        
        lr_config = MODEL_CONFIG["logistic_regression"]
        lr = LogisticRegression(**lr_config)
        lr.fit(X_train, y_train)
        
        self.models["Logistic Regression"] = lr
        logger.info("Logistic Regression trained")
    
    def train_decision_tree(self, X_train: Any, y_train: Any) -> None:
        """
        Train Decision Tree model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training Decision Tree")
        
        dt_config = MODEL_CONFIG["decision_tree"]
        dt = DecisionTreeClassifier(**dt_config)
        dt.fit(X_train, y_train)
        
        self.models["Decision Tree"] = dt
        logger.info("Decision Tree trained")
    
    def evaluate_models(self, X_test: Any, y_test: Any) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of metrics for each model
        """
        logger.info("Evaluating models")
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            metrics = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred), 4),
            }
            
            self.metrics[model_name] = metrics
            
            logger.info(f"\n{model_name} Results:")
            logger.info(f"  Accuracy:  {metrics['Accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['Precision']:.4f}")
            logger.info(f"  Recall:    {metrics['Recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['F1 Score']:.4f}")
        
        return self.metrics
    
    def save_models(self, output_path: str) -> None:
        """
        Save models and preprocessor to disk.
        
        Args:
            output_path: Path to save pickle file
        """
        logger.info(f"Saving models to {output_path}")
        
        model_data = {
            "models": self.models,
            "preprocessor": self.preprocessor,
            "metrics": self.metrics,
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info("Models saved successfully")
    
    def train_pipeline(self, data_path: str,
                      categorical_features: list,
                      numeric_features: list,
                      output_path: str) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            data_path: Path to input CSV
            categorical_features: List of categorical features
            numeric_features: List of numeric features
            output_path: Path to save models
            
        Returns:
            Dictionary with results
        """
        logger.info("Starting full training pipeline")
        
        # Load & preprocess
        X, y, feature_names = self.load_and_preprocess(
            data_path, categorical_features, numeric_features
        )
        
        # Split
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train
        self.train_logistic_regression(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_models(X_test, y_test)
        
        # Save
        self.save_models(output_path)
        
        return {
            "feature_names": feature_names,
            "metrics": metrics,
            "n_features": len(feature_names),
        }
