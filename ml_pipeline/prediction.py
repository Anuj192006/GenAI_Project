"""
Model Loading and Prediction Inference.
Loads saved models and performs predictions on new data.
"""

import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and manages pre-trained models from disk.
    """
    
    def __init__(self):
        """Initialize model loader."""
        self.models = {}
        self.preprocessor = None
        self.metrics = {}
        self.is_loaded = False
    
    def load(self, model_path: str) -> bool:
        """
        Load models from pickle file.
        
        Args:
            model_path: Path to saved model pickle file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading models from {model_path}")
            
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get("models", {})
            self.preprocessor = model_data.get("preprocessor", None)
            self.metrics = model_data.get("metrics", {})
            
            # Handle single model case (legacy format)
            if not self.models and "model" in model_data:
                self.models = {"Logistic Regression": model_data["model"]}
                if "scaler" in model_data:
                    self.scaler = model_data["scaler"]
                if "encoders" in model_data:
                    self.encoders = model_data["encoders"]
                if "feature_names" in model_data:
                    self.feature_names = model_data["feature_names"]
                
                # Create a preprocessor from components
                if hasattr(self, 'encoders') and hasattr(self, 'scaler') and hasattr(self, 'feature_names'):
                    from .preprocessing import DataPreprocessor
                    # We need to infer categorical/numeric features from encoders
                    categorical_features = list(self.encoders.keys())
                    # Assume remaining features are numeric
                    all_features = set(self.feature_names)
                    encoded_features = set()
                    for encoder in self.encoders.values():
                        if hasattr(encoder, 'classes_'):
                            encoded_features.update(encoder.classes_)
                    numeric_features = [f for f in all_features if f not in categorical_features]
                    
                    self.preprocessor = DataPreprocessor(categorical_features, numeric_features)
                    self.preprocessor.encoders = self.encoders
                    self.preprocessor.scaler = self.scaler
                    self.preprocessor.feature_names = self.feature_names
                    self.preprocessor.is_fitted = True
            
            self.is_loaded = True
            logger.info(f"Loaded {len(self.models)} models")
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get specific model by name.
        
        Args:
            model_name: Name of model (e.g., "Logistic Regression")
            
        Returns:
            Model object or None if not found
        """
        if not self.is_loaded:
            logger.warning("Models not loaded")
            return None
        
        return self.models.get(model_name, None)
    
    def get_available_models(self) -> list:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """Get metrics for a specific model."""
        return self.metrics.get(model_name, {})


class ChurnPredictor:
    """
    Makes churn predictions on customer data.
    """
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize predictor.
        
        Args:
            model_loader: Loaded ModelLoader instance
        """
        self.model_loader = model_loader
        if not model_loader.is_loaded:
            raise ValueError("ModelLoader not loaded with valid models")
    
    def predict(self, customer_data: Dict[str, Any],
               model_name: str = "Logistic Regression"
               ) -> Tuple[int, np.ndarray]:
        """
        Make churn prediction for a customer.
        
        Args:
            customer_data: Customer information dictionary
            model_name: Model to use for prediction
            
        Returns:
            Tuple of (prediction, probabilities)
            prediction: 0 (no churn) or 1 (churn)
            probabilities: Array of [prob_no_churn, prob_churn]
        """
        try:
            # Get model
            model = self.model_loader.get_model(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
            
            # Preprocess
            X_scaled = self._preprocess_input(customer_data)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            logger.info(f"Prediction for {model_name}: {prediction} "
                       f"(confidence: {max(probabilities):.2%})")
            
            return int(prediction), probabilities
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_batch(self, customers_df: pd.DataFrame,
                     model_name: str = "Logistic Regression"
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for multiple customers.
        
        Args:
            customers_df: DataFrame with customer data
            model_name: Model to use for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            model = self.model_loader.get_model(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
            
            X_scaled = self._preprocess_batch(customers_df)
            
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
            logger.info(f"Batch prediction complete: {len(predictions)} records")
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise
    
    def _preprocess_input(self, customer_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess single customer input.
        
        Args:
            customer_data: Customer dictionary
            
        Returns:
            Scaled feature array
        """
        preprocessor = self.model_loader.preprocessor
        if preprocessor is None:
            raise ValueError("Preprocessor not available")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Transform using preprocessor
        X_scaled = preprocessor.transform(df)
        
        return X_scaled
    
    def _preprocess_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess batch of customers.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            Scaled feature array
        """
        preprocessor = self.model_loader.preprocessor
        if preprocessor is None:
            raise ValueError("Preprocessor not available")
        
        X_scaled = preprocessor.transform(df)
        return X_scaled
    
    def get_feature_importance(self, model_name: str = "Logistic Regression",
                              top_n: int = 10) -> list:
        """
        Get feature importance for a model.
        
        Args:
            model_name: Model name
            top_n: Number of top features
            
        Returns:
            List of (feature_name, importance) tuples
        """
        model = self.model_loader.get_model(model_name)
        if model is None:
            return []
        
        preprocessor = self.model_loader.preprocessor
        if preprocessor is None:
            return []
        
        if hasattr(model, 'coef_'):
            # Linear model
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            # Tree model
            importance = model.feature_importances_
        else:
            return []
        
        # Get feature names from preprocessor
        feature_names = preprocessor.feature_names if preprocessor.feature_names else [f"Feature {i}" for i in range(len(importance))]
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1][:top_n]
        
        return [
            (str(feature_names[i]), float(importance[i]))
            for i in sorted_indices
        ]
