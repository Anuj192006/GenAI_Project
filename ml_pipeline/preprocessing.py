"""
ML Data Preprocessing Pipeline.
Handles data loading, cleaning, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Tuple, Any, List
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing for churn prediction.
    
    Responsibilities:
    - Load and clean data
    - Handle missing values
    - Encode categorical variables
    - Scale numeric variables
    """
    
    def __init__(self, categorical_features: List[str], 
                 numeric_features: List[str]):
        """
        Initialize preprocessor.
        
        Args:
            categorical_features: List of categorical column names
            numeric_features: List of numeric column names
        """
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: List[str] = []  # Store feature names in training order
        self.is_fitted = False
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and perform initial cleaning.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Drop customer ID as it's not a feature
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: handle missing values, convert types.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Handle TotalCharges - convert to numeric and fill NaN
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(
                df["TotalCharges"].mean()
            )
        
        # SeniorCitizen is stored as int (0/1) in training data.
        # Guard against "Yes"/"No" strings coming from the UI form.
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0})
            df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        
        # Handle Churn column
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def fit_encoders(self, df: pd.DataFrame) -> None:
        """
        Fit label encoders on categorical features.
        
        Args:
            df: Training DataFrame
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col])
                self.encoders[col] = le
                logger.info(f"Fitted encoder for {col} with {len(le.classes_)} classes")
        
        self.is_fitted = True
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using fitted encoders.
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded features
        """
        if not self.is_fitted:
            raise ValueError("Encoders not fitted. Call fit_encoders() first.")
        
        df = df.copy()
        
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError as e:
                    logger.warning(f"Could not encode {col}: {e}")
        
        return df
    
    def fit_scaler(self, X: pd.DataFrame) -> None:
        """
        Fit standard scaler on features.
        
        Args:
            X: Feature DataFrame
        """
        self.scaler.fit(X)
        logger.info("Scaler fitted")
    
    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Scale features using fitted scaler.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Scaled feature array
        """
        return self.scaler.transform(X)
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = "Churn"
                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline: clean, encode, scale.
        
        Args:
            df: Raw DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X_scaled, y, feature_names)
        """
        logger.info("Starting fit_transform pipeline")
        
        # Clean
        df = self.clean_data(df)
        
        # Encode
        self.fit_encoders(df)
        df = self.encode_features(df)
        
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        feature_names = X.columns.tolist()
        self.feature_names = feature_names  # Store for later use
        
        # Scale
        self.fit_scaler(X)
        X_scaled = self.scale_features(X)
        
        logger.info(f"Pipeline complete. Features: {len(feature_names)}")
        return X_scaled, y, feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Scaled feature array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        df = df.copy()
        df = self.clean_data(df)
        df = self.encode_features(df)
        
        # Ensure columns are in the correct order
        if self.feature_names:
            # Add missing columns with default values
            for col in self.feature_names:
                if col not in df.columns:
                    if col in self.categorical_features:
                        df[col] = 0  # Default for categorical
                    else:
                        df[col] = 0.0  # Default for numeric
            
            # Reorder columns to match training order
            df = df[self.feature_names]
        
        X_scaled = self.scale_features(df)
        return X_scaled
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get preprocessor state for serialization.
        
        Returns:
            Dictionary containing encoders and scaler
        """
        return {
            "encoders": self.encoders,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "feature_names": self.feature_names,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load preprocessor state from saved state.
        
        Args:
            state: State dictionary
        """
        self.encoders = state["encoders"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        self.categorical_features = state.get("categorical_features", [])
        self.numeric_features = state.get("numeric_features", [])
        self.feature_names = state.get("feature_names", [])
        logger.info("Preprocessor state loaded")
