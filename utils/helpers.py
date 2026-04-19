"""
Helper functions for ChurnPredictor AI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def setup_logging(name: str = "ChurnPredictor") -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)


def validate_customer_data(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate customer input data.
    
    Args:
        data: Customer data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate numeric fields
    if data["tenure"] < 0 or data["tenure"] > 72:
        return False, "Tenure must be between 0 and 72 months"
    
    if data["MonthlyCharges"] < 0:
        return False, "Monthly charges cannot be negative"
    
    if data["TotalCharges"] < 0:
        return False, "Total charges cannot be negative"
    
    return True, ""


def get_risk_level(probability: float) -> Tuple[str, str]:
    """
    Determine risk level from churn probability.
    
    Args:
        probability: Churn probability (0-1)
        
    Returns:
        Tuple of (risk_level, emoji)
    """
    if probability < 0.3:
        return "LOW", "🟢"
    elif probability < 0.6:
        return "MEDIUM", "🟡"
    else:
        return "HIGH", "🔴"


def format_probability(prob: float) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.2f}%"


def extract_feature_importance(
    model: Any, feature_names: List[str], top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Extract feature importance from model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        List of (feature_name, importance) tuples sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        if hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importance = np.abs(model.coef_[0])
        else:
            return []
    else:
        importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return list(zip(
        importance_df['feature'].head(top_n),
        importance_df['importance'].head(top_n)
    ))


def sample_to_dict(row: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
    """Convert DataFrame row to dictionary."""
    return {col: row[col] for col in feature_names}


def dict_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert dictionary to single-row DataFrame."""
    return pd.DataFrame([data])


def format_retention_budget(churn_prob: float, monthly_charge: float, 
                           retention_budget_pct: float = 0.15) -> float:
    """
    Calculate suggested retention budget for a customer.
    
    Args:
        churn_prob: Churn probability
        monthly_charge: Monthly charges
        retention_budget_pct: Budget as percentage of monthly charge
        
    Returns:
        Suggested budget in dollars
    """
    # Higher churn prob = higher budget allocation
    return monthly_charge * retention_budget_pct * (churn_prob + 0.5)


def create_churn_segments(probabilities: np.ndarray) -> Dict[str, int]:
    """
    Segment customers by churn risk.
    
    Args:
        probabilities: Array of churn probabilities
        
    Returns:
        Dictionary with segment counts
    """
    low = np.sum(probabilities < 0.3)
    medium = np.sum((probabilities >= 0.3) & (probabilities < 0.6))
    high = np.sum(probabilities >= 0.6)
    
    return {
        "Low Risk": int(low),
        "Medium Risk": int(medium),
        "High Risk": int(high),
    }
