"""
Embedding Generation for RAG System.
Uses sentence-transformers for semantic embeddings.
"""

import numpy as np
from typing import List, Union, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import json

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates semantic embeddings for churn cases and queries.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Hugging Face model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode single text string to embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, 
                                       batch_size=32, show_progress_bar=True)
        return embeddings
    
    def encode_customer_profile(self, customer: Dict[str, Any]) -> np.ndarray:
        """
        Encode customer profile to embedding.
        
        Creates a text representation of customer data and encodes it.
        
        Args:
            customer: Customer information dictionary
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Create text representation
        text_parts = []
        
        # Demographics
        for key in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
            if key in customer:
                text_parts.append(f"{key}: {customer[key]}")
        
        # Services
        services = ["PhoneService", "InternetService", "OnlineSecurity", 
                   "TechSupport", "StreamingTV", "StreamingMovies"]
        service_str = ", ".join([str(customer.get(s, "No")) 
                                 for s in services if s in customer])
        if service_str:
            text_parts.append(f"Services: {service_str}")
        
        # Contract & Billing
        for key in ["Contract", "PaymentMethod"]:
            if key in customer:
                text_parts.append(f"{key}: {customer[key]}")
        
        # Charges
        if "MonthlyCharges" in customer:
            try:
                monthly = float(customer['MonthlyCharges'])
                text_parts.append(f"Monthly charges: ${monthly:.2f}")
            except (ValueError, TypeError):
                text_parts.append(f"Monthly charges: {customer['MonthlyCharges']}")
        if "TotalCharges" in customer:
            try:
                total = float(customer['TotalCharges'])
                text_parts.append(f"Total charges: ${total:.2f}")
            except (ValueError, TypeError):
                text_parts.append(f"Total charges: {customer['TotalCharges']}")
        if "tenure" in customer:
            text_parts.append(f"Tenure: {customer['tenure']} months")
        
        profile_text = ". ".join(text_parts)
        
        return self.encode_text(profile_text)
    
    def encode_case_summary(self, case: Dict[str, Any]) -> np.ndarray:
        """
        Encode customer churn case to embedding.
        
        Args:
            case: Case dictionary with customer data and outcome
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Extract case information
        text_parts = []
        
        if "customer_features" in case:
            features = case["customer_features"]
            
            # Key demographics
            for key in ["gender", "Contract", "InternetService", "TechSupport"]:
                if key in features:
                    text_parts.append(f"{key} is {features[key]}")
            
            # Key metrics
            if "tenure" in features:
                text_parts.append(f"tenure {features['tenure']} months")
            if "MonthlyCharges" in features:
                text_parts.append(f"monthly charges ${features['MonthlyCharges']:.2f}")
        
        if "churn_outcome" in case:
            text_parts.append(f"Result: {'churned' if case['churn_outcome'] else 'retained'}")
        
        if "retention_strategies" in case and case.get("churn_outcome") == True:
            strategy = case["retention_strategies"][0] if case["retention_strategies"] else ""
            text_parts.append(f"Strategy: {strategy}")
        
        case_text = ". ".join(text_parts)
        
        return self.encode_text(case_text)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def similarity_score(self, embedding1: np.ndarray, 
                        embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(e1, e2)
        return float(similarity)
